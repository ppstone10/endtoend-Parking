"""正常感知/推理下保持带严格到达证书的同一网络参考对象。"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import numpy as np

from experiments import validation_matrix as matrix
from training.reporting import atomic_write_json
from verification.terminal_capture import terminal_scope
from verification.reference_certificate import certify_reference


def array_digest(points):
    array = np.asarray(points)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode()+array.tobytes()).hexdigest()


class CertifiedReferenceSource:
    def __init__(self, primary, engine, budget):
        self.primary, self.engine, self.budget = primary, engine, budget

    def begin(self, start, goal):
        self.primary.begin(start, goal)
        self.step = 0
        self.committed = None
        self.certificate = None
        self.certificates, self.queries, self.raw_references = [], [], {}
        self.attempts = 0

    def next_trajectory(self, state):
        self.budget.check()
        raw, elapsed = self.primary.next_trajectory(state)
        key = f'raw_{len(self.queries):04d}'
        self.raw_references[key] = np.asarray(raw.points).copy()
        holding = self.committed is not None
        if holding:
            self._check_expected_state(state)
        else:
            terminal, goal = self.engine.terminal, self.engine.mpc.goal
            points = np.asarray(raw.points)
            pos = np.linalg.norm(points[:, :2]-[goal.x, goal.y], axis=1)
            yaw = np.abs(np.arctan2(np.sin(points[:, 2]-goal.yaw), np.cos(points[:, 2]-goal.yaw)))
            if (terminal.pos_err(state, goal) <= 3*terminal.tol_pos
                    and np.any((pos < terminal.tol_pos)&(yaw < terminal.tol_yaw))
                    and self.engine.margin_checker.pose_free(state.x, state.y, state.yaw)):
                self.attempts += 1
                certificate = certify_reference(self.engine.mpc, raw, state, self.engine.collision_checker,
                                                self.engine.margin_checker, min(80, self.engine.max_steps-self.step),
                                                self.budget.check)
                if certificate is not None:
                    certificate.update(start_step=self.step, raw_key=key, reference_sha256=array_digest(raw.points))
                    self.certificates.append(certificate)
                    self.certificate, self.committed = certificate, raw
        selected = self.committed if self.committed is not None else raw
        self.queries.append({'step': self.step, 'raw_key': key, 'raw_sha256': array_digest(raw.points),
                             'used_sha256': array_digest(selected.points), 'raw_dt': raw.dt,
                             'holding_previous': holding,
                             'certificate': len(self.certificates)-1 if self.certificate is not None else None})
        return selected, elapsed

    def _check_expected_state(self, state):
        offset = self.step-self.certificate['start_step']
        if offset >= len(self.certificate['controls']):
            raise RuntimeError('到达证书已耗尽而引擎仍未结束')
        if not np.array_equal(state.to_array(), self.certificate['states_before'][offset]):
            raise RuntimeError('实际前态偏离确定性到达证书，停止实验')

    def observe_command(self, compute, trajectory, state):
        if self.certificate is not None:
            self._check_expected_state(state)
        command = compute(trajectory, state)
        if self.certificate is not None:
            offset = self.step-self.certificate['start_step']
            if not np.array_equal([command.v, command.omega], self.certificate['controls'][offset]):
                raise RuntimeError('实际控制偏离私有副本证书')
            predicted = self.engine.vehicle_model.step(state, command, self.engine.mpc.dt)
            if not np.array_equal(predicted.to_array(), self.certificate['states_after'][offset]):
                raise RuntimeError('实际预测后态偏离证书')
        self.step += 1
        return command


@contextmanager
def retention_scope(output, vehicle, expected, budget):
    original = matrix.ClosedLoopEngine
    ledgers = []

    class RetentionEngine(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.source = CertifiedReferenceSource(self.source, self, budget)

        def run(self, start, goal):
            compute = self.mpc.compute
            self.mpc.compute = lambda trajectory, state: self.source.observe_command(compute, trajectory, state)
            try:
                result = super().run(start, goal)
            finally:
                self.mpc.compute = compute
            ledger = {'dataset_index': self.meta['dataset_index'], 'steps': self.source.step,
                      'attempts': self.source.attempts, 'queries': self.source.queries,
                      'certificates': self.source.certificates,
                      'held_queries': sum(q['holding_previous'] for q in self.source.queries)}
            index = self.meta['dataset_index']
            atomic_write_json(Path(output)/f'retention-{index:04d}.json', ledger)
            np.savez_compressed(Path(output)/f'retention-{index:04d}.npz', **self.source.raw_references)
            ledgers.append(ledger)
            return result

    matrix.ClosedLoopEngine = RetentionEngine
    try:
        with terminal_scope(output, vehicle, expected, budget) as rows:
            yield rows, ledgers
    finally:
        matrix.ClosedLoopEngine = original
