"""固定回归的实际运行和只读审计；LT32只作冻结缓存对照。"""
from pathlib import Path
import json
import time
import numpy as np
import torch

from dataset import DatasetGenerator
from experiments import validation_matrix as matrix
from experiments.component_capture import VerificationBudget
from planner import RectangleFootprintCollisionChecker
from scripts.diagnose_closed_loop_failures import _sha256, first_blocked_index
from scripts.diagnose_reference_execution import code_identity
from scripts.prepare_recovery_balance_ab import runtime_identity
from sim import VehicleConfig
from training.reporting import atomic_write_json
from verification.strict_footprint_audit import audit_tape

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT/'runs/verification/strict-footprint-formal-20261028/results'
ARCHIVE = ROOT/'runs/prepared/safe-prefix-20261028/source-before'
REGRESSIONS = [80, 81, 196]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_baseline():
    config = read(BASELINE/'config.json')
    for filename, digest in config['code'].items():
        if _sha256(ARCHIVE/Path(filename).relative_to(ROOT)) != digest:
            raise ValueError('LT32源码快照身份漂移')
    for name, digest in read(BASELINE/'artifacts.json').items():
        path = (BASELINE/name).resolve()
        if not path.is_relative_to(BASELINE) or _sha256(path) != digest:
            raise ValueError('LT32产物身份漂移')
    for key in ('data', 'manifest', 'checkpoint', 'historical'):
        item = config['inputs'][key]
        if _sha256(Path(item['path'])) != item['sha256']:
            raise ValueError('输入身份漂移')
    return config, read(BASELINE/'report.json')


def execution_identity():
    return {**code_identity(), **{str(p.resolve()): _sha256(p) for p in
        (Path(__file__), ROOT/'verification/strict_footprint_audit.py', ROOT/'scripts/preflight_safe_prefix.py',
         ROOT/'verification/safe_prefix_handoff.py', ROOT/'scripts/validate_safe_prefix.py')}}


def select_indices(stage):
    if stage == 'regressions':
        return REGRESSIONS
    if stage != 'expanded':
        raise ValueError('此入口只允许固定短检，正式任务须另立计划')
    baseline = read(BASELINE/'strict/raw-report.json')
    failures = [r['meta']['dataset_index'] for r in baseline['episodes'] if not r['success']]
    first_successes = {}
    for row in baseline['episodes']:
        if row['success']:
            meta = row['meta']
            first_successes.setdefault((meta['scene_name'], meta['maneuver']), meta['dataset_index'])
    return sorted(set(failures) | set(first_successes.values()))


def audit_admissions(tape_path, decisions, checker, budget):
    modes = {}
    with np.load(tape_path, allow_pickle=False) as tape:
        if len(decisions) != int(tape['references']):
            raise ValueError('参考准入决策分母不完整')
        goal = tape['goal']
        for ordinal, decision in enumerate(decisions):
            budget.check()
            raw = tape[f'raw_{ordinal:04d}']
            emitted = tape[f'reference_{ordinal:04d}']
            state = tape[f'query_state_{ordinal:04d}']
            mode = decision['mode']
            modes[mode] = modes.get(mode, 0) + 1
            if mode in ('unchanged', 'safe_prefix'):
                if raw.dtype != emitted.dtype or not np.array_equal(raw[:len(emitted)], emitted):
                    raise ValueError('原样/前缀点值或dtype被改写')
                if decision['raw_dt'] != float(tape[f'reference_dt_{ordinal:04d}']):
                    raise ValueError('前缀dt被改写')
            if mode == 'unchanged' and len(raw) != len(emitted):
                raise ValueError('未修改参考长度不符')
            if mode == 'safe_prefix':
                blocked = first_blocked_index(checker, state, raw)
                if blocked != len(emitted) or blocked != decision['first_blocked_index']:
                    raise ValueError('不是首危险段前的严格原样前缀')
                if (len(emitted)-1)*float(tape[f'reference_dt_{ordinal:04d}']) < 1.:
                    raise ValueError('前缀未覆盖完整H10时间')
            pos = np.linalg.norm(state[:2]-goal[:2])
            yaw = abs(np.arctan2(np.sin(state[2]-goal[2]), np.cos(state[2]-goal[2])))
            tp, ty = float(tape['tol_pos']), float(tape['tol_yaw'])
            if pos <= 3*tp and not (pos < tp and yaw < ty):
                p = np.linalg.norm(emitted[:, :2]-goal[:2], axis=1)
                y = np.abs(np.arctan2(np.sin(emitted[:, 2]-goal[2]), np.cos(emitted[:, 2]-goal[2])))
                if not np.any((p < tp) & (y < ty)):
                    raise ValueError('近端发出参考没有原联合目标支撑')
    return modes


def summarize_candidate(rows, baseline, historical, *, formal=False):
    indices = [r['dataset_index'] for r in rows]
    if len(indices) != len(set(indices)):
        raise ValueError('任务重复')
    old = {r['dataset_index']: r for r in baseline['rows']['strict']}
    native = {r['dataset_index']: r for r in baseline['rows']['native']}
    historical_success = {r['meta']['dataset_index'] for r in historical['episodes'] if r['success']}
    lost = [r['dataset_index'] for r in rows if old[r['dataset_index']]['success'] and not r['success']]
    native_lost = [r['dataset_index'] for r in rows if native[r['dataset_index']]['success'] and not r['success']]
    history_lost = [r['dataset_index'] for r in rows if r['dataset_index'] in historical_success and not r['success']]
    gained = [r['dataset_index'] for r in rows if not old[r['dataset_index']]['success'] and r['success']]
    safe = all(not r['collision'] and all(r['audit'][k] == 0 for k in
        ('actual_contact_steps', 'planning_margin_violations', 'unsafe_selected_horizons', 'unsafe_emitted_references')) for r in rows)
    full = formal and indices == list(range(300))
    return {'successes': sum(r['success'] for r in rows), 'lost_lt32_successes': lost,
        'lost_native_successes_in_evaluated_set': native_lost, 'lost_historical_successes_in_evaluated_set': history_lost,
        'gained_successes': gained, 'safety_audit_clean': safe, 'full_300': full,
        'candidate_qualified': bool(full and safe and sum(r['success'] for r in rows) >= 210
                                    and not lost and not native_lost and not history_lost and gained),
        'production_qualified': False, 'all_five_modules_qualified': False}


def run(stage, output, *, formal_config=None):
    config, baseline = validate_baseline()
    if stage == 'formal':
        if formal_config is None or formal_config['code'] != execution_identity() or formal_config['indices'] != list(range(300)):
            raise ValueError('正式运行必须消费经预检的完整冻结配置')
        indices = formal_config['indices']
    else:
        indices = select_indices(stage)
    code = execution_identity()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if formal_config is not None:
        atomic_write_json(output/'formal-config.json', formal_config)
    identity = {'stage': stage, 'indices': indices, 'code': code, 'runtime': runtime_identity(),
        'baseline_config_sha256': _sha256(BASELINE/'config.json'), 'baseline_report_sha256': _sha256(BASELINE/'report.json'),
        'inputs': {k: config['inputs'][k] for k in ('data', 'manifest', 'checkpoint', 'historical')},
        'budget_seconds': 28800 if stage == 'formal' else 90 if stage == 'regressions' else 150,
        'production_qualified': False}
    atomic_write_json(output/'identity.json', identity)
    budget = VerificationBudget(identity['budget_seconds'])
    vehicle = VehicleConfig(**read(config['inputs']['manifest']['path'])['vehicle_model'])
    original = matrix.ClosedLoopEngine
    rows = []

    class ObservedEngine(original):
        def run(self, start, goal):
            plans, frames, raw_queries = [], [], []
            predict, primary_predict, compute = self.source.next_trajectory, self.source.primary.next_trajectory, self.mpc.compute
            def raw_prediction(state):
                raw, elapsed = primary_predict(state)
                raw_queries.append((state.to_array(), raw.points.copy(), raw.dt))
                return raw, elapsed
            def next_plan(state):
                budget.check()
                trajectory, elapsed = predict(state)
                budget.check()
                plans.append((state.to_array(), trajectory.points.copy(), trajectory.dt, raw_queries[-1][1]))
                return trajectory, elapsed
            def command(trajectory, state):
                budget.check()
                cmd = compute(trajectory, state)
                frames.append((state.to_array(), self.mpc._last_u.copy().reshape(-1, 2), self.mpc.safety_audit()))
                return cmd
            self.source.next_trajectory, self.source.primary.next_trajectory, self.mpc.compute = next_plan, raw_prediction, command
            try:
                result = super().run(start, goal)
            finally:
                self.source.next_trajectory, self.source.primary.next_trajectory, self.mpc.compute = predict, primary_predict, compute
            if len(frames) != result.steps:
                raise ValueError('实际转移观察不完整')
            i = self.meta['dataset_index']
            path = output/f'episode-{i:04d}'
            arrays = {'initial_state': start.to_array(), 'goal': np.array([goal.x, goal.y, goal.yaw]),
                'tol_pos': np.array(self.terminal.tol_pos), 'tol_yaw': np.array(self.terminal.tol_yaw),
                'states_before': np.asarray([f[0] for f in frames]).reshape(-1, 5),
                'states_after': np.asarray([s.to_array() for s in result.record.states]).reshape(-1, 5),
                'commands': np.asarray([[c.v, c.omega] for c in result.record.cmds]).reshape(-1, 2),
                'sequences': np.asarray([f[1] for f in frames]).reshape(-1, 10, 2), 'references': np.array(len(plans)),
                'terminal_steps': np.asarray([n for n, f in enumerate(frames) if 'terminal_certificate' in f[2]], dtype=int)}
            for n, (state, points, dt, raw) in enumerate(plans):
                arrays[f'reference_{n:04d}'], arrays[f'query_state_{n:04d}'] = points, state
                arrays[f'reference_dt_{n:04d}'], arrays[f'raw_{n:04d}'] = np.array(dt), raw
            for n, (state, points, dt) in enumerate(raw_queries[len(plans):]):
                arrays[f'unemitted_raw_{n:04d}'] = points
                arrays[f'unemitted_state_{n:04d}'], arrays[f'unemitted_dt_{n:04d}'] = state, np.array(dt)
            np.savez_compressed(path.with_suffix('.npz'), **arrays)
            audit = audit_tape(path.with_suffix('.npz'), vehicle, self.env, budget)
            if any(audit[k] for k in ('actual_contact_steps', 'planning_margin_violations', 'unsafe_selected_horizons', 'unsafe_emitted_references')):
                raise ValueError('安全约束违反')
            checker = self.source.checker.footprint_checker
            decisions = self.source.admission_records
            counts = audit_admissions(path.with_suffix('.npz'), decisions, checker, budget)
            row = {'dataset_index': i, **{k: getattr(result, k) for k in
                ('success', 'collision', 'failure', 'steps', 'final_pos_err', 'final_yaw_err')},
                'audit': audit, 'reference_policy_counts': counts, 'admissions': decisions,
                'raw_queries_not_emitted': len(raw_queries)-len(plans)}
            atomic_write_json(path.with_suffix('.json'), row)
            rows.append(row)
            return result

    matrix.ClosedLoopEngine = ObservedEngine
    report = {'status': 'running', 'stage': stage, 'production_qualified': False}
    atomic_write_json(output/'report.json', report)
    try:
        raw = matrix.run_validation_experiment(
            matrix.ExperimentSpec('safe_prefix', 'sensor', 'network', 'mpc_vehicle', 'safe_prefix'),
            data_path=config['inputs']['data']['path'], checkpoint_path=config['inputs']['checkpoint']['path'],
            indices=indices, samples=0, max_steps=600, replan_every=10, control_seed=0, output_path=output/'raw-report.json',
            progress=lambda n, total, b: print(f'[{n}/{total}] #{b.result.meta["dataset_index"]} {b.result.failure or "success"} {b.result.steps}', flush=True))
        if raw['reconstruct_failures'] or [r['dataset_index'] for r in rows] != indices or code != execution_identity():
            raise ValueError('分母或代码身份漂移')
        acceptance = summarize_candidate(rows, baseline, read(config['inputs']['historical']['path']), formal=stage == 'formal')
        report.update(status='completed', rows=rows, indices=indices, **acceptance,
            elapsed_seconds=time.perf_counter()-budget.started,
            all_three_regressions_restored=all(next(r['success'] for r in rows if r['dataset_index'] == i) for i in REGRESSIONS))
        atomic_write_json(output/'report.json', report)
        atomic_write_json(output/'artifacts.json', {p.name: _sha256(p) for p in output.iterdir() if p.is_file()})
        return report
    except BaseException as exc:
        report.update(status='incomplete', error=f'{type(exc).__name__}: {exc}', rows=rows)
        atomic_write_json(output/'report.json', report)
        raise
    finally:
        matrix.ClosedLoopEngine = original
