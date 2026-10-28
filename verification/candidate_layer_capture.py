"""围绕当前候选的只读全输入观察；不替换控制或重复调用感知。"""
from contextlib import contextmanager
from pathlib import Path
import numpy as np

from experiments import validation_matrix as matrix
from training.reporting import atomic_write_json
from verification.perception_calibration import pipeline_semantics
from verification.reference_retention import array_digest
from verification.typed_replay import scalar_type


@contextmanager
def capture_inputs_scope(output, budget):
    original_engine = matrix.ClosedLoopEngine
    captured = []

    class InputEngine(original_engine):
        def run(self, start, goal):
            source = self.source
            primary = source.primary
            pipeline, model = primary.sensor_pipeline, primary.model
            capture, predict, next_trajectory = pipeline.capture_bev, model.predict, source.next_trajectory
            packets, arrays, pending = [], {}, {}

            def observe_capture(x, y, yaw):
                budget.check()
                bev = capture(x, y, yaw)
                pending.update(bev=bev, data=bev.data.copy(), metadata=bev.to_metadata(),
                               pose=[x, y, yaw], types=[scalar_type(v) for v in (x, y, yaw)])
                return bev

            def observe_predict(bev, local_goal, state):
                budget.check()
                local = predict(bev, local_goal, state)
                ordinal = len(packets)
                if not np.array_equal(bev.data, pending['data']) or bev.to_metadata() != pending['metadata']:
                    raise ValueError('BEV在capture到predict之间改变')
                arrays[f'bev_{ordinal:04d}'] = pending['data']
                arrays[f'local_{ordinal:04d}'] = local.points.copy()
                arrays[f'model_state_{ordinal:04d}'] = state.to_array()
                packets.append({'ordinal': ordinal, 'step': source.step,
                                'capture_pose': pending['pose'], 'capture_types': pending['types'],
                                'state_types': [scalar_type(v) for v in (state.x, state.y, state.yaw, state.v, state.omega)],
                                'bev_metadata': pending['metadata'], 'goal_local': [local_goal.x, local_goal.y, local_goal.yaw],
                                'semantics': pipeline_semantics(pipeline), 'dt': local.dt})
                return local

            def observe_source(state):
                ordinal = len(packets)
                trajectory, elapsed = next_trajectory(state)
                if len(packets) != ordinal + 1:
                    raise ValueError('每次真实查询必须对应一次capture/predict')
                arrays[f'raw_{ordinal:04d}'] = source.raw_references[f'raw_{ordinal:04d}'].copy()
                packets[-1].update(used_sha256=array_digest(trajectory.points),
                                   raw_sha256=array_digest(arrays[f'raw_{ordinal:04d}']))
                return trajectory, elapsed

            pipeline.capture_bev, model.predict, source.next_trajectory = observe_capture, observe_predict, observe_source
            try:
                result = super().run(start, goal)
            finally:
                pipeline.capture_bev, model.predict, source.next_trajectory = capture, predict, next_trajectory
            if [p['step'] for p in packets] != list(range(0, result.steps, self.replan_every)):
                raise ValueError('全输入预测分母不完整')
            path = Path(output)/f'inputs-{self.meta["dataset_index"]:04d}'
            np.savez_compressed(path.with_suffix('.npz'), **arrays)
            atomic_write_json(path.with_suffix('.json'), {'status': 'completed', 'packets': packets})
            captured.append({'dataset_index': self.meta['dataset_index'], 'queries': len(packets)})
            return result

    matrix.ClosedLoopEngine = InputEngine
    try:
        yield captured
    finally:
        matrix.ClosedLoopEngine = original_engine
