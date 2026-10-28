"""只读迁移v1证据：恢复真实标量类型，正交呈现契约与轨迹质量。"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import json
import time
import numpy as np

from dataset import DatasetGenerator
from experiments import validation_matrix as matrix
from experiments.component_verification import acceptance, aggregate_layers
from scripts.diagnose_closed_loop_failures import _sha256
from scripts.verify_closed_loop_components import inspect_completed
from sim import VehicleConfig, DifferentialDriveModel
from training.reporting import atomic_write_json
from verification.typed_replay import reconstruct_capture_types, restore_pose
from verification.identity import execution_identity, verify_files

REVISION_FILES = ['verification/report_update.py', 'verification/typed_replay.py', 'scripts/review_component_verification.py']


def migrate_report(source, output, *, budget_seconds=90, contract_audit=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('修正版输出必须为新目录')
    started = time.perf_counter()
    code = execution_identity(REVISION_FILES)
    report = inspect_completed(source)
    config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    data = DatasetGenerator.load(config['inputs']['data']['path'])
    manifest = matrix.load_dataset_manifest(Path(config['inputs']['data']['path']))
    vehicle = VehicleConfig(**manifest['vehicle_model'])
    model = DifferentialDriveModel(**vehicle.vehicle_model_kwargs())
    rows, corrections, event_counts = [], [], Counter()
    payload = {'schema': 'component_verification_v2', 'status': 'running',
               'source_report': str(source / 'report.json'), 'source_sha256': _sha256(source / 'report.json'),
               'source_artifacts_sha256': _sha256(source / 'artifacts.json'), 'source_unchanged': True,
               'production_qualified': False, 'code': code}
    if contract_audit is not None:
        contract_audit = Path(contract_audit).resolve()
        correction = json.loads(contract_audit.read_text(encoding='utf-8'))
        verify_files(correction['inputs'])
        if (correction['status'] != 'completed' or correction['weights_or_runtime_inputs_changed']
                or _sha256(Path(correction['checkpoint'])) != correction['checkpoint_sha256']):
            raise ValueError('声明校正证据无效，不关联到v2报告')
        parent = config['inputs']['checkpoint']
        if correction['inputs'].get(parent['path']) != parent['sha256']:
            raise ValueError('声明校正不属于该历史模型')
        payload['deployment_contract_update'] = {
            'audit': str(contract_audit), 'audit_sha256': _sha256(contract_audit),
            'checkpoint': correction['checkpoint'], 'checkpoint_sha256': correction['checkpoint_sha256'],
            'target_channel': 'image', 'weights_or_runtime_inputs_changed': False,
            'historical_contract_axis_retained': True}
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / 'report.json', payload)
    try:
        for e in report['episodes']:
            if time.perf_counter() - started > budget_seconds:
                raise TimeoutError('只读迁移预算耗尽，不自动续跑')
            index = e['dataset_index']
            capture = json.loads((source / 'capture' / f'episode-{index:04d}.json').read_text(encoding='utf-8'))
            layers = deepcopy(e['layers'])
            with np.load(source / 'capture' / f'episode-{index:04d}.npz', allow_pickle=False) as tape:
                issue_steps = tape['plan_issue_steps']
                types = reconstruct_capture_types(tape['initial_state'], tape['states_before'], tape['states_after'],
                                                  [f['command'] for f in capture['frames']], issue_steps,
                                                  model, float(tape['control_dt']))
                l1_errors = layers['L1']['errors']
                if l1_errors:
                    restored = matrix.reconstruct_dataset_task(data['task_meta'][index], root_seed=manifest['seed'], vehicle=vehicle)
                    _, pipeline = matrix._build_pipeline(matrix.EXPERIMENT_SPECS['E5'], restored.task, vehicle)
                    pipeline.set_target_goals([restored.goal])
                    error_plans = {err['plan'] for err in l1_errors if err['error'] == 'production_sensor_replay'}
                    matched = set()
                    for ordinal in range(max(error_plans, default=-1) + 1):
                        step = int(issue_steps[ordinal])
                        pose = restore_pose(tape['plan_issue_states'][ordinal], types[step])
                        independent = pipeline.capture_bev(*pose)
                        if ordinal in error_plans:
                            archived = tape['issue_bevs'][ordinal]
                            metadata = capture['inputs'][ordinal]['bev_meta']
                            exact = (np.isfinite(archived).all() and np.array_equal(independent.data, archived)
                                     and independent.to_metadata() == metadata)
                            if exact:
                                matched.add(ordinal)
                            corrections.append({'dataset_index': index, 'plan': ordinal, 'step': step,
                                                'capture_scalar_types': types[step], 'exact_match': bool(exact)})
                    layers['L1']['errors'] = [err for err in l1_errors
                                             if not (err['error'] == 'production_sensor_replay' and err['plan'] in matched)]
                    layers['L1']['contract_status'] = 'failed' if layers['L1']['errors'] else 'passed'
            event = e['contact_signature'] if e['collision'] else 'unresolved_nonarrival' if not e['success'] else 'task_reached'
            event_counts[event] += 1
            rows.append({'dataset_index': index, 'scene_name': e['scene_name'],
                         **{k: e[k] for k in ('success', 'collision', 'failure', 'steps', 'final_pos_err', 'final_yaw_err')},
                         'layers': layers, 'typed_transitions_verified': len(capture['frames']),
                         'axes': {'contact_at_failure': event, 'reference_risk_any_time': e['reference_contact_risk'],
                                  'contract_errors': layers['L1_L2']['errors'] + layers['L2_L3']['errors'],
                                  'nonarrival': e['nonarrival']}})
        layers = aggregate_layers(rows)
        if code != execution_identity(REVISION_FILES):
            raise ValueError('迁移代码执行中漂移')
        payload.update(status='completed', episodes=rows, layers=layers, corrections=corrections,
                       event_counts=dict(event_counts), coverage=report['coverage'],
                       acceptance=acceptance(rows, full_coverage=report['mode'] == 'formal',
                                             checks_ok=all(v['contract_status'] == 'passed' for v in layers.values())),
                       elapsed_seconds=time.perf_counter() - started,
                       limits='契约错误与质量事件独立；只更正有精确类型/状态/BEV证据的误报，不修改旧报告或追认为策略收益。')
        atomic_write_json(output / 'report.json', payload)
        text = ['# 分层验证v2', '', f'状态：completed；{len(rows)}回合；真实转移类型重建{sum(e["typed_transitions_verified"] for e in rows)}步。', '',
                f'精确修正L1误报：{sum(c["exact_match"] for c in corrections)}；事件分组：{dict(event_counts)}。', '',
                'target声明冲突仍作为独立契约轴，不覆盖碰撞/未到达。原模型/报告/冻结源不改，正式资格仍以真实联合门槛为准。']
        (output / 'summary.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
        atomic_write_json(output / 'artifacts.json', {p.name: _sha256(p) for p in output.iterdir() if p.is_file()})
        return payload
    except BaseException as exc:
        payload.update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(output / 'report.json', payload)
        raise
