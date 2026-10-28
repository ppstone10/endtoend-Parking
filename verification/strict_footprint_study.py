"""严格车体模式的固定配对、身份冻结和只读动作审计；不授予原生资格。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

from dataset import DatasetGenerator
from experiments import validation_matrix as matrix
from experiments.component_capture import VerificationBudget
from scripts.diagnose_closed_loop_failures import _sha256
from scripts.diagnose_reference_execution import code_identity
from scripts.prepare_recovery_balance_ab import runtime_identity
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json
from verification.strict_footprint_audit import audit_tape

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'strict_footprint_pair_v1'
PILOT = [0, 163, 247, 292]
RETENTION = [55, 110, 162, 212, 245]
INPUTS = {
    'data': ROOT/'runs/verification/surround-target-rebuild-lineage-20261027/val/val.npz',
    'manifest': ROOT/'runs/verification/surround-target-rebuild-lineage-20261027/val/manifest.json',
    'checkpoint': ROOT/'runs/training/lt31-terminal-hold-20261027/control/last.pt',
    'historical': ROOT/'runs/validation/live-20261018-170806/full/E5/report.json',
    'identity_reference': ROOT/'runs/verification/native-continuation-precheck-v2-20261027/identity.json',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity():
    return {**code_identity(), **{str(p.resolve()): _sha256(p) for p in
            (Path(__file__), ROOT/'verification/strict_footprint_audit.py', ROOT/'scripts/validate_strict_footprint.py', ROOT/'scripts/run_validation_experiment.py')}}


def prepare(path, *, mode='smoke', pilot=None, retention_scout=None):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError('冻结配置不得覆盖')
    if mode not in ('smoke', 'retention', 'formal'):
        raise ValueError('未知模式')
    inputs = {k: {'path': str(p.resolve()), 'sha256': _sha256(p)} for k, p in INPUTS.items()}
    prior = read(INPUTS['identity_reference'])['inputs']
    for current, previous in (('data', 'data'), ('checkpoint', 'control'), ('manifest', 'manifest'), ('historical', 'historical_baseline')):
        if inputs[current]['sha256'] != prior[previous]['sha256']:
            raise ValueError('输入不属于已复核固定任务和原生模型')
    data = DatasetGenerator.load(inputs['data']['path'])
    historical = read(inputs['historical']['path'])
    actual_ids = [m['task_id'] for m in data['task_meta']]
    expected_ids = [r['meta']['task_id'] for r in historical['episodes']]
    if len(actual_ids) != 300 or actual_ids != expected_ids:
        raise ValueError('历史300条任务身份/顺序不符')
    if mode == 'formal':
        if pilot is None or not inspect(pilot)['acceptance']['pilot_eligible']:
            raise ValueError('必须有当前代码下完整、联合通过的固定短检')
        inputs['pilot_report'] = {'path': str((Path(pilot)/'report.json').resolve()),
                                  'sha256': _sha256(Path(pilot)/'report.json')}
        inputs['pilot_artifacts'] = {'path': str((Path(pilot)/'artifacts.json').resolve()),
                                     'sha256': _sha256(Path(pilot)/'artifacts.json')}
        if retention_scout is None or not inspect(retention_scout)['acceptance']['retention_eligible']:
            raise ValueError('历史成功抽查未全部保持，不得准备正式长测')
        for name in ('report', 'artifacts'):
            file = (Path(retention_scout)/f'{name}.json').resolve()
            inputs[f'retention_{name}'] = {'path': str(file), 'sha256': _sha256(file)}
    config = {'schema': SCHEMA, 'mode': mode,
              'indices': PILOT if mode == 'smoke' else RETENTION if mode == 'retention' else list(range(300)),
              'inputs': inputs, 'code': identity(), 'runtime': runtime_identity(),
              'budget_seconds': 28800 if mode == 'formal' else 180,
              'protocol': {'max_steps': 600, 'replan_every': 10, 'control_seed': 0,
                           'mpc_dt': .1, 'horizon': 10, 'stop_threshold': .5, 'margin_m': .2}}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, config)
    precheck(path)
    return config


def precheck(path):
    config = read(path)
    if config.get('schema') != SCHEMA or config['mode'] not in ('smoke', 'retention', 'formal'):
        raise ValueError('配置身份不符')
    expected = PILOT if config['mode'] == 'smoke' else RETENTION if config['mode'] == 'retention' else list(range(300))
    if config['indices'] != expected or config['budget_seconds'] != (28800 if config['mode'] == 'formal' else 180):
        raise ValueError('分母或预算漂移')
    if config['protocol'] != {'max_steps': 600, 'replan_every': 10, 'control_seed': 0,
                              'mpc_dt': .1, 'horizon': 10, 'stop_threshold': .5, 'margin_m': .2}:
        raise ValueError('协议漂移')
    if Path(sys.executable).resolve() != Path('D:/conda/envs/endtoend-parking/python.exe').resolve():
        raise ValueError('必须使用endtoend-parking环境')
    for item in config['inputs'].values():
        if _sha256(Path(item['path'])) != item['sha256']:
            raise ValueError('输入身份漂移')
    if config['code'] != identity() or config['runtime'] != runtime_identity():
        raise ValueError('代码或环境漂移；不可静默重冻身份')
    loaded = load_model_checkpoint(config['inputs']['checkpoint']['path'])
    if loaded.model.stop_threshold != .5 or loaded.model_config.get('camera_projection') != 'per_view_v2':
        raise ValueError('模型阈值或输入版本不符')
    manifest = read(config['inputs']['manifest']['path'])
    if manifest['vehicle_model']['collision_margin'] != .2:
        raise ValueError('车辆规划余量漂移')
    if config['mode'] == 'formal':
        pilot = Path(config['inputs']['pilot_report']['path']).parent
        if not inspect(pilot)['acceptance']['pilot_eligible']:
            raise ValueError('短检不满足正式前置条件')
        if not inspect(Path(config['inputs']['retention_report']['path']).parent)['acceptance']['retention_eligible']:
            raise ValueError('历史成功抽查存在回退')
    return config


def summarize(rows, historical, *, formal):
    native, strict = rows['native'], rows['strict']
    if [r['dataset_index'] for r in native] != [r['dataset_index'] for r in strict]:
        raise ValueError('配对分母不一致')
    old = {r['meta']['dataset_index'] for r in historical['episodes'] if r['success']}
    lost = [s['dataset_index'] for n, s in zip(native, strict, strict=True) if n['success'] and not s['success']]
    old_lost = [s['dataset_index'] for s in strict if s['dataset_index'] in old and not s['success']]
    gained = [s['dataset_index'] for n, s in zip(native, strict, strict=True) if not n['success'] and s['success']]
    clean = all(not s['collision'] and all(s['audit'][k] == 0 for k in
        ('actual_contact_steps', 'planning_margin_violations', 'unsafe_selected_horizons', 'unsafe_emitted_references')) for s in strict)
    recovery = any(s['dataset_index'] == 247 and s['success'] for s in strict)
    full = formal and len(strict) == 300 and [s['dataset_index'] for s in strict] == list(range(300))
    return {'native_successes': sum(s['success'] for s in native), 'strict_successes': sum(s['success'] for s in strict),
            'native_collisions': sum(s['collision'] for s in native), 'strict_collisions': sum(s['collision'] for s in strict),
            'paired_success_lost': lost, 'new_success_gained': gained, 'historical_successes_observed': sum(s['dataset_index'] in old for s in strict),
            'historical_success_lost_in_evaluated_set': old_lost, 'safety_audit_clean': clean,
            'pilot_eligible': bool(not formal and recovery and not lost and gained and clean),
            'retention_eligible': bool(not formal and [s['dataset_index'] for s in strict] == RETENTION
                                       and all(s['success'] for s in strict) and clean),
            'full_300': full, 'candidate_qualified': bool(full and not lost and not old_lost and clean
                and sum(s['success'] for s in strict) >= 210),
            'production_qualified': False, 'all_five_modules_qualified': False}


def run(config_path, output):
    config = precheck(config_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output/'config.json', config)
    report = {'schema': SCHEMA, 'status': 'running', 'mode': config['mode'], 'rows': {},
              'geometry_source': 'simulator_ground_truth', 'production_qualified': False}
    atomic_write_json(output/'report.json', report)
    budget = VerificationBudget(config['budget_seconds'])
    original = matrix.ClosedLoopEngine
    arm, active = '', []
    manifest = read(config['inputs']['manifest']['path'])
    vehicle = VehicleConfig(**manifest['vehicle_model'])

    class ObservedEngine(original):
        def run(self, start, goal):
            plans, frames = [], []
            predict, compute = self.source.next_trajectory, self.mpc.compute
            def next_plan(state):
                budget.check()
                trajectory, elapsed = predict(state)
                budget.check()
                plans.append((state.to_array(), trajectory.points.copy(), trajectory.dt))
                return trajectory, elapsed
            def command(trajectory, state):
                budget.check()
                cmd = compute(trajectory, state)
                audit = self.mpc.safety_audit() if hasattr(self.mpc, 'safety_audit') else {}
                frames.append((state.to_array(), self.mpc._last_u.copy().reshape(-1, 2), audit))
                return cmd
            self.source.next_trajectory, self.mpc.compute = next_plan, command
            try:
                result = super().run(start, goal)
            finally:
                self.source.next_trajectory, self.mpc.compute = predict, compute
            if len(frames) != result.steps:
                raise ValueError('实际控制与观察分母不一致')
            index = self.meta['dataset_index']
            path = output/arm/f'episode-{index:04d}'
            arrays = {'initial_state': start.to_array(), 'goal': np.array([goal.x, goal.y, goal.yaw]),
                'states_before': np.asarray([f[0] for f in frames]).reshape(-1, 5),
                'states_after': np.asarray([s.to_array() for s in result.record.states]).reshape(-1, 5),
                'commands': np.asarray([[c.v, c.omega] for c in result.record.cmds]).reshape(-1, 2),
                'sequences': np.asarray([f[1] for f in frames]).reshape(-1, 10, 2), 'references': np.array(len(plans)),
                'tol_pos': np.array(self.terminal.tol_pos), 'tol_yaw': np.array(self.terminal.tol_yaw),
                'terminal_steps': np.asarray([i for i, f in enumerate(frames) if 'terminal_certificate' in f[2]], dtype=int)}
            for ordinal, (state, points, dt) in enumerate(plans):
                arrays[f'reference_{ordinal:04d}'], arrays[f'query_state_{ordinal:04d}'] = points, state
                arrays[f'reference_dt_{ordinal:04d}'] = np.array(dt)
            np.savez_compressed(path.with_suffix('.npz'), **arrays)
            audit = audit_tape(path.with_suffix('.npz'), vehicle, self.env, budget)
            if arm == 'strict' and any(audit[k] for k in ('actual_contact_steps', 'planning_margin_violations',
                    'unsafe_selected_horizons', 'unsafe_emitted_references')):
                raise ValueError('严格安全约束违反，停止而非报告合格')
            row = {'dataset_index': index, **{k: getattr(result, k) for k in
                ('success', 'collision', 'failure', 'steps', 'final_pos_err', 'final_yaw_err')}, 'audit': audit,
                'terminal_certificates': [{'step': i, **f[2]['terminal_certificate']} for i, f in enumerate(frames)
                                          if 'terminal_certificate' in f[2]]}
            atomic_write_json(path.with_suffix('.json'), row)
            active.append(row)
            return result

    matrix.ClosedLoopEngine = ObservedEngine
    try:
        for arm, mode in (('native', 'none'), ('strict', 'strict_footprint')):
            precheck(config_path)
            (output/arm).mkdir()
            active = []
            spec = matrix.ExperimentSpec('E5' if arm == 'native' else 'strict_footprint', 'sensor', 'network', 'mpc_vehicle', mode)
            raw = matrix.run_validation_experiment(spec, data_path=config['inputs']['data']['path'],
                checkpoint_path=config['inputs']['checkpoint']['path'], indices=config['indices'], samples=0,
                output_path=output/arm/'raw-report.json', max_steps=600, replan_every=10, control_seed=0,
                progress=lambda n, total, b: print(f'[{arm} {n}/{total}] #{b.result.meta["dataset_index"]} {b.result.failure or "success"}', flush=True))
            if raw['reconstruct_failures'] or [r['dataset_index'] for r in active] != config['indices']:
                raise ValueError('分母缺失')
            report['rows'][arm] = active
            atomic_write_json(output/'report.json', report)
        precheck(config_path)
        report.update(status='completed', elapsed_seconds=time.perf_counter()-budget.started,
            acceptance=summarize(report['rows'], read(config['inputs']['historical']['path']), formal=config['mode'] == 'formal'))
        atomic_write_json(output/'report.json', report)
        atomic_write_json(output/'artifacts.json', {p.relative_to(output).as_posix(): _sha256(p)
                                                  for p in output.rglob('*') if p.is_file()})
        return report
    except BaseException as exc:
        report.update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(output/'report.json', report)
        raise
    finally:
        matrix.ClosedLoopEngine = original


def inspect(output):
    output = Path(output).resolve()
    report = read(output/'report.json')
    config = precheck(output/'config.json')
    artifacts = read(output/'artifacts.json')
    required = {'report.json', 'config.json'} | {f'{arm}/raw-report.json' for arm in ('native', 'strict')}
    required |= {f'{arm}/episode-{i:04d}.{suffix}' for arm in ('native', 'strict')
                 for i in config['indices'] for suffix in ('json', 'npz')}
    if not required <= artifacts.keys():
        raise ValueError('产物登记缺失')
    for name, digest in artifacts.items():
        path = (output/name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or _sha256(path) != digest:
            raise ValueError('产物身份漂移')
    for arm in ('native', 'strict'):
        if [r['dataset_index'] for r in report['rows'][arm]] != config['indices']:
            raise ValueError('逐回合分母不符')
        raw = read(output/arm/'raw-report.json')
        if raw['reconstruct_failures'] or raw['protocol']['evaluated_indices'] != config['indices']:
            raise ValueError('原始分母不符')
        for row, episode in zip(report['rows'][arm], raw['episodes'], strict=True):
            saved = read(output/arm/f'episode-{row["dataset_index"]:04d}.json')
            if saved != row or any(episode[k] != row[k] for k in ('success', 'collision', 'failure', 'steps', 'final_pos_err', 'final_yaw_err')):
                raise ValueError('逐例结果不一致')
    expected = summarize(report['rows'], read(config['inputs']['historical']['path']), formal=config['mode'] == 'formal')
    if report.get('status') != 'completed' or report.get('acceptance') != expected:
        raise ValueError('未完成或联合裁决不一致')
    return report
