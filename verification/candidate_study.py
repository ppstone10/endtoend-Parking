"""候选消融的配置、短检、正式人工计划与联合裁决。"""
from __future__ import annotations

from pathlib import Path
import json
import time
import torch

from experiments import validation_matrix as matrix
from experiments.component_capture import VerificationBudget
from scripts.verify_closed_loop_components import inspect_completed
from scripts.diagnose_closed_loop_failures import _sha256
from scripts.prepare_recovery_balance_ab import runtime_identity
from sim import VehicleConfig
from training.reporting import atomic_write_json
from verification.candidate_execution import ARMS, candidate_scope, compare_rows
from verification.identity import execution_identity, verify_files, ROOT

SOURCE = ROOT / 'runs/verification/component-v1-formal-20261023-v2/result'
SEMANTICS = ROOT / 'runs/verification/input-contract-corrected-20261023/report.json'
FILES = ['verification/candidate_execution.py', 'verification/safety_candidate.py',
         'verification/candidate_study.py', 'scripts/compare_footprint_execution.py']
SMOKE = [0, 123, 163, 292]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validated_pilot(pilot, audit_path):
    pilot, audit_path = Path(pilot).resolve(), Path(audit_path).resolve()
    artifacts = read(pilot.parent / 'artifacts.json')
    for name, digest in artifacts.items():
        path = (pilot.parent / name).resolve()
        if not path.is_relative_to(pilot.parent) or not path.is_file() or _sha256(path) != digest:
            raise ValueError('短检登记产物缺失或漂移')
    required = {'report.json', 'config.json', 'summary.md'}
    required.update(f'{arm}/episode-{i:04d}.{suffix}' for arm in ARMS for i in SMOKE for suffix in ('json', 'npz'))
    if not required <= set(artifacts):
        raise ValueError('短检产物登记不完整')
    result, audit = read(pilot), read(audit_path)
    if result.get('status') != 'completed' or result.get('mode') != 'smoke' or result['indices'] != SMOKE:
        raise ValueError('只接受完整固定短检作为正式前置')
    if audit.get('status') != 'completed' or audit['source_report_sha256'] != _sha256(pilot):
        raise ValueError('缺少匹配的独立审计')
    verify_files(audit['code'])
    if ([r['dataset_index'] for r in audit['episodes']] != SMOKE
            or not all(r['all_choices_costs_windows_and_states_exact'] and r['all_selected_horizons_contact_safe'] for r in audit['episodes'])):
        raise ValueError('独立审计覆盖/安全未通过')
    eligible = []
    for arm in ARMS[1:]:
        paired = compare_rows(result['rows']['native'], result['rows'][arm])
        if any(result['comparisons'][arm][k] != value for k, value in paired.items()):
            raise ValueError('短检汇总与逐任务配对不一致')
        if paired['pilot_benefit']:
            eligible.append(arm)
    return result, eligible


def prepare(destination, *, pilot=None, audit=None):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('配置已存在，不覆盖')
    source = inspect_completed(SOURCE)
    semantics = read(SEMANTICS)
    if semantics['status'] != 'completed' or semantics['weights_or_runtime_inputs_changed']:
        raise ValueError('输入声明校正未通过')
    verify_files(semantics['inputs'])
    if _sha256(Path(semantics['checkpoint'])) != semantics['checkpoint_sha256']:
        raise ValueError('声明校正副本漂移')
    inputs = {str(SOURCE / name): _sha256(SOURCE / name) for name in ('report.json', 'artifacts.json', 'config.json')}
    inputs.update(semantics['inputs'])
    inputs.update({str(SEMANTICS): _sha256(SEMANTICS), semantics['checkpoint']: semantics['checkpoint_sha256']})
    manifest = Path(source['identity']['inputs']['data']['path']).with_name('manifest.json')
    inputs[str(manifest)] = _sha256(manifest)
    eligible = []
    if pilot is not None:
        if audit is None:
            raise ValueError('正式prepare必须提供独立审计--audit')
        pilot = Path(pilot).resolve()
        audit = Path(audit).resolve()
        result, eligible = validated_pilot(pilot, audit)
        if result['identity']['code'] != execution_identity(FILES) or result['identity']['inputs'] != inputs:
            raise ValueError('短检身份与当前候选不一致')
        if not eligible:
            raise ValueError('短检没有兼顾新增到达和旧成功保持的收益，不启动正式长任务')
        inputs[str(pilot)] = _sha256(pilot)
        inputs[str(pilot.parent / 'artifacts.json')] = _sha256(pilot.parent / 'artifacts.json')
        inputs[str(audit)] = _sha256(audit)
    config = {'kind': 'footprint_execution_candidate_v1', 'purpose': 'formal' if pilot else 'smoke',
              'inputs': inputs, 'code': execution_identity(FILES), 'runtime': runtime_identity(),
              'source': str(SOURCE), 'checkpoint': semantics['checkpoint'],
              'data': source['identity']['inputs']['data']['path'],
              'indices': list(range(300)) if pilot else SMOKE, 'arms': eligible if pilot else list(ARMS),
              'budget_seconds': 28800 if pilot else 180,
              'control_seed': 0, 'max_steps': 600, 'replan_every': 10, 'mpc_dt': .1, 'mpc_horizon': 10,
              'pilot': str(pilot) if pilot else None, 'independent_audit': str(audit) if audit else None,
              'max_candidate_checks': 16, 'physical_margin': 0., 'reference_margin': .2,
              'reference_max_offset': 1., 'reference_rounds': 10,
              'automatic_deployment': False, 'training_performed': False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, config)
    return config


def precheck(path):
    config = read(path)
    if config['kind'] != 'footprint_execution_candidate_v1':
        raise ValueError('候选配置协议无效')
    verify_files(config['inputs'])
    if config['code'] != execution_identity(FILES) or config['runtime'] != runtime_identity():
        raise ValueError('候选代码/环境漂移')
    expected = {'control_seed': 0, 'max_steps': 600, 'replan_every': 10, 'mpc_dt': .1, 'mpc_horizon': 10,
                'max_candidate_checks': 16, 'physical_margin': 0., 'reference_margin': .2,
                'reference_max_offset': 1., 'reference_rounds': 10}
    if any(config[k] != value for k, value in expected.items()):
        raise ValueError('固定参数变化，不静默执行其他候选')
    if config['purpose'] == 'smoke':
        if config['indices'] != SMOKE or config['arms'] != list(ARMS) or config['budget_seconds'] != 180:
            raise ValueError('短检范围/预算变化')
    elif config['purpose'] == 'formal':
        if config['indices'] != list(range(300)) or not config['arms'] or any(a not in ARMS[1:] for a in config['arms']):
            raise ValueError('正式分母/候选不完整')
        if config['budget_seconds'] != 28800 or not config.get('pilot') or not config.get('independent_audit'):
            raise ValueError('正式预算/前置证据缺失')
        _, eligible = validated_pilot(config['pilot'], config['independent_audit'])
        if config['arms'] != eligible:
            raise ValueError('正式臂与已验证收益不一致')
    else:
        raise ValueError('未知用途')
    return config


def native_rows(source):
    rows = []
    report = read(Path(source) / 'report.json')
    for e in report['episodes']:
        capture = read(Path(source) / 'capture' / f'episode-{e["dataset_index"]:04d}.json')
        rows.append({k: e[k] for k in ('dataset_index', 'success', 'collision', 'failure', 'steps', 'final_pos_err', 'final_yaw_err')})
        rows[-1].update(zero_hold_steps=0, intervened_steps=0,
                        planning_margin_violations=sum(not f['planning']['actual_transition_safe'] for f in capture['frames']))
    return rows


def run(path, *, formal=False):
    config = precheck(path)
    if formal != (config['purpose'] == 'formal'):
        raise ValueError('短检与正式配置必须显式分开')
    torch.set_num_threads(1)
    output = Path(path).resolve().parent / ('results' if formal else 'smoke')
    output.mkdir(parents=True, exist_ok=False)
    report = {'kind': config['kind'], 'status': 'running', 'mode': 'formal' if formal else 'smoke',
              'indices': config['indices'], 'identity': {'config_sha256': _sha256(Path(path)),
              'inputs': config['inputs'], 'code': config['code']}, 'comparisons': {}, 'rows': {},
              'candidate_qualified': False, 'pure_network_learning_benefit': False,
              'training_performed': False, 'automatic_deployment': False}
    atomic_write_json(output / 'config.json', config)
    atomic_write_json(output / 'report.json', report)
    baseline = native_rows(config['source'])
    expected = {r['dataset_index']: r for r in baseline}
    vehicle = VehicleConfig(**matrix.load_dataset_manifest(Path(config['data']))['vehicle_model'])
    budget = VerificationBudget(config['budget_seconds'])
    if formal:
        report['rows']['native'] = baseline
    try:
        for arm in config['arms']:
            budget.check()
            folder = output / arm
            folder.mkdir()
            print(f'ARM {arm} BEGIN {len(config["indices"])} tasks', flush=True)
            spec = 'E5' if arm == 'native' else matrix.ExperimentSpec(
                name=f'candidate_{arm}', bev_source='sensor', trajectory_source='network', executor='mpc_vehicle',
                description=f'explicit footprint constraint candidate: {arm}; not unchanged native E5')
            with candidate_scope(arm, folder, vehicle, expected, budget) as rows:
                raw = matrix.run_validation_experiment(
                    spec, data_path=config['data'], checkpoint_path=config['checkpoint'],
                    output_path=folder / 'raw-report.json', indices=config['indices'], samples=0,
                    max_steps=600, replan_every=10, control_seed=0,
                    progress=lambda n, total, bundle: print(f'[{arm} {n}/{total}] #{bundle.result.meta["dataset_index"]} {bundle.result.failure or "success"}', flush=True))
            if raw['reconstruct_failures'] or len(rows) != len(config['indices']):
                raise ValueError('候选执行分母不完整')
            report['rows'][arm] = rows
            if arm != 'native':
                report['comparisons'][arm] = compare_rows(report['rows']['native'], rows)
            atomic_write_json(output / 'report.json', report)
        precheck(path)
        report.update(status='completed', elapsed_seconds=time.perf_counter() - budget.started)
        for arm, result in report['comparisons'].items():
            result['formal_joint_pass'] = bool(formal and result['samples'] == 300 and result['successes'] >= 210
                                               and result['collisions'] <= 30 and not result['old_success_lost'])
        report['candidate_qualified'] = any(r['formal_joint_pass'] for r in report['comparisons'].values())
        atomic_write_json(output / 'report.json', report)
        summary = ['# 完整车体约束执行候选', '', f'状态：completed；模式：{report["mode"]}；总体资格：{report["candidate_qualified"]}。', '',
                   '| 臂 | 成功 | 碰撞 | 旧成功丢失 | 新成功 | 零动作步 | 规划余量侵入步 |', '|---|---:|---:|---|---|---:|---:|']
        for arm, r in report['comparisons'].items():
            summary.append(f'| {arm} | {r["successes"]} | {r["collisions"]} | {r["old_success_lost"]} | {r["new_success_gained"]} | {r["zero_hold_steps"]} | {r["planning_margin_violations"]} |')
        summary += ['', '参考过滤/控制约束是显式系统干预，不是纯网络学习收益；安全停车不能替代到达。']
        (output / 'summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')
        atomic_write_json(output / 'artifacts.json', {p.relative_to(output).as_posix(): _sha256(p)
                                                    for p in output.rglob('*') if p.is_file()})
        return report
    except BaseException as exc:
        report.update(status='incomplete', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.perf_counter() - budget.started)
        atomic_write_json(output / 'report.json', report)
        raise
