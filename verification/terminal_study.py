"""显式终端接管候选的冻结、固定短检及同300人工长任务。"""
from pathlib import Path
import time
import torch

from experiments import validation_matrix as matrix
from experiments.component_capture import VerificationBudget
from scripts.diagnose_closed_loop_failures import _sha256
from scripts.prepare_recovery_balance_ab import runtime_identity
from scripts.inspect_footprint_execution import inspect as inspect_lt22
from training.reporting import atomic_write_json
from sim import VehicleConfig
from verification.candidate_study import read, FILES as PREVIOUS_FILES
from verification.candidate_execution import compare_rows
from verification.identity import execution_identity, verify_files, ROOT
from verification.terminal_capture import terminal_scope

SOURCE = ROOT / 'runs/verification/footprint-control-formal-20261023/results'
SMOKE = [0, 123, 166, 168, 247, 292]
FILES = [*PREVIOUS_FILES, 'verification/terminal_capture.py', 'verification/terminal_study.py',
         'verification/terminal_audit.py', 'verification/terminal_diagnosis.py',
         'scripts/compare_terminal_capture.py']
PARAMETERS = {'max_steps': 600, 'replan_every': 10, 'control_seed': 0, 'mpc_dt': .1,
              'mpc_horizon': 10, 'physical_margin': 0., 'planning_margin': .2,
              'geometry_source': 'simulator_ground_truth', 'production_qualified': False}


def check_artifacts(source):
    source = Path(source).resolve()
    report = read(source / 'report.json')
    if report['status'] != 'completed':
        raise ValueError('结果未完整完成')
    artifacts = read(source / 'artifacts.json')
    required = {'report.json', 'config.json', 'summary.md', 'control/raw-report.json'}
    required.update(f'control/episode-{i:04d}.{suffix}' for i in report['indices'] for suffix in ('json', 'npz'))
    if not required <= set(artifacts):
        raise ValueError('登记产物不完整')
    for name, digest in artifacts.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or _sha256(path) != digest:
            raise ValueError('登记产物缺失或漂移')
    return report


def comparisons(native, lt22, rows):
    first, second = compare_rows(native, rows), compare_rows(lt22, rows)
    restored = next((r['success'] for r in rows if r['dataset_index'] == 168), False)
    margin_not_increased = second['planning_margin_violations'] <= sum(r['planning_margin_violations'] for r in lt22)
    eligible = bool(restored and not first['old_success_lost'] and not second['old_success_lost']
                    and second['new_success_gained'] and second['collisions'] == 0 and margin_not_increased)
    return {'vs_native': first, 'vs_lt22': second, 'regression_168_restored': restored,
            'margin_not_increased': margin_not_increased, 'pilot_eligible': eligible}


def validate_pilot(pilot, audit_path):
    pilot = Path(pilot).resolve()
    result = inspect(pilot)
    audit = read(audit_path)
    if (result['mode'] != 'smoke' or result['indices'] != SMOKE or not result['comparison']['pilot_eligible']
            or audit['status'] != 'completed' or audit['source_report_sha256'] != _sha256(pilot / 'report.json')
            or audit['indices'] != SMOKE or not audit['all_actual_states_and_sweeps_exact']
            or not audit['all_terminal_sequences_and_certificates_exact']):
        raise ValueError('固定短检或匹配的独立审计未通过')
    if audit['code'] != execution_identity(FILES):
        raise ValueError('独立审计代码漂移')
    return result


def prepare(path, *, pilot=None, audit=None):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError('新配置禁止覆盖')
    inspect_lt22(SOURCE)
    prior = read(SOURCE / 'config.json')
    inputs = dict(prior['inputs'])
    inputs.update({str(SOURCE / name): _sha256(SOURCE / name) for name in ('report.json', 'config.json', 'artifacts.json')})
    if pilot is not None:
        if audit is None:
            raise ValueError('正式配置必须有独立审计')
        result = validate_pilot(pilot, audit)
        if result['identity']['code'] != execution_identity(FILES) or result['identity']['inputs'] != inputs:
            raise ValueError('短检身份不属于当前候选')
        inputs.update({str(Path(pilot).resolve() / name): _sha256(Path(pilot) / name)
                       for name in ('report.json', 'artifacts.json')})
        inputs[str(Path(audit).resolve())] = _sha256(Path(audit))
    config = {'kind': 'terminal_capture_v1', 'purpose': 'formal' if pilot else 'smoke',
              'source': str(SOURCE), 'inputs': inputs, 'code': execution_identity(FILES), 'runtime': runtime_identity(),
              'data': prior['data'], 'checkpoint': prior['checkpoint'],
              'indices': list(range(300)) if pilot else SMOKE, 'budget_seconds': 28800 if pilot else 180,
              'pilot': str(Path(pilot).resolve()) if pilot else None,
              'audit': str(Path(audit).resolve()) if audit else None, **PARAMETERS}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, config)
    return config


def precheck(path):
    config = read(path)
    if config['kind'] != 'terminal_capture_v1' or Path(config['source']).resolve() != SOURCE:
        raise ValueError('终端候选协议或原始来源不符')
    verify_files(config['inputs'])
    if config['code'] != execution_identity(FILES) or config['runtime'] != runtime_identity():
        raise ValueError('候选代码或运行环境漂移')
    if any(config[k] != value for k, value in PARAMETERS.items()):
        raise ValueError('候选固定参数被更改')
    prior = read(SOURCE / 'config.json')
    if config['data'] != prior['data'] or config['checkpoint'] != prior['checkpoint']:
        raise ValueError('数据/模型路径被更改')
    required = dict(prior['inputs'])
    required.update({str(SOURCE / name): _sha256(SOURCE / name) for name in ('report.json', 'config.json', 'artifacts.json')})
    if any(config['inputs'].get(k) != value for k, value in required.items()):
        raise ValueError('输入身份登记不完整')
    check_artifacts(SOURCE)
    if config['purpose'] == 'smoke':
        if config['indices'] != SMOKE or config['budget_seconds'] != 180:
            raise ValueError('固定短检范围漂移')
    elif config['purpose'] == 'formal':
        if config['indices'] != list(range(300)) or config['budget_seconds'] != 28800:
            raise ValueError('正式分母/预算漂移')
        validate_pilot(config['pilot'], config['audit'])
        for p in (Path(config['pilot'])/'report.json', Path(config['pilot'])/'artifacts.json', Path(config['audit'])):
            if config['inputs'].get(str(p.resolve())) != _sha256(p):
                raise ValueError('正式前置输入未冻结')
    else:
        raise ValueError('未知用途')
    return config


def baseline_rows(config):
    selected = set(config['indices'])
    cached = read(SOURCE / 'report.json')['rows']
    lt22, original = cached['control'], cached['native']
    return [r for r in original if r['dataset_index'] in selected], [r for r in lt22 if r['dataset_index'] in selected]


def inspect(source):
    source = Path(source).resolve()
    config = precheck(source / 'config.json')
    report = check_artifacts(source)
    if report['indices'] != config['indices'] or report['mode'] != config['purpose']:
        raise ValueError('分母或模式不一致')
    if report['identity'] != {'config_sha256': _sha256(source/'config.json'), 'code': config['code'], 'inputs': config['inputs']}:
        raise ValueError('报告身份不符')
    rows = report['rows']
    if [r['dataset_index'] for r in rows] != config['indices']:
        raise ValueError('完整回合账本缺失')
    for row in rows:
        episode = read(source/'control'/f'episode-{row["dataset_index"]:04d}.json')
        if episode['summary'] != row or len(episode['frames']) != row['steps']:
            raise ValueError('逐回合账本不一致')
        if not all(f['actual_contact_safe'] and f['selected_horizon_safe'] for f in episode['frames']):
            raise ValueError('约束账本出现非安全声明')
    actual = comparisons(*baseline_rows(config), rows)
    if actual != report['comparison'] or report['production_qualified'] is not False:
        raise ValueError('配对裁决/生产范围不符')
    qualified = bool(config['purpose'] == 'formal' and actual['pilot_eligible'] and len(rows) == 300
                     and actual['vs_native']['successes'] >= 210 and actual['vs_native']['collisions'] <= 30)
    if report['candidate_qualified'] != qualified:
        raise ValueError('正式联合裁决不符')
    return report


def run(path, *, formal=False):
    config = precheck(path)
    if formal != (config['purpose'] == 'formal'):
        raise ValueError('正式与短检不能混用')
    torch.set_num_threads(1)
    output = Path(path).resolve().parent / ('results' if formal else 'smoke')
    output.mkdir(parents=True, exist_ok=False)
    folder = output/'control'; folder.mkdir()
    atomic_write_json(output/'config.json', config)
    report = {'kind': config['kind'], 'status': 'running', 'mode': config['purpose'], 'indices': config['indices'],
              'identity': {'config_sha256': _sha256(Path(path)), 'code': config['code'], 'inputs': config['inputs']},
              'geometry_source': config['geometry_source'], 'production_qualified': False,
              'candidate_qualified': False, 'training_performed': False, 'automatic_deployment': False}
    atomic_write_json(output/'report.json', report)
    budget = VerificationBudget(config['budget_seconds'])
    native, previous = baseline_rows(config)
    vehicle = VehicleConfig(**matrix.load_dataset_manifest(Path(config['data']))['vehicle_model'])
    try:
        spec = matrix.ExperimentSpec(name='terminal_capture_control', bev_source='sensor', trajectory_source='network',
                                     executor='mpc_vehicle', description='GT footprint constrained MPC plus explicit terminal rotation')
        with terminal_scope(folder, vehicle, {r['dataset_index']: r for r in native}, budget) as rows:
            raw = matrix.run_validation_experiment(
                spec, data_path=config['data'], checkpoint_path=config['checkpoint'], output_path=folder/'raw-report.json',
                indices=config['indices'], samples=0, max_steps=600, replan_every=10, control_seed=0,
                progress=lambda n,total,b: print(f'[terminal {n}/{total}] #{b.result.meta["dataset_index"]} {b.result.failure or "success"}', flush=True))
        if raw['reconstruct_failures'] or len(rows) != len(config['indices']):
            raise ValueError('候选执行未覆盖完整分母')
        precheck(path)
        paired = comparisons(native, previous, rows)
        qualified = bool(formal and paired['pilot_eligible'] and len(rows) == 300
                         and paired['vs_native']['successes'] >= 210 and paired['vs_native']['collisions'] <= 30)
        report.update(status='completed', rows=rows, comparison=paired, candidate_qualified=qualified,
                      elapsed_seconds=time.perf_counter()-budget.started)
        atomic_write_json(output/'report.json', report)
        text = ['# 终端朝向接管候选', '', f'状态completed；模式{config["purpose"]}；candidate_qualified={qualified}。', '',
                f'原生对照：{paired["vs_native"]}', f'LT22对照：{paired["vs_lt22"]}', '',
                '约束来自仿真完整几何；production_qualified=false。终端系统接管不是纯网络学习收益，S6远端参考仍未修复。']
        (output/'summary.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
        atomic_write_json(output/'artifacts.json', {p.relative_to(output).as_posix(): _sha256(p)
                                                   for p in output.rglob('*') if p.is_file()})
        return report
    except BaseException as exc:
        report.update(status='incomplete', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.perf_counter()-budget.started)
        atomic_write_json(output/'report.json', report)
        raise
