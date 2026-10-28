"""全量安全前缀验证的冻结交接；正式run仅由用户启动。"""
from pathlib import Path
import sys
from scripts.diagnose_closed_loop_failures import _sha256
from scripts.prepare_recovery_balance_ab import runtime_identity
from training.reporting import atomic_write_json
from verification import safe_prefix_preflight as study


def inspect(output):
    output = Path(output).resolve()
    report, identity = study.read(output/'report.json'), study.read(output/'identity.json')
    artifacts = study.read(output/'artifacts.json')
    required = {'report.json', 'identity.json', 'raw-report.json'}
    required |= {f'episode-{i:04d}.{suffix}' for i in identity['indices'] for suffix in ('json', 'npz')}
    if report.get('stage') == 'formal':
        required.add('formal-config.json')
        precheck(output/'formal-config.json')
    if not required <= artifacts.keys() or report.get('status') != 'completed':
        raise ValueError('未完成或产物登记缺失')
    for name, sha in artifacts.items():
        path = (output/name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or _sha256(path) != sha:
            raise ValueError('产物身份漂移')
    if identity['code'] != study.execution_identity() or identity['runtime'] != runtime_identity():
        raise ValueError('短检代码或环境已漂移，需版本化重检，不能重冻旧结果')
    config, baseline = study.validate_baseline()
    if identity['inputs'] != {k: config['inputs'][k] for k in ('data', 'manifest', 'checkpoint', 'historical')}:
        raise ValueError('短检输入身份不一致')
    raw = study.read(output/'raw-report.json')
    indices = [r['dataset_index'] for r in report['rows']]
    if (indices != identity['indices'] or indices != report['indices']
            or raw['protocol']['evaluated_indices'] != indices or raw['reconstruct_failures']):
        raise ValueError('分母不一致')
    for row, episode in zip(report['rows'], raw['episodes'], strict=True):
        if study.read(output/f'episode-{row["dataset_index"]:04d}.json') != row or any(row[k] != episode[k]
                for k in ('success', 'collision', 'failure', 'steps', 'final_pos_err', 'final_yaw_err')):
            raise ValueError('原始/逐例/汇总不一致')
    acceptance = study.summarize_candidate(report['rows'], baseline, study.read(config['inputs']['historical']['path']),
                                          formal=report['stage'] == 'formal')
    if any(report.get(k) != v for k, v in acceptance.items()):
        raise ValueError('联合裁决与逐例证据不一致')
    return report


def prepare(path, pilot):
    path, pilot = Path(path).resolve(), Path(pilot).resolve()
    if path.exists():
        raise FileExistsError('冻结配置不得覆盖')
    report = inspect(pilot)
    if (report['stage'] != 'expanded' or report['indices'] != study.select_indices('expanded')
            or report['lost_lt32_successes'] or report['lost_native_successes_in_evaluated_set']
            or report['lost_historical_successes_in_evaluated_set'] or not report['safety_audit_clean']
            or not report['gained_successes'] or not report['all_three_regressions_restored']):
        raise ValueError('固定扩展短检尚有旧能力回退/安全失败，不能准备正式任务')
    config, _ = study.validate_baseline()
    inputs = {k: config['inputs'][k] for k in ('data', 'manifest', 'checkpoint', 'historical')}
    for key, p in {'baseline_report': study.BASELINE/'report.json', 'baseline_artifacts': study.BASELINE/'artifacts.json',
                   'baseline_config': study.BASELINE/'config.json', 'pilot_report': pilot/'report.json',
                   'pilot_identity': pilot/'identity.json', 'pilot_artifacts': pilot/'artifacts.json'}.items():
        inputs[key] = {'path': str(p.resolve()), 'sha256': _sha256(p)}
    frozen = {'schema': 'safe_prefix_formal_v1', 'indices': list(range(300)), 'inputs': inputs,
        'code': study.execution_identity(), 'runtime': runtime_identity(), 'budget_seconds': 28800,
        'protocol': {'control_seed': 0, 'max_steps': 600, 'replan_every': 10, 'mpc_dt': .1,
                     'horizon': 10, 'model_dt': .2, 'stop_threshold': .5, 'required_margin_m': .2}}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, frozen)
    return precheck(path)


def precheck(path):
    config = study.read(path)
    if config.get('schema') != 'safe_prefix_formal_v1' or config['indices'] != list(range(300)) or config['budget_seconds'] != 28800:
        raise ValueError('正式身份/分母/预算不符')
    if config['protocol'] != {'control_seed': 0, 'max_steps': 600, 'replan_every': 10, 'mpc_dt': .1,
        'horizon': 10, 'model_dt': .2, 'stop_threshold': .5, 'required_margin_m': .2}:
        raise ValueError('正式协议漂移')
    if Path(sys.executable).resolve() != Path('D:/conda/envs/endtoend-parking/python.exe').resolve():
        raise ValueError('必须使用endtoend-parking环境')
    if config['code'] != study.execution_identity() or config['runtime'] != runtime_identity():
        raise ValueError('正式代码/环境身份漂移')
    for item in config['inputs'].values():
        if _sha256(Path(item['path'])) != item['sha256']:
            raise ValueError('冻结输入漂移')
    baseline_config, _ = study.validate_baseline()
    if any(config['inputs'][k] != baseline_config['inputs'][k] for k in ('data', 'manifest', 'checkpoint', 'historical')):
        raise ValueError('正式输入不属于LT32同模型/同任务配对')
    inspect(Path(config['inputs']['pilot_report']['path']).parent)
    return config


def run(config_path, output):
    config = precheck(config_path)
    result = study.run('formal', output, formal_config=config)
    try:
        precheck(config_path)
    except BaseException as exc:
        result.update(status='incomplete', error=f'正式运行后身份复核失败：{type(exc).__name__}: {exc}')
        atomic_write_json(Path(output)/'report.json', result)
        raise
    return result
