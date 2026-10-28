"""统一生产分层验证入口；prepare/precheck不运行回合，正式批量任务由用户执行。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from experiments.component_capture import VerificationBudget, capture_native_e5
from experiments.component_analysis import analyze_captures
from experiments.component_verification import SCHEMA, acceptance, aggregate_layers, validate_baseline
from scripts.diagnose_closed_loop_failures import _sha256
from scripts.diagnose_reference_execution import code_identity
from scripts.prepare_recovery_balance_ab import runtime_identity
from training.reporting import atomic_write_json

BASELINE = ROOT / 'runs/validation/live-20261018-170806/full/E5/report.json'
INPUT_SHA = {'data': 'ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963',
             'checkpoint': 'd7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8'}
ENTRY_FILES = ['scripts/verify_closed_loop_components.py', 'scripts/evaluate_bev_fidelity.py',
               'scripts/evaluate_visited_state_trajectory.py', 'scripts/evaluate_single_cycle_tracking.py',
               'scripts/run_validation_suite.py']


def verification_code_identity():
    return {**code_identity(), **{str((ROOT / p).resolve()): _sha256(ROOT / p) for p in ENTRY_FILES}}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(config_path, baseline_path=BASELINE, mode='smoke'):
    destination = Path(config_path).resolve()
    if destination.exists():
        raise FileExistsError('配置已存在；不得覆盖冻结身份')
    baseline_path = Path(baseline_path).resolve()
    baseline = read_json(baseline_path)
    all_indices = [r['meta']['dataset_index'] for r in baseline['episodes']]
    validate_baseline(baseline, all_indices)
    if sorted(all_indices) != list(range(300)):
        raise ValueError('v1要求已归档同300纯E5基线')
    if mode == 'smoke':
        indices = [0, 163, 247, 292]  # 成功、S4碰撞、S6振荡、S8碰撞；看结果前固定。
    elif mode == 'regression':
        indices = [r['meta']['dataset_index'] for r in baseline['episodes'] if not r['success']]
    elif mode == 'formal':
        indices = all_indices
    else:
        raise ValueError('未知运行模式')
    inputs = {'baseline': baseline_path, 'data': Path(baseline['protocol']['data']),
              'checkpoint': Path(baseline['protocol']['checkpoint']),
              'manifest': Path(baseline['protocol']['data']).with_name('manifest.json')}
    frozen = {name: {'path': str(p.resolve()), 'sha256': _sha256(p)} for name, p in inputs.items()}
    if frozen['data']['sha256'] != INPUT_SHA['data']:
        raise ValueError('v1要求当前同300验证数据身份')
    if baseline_path == BASELINE.resolve() and frozen['checkpoint']['sha256'] != INPUT_SHA['checkpoint']:
        raise ValueError('R0模型与已知归档身份不符')
    config = {'schema': SCHEMA, 'mode': mode, 'indices': indices, 'inputs': frozen,
              'code': verification_code_identity(), 'runtime': runtime_identity(),
              'budget_seconds': 180 if mode == 'smoke' else 28800,
              'protocol': baseline['protocol'], 'vehicle_model': baseline['vehicle_model'],
              'failure_counts': dict(Counter(r['failure'] or 'success' for r in baseline['episodes'])),
              'production_qualified': False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, config)
    return config


def precheck(config_path):
    config = read_json(config_path)
    if config.get('schema') != SCHEMA or config['mode'] not in ('smoke', 'regression', 'formal'):
        raise ValueError('配置协议不受支持')
    expected_python = Path('D:/conda/envs/endtoend-parking/python.exe').resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise ValueError('必须使用endtoend-parking conda Python')
    for name, item in config['inputs'].items():
        if _sha256(Path(item['path'])) != item['sha256']:
            raise ValueError(f'冻结输入漂移：{name}')
    if config['code'] != verification_code_identity() or config['runtime'] != runtime_identity():
        raise ValueError('冻结代码或运行环境漂移，请审查后另建配置；禁止静默更新身份')
    baseline = read_json(config['inputs']['baseline']['path'])
    validate_baseline(baseline, config['indices'])
    if config['protocol'] != baseline['protocol'] or config['vehicle_model'] != baseline['vehicle_model']:
        raise ValueError('配置与基线不一致')
    if config['mode'] == 'formal' and config['indices'] != baseline['protocol']['evaluated_indices']:
        raise ValueError('正式运行必须完整同300分母和顺序')
    if config['mode'] == 'regression' and config['indices'] != [r['meta']['dataset_index'] for r in baseline['episodes'] if not r['success']]:
        raise ValueError('失败回归必须包含所有已知失败，不得择样')
    if not 0 < config['budget_seconds'] <= 28800:
        raise ValueError('运行预算无效')
    if config['mode'] == 'smoke' and (config['indices'] != [0, 163, 247, 292] or config['budget_seconds'] > 180):
        raise ValueError('冒烟范围/预算不可扩大')
    return config, baseline


def inspect_completed(output):
    """查看结果前核对完整产物和冻结代码；旧/损坏/未完成结果不能被入口背书。"""
    output = Path(output)
    report = read_json(output / 'report.json')
    if report.get('schema') != SCHEMA or report.get('status') != 'completed':
        raise ValueError('仅可查看完整统一验证报告，旧版诊断/部分结果不可作为通过证据')
    artifacts = {Path(name).as_posix(): digest for name, digest in read_json(output / 'artifacts.json').items()}
    required = ['config.json', 'report.json', 'capture/replayed-report.json', 'summary.md']
    required += [f'layers/{name}.json' for name in ('L1', 'L2', 'L3', 'L1_L2', 'L2_L3')]
    for index in report['coverage']['requested_indices']:
        required += [f'capture/episode-{index:04d}.{suffix}' for suffix in ('npz', 'json')]
    if not set(required) <= set(artifacts):
        raise ValueError('产物登记不完整')
    for name, digest in artifacts.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or _sha256(path) != digest:
            raise ValueError(f'产物缺失/漂移：{name}')
    config, _ = precheck(output / 'config.json')
    if _sha256(output / 'config.json') != report['identity']['config_sha256']:
        raise ValueError('报告与冻结配置身份不一致')
    actual_indices = [e['dataset_index'] for e in report['episodes']]
    if actual_indices != config['indices'] or actual_indices != report['coverage']['evaluated_indices']:
        raise ValueError('结果分母不完整')
    expected_layers = aggregate_layers(report['episodes'])
    expected_acceptance = acceptance(report['episodes'], full_coverage=config['mode'] == 'formal',
                                     checks_ok=all(l['contract_status'] == 'passed' for l in expected_layers.values()))
    if report['layers'] != expected_layers or report['acceptance'] != expected_acceptance:
        raise ValueError('报告汇总与逐回合证据不一致')
    return report


def write_summary(output, report):
    lines = ['# 生产闭环分层验证', '',
             f"运行状态：{report['status']}；模式：{report['mode']}；正式资格：{report['acceptance']['production_qualified']}。", '',
             '模块的结构重放通过不能代替安全与到达；boundary_mismatch只说明已测契约不一致，不能单独证明其导致全部失败。', '',
             '| 索引 | 场景 | E5结果 | 接触事件签名 | 已测接口错误数 | 最终位置/航向 |',
             '|---|---|---|---|---|---|']
    for e in report['episodes']:
        errors = sum(len(e['layers'][k]['errors']) for k in ('L1_L2', 'L2_L3'))
        lines.append(f"| {e['dataset_index']} | {e['scene_name']} | {e['failure'] or 'success'} | {e['contact_signature'] or '无接触'} | {errors} | {e['final_pos_err']:.4f}m / {e['final_yaw_err']:.4f}rad |")
    lines += ['', '完整细节：report.json的layers/episodes、layers/*.json与capture/episode-*.json/npz。',
              '输入声明冲突不等于已证实训练/部署分布错配；还须查看initial_dataset_bev_equal及数据生成身份。',
              'IoU、计划远端风险与单步反事实均为诊断量；本工具不修改生产链路、不训练、不自动晋升。']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(config_path, output):
    config, baseline = precheck(config_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('输出目录必须不存在；不覆盖、不自动恢复或续跑')
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / 'config.json', config)
    report = {'schema': SCHEMA, 'status': 'running', 'mode': config['mode'],
              'identity': {'config_sha256': _sha256(Path(config_path)), 'inputs': config['inputs'], 'code': config['code']},
              'acceptance': {'production_qualified': False}, 'episodes': []}
    atomic_write_json(output / 'report.json', report)
    budget = VerificationBudget(config['budget_seconds'])
    try:
        capture_native_e5(baseline, output / 'capture', config['indices'], budget)
        episodes = analyze_captures(output / 'capture', baseline, config['indices'], budget)
        precheck(config_path)
        layers = aggregate_layers(episodes)
        checks_ok = all(l['contract_status'] == 'passed' for l in layers.values())
        report.update(status='completed', episodes=episodes, layers=layers,
                      coverage={'requested_indices': config['indices'], 'evaluated_indices': [e['dataset_index'] for e in episodes],
                                'controls': sum(len(e['controls']) for e in episodes),
                                'predictions': sum(len(e['plans']) for e in episodes),
                                'all_control_steps_checked': True, 'all_prediction_inputs_checked': True},
                      acceptance=acceptance(episodes, full_coverage=config['mode'] == 'formal', checks_ok=checks_ok),
                      signature_counts=dict(Counter(e['observed_signature'] for e in episodes)),
                      contact_signature_counts=dict(Counter(e['contact_signature'] for e in episodes if e['collision'])),
                      elapsed_seconds=time.perf_counter() - budget.started)
        (output / 'layers').mkdir()
        for name, result in layers.items():
            atomic_write_json(output / 'layers' / f'{name}.json',
                              {'schema': SCHEMA, 'layer': name, 'identity': report['identity'], **result,
                               'acceptance': report['acceptance'],
                               'episodes': [{'dataset_index': e['dataset_index'], **e['layers'][name],
                                             'native_success': e['success'], 'native_collision': e['collision'],
                                             'native_failure': e['failure']} for e in episodes]})
        atomic_write_json(output / 'report.json', report)
        write_summary(output, report)
        paths = [p for p in output.rglob('*') if p.is_file()]
        atomic_write_json(output / 'artifacts.json', {p.relative_to(output).as_posix(): _sha256(p) for p in paths})
        return report
    except BaseException as exc:
        report.update(status='incomplete', error=f'{type(exc).__name__}: {exc}',
                      elapsed_seconds=time.perf_counter() - budget.started)
        atomic_write_json(output / 'report.json', report)
        raise


def main(stage='all', argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'precheck', 'run', 'inspect'], required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--mode', choices=['smoke', 'regression', 'formal'], default='smoke')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.phase != 'inspect' and args.config is None:
        parser.error('--config必填；旧参数请显式加--legacy-diagnostic')
    if args.phase in ('run', 'inspect') and args.output is None:
        parser.error('--output必填')
    if args.phase == 'prepare':
        config = prepare(args.config, args.baseline, args.mode)
        print(f'prepared {args.config} ({len(config["indices"])} tasks); no episodes run', flush=True)
    elif args.phase == 'precheck':
        config, _ = precheck(args.config)
        print(f'precheck passed ({len(config["indices"])} tasks); no episodes run', flush=True)
    else:
        report = run(args.config, args.output) if args.phase == 'run' else inspect_completed(args.output)
        if report.get('schema') != SCHEMA or report.get('status') != 'completed':
            raise ValueError('仅可查看完整统一验证报告，旧版诊断/部分结果不可作为通过证据')
        print(json.dumps({'status': report['status'], 'acceptance': report['acceptance'],
                          'layer': report['layers'] if stage == 'all' else report['layers'][stage]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
