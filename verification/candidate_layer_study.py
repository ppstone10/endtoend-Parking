"""当前候选五层采集/分析的冻结与独立阶段编排。"""
from pathlib import Path
import time
import numpy as np
import torch

from experiments import validation_matrix as matrix
from experiments.component_capture import VerificationBudget
from sim import VehicleConfig
from training.reporting import atomic_write_json
from scripts.diagnose_closed_loop_failures import _sha256
from scripts.prepare_recovery_balance_ab import runtime_identity
from verification.identity import ROOT, execution_identity, verify_files
from verification.terminal_study import read
from verification.feasible_reference_study import FILES as PRIOR_FILES, SMOKE, inspect as inspect_source
from verification.feasible_reference_candidate import feasible_reference_scope
from verification.candidate_layer_capture import capture_inputs_scope

SOURCE = ROOT/'runs/verification/feasible-reference-pilot-20261026/smoke'
TRAIN_REPORT = ROOT/'runs/training/v40-lineage/net-v1/report.json'
FILES = [*PRIOR_FILES, 'verification/candidate_layer_quality.py', 'verification/perception_calibration.py',
         'verification/candidate_layer_capture.py', 'verification/candidate_layer_analysis.py',
         'verification/candidate_layer_study.py', 'scripts/verify_candidate_layers.py',
         'scripts/calibrate_perception_acceptance.py']
FIXED = {'kind': 'candidate_five_layer_acceptance_v1', 'max_steps': 600, 'replan_every': 10,
         'control_seed': 0, 'mpc_dt': .1, 'mpc_horizon': 10, 'planning_margin': .2,
         'near_goal_position_tolerances': 3, 'geometry_source': 'simulator_ground_truth',
         'production_qualified': False}


def expected_inputs():
    prior = read(SOURCE/'config.json');inputs = dict(prior['inputs'])
    train = Path(read(TRAIN_REPORT)['data']['train']).resolve()
    for path in (SOURCE/'config.json', SOURCE/'report.json', SOURCE/'artifacts.json', TRAIN_REPORT, train):
        inputs[str(path)] = _sha256(path)
    return inputs, prior, train


def prepare(path, *, purpose='regression', profile=None):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError('新配置禁止覆盖')
    if purpose not in ('pilot', 'regression'):
        raise ValueError('仅支持短预检或固定14验收，不自动准备全300')
    inspect_source(SOURCE)
    inputs, prior, train = expected_inputs()
    profile = Path(profile).resolve() if profile is not None else None
    profile_files = {} if profile is None else {str(profile/name): _sha256(profile/name)
                                               for name in ('profile.json', 'report.json', 'config.json', 'artifacts.json')}
    c = {**FIXED, 'purpose': purpose, 'source': str(SOURCE), 'inputs': inputs,
         'data': prior['data'], 'checkpoint': prior['checkpoint'], 'train': str(train),
         'code': execution_identity(FILES), 'runtime': runtime_identity(),
         'indices': [0] if purpose == 'pilot' else SMOKE, 'profile': str(profile) if profile else None,
         'profile_files': profile_files}
    if profile:
        from verification.perception_calibration import load_profile
        load_profile(profile, c)
    path.parent.mkdir(parents=True, exist_ok=True);atomic_write_json(path, c)
    return c


def precheck(path):
    c = read(path)
    if any(c.get(k) != v for k, v in FIXED.items()) or c['source'] != str(SOURCE):
        raise ValueError('五层协议/来源漂移')
    if c['purpose'] not in ('pilot', 'regression') or c['indices'] != ([0] if c['purpose']=='pilot' else SMOKE):
        raise ValueError('分母/模式漂移')
    if c['code'] != execution_identity(FILES) or c['runtime'] != runtime_identity():
        raise ValueError('代码/运行时漂移')
    inputs, prior, train = expected_inputs()
    if c['inputs'] != inputs or c['train'] != str(train) or c['data'] != prior['data'] or c['checkpoint'] != prior['checkpoint']:
        raise ValueError('输入身份/路径漂移')
    verify_files(c['inputs']);verify_files(c['profile_files'])
    if c['profile']:
        names = ('profile.json', 'report.json', 'config.json', 'artifacts.json')
        if set(c['profile_files']) != {str(Path(c['profile'])/name) for name in names}:
            raise ValueError('校准冻结文件不完整')
        from verification.perception_calibration import load_profile
        load_profile(c['profile'], c)
    elif c['profile_files']:
        raise ValueError('无校准路径但存在校准文件')
    inspect_source(SOURCE)
    return c


def write_manifest(folder):
    atomic_write_json(folder/'artifacts.json', {p.relative_to(folder).as_posix(): _sha256(p)
                                               for p in folder.rglob('*') if p.is_file() and p.name != 'artifacts.json'})


def verify_manifest(folder, required):
    folder = Path(folder).resolve();manifest = read(folder/'artifacts.json')
    if not set(required) <= set(manifest):
        raise ValueError('五层登记文件缺失')
    for name, digest in manifest.items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or _sha256(path) != digest:
            raise ValueError('五层文件缺失或SHA漂移')
    return manifest


def assert_observer_invariance(folder, indices):
    original = read(SOURCE/'report.json')
    expected = {r['dataset_index']: r for r in original['rows']}
    for index in indices:
        old = SOURCE/'control';prefix = f'episode-{index:04d}'
        if read(folder/f'{prefix}.json')['summary'] != expected[index]:
            raise ValueError('只读观察改变原任务结果/计数')
        for stem in (prefix, f'retention-{index:04d}'):
            with np.load(folder/f'{stem}.npz', allow_pickle=False) as a, np.load(old/f'{stem}.npz', allow_pickle=False) as b:
                if set(a.files) != set(b.files) or any(not np.array_equal(a[key], b[key]) for key in a.files):
                    raise ValueError(f'#{index}观察改变原动作/状态/原始或使用参考')
        if read(folder/f'retention-{index:04d}.json') != read(old/f'retention-{index:04d}.json'):
            raise ValueError('观察改变证书/参考保持账本')


def capture(path):
    c = precheck(path);out = Path(path).resolve().parent/'capture'
    out.mkdir(parents=True, exist_ok=False);folder = out/'control';folder.mkdir()
    atomic_write_json(out/'config.json', c)
    report = {'status': 'running', 'indices': c['indices'], 'config_sha256': _sha256(Path(path)),
              'code': c['code'], 'production_qualified': False}
    atomic_write_json(out/'report.json', report)
    budget = VerificationBudget(90 if c['purpose']=='pilot' else 900)
    torch.set_num_threads(1)
    vehicle = VehicleConfig(**matrix.load_dataset_manifest(Path(c['data']))['vehicle_model'])
    try:
        spec = matrix.ExperimentSpec(name='candidate_five_layer_capture', bev_source='sensor',
                                     trajectory_source='network', executor='mpc_vehicle',
                                     description='LT26 unchanged candidate with actual five-layer input observation')
        with feasible_reference_scope(folder, vehicle, {}, budget) as (rows, ledgers):
            with capture_inputs_scope(folder, budget) as inputs:
                raw = matrix.run_validation_experiment(spec, data_path=c['data'], checkpoint_path=c['checkpoint'],
                        output_path=folder/'raw-report.json', indices=c['indices'], samples=0,
                        max_steps=600, replan_every=10, control_seed=0,
                        progress=lambda n, total, b: print(f'[layer-capture {n}/{total}] #{b.result.meta["dataset_index"]}', flush=True))
        if raw['reconstruct_failures'] or len(rows) != len(c['indices']) or len(inputs) != len(rows) or len(ledgers) != len(rows):
            raise ValueError('采集分母不完整')
        assert_observer_invariance(folder, c['indices']);precheck(path)
        report.update(status='completed', observer_exact_to_LT26=True, rows=rows, inputs=inputs,
                      elapsed_seconds=time.perf_counter()-budget.started)
        atomic_write_json(out/'report.json', report);write_manifest(out)
        return report
    except BaseException as exc:
        report.update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(out/'report.json', report);raise


def inspect_capture(path):
    c = precheck(path);out = Path(path).resolve().parent/'capture';r = read(out/'report.json')
    required = {'config.json', 'report.json', 'control/raw-report.json'}
    required.update(f'control/{stem}-{i:04d}.{ext}' for i in c['indices']
                    for stem in ('episode', 'retention', 'inputs') for ext in ('json', 'npz'))
    verify_manifest(out, required)
    if (r['status'] != 'completed' or r['indices'] != c['indices'] or r['code'] != c['code']
            or read(out/'config.json') != c or r['config_sha256'] != _sha256(Path(path))
            or not r['observer_exact_to_LT26'] or [x['dataset_index'] for x in r['rows']] != c['indices']):
        raise ValueError('采集完成/配置/分母不一致')
    assert_observer_invariance(out/'control', c['indices'])
    return r


def analyze(path):
    from verification.candidate_layer_analysis import analyze_episode
    from verification.candidate_layer_quality import aggregate_verdicts
    from verification.perception_calibration import load_profile
    from dataset import DatasetGenerator
    from training.checkpoint import load_model_checkpoint
    c = precheck(path);captured = inspect_capture(path);out = Path(path).resolve().parent/'analysis'
    out.mkdir(parents=True, exist_ok=False);atomic_write_json(out/'config.json', c)
    r = {'status': 'running', 'config_sha256': _sha256(Path(path)), 'indices': c['indices'],
         'capture_report_sha256': _sha256(out.parent/'capture/report.json'), 'code': c['code'],
         'production_qualified': False}
    atomic_write_json(out/'report.json', r);budget = VerificationBudget(120 if c['purpose']=='pilot' else 1800)
    try:
        torch.set_num_threads(1);data = DatasetGenerator.load(c['data'])
        manifest = matrix.load_dataset_manifest(Path(c['data']));vehicle = VehicleConfig(**manifest['vehicle_model'])
        loaded = load_model_checkpoint(c['checkpoint']);profile = load_profile(c['profile'], c)
        if profile is not None and profile['vehicle'] != vehicle.to_metadata():
            raise ValueError('校准车辆不匹配')
        rows = []
        for index in c['indices']:
            row = analyze_episode(out.parent/'capture/control', index, data, manifest, vehicle, loaded, profile, budget)
            rows.append(row);atomic_write_json(out/f'episode-{index:04d}.json', row)
            print(f'[layer-analysis {len(rows)}/{len(c["indices"])}] #{index}', flush=True)
        precheck(path)
        r.update(status='completed', rows=rows, layers=aggregate_verdicts(rows),
                 scope='one_case_preflight' if c['purpose']=='pilot' else 'fixed_14_known_regression',
                 elapsed_seconds=time.perf_counter()-budget.started,
                 successes=sum(row['facts']['success'] for row in rows),
                 all_five_modules_qualified=all(row['layers'][name]['module_qualified'] for row in rows for name in row['layers']))
        atomic_write_json(out/'report.json', r);write_manifest(out)
        return r
    except BaseException as exc:
        r.update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(out/'report.json', r);raise


def inspect(path):
    from verification.candidate_layer_quality import aggregate_verdicts, episode_verdict
    c = precheck(path);inspect_capture(path);out = Path(path).resolve().parent/'analysis'
    required = ['report.json', 'config.json', *(f'episode-{i:04d}.json' for i in c['indices'])]
    verify_manifest(out, required);r = read(out/'report.json')
    if (r['status'] != 'completed' or r['indices'] != c['indices'] or r['code'] != c['code']
            or read(out/'config.json') != c or r['config_sha256'] != _sha256(Path(path))
            or r['capture_report_sha256'] != _sha256(out.parent/'capture/report.json')
            or [x['dataset_index'] for x in r['rows']] != c['indices']
            or r['layers'] != aggregate_verdicts(r['rows']) or r['production_qualified'] is not False):
        raise ValueError('分析身份/分母/判定漂移')
    for row in r['rows']:
        if read(out/f'episode-{row["dataset_index"]:04d}.json') != row:
            raise ValueError('逐例分析与汇总不一致')
        errors = {name: value['contract_errors'] for name, value in row['layers'].items()}
        profiles = [q['perception_reference_conformance'] for q in row['queries']]
        if row['layers'] != episode_verdict(row['facts'], errors, profiles):
            raise ValueError('五层质量判定不能从事实与错误复现')
    expected_all = all(v['module_qualified'] for row in r['rows'] for v in row['layers'].values())
    if r['all_five_modules_qualified'] != expected_all or r['successes'] != sum(row['facts']['success'] for row in r['rows']):
        raise ValueError('总体资格/成功数与逐层判定不符')
    return r
