"""STOP-ACCEPT-005: paired gate/accept fitting and complete frozen audit."""
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from experiments.emitted_output_data import RECIPE, read_json, file_digest
from experiments.emitted_output_study import batch_plan
from scripts.accepted_native_objectives import ARMS, update
from scripts.recertified_emitted_fit import native_evaluate
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


def expected_files():
    names = ['config.json', 'report.json']
    for seed in RECIPE['seeds']:
        names.append(f'plan-{seed}.json')
        for arm in ARMS:
            names.append(f'{seed}-{arm}/history.json')
            for step in RECIPE['checkpoints']:
                names.extend(f'{seed}-{arm}/{name}-{step:04d}.{ext}' for name, ext in
                             (('native', 'json'), ('native', 'npz'), ('update', 'pt')))
    return names


def baseline(config):
    previous = config['source']
    old = read_json(Path(previous['lt43_manifest']['path']).parent / 'report.json')
    older = read_json(Path(previous['lt43_config']['source']['baseline_manifest']['path']).parent / 'report.json')
    return ([dict(seed=a['seed'], arm='LT43_AB', final=a['final']) for a in old['arms'] if a['arm'] == 'AB']
            + [dict(seed=a['seed'], arm='LT42_joint', final=a['final']) for a in older['arms'] if a['arm'] == 'joint'])


def decision(arms):
    if len(arms) != 4 or {(a['seed'], a['arm']) for a in arms} != {(s, a) for s in RECIPE['seeds'] for a in ARMS}:
        raise ValueError('four complete new arms required')
    passed = all(a['final']['passed'] for a in arms if a['arm'] == 'accept')
    return dict(passed=passed, candidate_arm='accept',
                decision='ready_for_separate_fixed_and_rolling_plan' if passed else 'stop_acceptance_recipe_no_more_dose',
                diagnostic_passed=all(a['final']['passed'] for a in arms if a['arm'] == 'gate'),
                candidate_qualified=False, production_qualified=False)


def comparison(arms, reused):
    keys = lambda final: {(r['source_index'], r['frame_index']) for r in final['rows'] if r['passed']}
    rows = []
    for a in arms:
        for old in reused:
            if old['seed'] != a['seed']:
                continue
            before, after = keys(old['final']), keys(a['final'])
            rows.append(dict(seed=a['seed'], arm=a['arm'], baseline_arm=old['arm'],
                             gained=[list(k) for k in sorted(after - before)], lost=[list(k) for k in sorted(before - after)]))
    return rows


def coverage(history):
    counts = {}
    for row in history:
        if len(row['samples']) != 8 or len(row['stop_rows']) != 8:
            raise ValueError('eight samples and stop records required')
        for sample, stop in zip(row['samples'], row['stop_rows']):
            if (stop['status'] not in ('accepted', 'early', 'late_or_gap', 'no_window')
                    or not 1 <= stop['native_length'] <= 320 or not 0 <= stop['eligible_count'] <= 320):
                raise ValueError('invalid stop diagnostic')
            key = f"{sample['source_index']}:{sample['frame_index']}"
            count = counts.setdefault(key, dict(exposures=0, accepted=0, early=0, late_or_gap=0, no_window=0))
            count['exposures'] += 1
            count[stop['status']] += 1
        if (row['stop_deferred'] != sum(r['status'] == 'no_window' for r in row['stop_rows'])
                or row['already_accepted'] != sum(r['status'] == 'accepted' for r in row['stop_rows'])):
            raise ValueError('coverage totals mismatch')
    return counts


def check_origin(config, seed, evaluation, output):
    old = Path(config['source']['lt43_manifest']['path']).parent / f'{seed}-AB/native-0000'
    if evaluation != read_json(old.with_suffix('.json')):
        raise ValueError('original native origin changed')
    with np.load(Path(output).with_suffix('.npz')) as new, np.load(old.with_suffix('.npz')) as saved:
        if set(new.files) != set(saved.files):
            raise ValueError('origin fields changed')
        for key in new.files:
            np.testing.assert_array_equal(new[key], saved[key])


def execute(config, samples, data, cases, output, budget, final_check):
    reused = baseline(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', config_sha256=config['config_sha256'], arms=[], baseline=reused,
                  new_arms=4, reused_arms=4, training_performed=False, validation_used_for_fitting=False,
                  closed_loop_evaluated=False, original_lt34_fit_unlocked=False,
                  candidate_qualified=False, production_qualified=False)
    atomic_write_json(output / 'config.json', config)
    atomic_write_json(output / 'report.json', report)
    try:
        loaded = load_model_checkpoint(config['checkpoint']['path'])
        for seed in RECIPE['seeds']:
            plan = batch_plan(samples, seed, 200)
            atomic_write_json(output / f'plan-{seed}.json', plan)
            for arm in ARMS:
                budget.check()
                torch.manual_seed(seed)
                model = deepcopy(loaded.model)
                optimizer = torch.optim.Adam(model.parameters(), lr=RECIPE['learning_rate'])
                folder = output / f'{seed}-{arm}'
                folder.mkdir()
                history, snapshots = [], []
                for step in range(201):
                    budget.check()
                    if step:
                        history.append(update(model, optimizer, [samples[j] for j in plan[step - 1]], cases, arm, seed, step))
                        report['training_performed'] = True
                    if step in RECIPE['checkpoints']:
                        evaluation = native_evaluate(model, samples, data, cases, folder / f'native-{step:04d}')
                        if step == 0:
                            check_origin(config, seed, evaluation, folder / 'native-0000')
                        torch.save(dict(schema_version=1, model_name=loaded.model_name, model_config=loaded.model_config,
                                        model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), epoch=-1,
                                        experiment='STOP-ACCEPT', experimental_only=True, seed=seed, arm=arm, updates=step,
                                        config_sha256=config['config_sha256']), folder / f'update-{step:04d}.pt')
                        snapshots.append(dict(step=step, **evaluation))
                        atomic_write_json(folder / 'history.json', history)
                        print(f'FIT {seed} {arm} step={step}/200 initial={evaluation["initial_passed"]}/8 all={evaluation["all_frames_passed"]}/32', flush=True)
                report['arms'].append(dict(seed=seed, arm=arm, updates=200, checkpoints=snapshots,
                                           final=snapshots[-1], coverage=coverage(history)))
                atomic_write_json(output / 'report.json', report)
        final_check()
        budget.check()
        report.update(status='completed', gate=decision(report['arms']), comparison=comparison(report['arms'], reused),
                      elapsed_seconds=time.perf_counter() - budget.started)
        atomic_write_json(output / 'report.json', report)
        atomic_write_json(output / 'artifacts.json', {str(p.resolve()): file_digest(p) for p in output.rglob('*') if p.is_file()})
        return report
    except BaseException as error:
        report.update(status='incomplete', error=str(error))
        atomic_write_json(output / 'report.json', report)
        raise


def inspect(config, samples, data, cases, output, budget):
    output = Path(output).resolve()
    report, manifest = read_json(output / 'report.json'), read_json(output / 'artifacts.json')
    expected = {str((output / name).resolve()) for name in expected_files()}
    actual = {str(p.resolve()) for p in output.rglob('*') if p.is_file() and p.name != 'artifacts.json'}
    if (set(manifest) != expected or actual != expected or read_json(output / 'config.json') != config
            or any(file_digest(p) != h for p, h in manifest.items())):
        raise ValueError('complete immutable 56 artifacts/config required')
    if (report['status'] != 'completed' or report['config_sha256'] != config['config_sha256']
            or report['baseline'] != baseline(config) or report['new_arms'] != 4 or report['reused_arms'] != 4
            or not report['training_performed'] or any(report[k] for k in
            ('validation_used_for_fitting', 'closed_loop_evaluated', 'original_lt34_fit_unlocked', 'candidate_qualified', 'production_qualified'))):
        raise ValueError('completion/qualification mismatch')
    parent = load_model_checkpoint(config['checkpoint']['path']).model.state_dict()
    rebuilt, evaluations = [], 0
    with tempfile.TemporaryDirectory(prefix='accepted-native-audit-') as temporary:
        for seed in RECIPE['seeds']:
            plan = batch_plan(samples, seed, 200)
            if read_json(output / f'plan-{seed}.json') != plan:
                raise ValueError('sample plan changed')
            for arm in ARMS:
                folder = output / f'{seed}-{arm}'
                history = read_json(folder / 'history.json')
                if len(history) != 200:
                    raise ValueError('fixed update budget changed')
                for step, row in enumerate(history, 1):
                    expected_samples = [dict(source_index=samples[j]['source_index'], frame_index=samples[j]['frame_index']) for j in plan[step - 1]]
                    values = [v for k, v in row.items() if k.endswith('loss') or k == 'unclipped_gradient_norm']
                    if row['step'] != step or row['samples'] != expected_samples or not np.isfinite(values).all():
                        raise ValueError('history/finite/sample mismatch')
                if (len(history[0]['head_gradient_norms']) != 2 or not np.isfinite(history[0]['head_gradient_norms']).all()
                        or min(history[0]['head_gradient_norms']) <= 0):
                    raise ValueError('missing head gradients')
                snapshots = []
                for step in RECIPE['checkpoints']:
                    budget.check()
                    path = folder / f'update-{step:04d}.pt'
                    payload = torch.load(path, map_location='cpu', weights_only=False)
                    identity = dict(experiment='STOP-ACCEPT', seed=seed, arm=arm, updates=step, config_sha256=config['config_sha256'])
                    if any(payload.get(k) != v for k, v in identity.items()) or not all(torch.isfinite(v).all() for v in payload['model_state'].values()):
                        raise ValueError('checkpoint identity/finite mismatch')
                    state = payload['optimizer_state']['state']
                    if step == 0:
                        if state or set(parent) != set(payload['model_state']):
                            raise ValueError('origin/reset mismatch')
                        for k, value in parent.items():
                            torch.testing.assert_close(value, payload['model_state'][k], rtol=0, atol=0)
                    elif not state or any(int(v['step']) != step for v in state.values()):
                        raise ValueError('Adam update count changed')
                    target = Path(temporary) / 'native'
                    evaluation = native_evaluate(load_model_checkpoint(path).model, samples, data, cases, target)
                    if evaluation != read_json(folder / f'native-{step:04d}.json'):
                        raise ValueError('hard evaluation mismatch')
                    with np.load(target.with_suffix('.npz')) as fresh, np.load(folder / f'native-{step:04d}.npz') as saved:
                        if set(fresh.files) != set(saved.files):
                            raise ValueError('native fields changed')
                        for key in fresh.files:
                            np.testing.assert_array_equal(fresh[key], saved[key])
                    if step == 0:
                        check_origin(config, seed, evaluation, target)
                    snapshots.append(dict(step=step, **evaluation))
                    evaluations += 32
                    print(f'AUDIT {seed} {arm} step={step} inputs=32', flush=True)
                rebuilt.append(dict(seed=seed, arm=arm, updates=200, checkpoints=snapshots,
                                    final=snapshots[-1], coverage=coverage(history)))
    if (report['arms'] != rebuilt or report['gate'] != decision(rebuilt)
            or report['comparison'] != comparison(rebuilt, report['baseline'])):
        raise ValueError('final decision/coverage mismatch')
    return dict(status='verified', tasks=8, samples=32, new_arms=4, reused_arms=4, updates=800,
                checkpoints=16, native_evaluations=evaluations, artifacts=56, gate=report['gate'], closed_loop_evaluated=False)
