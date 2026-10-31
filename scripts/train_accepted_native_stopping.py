"""LT44: acceptance-consistent stopping; formal fit/inspect are human-run."""
from pathlib import Path
import argparse
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.emitted_output_data import RECIPE, read_json, file_digest, file_identity
from scripts import train_native_output_contract as previous
from scripts import native_output_fit as old_fitting
from scripts import accepted_native_fit as fitting
from scripts.accepted_native_objectives import ARMS, OBJECTIVE
from scripts.accepted_native_preflight import old_output_regression, paired_short_updates
from scripts.diagnose_fixed_reference import read_shell_json
from scripts.train_emitted_output_joint import StageBudget
from training.reporting import atomic_write_json

OLD_PREPARED = ROOT / 'runs/prepared/native-output-contract-20261030-v1'
OLD = ROOT / 'runs/training/lt43-native-output-contract-20261030/fit'
PREFLIGHT = ROOT / 'runs/verification/accepted-native-20261031/preflight-v2'
FILES = ('accepted_native_objectives.py', 'accepted_native_fit.py', 'accepted_native_preflight.py',
         'train_accepted_native_stopping.py')
PROTOCOL = dict(arms=list(ARMS), seeds=RECIPE['seeds'], updates_per_arm=200, new_updates=800,
                checkpoints=[0, 50, 100, 200], candidate='accept', objective=OBJECTIVE,
                reused='LT43_AB_and_LT42_joint_read_only', preflight_seconds=90, fit_seconds=7200,
                inspect_seconds=7200, original_lt34_fit_unlocked=False, lt40_rejection_preserved=True,
                validation_used_for_fitting=False, automatic_next_stage=False)


def source_identity():
    old_config = previous.precheck(OLD_PREPARED)
    expected = {str((OLD / n).resolve()) for n in old_fitting.expected_files()}
    manifest = read_json(OLD / 'artifacts.json')
    actual = {str(p.resolve()) for p in OLD.rglob('*') if p.is_file() and p.name != 'artifacts.json'}
    if set(manifest) != expected or actual != expected or any(file_digest(p) != h for p, h in manifest.items()):
        raise ValueError('complete immutable LT43 artifacts required')
    execution = dict(old_config, config_sha256=file_digest(OLD_PREPARED / 'config.json'))
    report = read_json(OLD / 'report.json')
    inspected = read_shell_json(OLD.parent / 'inspect.log', last_line=True)
    if (read_json(OLD / 'config.json') != execution or report['status'] != 'completed' or not report['training_performed']
            or report['gate']['passed'] or report['validation_used_for_fitting'] or report['closed_loop_evaluated']
            or report['candidate_qualified'] or report['production_qualified']
            or inspected.get('status') != 'verified' or inspected.get('gate') != report['gate']
            or any(inspected.get(k) != v for k, v in dict(tasks=8, samples=32, new_arms=6, reused_arms=2,
                                                       updates=1200, checkpoints=24, native_evaluations=768, artifacts=82).items())):
        raise ValueError('verified negative LT43 identity required')
    if read_shell_json(OLD.parent / 'console.log', last_line=True)['gate'] != report['gate']:
        raise ValueError('LT43 console decision mismatch')
    for stage, name in (('fit', 'exit.json'), ('inspect', 'inspect-exit.json')):
        if read_shell_json(OLD.parent / name) != dict(task='LT43', stage=stage, exit_code=0):
            raise ValueError('LT43 stage identity mismatch')
    for seed in RECIPE['seeds']:
        if read_json(OLD / f'plan-{seed}.json') != read_json(previous.OLD / f'plan-{seed}.json'):
            raise ValueError('paired original batch plan changed')
    code = {**old_config['source']['code'], **{str((ROOT / 'scripts' / name).resolve()): file_digest(ROOT / 'scripts' / name) for name in FILES}}
    return dict(schema='accepted_native_source_v1', code=code, protocol=PROTOCOL, lt43_config=old_config,
                lt43_prepared=file_identity(OLD_PREPARED / 'config.json'), lt43_manifest=file_identity(OLD / 'artifacts.json'),
                lt43_inspect=file_identity(OLD.parent / 'inspect.log'),
                lt43_exits=[file_identity(OLD.parent / name) for name in ('exit.json', 'inspect-exit.json')],
                checkpoint=old_config['checkpoint'])


def supervision(source):
    return previous.supervision(source['lt43_config']['source'])


def preflight(output):
    source = source_identity()
    samples = supervision(source)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    budget = StageBudget(90)
    report = dict(status='running', source=source, formal_training=False, weights_reusable_as_formal_origin=False)
    atomic_write_json(output / 'context.json', source)
    atomic_write_json(output / 'report.json', report)
    try:
        data, cases = previous.previous.training_cases()
        static = old_output_regression(samples, cases, OLD, budget)
        atomic_write_json(output / 'static-native.json', static)
        parent, updates = paired_short_updates(source, samples, data, cases, output, budget)
        if source_identity() != source:
            raise ValueError('source changed during preflight')
        supervision(source)
        budget.check()
        report.update(status='passed', parent=parent, paired_updates=updates,
                      static_summary={k: static[k] for k in ('passed', 'old_inputs', 'objective_checks', 'accepted_zero_pressure')},
                      elapsed_seconds=time.perf_counter() - budget.started)
        atomic_write_json(output / 'report.json', report)
        atomic_write_json(output / 'artifacts.json', {str(p.resolve()): file_digest(p) for p in output.iterdir() if p.is_file()})
        return report
    except BaseException as error:
        report.update(status='incomplete', error=str(error))
        atomic_write_json(output / 'report.json', report)
        raise


def frozen_config():
    source = source_identity()
    supervision(source)
    names = ['context.json', 'report.json', 'static-native.json'] + [f'{name}.{ext}' for name in
            ('parent-native', 'gate-post-native', 'accept-post-native') for ext in ('json', 'npz')]
    manifest = read_json(PREFLIGHT / 'artifacts.json')
    expected = {str((PREFLIGHT / n).resolve()) for n in names}
    actual = {str(p.resolve()) for p in PREFLIGHT.iterdir() if p.is_file() and p.name != 'artifacts.json'}
    if set(manifest) != expected or actual != expected or any(file_digest(p) != h for p, h in manifest.items()):
        raise ValueError('complete current preflight artifacts required')
    report, static = read_json(PREFLIGHT / 'report.json'), read_json(PREFLIGHT / 'static-native.json')
    if (report['status'] != 'passed' or report['source'] != source or read_json(PREFLIGHT / 'context.json') != source
            or report['formal_training'] or report['weights_reusable_as_formal_origin'] or not static['passed']
            or static['old_inputs'] != 64 or static['objective_checks'] != 128
            or [r['arm'] for r in report['paired_updates']] != list(ARMS)
            or any(r['record']['step'] != 1 or not r['non_stop_components_unchanged'] for r in report['paired_updates'])):
        raise ValueError('bounded preflight authorization failed')
    return dict(schema='accepted_native_fit_v1', source=source, checkpoint=source['checkpoint'],
                preflight_manifest=file_identity(PREFLIGHT / 'artifacts.json'), candidate_qualified=False, automatic_training=False)


def prepare(directory):
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError('new prepared directory required')
    config = frozen_config()
    directory.mkdir(parents=True, exist_ok=False)
    atomic_write_json(directory / 'config.json', config)
    return config


def precheck(directory):
    config = read_json(Path(directory) / 'config.json')
    if config != frozen_config():
        raise ValueError('source/input/preflight drift; never edit hashes')
    return config


def fit(directory, output, *, audit=False):
    config = precheck(directory)
    samples = supervision(config['source'])  # authorization precedes output/optimizer creation
    config = dict(config, config_sha256=file_digest(Path(directory) / 'config.json'))
    data, cases = previous.previous.training_cases()
    if audit:
        result = fitting.inspect(config, samples, data, cases, output, StageBudget(7200))
        precheck(directory)
        return result
    return fitting.execute(config, samples, data, cases, output, StageBudget(7200),
                           lambda: (precheck(directory), supervision(config['source'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('preflight', 'prepare', 'precheck', 'fit', 'inspect'))
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == 'preflight':
        if args.output is None:
            parser.error('--output required')
        r = preflight(args.output)
        result = dict(status=r['status'], updates=2, static=r['static_summary'], elapsed_seconds=r['elapsed_seconds'])
    else:
        if args.prepared is None:
            parser.error('--prepared required')
        if args.phase in ('prepare', 'precheck'):
            r = prepare(args.prepared) if args.phase == 'prepare' else precheck(args.prepared)
            result = dict(status='prepared' if args.phase == 'prepare' else 'passed', tasks=8, samples=32,
                          new_arms=4, reused_arms=4, updates=800, source_files=len(r['source']['code']))
        else:
            if args.output is None:
                parser.error('--output required')
            r = fit(args.prepared, args.output, audit=args.phase == 'inspect')
            result = r if args.phase == 'inspect' else {k: r[k] for k in ('status', 'gate', 'elapsed_seconds')}
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
