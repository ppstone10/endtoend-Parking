"""EMIT-JOINT staged experiment: prepare/precheck/smoke/teacher/fit/inspect.

Formal stages are manual, finite, fresh-output only. No automatic next stage.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.emitted_output_data import (
    prepare, precheck, run_teacher, load_cohort, initial_samples, read_json,
    file_digest, atomic_write_json, RECIPE,
)
from experiments.emitted_output_joint import native_lengths
from experiments.emitted_output_study import update, run_fit, fitting_decision
from training.checkpoint import load_model_checkpoint


class StageBudget:
    def __init__(self, seconds):
        self.started = time.perf_counter()
        self.deadline = self.started + seconds

    def check(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError('declared stage wall-time exhausted; retain incomplete artifacts, no retry')


def smoke(prepared, output):
    budget = StageBudget(120)
    config = precheck(prepared)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', formal_training=False, candidate_qualified=False,
                  config_sha256=file_digest(Path(prepared)/'config.json'))
    atomic_write_json(output/'report.json', report)
    try:
        teacher = run_teacher(prepared, output/'teacher-smoke', budget, rows_to_run=1)
        report['teacher_smoke'] = teacher
        if not teacher['rows'][0]['passed']:
            report.update(status='completed', passed=False, decision='teacher_smoke_failed_no_update')
            atomic_write_json(output/'report.json', report)
            return report
        data, _, _ = load_cohort(prepared)
        # This one real update uses only the smoke teacher's verified initial input.
        sample = initial_samples(data)[0]
        loaded = load_model_checkpoint(config['inputs']['parent']['path'])
        report['updates'] = []
        for arm, weight in [('control',0.), ('joint',RECIPE['joint_weight'])]:
            budget.check(); torch.manual_seed(RECIPE['seeds'][0]); model = deepcopy(loaded.model)
            optimizer = torch.optim.Adam(model.parameters(), lr=RECIPE['learning_rate'])
            row = update(model,optimizer,[sample],seed=RECIPE['seeds'][0],step=1,
                         joint_weight=weight,inspect_gradients=bool(weight))
            row['arm'] = arm
            model.eval()
            with torch.no_grad():
                out = model.forward_with_stop(sample['bev'][None],sample['goal'][None],sample['state'][None])
            np.savez_compressed(output/f'{arm}-one-update.npz',points=out.points.numpy(),
                                logits=out.stop_logits.numpy(),native_lengths=native_lengths(out.stop_logits).numpy())
            row['head_parameter_delta_norms'] = [float((getattr(model.decoder,k).weight.detach()
                -getattr(loaded.model.decoder,k).weight.detach()).norm()) for k in ('point_head','stop_head')]
            if min(row['head_parameter_delta_norms']) <= 0:
                raise ValueError('real update did not affect both heads')
            report['updates'].append(row)
        base_control = report['updates'][0]['total_loss']
        base_joint = report['updates'][1]['base_loss']
        if not np.isclose(base_control,base_joint,rtol=2e-6,atol=2e-6):
            raise ValueError('paired original loss/data/TF initialization mismatch')
        precheck(prepared); budget.check()
        report.update(status='completed',passed=True, optimizer_steps_per_arm=1,
                      elapsed_seconds=time.perf_counter()-budget.started,
                      full_teacher_gate_passed=False, local_learning_gain_established=False)
        atomic_write_json(output/'report.json', report)
        return report
    except BaseException as exc:
        report.update(status='incomplete',passed=False,error=str(exc))
        atomic_write_json(output/'report.json', report)
        raise


def inspect(prepared, output):
    precheck(prepared)
    output = Path(output).resolve()
    report = read_json(output/'report.json')
    if report['status'] != 'completed' or report['config_sha256'] != file_digest(Path(prepared)/'config.json'):
        raise ValueError('stage incomplete or config mismatch')
    artifacts = read_json(output/'artifacts.json')
    actual_files = {str(p.resolve()) for p in output.rglob('*') if p.is_file() and p.name != 'artifacts.json'}
    if set(artifacts) != actual_files:
        raise ValueError('stage artifact set incomplete or changed')
    for path,digest in artifacts.items():
        if file_digest(path) != digest:
            raise ValueError(f'artifact drift: {path}')
    if 'arms' in report:
        if report['gate'] != fitting_decision(report['arms']):
            raise ValueError('fit decision does not match both fixed seeds')
        for arm in report['arms']:
            if arm['updates'] != 200 or [c['step'] for c in arm['checkpoints']] != [0,50,100,200]:
                raise ValueError('update/checkpoint denominator drift')
    else:
        from experiments.emitted_output_data import teacher_gate
        if not report['formal'] or report['gate'] != teacher_gate(report['rows']):
            raise ValueError('not a complete formal teacher gate')
    return dict(status='verified',gate=report['gate'],registered_artifacts=len(artifacts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['prepare','precheck','smoke','teacher','fit','inspect'],required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--teacher',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == 'prepare':
        result = prepare(args.prepared)
        result = dict(prepared=str(args.prepared),sources=result['sources'],regression=result['regression'])
    elif args.phase == 'precheck':
        config = precheck(args.prepared)
        result = dict(status='passed',sources=config['sources'],recipe=config['recipe'])
    else:
        if args.output is None:
            parser.error('--output required for this phase')
        if args.phase == 'smoke':
            result = smoke(args.prepared,args.output)
        elif args.phase == 'teacher':
            result = run_teacher(args.prepared,args.output,StageBudget(1800))
        elif args.phase == 'fit':
            if args.teacher is None:
                parser.error('--teacher required for fitting')
            result = run_fit(args.prepared,args.teacher,args.output,StageBudget(3600))
        else:
            result = inspect(args.prepared,args.output)
    if args.phase in ('smoke','teacher','fit'):
        result = {key:result[key] for key in ('status','passed','gate','decision','elapsed_seconds') if key in result}
        result['report'] = str(args.output/'report.json')
    import json
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
