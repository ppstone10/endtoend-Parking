"""Paired fixed-dose fitting and hard native-output checks for EMIT-JOINT.

No best-checkpoint selection, threshold calibration, retries or rollout expansion.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from experiments.emitted_output_data import (
    INDICES, RECIPE, precheck, load_cohort, case_at, pose_pass, reference_audit,
    load_authorized_samples, file_digest, atomic_write_json, read_json,
)
from experiments.emitted_output_joint import paired_objective, emitted_pose_loss, native_lengths
from interfaces import BEVTensor, GoalPose, VehicleState
from training.checkpoint import load_model_checkpoint


def batch_plan(samples, seed, updates):
    """One frame from each source per update, identical order and TF RNG in both arms."""
    rng = np.random.default_rng(seed)
    pools = {i:[j for j,s in enumerate(samples) if s['source_index'] == i] for i in INDICES}
    if any(not pool for pool in pools.values()):
        raise ValueError('missing fixed training source')
    return [[int(rng.choice(pools[int(i)])) for i in rng.permutation(INDICES)] for _ in range(updates)]


def update(model, optimizer, batch, *, seed, step, joint_weight, inspect_gradients=False):
    model.train()
    stack = lambda key: torch.stack([s[key] for s in batch])
    generator = torch.Generator().manual_seed(seed + step)
    optimizer.zero_grad(set_to_none=True)
    guided = model.forward_with_stop(stack('bev'), stack('goal'), stack('state'),
        teacher_points=stack('points'), teacher_forcing_ratio=.2, sampling_generator=generator)
    free = model.forward_with_stop(stack('bev'), stack('goal'), stack('state')) if joint_weight else None
    loss = paired_objective(guided.points, guided.stop_logits, stack('points'), stack('mask'),
        joint_weight=joint_weight, free_points=None if free is None else free.points,
        free_logits=None if free is None else free.stop_logits, goals=stack('goal'), tolerances=stack('tolerances'))
    record = dict(step=step, total_loss=float(loss.detach()),
                  samples=[dict(source_index=s['source_index'], frame_index=s['frame_index']) for s in batch])
    if free is not None:
        emitted = emitted_pose_loss(free.points, free.stop_logits, stack('goal'), stack('tolerances'))
        record.update(emitted_loss=float(emitted.detach()), base_loss=float((loss-joint_weight*emitted).detach()))
        if inspect_gradients:
            heads = (model.decoder.point_head.weight, model.decoder.stop_head.weight)
            gradients = torch.autograd.grad(emitted, heads, retain_graph=True)
            if any(not torch.isfinite(g).all() or not g.abs().sum() for g in gradients):
                raise ValueError('joint loss must have finite nonzero gradients in both actual heads')
            record['joint_head_gradient_norms'] = [float(g.norm()) for g in gradients]
    if not torch.isfinite(loss):
        raise ValueError('nonfinite loss; stop recipe, never retry automatically')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    record['unclipped_gradient_norm'] = float(norm)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite parameters after update')
    return record


def evaluate(model, samples, data, cases, destination):
    """B1 free outputs, checked against public predict. Preserve full horizon and logits."""
    model.eval(); rows = []; points = []; logits = []
    meta = data['bev_meta']
    with torch.no_grad():
        for sample in samples:
            out = model.forward_with_stop(sample['bev'][None], sample['goal'][None], sample['state'][None])
            length = int(native_lengths(out.stop_logits)[0])
            pred = out.points[0].numpy(); stop = out.stop_logits[0].numpy()
            native = model.predict(BEVTensor(sample['bev'].numpy(), resolution=meta['resolution'],
                extent=tuple(meta['extent']), channels=meta['channels'], camera_projection='per_view_v2'),
                GoalPose(*sample['goal'].tolist()), VehicleState(*sample['pose'], *sample['state'].tolist()))
            if native.horizon != length or native.dt != .2:
                raise ValueError('native inference length/dt mismatch')
            np.testing.assert_array_equal(native.points, pred[:length])
            goal = sample['goal'].numpy(); tolerances = sample['tolerances'].tolist()
            endpoint = pose_pass(pred[length-1], goal, *tolerances)
            expert_index = int(sample['mask'].sum()) - 1
            safety = reference_audit(pred[:length], sample['pose'], cases[sample['source_index']])
            row = dict(source_index=sample['source_index'], frame_index=sample['frame_index'],
                       native_length=length, teacher_length=expert_index+1, endpoint=endpoint,
                       teacher_index_endpoint=pose_pass(pred[expert_index], goal, *tolerances),
                       **safety)
            row['passed'] = bool(endpoint['joint_passed'] and all(safety.values()))
            rows.append(row); points.append(pred); logits.append(stop)
    np.savez_compressed(destination.with_suffix('.npz'), points=np.stack(points), logits=np.stack(logits),
                        goals=np.stack([s['goal'].numpy() for s in samples]),
                        targets=np.stack([s['points'].numpy() for s in samples]),
                        masks=np.stack([s['mask'].numpy() for s in samples]))
    initial = [r for r in rows if r['frame_index'] == 0]
    if len(initial) != 8 or {r['source_index'] for r in initial} != set(INDICES):
        raise ValueError('initial eight-task denominator changed')
    report = dict(initial_tasks=8, initial_passed=sum(r['passed'] for r in initial),
                  all_frames=len(rows), all_frames_passed=sum(r['passed'] for r in rows),
                  passed=all(r['passed'] for r in rows), rows=rows,
                  rolling_execution_evaluated=False)
    atomic_write_json(destination.with_suffix('.json'), report)
    return report


def fitting_decision(results):
    expected = {(seed, arm) for seed in RECIPE['seeds'] for arm in ('control','joint')}
    if {(r['seed'],r['arm']) for r in results} != expected or len(results) != 4:
        raise ValueError('paired seed/arm denominator incomplete')
    joint = [r for r in results if r['arm'] == 'joint']
    passed = all(r['final']['passed'] for r in joint)
    return dict(passed=passed, decision='ready_for_separate_fixed_and_rolling_plan' if passed else
                'stop_recipe_check_gradients_objective_conflict_representation',
                control_passed_by_seed={str(r['seed']):r['final']['passed'] for r in results if r['arm']=='control'},
                candidate_qualified=False, production_qualified=False)


def run_fit(prepared, teacher, destination, budget):
    config = precheck(prepared)
    samples = load_authorized_samples(prepared, teacher)  # before any optimizer or output creation
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', config_sha256=file_digest(Path(prepared)/'config.json'),
                  teacher_artifacts_sha256=file_digest(Path(teacher)/'artifacts.json'), arms=[], gate=None,
                  candidate_qualified=False, automatic_next_stage=False)
    atomic_write_json(destination/'report.json', report)
    try:
        data, manifest, vehicle = load_cohort(prepared)
        cases = {i:case_at(data,manifest,vehicle,j) for j,i in enumerate(INDICES)}
        loaded = load_model_checkpoint(config['inputs']['parent']['path'])
        for seed in RECIPE['seeds']:
            plan = batch_plan(samples, seed, RECIPE['updates'])
            atomic_write_json(destination/f'plan-{seed}.json', plan)
            for arm in ('control','joint'):
                budget.check(); torch.manual_seed(seed)
                model = deepcopy(loaded.model)
                optimizer = torch.optim.Adam(model.parameters(), lr=RECIPE['learning_rate'])
                arm_dir = destination/f'{seed}-{arm}'; arm_dir.mkdir()
                history = []; snapshots = []
                for step in range(RECIPE['updates']+1):
                    budget.check()
                    if step:
                        history.append(update(model,optimizer,[samples[j] for j in plan[step-1]],
                            seed=seed,step=step,joint_weight=RECIPE['joint_weight'] if arm=='joint' else 0.,
                            inspect_gradients=step==1 and arm=='joint'))
                    if step in RECIPE['checkpoints']:
                        diagnostic = evaluate(model,samples,data,cases,arm_dir/f'native-{step:04d}')
                        torch.save(dict(schema_version=1,model_name=loaded.model_name, model_config=loaded.model_config,
                            model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), epoch=-1,
                            experiment='EMIT-JOINT', seed=seed, arm=arm, updates=step,
                            config_sha256=report['config_sha256'], experimental_only=True),
                            arm_dir/f'update-{step:04d}.pt')
                        snapshots.append(dict(step=step, **diagnostic))
                        atomic_write_json(arm_dir/'history.json', history)
                        print(f'FIT {seed} {arm} step={step}/200 initial={diagnostic["initial_passed"]}/8 '
                              f'all={diagnostic["all_frames_passed"]}/{len(samples)}', flush=True)
                # Always use predeclared final step; earlier passing checkpoints cannot rescue failure.
                report['arms'].append(dict(seed=seed, arm=arm, updates=RECIPE['updates'],
                                           checkpoints=snapshots, final=snapshots[-1]))
                atomic_write_json(destination/'report.json', report)
        precheck(prepared); load_authorized_samples(prepared,teacher)
        report.update(status='completed', gate=fitting_decision(report['arms']))
        atomic_write_json(destination/'report.json', report)
        artifacts = {str(p.resolve()):file_digest(p) for p in destination.rglob('*') if p.is_file()}
        atomic_write_json(destination/'artifacts.json', artifacts)
        return report
    except BaseException as exc:
        report.update(status='incomplete', error=str(exc))
        atomic_write_json(destination/'report.json', report)
        raise
