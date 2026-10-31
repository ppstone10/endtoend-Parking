"""STOP-ACCEPT-004: immutable old-output regression and bounded two-update checks."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import torch
from experiments.emitted_output_data import RECIPE, read_json
from experiments.emitted_output_study import batch_plan
from scripts.accepted_native_objectives import ARMS, acceptance_stop_loss, qualified_points, objective, update
from scripts.native_output_geometry import NativeGeometry
from scripts.native_output_objectives import objective as old_objective, stop_window_loss
from scripts.recertified_emitted_fit import native_evaluate
from training.checkpoint import load_model_checkpoint


def old_output_regression(samples, cases, folder, budget):
    rows = []
    for seed in RECIPE['seeds']:
        native = Path(folder) / f'{seed}-AB/native-0200'
        saved_rows = read_json(native.with_suffix('.json'))['rows']
        with np.load(native.with_suffix('.npz')) as saved:
            for j, (sample, previous) in enumerate(zip(samples, saved_rows)):
                budget.check()
                geo = NativeGeometry(cases[sample['source_index']]['planner']._collision_checker)
                points = torch.tensor(saved['points'][j])
                valid = qualified_points(points, sample, geo)[None]
                eligible = torch.nonzero(valid[0], as_tuple=False).flatten()
                index = int(eligible[0]) if len(eligible) else None
                if index != geo.terminal_index(points, sample['pose'], sample['goal'].numpy(), sample['tolerances'].numpy()):
                    raise ValueError('full set/old earliest hard window mismatch')
                for arm in ARMS:
                    logits = torch.tensor(saved['logits'][j:j + 1], requires_grad=True)
                    loss, info = acceptance_stop_loss(logits, valid, arm)
                    gradient = torch.autograd.grad(loss, logits)[0]
                    accepted = info[0]['status'] == 'accepted'
                    if accepted != previous['passed']:
                        raise ValueError('original native acceptance changed')
                    if accepted and (float(loss.detach()) != 0 or bool(gradient.any())):
                        raise ValueError('accepted native output must have exactly zero stop pressure')
                    if arm == 'gate' and not accepted:
                        torch.testing.assert_close(loss, stop_window_loss(logits, [index]), rtol=0, atol=0)
                    if index is not None and not accepted and float(loss.detach()) <= 0:
                        raise ValueError('covered zero-loss output must be accepted')
                    if not torch.isfinite(gradient).all():
                        raise ValueError('nonfinite static stop gradient')
                    rows.append(dict(seed=seed, arm=arm, source_index=sample['source_index'], frame_index=sample['frame_index'],
                                     old_passed=previous['passed'], **info[0], loss=float(loss.detach()), gradient_l1=float(gradient.abs().sum())))
    if len(rows) != 128:
        raise ValueError('all 64 old inputs must be checked for both new objectives')
    return dict(passed=True, old_inputs=64, objective_checks=128,
                accepted_zero_pressure=sum(r['old_passed'] for r in rows), rows=rows)


def paired_short_updates(config, samples, data, cases, output, budget):
    loaded = load_model_checkpoint(config['checkpoint']['path'])
    if (loaded.model.stop_threshold != .5 or loaded.model.dt != .2 or loaded.model.max_horizon != 320):
        raise ValueError('original native protocol required')
    parent = native_evaluate(loaded.model, samples, data, cases, Path(output) / 'parent-native')
    old = Path(config['lt43_manifest']['path']).parent / '20261028-AB/native-0000.json'
    if parent != read_json(old):
        raise ValueError('parent native result changed')
    seed = RECIPE['seeds'][0]
    indices = batch_plan(samples, seed, 1)[0]
    batch = [samples[j] for j in indices]
    _, old_record = old_objective(loaded.model, batch, cases, 'AB', seed, 1)
    results = []
    for arm in ARMS:
        budget.check()
        _, new_record = objective(loaded.model, batch, cases, arm, seed, 1)
        for key in ('imitation_loss', 'native_pose_loss', 'geometry_loss', 'geometry_native_lengths'):
            if old_record[key] != new_record[key]:
                raise ValueError(f'non-stop component drift: {key}')
        residual = new_record['total_loss'] - old_record['total_loss']
        # The original imitation + weighted stop addition occurs in float32,
        # before promotion by the float64 tolerance-normalized pose term.
        # Preserve that rounding when checking the algebraic difference.
        rounded_base = lambda stop: float(torch.tensor(old_record['imitation_loss'], dtype=torch.float32)
                                          + .2 * torch.tensor(stop, dtype=torch.float32))
        expected = rounded_base(new_record['stop_loss']) - rounded_base(old_record['window_loss'])
        if not np.isclose(residual, expected, rtol=0, atol=1e-10):
            raise ValueError('only stop objective may change')
        torch.manual_seed(seed)
        model = deepcopy(loaded.model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        began = time.perf_counter()
        record = update(model, optimizer, batch, cases, arm, seed, 1)
        duration = time.perf_counter() - began
        evaluation = native_evaluate(model, samples, data, cases, Path(output) / f'{arm}-post-native')
        results.append(dict(arm=arm, record=record, update_seconds=duration, evaluation=evaluation,
                            non_stop_components_unchanged=True))
    return parent, results
