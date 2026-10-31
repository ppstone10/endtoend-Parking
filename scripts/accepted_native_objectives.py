"""STOP-ACCEPT-001..003: change only LT43 AB's stop supervision."""
import numpy as np
import torch
from torch.nn import functional as F
from experiments.emitted_output_joint import native_lengths
from model.network import loss_fn, endpoint_alignment_loss, terminal_pose_loss
from scripts.native_output_geometry import NativeGeometry
from scripts.native_output_objectives import native_pose_loss

ARMS = ('gate', 'accept')
OBJECTIVE = dict(stop_weight=.2, logit_margin=.5, native_pose_weight=.1,
                 geometry_weight=1., negative_reduction='mean',
                 empty_set='defer_stop_keep_all_AB_coordinate_and_geometry',
                 accepted='zero_stop_correction', candidate='accept')


def qualified_points(points, sample, geometry):
    """Detached strict pose AND the original complete swept safe prefix."""
    raw = points.detach().cpu().numpy().astype(np.float64)
    goal = sample['goal'].detach().cpu().numpy().astype(np.float64)
    tol = sample['tolerances'].detach().cpu().numpy().astype(np.float64)
    if (raw.ndim != 2 or raw.shape[1] != 3 or not len(raw)
            or goal.shape != (3,) or tol.shape != (2,)
            or not all(np.isfinite(v).all() for v in (raw, goal, tol)) or (tol <= 0).any()):
        raise ValueError('finite poses/goal and positive strict tolerances required')
    delta = raw - goal
    pose_ok = ((np.linalg.norm(delta[:, :2], axis=1) < tol[0])
               & (np.abs(np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))) < tol[1]))
    valid = pose_ok & geometry.safe_prefix(points.detach(), sample['pose'])
    return torch.as_tensor(valid, dtype=torch.bool, device=points.device)


def acceptance_stop_loss(logits, feasible, arm):
    if (arm not in ARMS or logits.ndim != 2 or min(logits.shape) < 1
            or not logits.is_floating_point() or not torch.isfinite(logits).all()
            or feasible.shape != logits.shape or feasible.dtype != torch.bool
            or feasible.device != logits.device):
        raise ValueError('gate/accept and finite B,H logits with same-device bool feasibility required')
    lengths = native_lengths(logits).detach().tolist()
    values, rows = [], []
    for scores, valid, length in zip(logits, feasible, lengths):
        eligible = torch.nonzero(valid, as_tuple=False).flatten()
        info = dict(native_length=length, eligible_count=len(eligible), selected_index=None)
        if not len(eligible):
            values.append(scores.sum() * 0)
            info['status'] = 'no_window'
        elif bool(valid[length - 1]):
            values.append(scores.sum() * 0)
            info['status'] = 'accepted'
        else:
            first = int(eligible[0])
            # Selection is discrete. Only the selected logit carries positive
            # gradient; equal scores choose the first eligible point.
            index = first if arm == 'gate' else int(eligible[scores.detach()[eligible].argmax()])
            previous = scores[:index]
            if arm == 'accept':
                previous = previous[~valid[:index]]
            negative = F.relu(previous + .5).mean() if len(previous) else scores.sum() * 0
            values.append(negative + F.relu(.5 - scores[index]))
            info.update(selected_index=index, status='early' if length - 1 < first else 'late_or_gap')
        rows.append(info)
    return torch.stack(values).mean(), rows


def objective(model, batch, cases, arm, seed, step):
    if arm not in ARMS:
        raise ValueError('predeclared gate/accept only')
    stack = lambda key: torch.stack([s[key] for s in batch])
    free = model.forward_with_stop(stack('bev'), stack('goal'), stack('state'))
    geometries = [NativeGeometry(cases[s['source_index']]['planner']._collision_checker) for s in batch]
    feasible = torch.stack([qualified_points(free.points[j], s, geo)
                            for j, (s, geo) in enumerate(zip(batch, geometries))])
    stopping, rows = acceptance_stop_loss(free.stop_logits, feasible, arm)
    imitation = (loss_fn(free.points, stack('points'), stack('mask'))
                 + .5 * endpoint_alignment_loss(free.points, stack('points'), stack('mask'), tail_points=8)
                 + terminal_pose_loss(free.points, stack('points'), stack('mask'), yaw_weight=10.))
    terminal = native_pose_loss(free.points, free.stop_logits, stack('goal'), stack('tolerances'))
    lengths = native_lengths(free.stop_logits).detach().tolist()
    safety = torch.stack([geo.loss(free.points[j, :lengths[j]], s['pose'])
                          for j, (s, geo) in enumerate(zip(batch, geometries))]).mean()
    total = imitation + .2 * stopping + .1 * terminal + safety
    record = dict(step=step, samples=[dict(source_index=s['source_index'], frame_index=s['frame_index']) for s in batch],
                  imitation_loss=float(imitation.detach()), stop_loss=float(stopping.detach()),
                  native_pose_loss=float(terminal.detach()), geometry_loss=float(safety.detach()),
                  total_loss=float(total.detach()), stop_rows=rows,
                  stop_deferred=sum(r['status'] == 'no_window' for r in rows),
                  already_accepted=sum(r['status'] == 'accepted' for r in rows),
                  geometry_native_lengths=lengths)
    return total, record


def update(model, optimizer, batch, cases, arm, seed, step):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, record = objective(model, batch, cases, arm, seed, step)
    if not torch.isfinite(loss):
        raise ValueError('nonfinite accepted native objective')
    loss.backward()
    if step == 1:
        grads = [p.grad for p in (model.decoder.point_head.weight, model.decoder.stop_head.weight)]
        record['head_gradient_norms'] = [float(g.norm()) if g is not None else 0. for g in grads]
        if not np.isfinite(record['head_gradient_norms']).all() or min(record['head_gradient_norms']) <= 0:
            raise ValueError('first fixed batch must have finite nonzero gradients in both heads')
    record['unclipped_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite parameters')
    return record
