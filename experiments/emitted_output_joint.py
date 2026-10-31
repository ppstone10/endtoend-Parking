"""EMIT-JOINT-001: experiment-only differentiable first-stop pose objective.

The hazard distribution is a surrogate, never an inference length selector.
The last step absorbs all survival mass, exactly as native no-stop fallback.
No default Trainer or model behavior is changed.
"""
import math

import torch
from torch.nn import functional as F

from model.network import variable_loss_fn, endpoint_alignment_loss, terminal_pose_loss


def _check_logits(logits):
    if (logits.ndim != 2 or min(logits.shape) < 1 or not logits.is_floating_point()
            or not bool(torch.isfinite(logits).all())):
        raise ValueError('stop logits must be finite floating (B,H), B/H positive')


def native_lengths(logits):
    """Same dtype, sigmoid, >=0.5 and horizon fallback as model.predict."""
    _check_logits(logits)
    crosses = logits.sigmoid() >= .5
    first = crosses.to(torch.int64).argmax(1) + 1
    return torch.where(crosses.any(1), first, first.new_full(first.shape, logits.shape[1]))


def first_stop_distribution(logits, *, temperature=.25):
    _check_logits(logits)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    # Ignore final hazard: stopping there and never stopping emit the same endpoint.
    scaled = logits[:, :-1] / temperature
    survival = F.logsigmoid(-scaled)
    prefix = torch.cat((logits[:, :1] * 0., survival.cumsum(1)), dim=1)
    early = (prefix[:, :-1] + F.logsigmoid(scaled)).exp()
    return torch.cat((early, prefix[:, -1:].exp()), dim=1)


def emitted_pose_loss(points, logits, goals, tolerances, *, temperature=.25):
    """Expected squared goal errors in units of each original task tolerance."""
    _check_logits(logits)
    if (points.shape != (*logits.shape, 3) or goals.shape != (len(points), 3)
            or tolerances.shape != (len(points), 2)
            or not all(bool(torch.isfinite(x).all()) for x in (points, goals, tolerances))
            or bool((tolerances <= 0).any())):
        raise ValueError('finite pose/goal and positive per-task tolerances required')
    delta = points - goals[:, None, :]
    xy_error = delta[..., :2].square().sum(-1) / tolerances[:, :1].square()
    yaw = torch.atan2(torch.sin(delta[..., 2]), torch.cos(delta[..., 2]))
    errors = xy_error + yaw.square() / tolerances[:, 1:].square()
    return (first_stop_distribution(logits, temperature=temperature) * errors).sum(1).mean()


def paired_objective(points, logits, targets, mask, *, joint_weight=0., free_points=None,
                     free_logits=None, goals=None, tolerances=None):
    """LT31 control loss exactly; only the explicit experiment adds a free branch."""
    if not math.isfinite(joint_weight) or joint_weight < 0:
        raise ValueError('joint_weight must be finite and nonnegative')
    base = (variable_loss_fn(points, logits, targets, mask, stop_weight=.2,
                             balance_stop=False, stop_target_mode='cumulative')
            + .5 * endpoint_alignment_loss(points, targets, mask, tail_points=8)
            + terminal_pose_loss(points, targets, mask, yaw_weight=10.))
    if joint_weight == 0:
        return base
    if any(x is None for x in (free_points, free_logits, goals, tolerances)):
        raise ValueError('joint objective requires the same input freely decoded output')
    return base + joint_weight * emitted_pose_loss(free_points, free_logits, goals, tolerances)
