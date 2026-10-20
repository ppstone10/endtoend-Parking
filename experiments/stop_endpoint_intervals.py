"""固定仿射logits的首次停点区间；没有步长网格或模型更新。"""
import math

import torch


def affine_cache_logits(features, weight, bias, direction, *, count, chunk_size=64):
    if (features.ndim != 3 or not 0 < count <= len(features) or not isinstance(chunk_size,int) or chunk_size < 1
            or weight.shape != (1,features.shape[-1]) or bias.shape != (1,)
            or direction.shape != (features.shape[-1]+1,)):
        raise ValueError('非法仿射缓存范围')
    bases=[]; velocities=[]
    with torch.no_grad():
        for begin in range(0,count,chunk_size):
            x=features[begin:min(begin+chunk_size,count)].double()
            bases.append((x@weight.double().T).squeeze(-1)+bias.double())
            velocities.append(x@direction[:-1]+direction[-1])
    return torch.cat(bases),torch.cat(velocities)


def first_stop_intervals(logits, velocity, *, threshold):
    if (logits.ndim != 2 or not logits.numel() or velocity.shape != logits.shape
            or not logits.is_floating_point() or not math.isfinite(threshold) or not 0 < threshold < 1
            or not torch.isfinite(logits).all() or not torch.isfinite(velocity).all()):
        raise ValueError('非法仿射停点输入')
    z = logits.detach().cpu().double(); v = velocity.detach().cpu().double()
    cutoff = math.log(threshold / (1 - threshold))
    roots = (cutoff - z) / torch.where(v != 0, v, torch.ones_like(v))
    lower_roots = torch.where(v < 0, roots, torch.full_like(roots, -torch.inf))
    upper_roots = torch.where(v > 0, roots, torch.full_like(roots, torch.inf))
    prefix_lower = torch.cat((torch.full_like(z[:, :1], -torch.inf), lower_roots.cummax(1).values[:, :-1]), 1)
    prefix_upper = torch.cat((torch.full_like(z[:, :1], torch.inf), upper_roots.cummin(1).values[:, :-1]), 1)
    static_block = (v == 0) & (z >= cutoff)
    prefix_block = torch.cat((torch.zeros_like(static_block[:, :1]), static_block.long().cummax(1).values[:, :-1].bool()), 1)
    lower = torch.maximum(prefix_lower, torch.zeros_like(z))
    upper = prefix_upper.clone()
    lower_closed = prefix_lower < 0  # 任何有效的严格前缀下界均开；eta=0本身闭。
    upper_closed = torch.zeros_like(prefix_block)
    increasing = v > 0; decreasing = v < 0
    increasing[:, -1] = False; decreasing[:, -1] = False
    event_lower = torch.where(increasing, roots, torch.full_like(z, -torch.inf))
    event_upper = torch.where(decreasing, roots, torch.full_like(z, torch.inf))
    lower_closed = torch.where(event_lower > lower, torch.ones_like(lower_closed), lower_closed)
    upper_closed = torch.where(event_upper < upper, torch.ones_like(upper_closed), upper_closed)
    lower = torch.maximum(lower, event_lower)
    upper = torch.minimum(upper, event_upper)
    event_block = (v == 0) & (z < cutoff)
    event_block[:, -1] = False  # H−1可以来自无越阈回退，不能要求它自己越阈。
    nonempty = (~prefix_block & ~event_block & torch.isfinite(lower)
                & ((lower < upper) | ((lower == upper) & lower_closed & upper_closed)))
    return {'lower': lower, 'upper': upper, 'lower_closed': lower_closed,
            'upper_closed': upper_closed, 'nonempty': nonempty}


def earliest_selected_interval(intervals, selection):
    if selection.shape != intervals['lower'].shape or selection.dtype != torch.bool:
        raise ValueError('非法终点集合')
    selected = selection & intervals['nonempty']
    starts = torch.where(selected, intervals['lower'], torch.full_like(intervals['lower'], torch.inf))
    first = starts.min(1).values
    at_first = selected & (starts == first[:, None])
    closed = (at_first & intervals['lower_closed']).any(1)
    preferred = at_first & (~closed[:, None] | intervals['lower_closed'])
    indices = preferred.long().argmax(1)
    return {'start': first, 'point_index': torch.where(torch.isfinite(first), indices, -1), 'lower_closed': closed}


def event_record(intervals, events, sample):
    start = float(events['start'][sample])
    if not math.isfinite(start):
        return None
    index = int(events['point_index'][sample])
    end = float(intervals['upper'][sample, index])
    return {'sample_offset': sample, 'point_index': index, 'start': start,
            'start_closed': bool(events['lower_closed'][sample]),
            'end': end if math.isfinite(end) else None,
            'end_closed': bool(intervals['upper_closed'][sample, index]),
            'last_point_includes_no_hit_fallback': index == intervals['lower'].shape[1] - 1}
