"""优化路径只读统计与单步Adam副本，不拥有训练排程或部署。"""
from copy import deepcopy

import torch


def gradient_list(loss, parameters):
    values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return [torch.zeros_like(p) if g is None else g.detach().clone()
            for p,g in zip(parameters,values,strict=True)]


def vector_dot(left,right):
    if len(left)!=len(right) or any(a.shape!=b.shape for a,b in zip(left,right,strict=True)):
        raise ValueError('梯度/位移分量不匹配')
    return sum(float((a.double()*b.double()).sum()) for a,b in zip(left,right,strict=True))


def state_equal(left,right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right)
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(state_equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):
        return type(left)==type(right) and len(left)==len(right) and all(state_equal(a,b) for a,b in zip(left,right,strict=True))
    return type(left)==type(right) and left==right


def adam_counterfactual(model,optimizer_state,clipped_gradients,mode):
    if mode not in ('actual','zero_current','zero_first_moment'):
        raise ValueError('未知Adam单步反事实')
    candidate=deepcopy(model).cpu()
    optimizer=torch.optim.Adam(candidate.parameters(),lr=1e-5)
    optimizer.load_state_dict(deepcopy(optimizer_state))
    for parameter,gradient in zip(candidate.parameters(),clipped_gradients,strict=True):
        if gradient.shape!=parameter.shape or not torch.isfinite(gradient).all():
            raise ValueError('梯度维度或有限性错误')
        parameter.grad=torch.zeros_like(gradient) if mode=='zero_current' else gradient.clone()
        if mode=='zero_first_moment':
            optimizer.state[parameter]['exp_avg'].zero_()
    optimizer.step()
    return candidate,optimizer.state_dict()
