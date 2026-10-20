"""只按原MPC代价精修当前CEM均值，不读取环境或几何。"""
from copy import deepcopy

import numpy as np
import torch

from controller import MPCController


class _EvaluationBudgetReached(Exception):
    pass


class _NonfinitePolish(Exception):
    pass


def torch_original_cost(mpc,controls,reference,state):
    target=torch.as_tensor(reference,dtype=controls.dtype,device=controls.device)
    x,y,yaw=[controls.new_tensor(v) for v in (state.x,state.y,state.yaw)]
    cost=controls.new_zeros(())
    for step in range(mpc.horizon):
        v,omega=controls[2*step],controls[2*step+1]
        x=x+v*torch.cos(yaw)*mpc.dt
        y=y+v*torch.sin(yaw)*mpc.dt
        yaw=yaw+omega*mpc.dt
        dx,dy=x-target[step,0],y-target[step,1]
        error=yaw-target[step,2]
        yaw_error=torch.atan2(torch.sin(error),torch.cos(error))
        cost=cost+mpc.w_xy*(dx*dx+dy*dy)+mpc.w_yaw*yaw_error*yaw_error
    return (cost+mpc.w_terminal*(dx*dx+dy*dy+mpc.w_yaw*yaw_error*yaw_error)
            +mpc.w_v*(controls[0::2]**2).sum()+mpc.w_omega*(controls[1::2]**2).sum())


class CEMPolishedMPC(MPCController):
    def __init__(self,*,max_polish_iterations=60,max_evaluations=80,**kwargs):
        if (not isinstance(max_polish_iterations,int) or not isinstance(max_evaluations,int)
                or max_polish_iterations<1 or max_evaluations<0):
            raise ValueError('精修迭代须正整数，评估预算须非负整数')
        super().__init__(**kwargs)
        if not np.isfinite([self.max_v,self.max_omega]).all() or min(self.max_v,self.max_omega)<=0:
            raise ValueError('控制上限须正且有限')
        self.max_polish_iterations=max_polish_iterations
        self.max_evaluations=max_evaluations
        self.polish_audits=[]

    @classmethod
    def from_controller(cls,controller,**polish_kwargs):
        if type(controller) is not MPCController:raise ValueError('只允许复制原MPC')
        keys=('dt','horizon','max_v','max_omega','w_xy','w_yaw','w_terminal','w_v','w_omega','population','elite_frac','iterations')
        clone=cls(**{k:getattr(controller,k) for k in keys},catchup_window=controller._catchup_window,**polish_kwargs)
        clone.__dict__.update(deepcopy(controller.__dict__))
        return clone

    def polish_controls(self,original,reference,state):
        original=np.asarray(original,dtype=float)
        reference=np.asarray(reference,dtype=float)
        if (original.shape!=(2*self.horizon,) or reference.shape!=(self.horizon,3)
                or not np.isfinite(original).all() or not np.isfinite(reference).all()
                or not np.isfinite(state.to_array()).all()):
            raise ValueError('精修输入形状/有限值不符')
        limits=np.tile([self.max_v,self.max_omega],self.horizon)
        if np.any(np.abs(original)>limits+1e-12):raise ValueError('原序列超出控制上限')
        original_cost=float(self._batch_cost(original[None,:],reference,state)[0])
        if not np.isfinite(original_cost):raise ValueError('原代价非有限值')
        best=original.copy();best_cost=original_cost;evaluations=0;reason='disabled'
        if self.max_evaluations:
            scale=torch.tensor(limits,dtype=torch.float64)
            z=torch.atanh(torch.clamp(torch.tensor(original)/scale,-1+1e-8,1-1e-8)).detach().requires_grad_(True)
            optimizer=torch.optim.LBFGS([z],lr=1.,max_iter=self.max_polish_iterations,max_eval=self.max_evaluations,
                       history_size=10,tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
            reason='optimizer_finished'
            def closure():
                nonlocal best,best_cost,evaluations
                if evaluations>=self.max_evaluations:raise _EvaluationBudgetReached()
                optimizer.zero_grad()
                controls=scale*torch.tanh(z)
                loss=torch_original_cost(self,controls,reference,state)
                evaluations+=1
                if not bool(torch.isfinite(loss)):raise _NonfinitePolish()
                loss.backward()
                if z.grad is None or not bool(torch.isfinite(z.grad).all()):raise _NonfinitePolish()
                value=float(loss.detach())
                if value<best_cost:
                    best=controls.detach().numpy().copy();best_cost=value
                return loss
            try:optimizer.step(closure)
            except _EvaluationBudgetReached:reason='evaluation_budget'
            except _NonfinitePolish:
                reason='nonfinite';best=original.copy()
        # 只用原NumPy cost接受，不依赖优化器返回值或任何安全判定。
        verified=float(self._batch_cost(best[None,:],reference,state)[0])
        accepted=bool(np.isfinite(best).all() and np.all(np.abs(best)<=limits+1e-12)
                      and np.isfinite(verified) and verified<original_cost)
        selected=best if accepted else original.copy()
        self.polish_audits.append({'original_controls':original.tolist(),'original_cost':original_cost,
            'selected_cost':verified if accepted else original_cost,'accepted':accepted,'evaluations':evaluations,
            'max_evaluations':self.max_evaluations,'stop_reason':reason})
        return selected

    def _solve(self,initial,reference,state):
        return self.polish_controls(super()._solve(initial,reference,state),reference,state)
