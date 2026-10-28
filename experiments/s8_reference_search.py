"""S8离线参考航向参数与固定DE算子；不拥有控制器或文件。"""
from copy import deepcopy

import numpy as np

from interfaces import Trajectory


KNOTS=np.array([0.,.025,.10,.25,.50,.75,.90,.975,1.])


class YawLimitError(ValueError):
    """候选相对未补偿参考的航向总偏移超出预声明范围。"""


def shape_reference(base,original,parameters):
    parameters=np.asarray(parameters,dtype=np.float64)
    points=np.asarray(base.points)
    if (parameters.shape!=(7,) or not np.isfinite(parameters).all() or (np.abs(parameters)>.03).any()
            or points.shape!=original.points.shape or not 2<=len(points)<=320
            or not np.isfinite(points).all() or not np.isfinite(original.points).all()
            or abs(base.dt-.2)>1e-12 or base.dt!=original.dt
            or not np.array_equal(points[:,:2],original.points[:,:2])):
        raise ValueError('非法七参数/基参考/原参考')
    offset=np.interp(np.linspace(0.,1.,len(points)),KNOTS,np.r_[0.,parameters,0.])
    updated=points.copy();chosen=offset!=0.
    angles=updated[chosen,2].astype(np.float64)+offset[chosen]
    updated[chosen,2]=np.arctan2(np.sin(angles),np.cos(angles))
    updated[[0,-1]]=points[[0,-1]]
    delta=updated[:,2].astype(np.float64)-original.points[:,2].astype(np.float64)
    if (np.abs(np.arctan2(np.sin(delta),np.cos(delta)))>.1+1e-7).any():
        raise YawLimitError('相对未补偿参考航向超过0.1rad（float32容差1e-7）')
    return Trajectory(updated,base.dt)


def execution_rank(summary,penetration,parameters):
    if summary['truncated_preflight']:raise ValueError('截断执行不能作为搜索排名')
    stage=(0 if summary['reached_with_margin'] else 3 if summary['collision'] else 1 if summary['success'] else 2)
    rank=(stage,int(summary['margin_violation_steps']),float(penetration['max_penetration']),
        float(penetration['sum_sq_penetration']),float(summary['final_pos_err']),
        float(summary['final_yaw_err']),float(np.linalg.norm(parameters)))
    if not np.isfinite(rank).all():raise ValueError('非有限搜索排名')
    return rank


def search_reference_parameters(evaluate,budget,seed,*,dimension=7,population_size=28,generations=5,on_generation=None):
    """独立实现固定rand/1/bin；调用方拥有真实执行及首次严格后的核验。"""
    if dimension<1 or population_size<4 or generations<0:raise ValueError('非法DE规模')
    rng=np.random.default_rng(seed)
    population=rng.uniform(-.03,.03,(population_size,dimension));population[0]=0.
    scores=[];evaluations=0;best_parameters=None;best_score=None
    def measured(parameters):
        nonlocal evaluations,best_parameters,best_score
        budget.check();score=tuple(evaluate(parameters.copy()));evaluations+=1
        if len(score)!=7 or not np.isfinite(score).all():raise ValueError('非法DE排名')
        if best_score is None or score<best_score:best_parameters=parameters.copy();best_score=score
        return score
    def finished(decision):
        return dict(decision=decision,parameters=best_parameters.copy(),score=best_score,evaluations=evaluations)
    def checkpoint(generation):
        if on_generation is not None:
            on_generation(generation,population.copy(),np.asarray(scores,dtype=np.float64),deepcopy(rng.bit_generator.state))
    for parameters in population:
        score=measured(parameters);scores.append(score)
        if score[0]==0:return finished('strict_found')
    checkpoint(0)
    for generation in range(1,generations+1):
        for index in range(population_size):
            others=np.delete(np.arange(population_size),index)
            a,b,c=rng.choice(others,3,replace=False)
            mutant=np.clip(population[a]+.5*(population[b]-population[c]),-.03,.03)
            selected=rng.random(dimension)<.7;selected[rng.integers(dimension)]=True
            trial=np.where(selected,mutant,population[index])
            score=measured(trial)
            if score[0]==0:return finished('strict_found')
            if score<scores[index]:population[index]=trial;scores[index]=score
        checkpoint(generation)
    return finished('budget_complete_no_strict')
