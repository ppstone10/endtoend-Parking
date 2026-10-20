"""只用真实闭环逐回合结果判断净收益，不参与模型或控制计算。"""
from collections import Counter
import math


def outcome(row):
    return 'success' if row['success'] else row['failure']


def counts(rows):
    values=list(rows)
    return {'samples':len(values),'success':sum(r['success'] for r in values),
            'collision':sum(r['collision'] for r in values),
            'failures':dict(sorted(Counter(r['failure'] for r in values if not r['success']).items()))}


def validate_report(report, indices):
    """检查分母、纯E5语义及逐回合/汇总一致；返回按来源索引映射。"""
    spec=report['experiment']; protocol=report['protocol']; overall=report['overall']
    expected={'name':'E5','bev_source':'sensor','trajectory_source':'network',
              'executor':'mpc_vehicle','safety_mode':'none'}
    if (not indices or len(set(indices))!=len(indices) or report.get('status')!='completed'
            or any(spec.get(k)!=v for k,v in expected.items())
            or report.get('reconstruct_failures') or overall.get('reconstruct_failures')!=0
            or protocol.get('controller_candidate') not in (None,'baseline')
            or report.get('probe',{}).get('kind')=='diagnostic_controller_counterfactual'):
        raise ValueError('必须为完整的无干预原控制器E5')
    rows=report['episodes']; mapped={r['meta']['dataset_index']:r for r in rows}
    if (len(rows)!=len(indices) or len(mapped)!=len(rows) or set(mapped)!=set(indices)
            or protocol['evaluated_indices']!=indices or protocol['selected_indices']!=indices
            or overall['evaluated_samples']!=len(indices) or overall['requested_indices']!=len(indices)):
        raise ValueError('闭环分母/索引不完整或重复')
    for row in rows:
        if (type(row['success']) is not bool or type(row['collision']) is not bool
                or row['success'] and (row['collision'] or row['failure'] is not None)
                or not row['success'] and (not isinstance(row['failure'],str) or not row['failure'])
                or row['collision']!=(row['failure']=='collision')
                or type(row['steps']) is not int or not 0<=row['steps']<=protocol['max_steps']
                or any(not math.isfinite(row[k]) or row[k]<0 for k in ('final_pos_err','final_yaw_err'))
                or any(not math.isfinite(row['meta'][k]) or row['meta'][k]<=0 for k in ('tol_pos','tol_yaw'))):
            raise ValueError('非法逐回合结果')
    total=counts(rows)
    for name in ('success','collision'):
        if not math.isclose(overall[name+'_rate'],total[name]/len(indices),abs_tol=1e-12,rel_tol=0):
            raise ValueError('汇总比例与逐回合不一致')
    if ('episodes' in overall and overall['episodes']!=len(indices)
            or 'failures' in overall and overall['failures']!=total['failures']):
        raise ValueError('汇总分母/失败构成不一致')
    return mapped


def summarize_pair(baseline, candidate, indices, *, meaningful_count=9, smoke=False):
    if type(meaningful_count) is not int or meaningful_count<1:
        raise ValueError('阶段收益条数须为正整数')
    before=validate_report(baseline,indices); after=validate_report(candidate,indices)
    a={k:v for k,v in baseline['protocol'].items() if k!='checkpoint'}
    b={k:v for k,v in candidate['protocol'].items() if k!='checkpoint'}
    if a!=b or baseline['vehicle_model']!=candidate['vehicle_model'] or baseline['experiment']!=candidate['experiment']:
        raise ValueError('两臂协议/车辆/实验配置不一致')
    for i in indices:
        if any(before[i]['meta'][k]!=after[i]['meta'][k] for k in ('task_id','scene_name','tol_pos','tol_yaw')):
            raise ValueError('配对任务身份或容差不一致')
    old=counts(before.values()); new=counts(after.values())
    ds=new['success']-old['success']; dc=new['collision']-old['collision']
    scenes={}
    for scene in sorted({row['meta']['scene_name'] for row in before.values()}):
        selected=[i for i in indices if before[i]['meta']['scene_name']==scene]
        ca=counts(before[i] for i in selected); cb=counts(after[i] for i in selected)
        scenes[scene]={'baseline':ca,'candidate':cb,'success_delta':cb['success']-ca['success'],
                       'collision_delta':cb['collision']-ca['collision'],
                       'lost_success_indices':[i for i in selected if before[i]['success'] and not after[i]['success']],
                       'new_success_indices':[i for i in selected if not before[i]['success'] and after[i]['success']]}
    joint=not smoke and ds>0 and dc<0
    meaningful=joint and ds>=meaningful_count and dc<=-meaningful_count
    target=not smoke and new['success']*10>=len(indices)*7 and new['collision']*10<=len(indices)
    if smoke: decision='bounded_smoke_not_formal_evidence'
    elif target: decision='selection_target_met_requires_main_protocol'
    elif meaningful: decision='meaningful_joint_gain_target_not_met'
    elif joint: decision='small_joint_gain_not_sufficient'
    else: decision='no_joint_gain_stop'
    regressions=[scene for scene,value in scenes.items() if value['success_delta']<0 or value['collision_delta']>0]
    return {'baseline':old,'candidate':new,'success_delta':ds,'collision_delta':dc,
            'success_delta_pp':100*ds/len(indices),'collision_delta_pp':100*dc/len(indices),
            'joint_gain':joint,'meaningful_joint_gain':meaningful,'selection_target_met':target,
            'decision':decision,'meaningful_count':meaningful_count,'by_scene':scenes,
            'scene_regressions':regressions,'requires_scene_review':bool(regressions),
            'retained_success_indices':[i for i in indices if before[i]['success'] and after[i]['success']],
            'lost_success_indices':[i for i in indices if before[i]['success'] and not after[i]['success']],
            'new_success_indices':[i for i in indices if not before[i]['success'] and after[i]['success']],
            'transitions':dict(sorted(Counter(outcome(before[i])+'->'+outcome(after[i]) for i in indices).items())),
            'main_protocol_passed':False,'automatic_deployment':False,'automatic_training':False}
