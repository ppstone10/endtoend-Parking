"""LT20：S8两固定起点的原MPC反馈参考搜索；Agent只运行prepare/precheck/smoke。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from dataset import DatasetGenerator
from experiments.behavior_exploration_data import run_case,save_record
from experiments.executable_behavior import PreflightBudget,prepare_supervision_frames
from experiments.exploration_store import artifact_path,exploration_lock
from experiments.recovery_execution import record_arrays
from experiments.retention_quality import audit_source_windows
from experiments.s8_clearance_surrogate import signed_rectangle_gap,swept_penetration_summary
from experiments.s8_reference_search import KNOTS,YawLimitError,execution_rank,shape_reference,search_reference_parameters
from interfaces import Trajectory
from runtime.sources import NetworkSource
from scripts.preflight_behavior_repair import INPUTS,build_case,matrix,result_row
from scripts.preflight_learner_failure_teacher import verify_config
from scripts.preflight_retention_sampling import verify_files
from scripts.prepare_recovery_balance_ab import file_digest
from training.reporting import atomic_write_json
from sim import VehicleConfig


PREVIOUS=ROOT/'runs/diagnostics/s8-yaw-pulse-full-validation-20261022-v1'
ENTRY=ROOT/'runs/diagnostics/s8-entry-recovery-preflight-20261022-v1'
HASHES={'config.json':'895c1fb3c83d6dca9f620df062eda03632ff65d1f71321640c3c6e390c1e3760',
    'report.json':'1582f3f01c562a9b1080c4f70a0007357b512cd2e33b89f6fbd4ee75e3bc8195'}
SMOKE=ROOT/'runs/diagnostics/s8-start-reference-search-smoke-20261023-v3'
SMOKE_HASHES={'config.json':'3d19e48ff98bd6d01dd12f8219793c8478112f8538ec8b4d1ca50bbccab7897b',
    'report.json':'a691b513f586c862e22da56ace5cbf6e6ed3f6fc2332a6173aa727d7554c4f56'}
PROTOCOL=dict(sources=[dict(index=2338,update=1,baseline_steps=240,baseline_margin_steps=37,max_attempts=168),
    dict(index=2351,update=2,baseline_steps=245,baseline_margin_steps=2,max_attempts=2,fixed_witness=-.001)],knots=KNOTS.tolist(),
    parameter_bound=.03,total_yaw_bound=.1,algorithm='numpy_de_rand1bin_immediate_strict_improvement',
    dimension=7,population_size=28,generations=5,mutation=.5,crossover=.7,
    seed_rule='20261023+source_index',control_seeds=[0,1,2],max_steps=600,planning_margin=.2,
    formal_seconds=1800,smoke_seconds=90,smoke_parameters={'2338':.001,'2351':-.001})


class UserSearchBudget:
    def __init__(self):self.deadline=time.perf_counter()+1800.
    def check(self):
        if time.perf_counter()>=self.deadline:raise TimeoutError('LT20总预算1800秒耗尽；不自动续跑或扩预算')


def prepare_config(budget):
    verify_files({str(PREVIOUS/k):v for k,v in HASHES.items()})
    report=json.loads((PREVIOUS/'report.json').read_text(encoding='utf-8'))
    old=json.loads((PREVIOUS/'config.json').read_text(encoding='utf-8'));verify_config(old,budget)
    if report['status']!='completed' or not report['execution']['reached_with_margin'] or report['teacher_labels_created']:
        raise ValueError('不是完整无训练银行的上一阶段证据')
    for name,digest in report['artifacts_sha256'].items():
        budget.check();verify_files({str(artifact_path(PREVIOUS,name)):digest})
    from experiments import s8_reference_search,s8_clearance_surrogate
    code=dict(old['code']);files=dict(old['frozen_files'])
    files.update({str(PREVIOUS/k):v for k,v in HASHES.items()})
    files.update({str(artifact_path(PREVIOUS,k)):v for k,v in report['artifacts_sha256'].items()})
    for p in (Path(__file__).resolve(),Path(s8_reference_search.__file__).resolve(),Path(s8_clearance_surrogate.__file__).resolve()):
        code[str(p)]=file_digest(p)
    verify_files({str(SMOKE/k):v for k,v in SMOKE_HASHES.items()})
    smoke_report=json.loads((SMOKE/'report.json').read_text(encoding='utf-8'))
    smoke_config=json.loads((SMOKE/'config.json').read_text(encoding='utf-8'))
    if (smoke_report['status']!='completed' or smoke_report['mode']!='smoke' or smoke_report['complete_sources']!=2
            or smoke_report['search_performed'] or smoke_report['teacher_bank_created']):
        raise ValueError('不是完整零搜索的起点smoke')
    archived=deepcopy(smoke_config);archived_code={}
    for p,digest in smoke_config['code'].items():
        mapped=SMOKE/'frozen-new-code'/Path(p).name if p in code and p not in old['code'] else Path(p)
        archived_code[str(mapped)]=digest
        if mapped!=Path(p):files[str(mapped)]=digest
    archived['code']=archived_code;verify_config(archived,budget)
    files.update({str(SMOKE/k):v for k,v in SMOKE_HASHES.items()})
    for name,digest in smoke_report['artifacts_sha256'].items():
        path=artifact_path(SMOKE,name);budget.check();verify_files({str(path):digest});files[str(path)]=digest
    witness=next(s for s in smoke_report['sources'] if s['index']==2351)['attempts'][1]
    if witness['parameters']!=[-.001]*7 or not witness['execution']['reached_with_margin']:
        raise ValueError('2351固定冒烟见证身份漂移')
    config=dict(kind='s8_start_reference_search_v1',code=code,frozen_files=files,runtime=old['runtime'],
        protocol=deepcopy(PROTOCOL),training_updates=0,teacher_bank_created=False,automatic_deployment=False,
        allowed_to_prepare_training_plan=False,main_protocol_passed=False)
    for source in PROTOCOL['sources']:
        for update in (0,source['update']):
            p=ENTRY/f"source-20261020-{source['index']}/query-0/teacher-{update}.npz"
            if str(p) not in files:raise ValueError('起点基参考未被冻结')
    verify_config(config,budget);return config


def checked_config(path,budget):
    config=json.loads(Path(path).read_text(encoding='utf-8'))
    if config!=prepare_config(budget):raise ValueError('准备配置与固定身份/协议无法精确重建')
    return config


def load_sources():
    data=DatasetGenerator.load(INPUTS['train']);manifest=matrix.load_dataset_manifest(INPUTS['train'])
    vehicle=VehicleConfig(**manifest['vehicle_model']);sources=[]
    for source in PROTOCOL['sources']:
        index=source['index'];case=build_case(data,manifest,index,vehicle)
        if case['planner'].collision_margin!=.2:raise ValueError('原执行余量不是0.2m')
        signed_rectangle_gap(case['start'].to_array()[None,:3],case['planner']._collision_checker)
        directory=ENTRY/f'source-20261020-{index}/query-0'
        archive=directory/f"teacher-{source['update']}.npz"
        with np.load(archive) as z:base=Trajectory(z['reference'].copy(),float(z['dt']))
        with np.load(directory/'teacher-0.npz') as z:original=Trajectory(z['reference'].copy(),float(z['dt']))
        shape_reference(base,original,np.zeros(7))
        sources.append((source,case,archive,base,original))
    return data,sources


class ReferenceEvaluator:
    """一次候选从参数到原MPC执行；尝试与最好参考由本新目录独占。"""
    def __init__(self,definition,case,archive,base,original,budget,directory,row,save):
        self.definition=definition;self.case=case;self.archive=archive;self.base=base;self.original=original
        self.budget=budget;self.directory=directory;self.row=row;self.save=save;self.best_score=None
        self.last_execution=None

    def __call__(self,parameters):
        self.budget.check();number=len(self.row['attempts'])
        if number>=self.definition['max_attempts']:raise ValueError('候选次数达到固定上限，不续跑')
        directory=self.directory/f'eval-{number:04d}'
        directory.mkdir();attempt=dict(number=number,parameters=np.asarray(parameters).tolist(),status='running')
        self.row['attempts'].append(attempt);self.save();started=time.perf_counter()
        try:
            reference=shape_reference(self.base,self.original,parameters)
        except YawLimitError as exc:
            score=(5,0,0.,0.,0.,0.,float(np.linalg.norm(parameters)))
            attempt.update(decision='yaw_bound_rejected',error=str(exc));reference=None
        if reference is not None:
            checker=self.case['planner']._collision_checker;previous=self.case['start'].to_array()[:3];bad=[]
            for point in reference.points:
                self.budget.check();bad.append(not checker.swept_segment_free(previous,point));previous=point
            geometry=swept_penetration_summary(self.case['start'].to_array(),reference.points,checker,np.asarray(bad))
            attempt.update(reference_unsafe_segments=sum(bad),reference_proxy=geometry)
            np.savez_compressed(directory/'reference.npz',reference=reference.points,dt=np.asarray(reference.dt),parameters=parameters)
            if any(bad):
                score=(4,sum(bad),geometry['max_penetration'],geometry['sum_sq_penetration'],0.,0.,float(np.linalg.norm(parameters)))
                attempt['decision']='reference_geometry_rejected'
            else:
                result,engine,frames=run_case(self.case,None,self.budget,trajectory=reference)
                states,commands=record_arrays(result);summary=result_row(self.case,result,engine)
                penetration=swept_penetration_summary(self.case['start'].to_array(),states,checker,np.asarray(engine.margin_violations))
                score=execution_rank(summary,penetration,parameters)
                save_record(directory,'execution',result,reference=reference.points,dt=np.asarray(reference.dt),parameters=parameters,
                    margin_violations=np.asarray(engine.margin_violations),progress_trace=np.asarray(engine.mpc.execution_trace))
                attempt.update(execution=summary,actual_penetration_proxy=penetration,decision='actual_mpc_evaluated')
                self.last_execution=dict(number=number,reference=reference,result=result,engine=engine,frames=frames,states=states)
                if number==0:
                    if np.any(parameters):raise ValueError('首次候选必须为原基线零参数')
                    with np.load(self.archive) as z:
                        if not all((np.array_equal(states,z['states']),np.array_equal(commands,z['commands']),
                                np.array_equal(np.asarray(engine.margin_violations),z['margin_violations']))):
                            raise ValueError('fresh起点基线未逐元素复现')
                    if summary['steps']!=self.definition['baseline_steps'] or summary['margin_violation_steps']!=self.definition['baseline_margin_steps']:
                        raise ValueError('起点基线指标漂移')
                    self.row['baseline_exact_verified']=True
                if self.case['index']==2351 and number==1:
                    if not np.array_equal(parameters,np.full(7,-.001)):raise ValueError('固定见证参数漂移')
                    with np.load(SMOKE/'source-2351/eval-0001/execution.npz') as z:
                        if not all((np.array_equal(reference.points,z['reference']),np.array_equal(states,z['states']),
                                np.array_equal(commands,z['commands']),np.array_equal(engine.margin_violations,z['margin_violations']))):
                            raise ValueError('固定起点见证未逐元素复现')
                    self.row['fixed_witness_exact_verified']=True
        attempt.update(status='completed',rank=list(score),elapsed_seconds=time.perf_counter()-started)
        atomic_write_json(directory/'attempt.json',attempt)
        if self.best_score is None or score<self.best_score:
            self.best_score=score;self.row['best_attempt']=number
            atomic_write_json(self.directory/'best.json',attempt)
            if reference is not None:np.savez_compressed(self.directory/'best-reference.npz',reference=reference.points,dt=np.asarray(reference.dt),parameters=parameters)
        self.save();print('EVAL',self.case['index'],number,attempt['decision'],score[:4],flush=True)
        return score


def check_first_window(case,data,frames,states,budget):
    if not frames or not frames[0]['mask'].any():raise ValueError('严格候选缺真实首帧/消费')
    frame=frames[0];index=case['index']
    runtime_source=NetworkSource(None,None);runtime_source.begin(case['start'],case['goal'])
    local_goal=runtime_source._to_local_goal(case['start']).astype(np.float32)
    if (not np.array_equal(np.asarray(frame['pose']),case['start'].to_array()[:3])
            or not np.array_equal(frame['bev'],data['bevs'][index])
            or not np.array_equal(frame['state'],np.asarray(case['start'].to_array()[3:],dtype=np.float32))
            or not np.allclose(frame['goal'],local_goal,rtol=0,atol=2e-5)):
        raise ValueError('严格候选首帧不复现训练起点输入')
    sample=prepare_supervision_frames(frames,None,'recovery',index,frame_limit=1)[0]
    # 与既有query_sample一致：容差核对后用原运行时输入，避免专家全局float32转换的舍入别名。
    sample['goal']=torch.from_numpy(local_goal.copy())
    window=audit_source_windows([dict(sample,role='anchor')],states,case,budget.check)[0]
    return sample,window


def confirm_candidate(case,base,reference,frames,states,data,budget,directory):
    sample,window=check_first_window(case,data,frames,states,budget)
    torch.save(dict(purpose='diagnostic_first_frame_not_teacher_bank_or_gradient_authorization',frame=frames[0]),directory/'first-frame.pt')
    exported=Trajectory(NetworkSource._to_global(np.asarray(sample['points'])[:frames[0]['length']],case['start']),reference.dt)
    result,engine,_=run_case(case,None,budget,trajectory=exported)
    save_record(directory,'float32-export-control0',result,reference=exported.points,dt=np.asarray(exported.dt),
        margin_violations=np.asarray(engine.margin_violations))
    export=result_row(case,result,engine);paired=[]
    for control_seed in (1,2):
        for name,trajectory in (('baseline',base),('candidate',reference)):
            budget.check();result,engine,_=run_case(case,None,budget,trajectory=trajectory,control_seed=control_seed)
            save_record(directory,f'{name}-control{control_seed}',result,reference=trajectory.points,dt=np.asarray(trajectory.dt),
                margin_violations=np.asarray(engine.margin_violations))
            paired.append(dict(strategy=name,control_seed=control_seed,**result_row(case,result,engine)))
    passed=bool(window['clean'] and export['reached_with_margin'] and
        all(row['reached_with_margin'] for row in paired if row['strategy']=='candidate'))
    return dict(first_window=window,float32_export=export,paired_independent_controls=paired,qualified=passed)


def search_source(evaluate,definition,budget,on_generation):
    if 'fixed_witness' in definition:
        evaluate(np.zeros(7));parameters=np.full(7,definition['fixed_witness']);score=evaluate(parameters)
        if score[0]!=0:raise ValueError('固定起点严格见证指标漂移')
        return dict(decision='strict_found',parameters=parameters,score=score,evaluations=2)
    return search_reference_parameters(evaluate,budget,20261023+definition['index'],on_generation=on_generation)


def run(config_path,output,*,smoke=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();budget=PreflightBudget(90) if smoke else UserSearchBudget();torch.set_num_threads(1)
    report=dict(status='running',stage='identity',mode='smoke' if smoke else 'formal_user_search',sources=[],
        training_updates=0,teacher_bank_created=False,search_performed=not smoke,main_protocol_passed=False,
        automatic_deployment=False,allowed_to_prepare_training_plan=False,qualification_passed=False)
    def save():
        report['elapsed_seconds']=time.perf_counter()-started;atomic_write_json(output/'report.json',report)
    with exploration_lock(output):
        save()
        try:
            config=checked_config(config_path,budget);atomic_write_json(output/'config.json',config)
            report['config_sha256']=file_digest(output/'config.json');data,sources=load_sources()
            for definition,case,archive,base,original in sources:
                budget.check();index=case['index'];report['stage']=f'source-{index}';save()
                directory=output/f'source-{index}';directory.mkdir()
                row=dict(index=index,baseline_archive=str(archive),attempts=[],qualified=False)
                report['sources'].append(row);evaluate=ReferenceEvaluator(definition,case,archive,base,original,budget,directory,row,save)
                if smoke:
                    evaluate(np.zeros(7));evaluate(np.full(7,PROTOCOL['smoke_parameters'][str(index)]))
                    observed=evaluate.last_execution
                    _,window=check_first_window(case,data,observed['frames'],observed['states'],budget)
                    row.update(smoke_window_diagnostic=window,smoke_window_attempt=observed['number'])
                    row['decision']='fixed_smoke_only_no_search_or_qualification'
                else:
                    def checkpoint(generation,population,scores,rng):
                        np.savez_compressed(directory/f'generation-{generation:02d}.npz',population=population,ranks=scores)
                        atomic_write_json(directory/f'generation-{generation:02d}.json',dict(generation=generation,
                            rng_state=rng,attempts=len(row['attempts']),best_attempt=row['best_attempt']))
                    search=search_source(evaluate,definition,budget,checkpoint)
                    row['search_result']={**search,'parameters':search['parameters'].tolist(),'score':list(search['score'])}
                    if search['decision']=='strict_found':
                        selected=evaluate.last_execution
                        if selected['number']!=row['best_attempt']:raise ValueError('首次严格候选与执行帧不一致')
                        row['verification']=confirm_candidate(case,base,selected['reference'],selected['frames'],selected['states'],data,budget,directory)
                        row['qualified']=row['verification']['qualified']
                        row['decision']='qualified_start_teacher' if row['qualified'] else 'strict_search_witness_failed_independent_checks'
                    else:row['decision']='fixed_search_exhausted_no_strict_teacher'
                row['attempt_count']=len(row['attempts']);row['executed_episodes']=sum('execution' in a for a in row['attempts'])
                if not row.get('baseline_exact_verified') or row['attempt_count']>(2 if smoke else definition['max_attempts']):
                    raise ValueError('基线或预算计数不符')
                save()
            verify_config(config,budget);budget.check()
            report['qualification_passed']=not smoke and len(report['sources'])==2 and all(s['qualified'] for s in report['sources'])
            report['artifacts_sha256']={str(p.relative_to(output)):file_digest(p) for p in output.rglob('*')
                if p.is_file() and p!=output/'report.json' and p.name not in ('run.lock','console.log')}
            budget.check();report.update(status='completed',stage='decision',complete_sources=len(report['sources']),
                decision='smoke_only_not_search_result' if smoke else 'teacher_search_only_not_learning_or_deployment')
            save();print('FINISHED',report['decision'],report['qualification_passed'],report['elapsed_seconds'],flush=True);return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else
                'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),decision='incomplete_do_not_infer')
            save();raise


def prepare(path):
    config=prepare_config(PreflightBudget(45));path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(config,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print('PREPARED',path,file_digest(path),'ZERO ROLLOUT/SEARCH/TRAINING',flush=True)


def precheck(path):
    checked_config(path,PreflightBudget(45));print('LT20 INPUTS/PROTOCOL VERIFIED; ZERO ROLLOUT/SEARCH/TRAINING',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',type=Path);action.add_argument('--precheck',action='store_true')
    action.add_argument('--smoke',action='store_true');action.add_argument('--run',action='store_true')
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.prepare:
        if args.config or args.output:parser.error('prepare只接受新配置路径')
        prepare(args.prepare)
    elif args.config is None:parser.error('需要--config冻结配置')
    elif args.precheck:
        if args.output:parser.error('precheck不接受output')
        precheck(args.config)
    elif args.output is None:parser.error('smoke/run需要不存在的新output')
    else:run(args.config,args.output,smoke=args.smoke)
