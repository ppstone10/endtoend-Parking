"""实际参考导出的完整余量控制候选：独立版本实验与回归门禁。"""
from pathlib import Path
import time
import torch

from experiments import validation_matrix as matrix
from experiments.component_capture import VerificationBudget
from sim import VehicleConfig
from training.reporting import atomic_write_json
from scripts.diagnose_closed_loop_failures import _sha256
from scripts.prepare_recovery_balance_ab import runtime_identity
from verification.identity import execution_identity, verify_files, ROOT
from verification.candidate_execution import compare_rows
from verification.terminal_study import read, check_artifacts
from verification.planning_study import inspect as inspect_baseline
from verification.reference_teacher import validate_teacher
from verification.feasible_reference_candidate import feasible_reference_scope
from verification.reference_study import FILES as V1_FILES, inspect as inspect_v1

SOURCE = ROOT/'runs/verification/planning-margin-formal-20261026/results'
OLDER = ROOT/'runs/verification/reference-retention-formal-20261026/results'
SMOKE = [0,131,132,139,161,167,168,172,183,267,268,279,292,296]
FILES = [*V1_FILES, 'verification/feasible_reference_candidate.py',
         'verification/feasible_reference_study.py', 'verification/feasible_reference_audit.py',
         'scripts/compare_feasible_reference_candidate.py']
V1_SOURCE = ROOT/'runs/verification/reference-candidate-pilot-20261026/smoke'
V1_AUDIT = ROOT/'runs/verification/reference-candidate-audit-20261026/report.json'


def negative_evidence():
    report = inspect_v1(V1_SOURCE)
    evidence = read(V1_AUDIT)
    if (report['comparison']['old_success_lost'] != [292]
            or evidence['status'] != 'completed'
            or evidence['source_report_sha256'] != _sha256(V1_SOURCE/'report.json')
            or evidence['code'] != execution_identity(V1_FILES)
            or not evidence['all_states_commands_and_reference_candidates_exact']):
        raise ValueError('v1 负结果或独立复核身份不符')
    return {str(path): _sha256(path) for path in
            (V1_SOURCE/'report.json', V1_SOURCE/'artifacts.json', V1_AUDIT)}

FIXED = {'max_steps':600, 'replan_every':10, 'control_seed':0, 'preview_steps':80,
         'capture_radius_in_position_tolerances':3, 'physical_margin':0.,'planning_margin':.2,
         'geometry_source':'simulator_ground_truth','production_qualified':False,
         'zero_recovery_sample_limit':400,'zero_recovery_only_all_zero':True,'all_output_branches_margin_constrained':True,'initial_invasion_recovery':False,'reference_candidate_rule':'projected_differential_model_consumed_window_v2','require_safe_consumed_window':True}


def recovery_statistics(folder, rows):
    frames=[frame for row in rows for frame in read(Path(folder)/f"episode-{row['dataset_index']:04d}.json")['frames']]
    expanded=[f for f in frames if f['selected']=='expanded_safe_sample']
    attempts=[f for f in frames if 'zero_recovery_context' in f]
    steps=len(frames);violations=sum(not f['planning_margin_safe'] for f in frames)
    reference=[f['reference_candidate'] for f in frames if 'reference_candidate' in f]
    return {'unsafe_window_skips':sum(f.get('reference_candidate_skipped')=='unsafe_consumed_window' for f in frames),'reference_candidate_attempts':len(reference),'reference_candidate_accepts':sum(x['accepted'] for x in reference),
            'reference_candidate_unsafe':sum(x['reason']=='unsafe' for x in reference),
            'invocations':len(attempts),'expanded_choices':len(expanded),
            'checked_samples':sum(f['zero_recovery_checked'] for f in attempts),
            'chosen_ranks':[f['zero_recovery_rank'] for f in expanded],
            'cost_reductions':[f['zero_recovery_context']['original_cost']-f['selected_cost'] for f in expanded],
            'terminal_rotations':sum(f['selected']=='terminal_rotation' for f in frames),
            'actual_steps':steps,'planning_margin_rate':violations/steps if steps else 0.}


def compare(baseline, rows):
    paired = compare_rows(baseline, rows)
    paired['margin_not_increased'] = paired['planning_margin_violations'] <= sum(r['planning_margin_violations'] for r in baseline)
    chosen={r['dataset_index'] for r in rows}
    older=[r for r in read(OLDER/'report.json')['rows'] if r['dataset_index'] in chosen]
    paired['vs_lt24']=compare_rows(older,rows)
    paired['reference_candidate_benefit'] = bool(paired['new_success_gained'] and not paired['old_success_lost']
                                     and paired['collisions']==0 and paired['planning_margin_violations']==0)
    paired['formal_joint_benefit']=bool(paired['reference_candidate_benefit'] and not paired['vs_lt24']['old_success_lost'])
    return paired


def baseline_rows(indices):
    chosen = set(indices)
    return [r for r in read(SOURCE/'report.json')['rows'] if r['dataset_index'] in chosen]


def inspect(source):
    source = Path(source).resolve();config = precheck(source/'config.json')
    report = check_artifacts(source)
    if report['indices'] != config['indices'] or report['mode'] != config['purpose']:
        raise ValueError('补查分母或模式不符')
    if report['identity'] != {'code':config['code'],'inputs':config['inputs'],'config_sha256':_sha256(source/'config.json')}:
        raise ValueError('补查报告身份不符')
    artifacts = read(source/'artifacts.json')
    if [r['dataset_index'] for r in report['rows']] != config['indices']:
        raise ValueError('逐任务账本不完整')
    for row in report['rows']:
        index=row['dataset_index'];folder=source/'control'
        for suffix in ('json','npz'):
            if f'control/retention-{index:04d}.{suffix}' not in artifacts:
                raise ValueError('原始网络参考/证书账本缺失')
        episode=read(folder/f'episode-{index:04d}.json');ledger=read(folder/f'retention-{index:04d}.json')
        if episode['summary'] != row or len(episode['frames'])!=row['steps'] or ledger['steps']!=row['steps']:
            raise ValueError('实际控制/证书步数不一致')
        if [q['step'] for q in ledger['queries']] != list(range(0,row['steps'],10)):
            raise ValueError('感知/推理时序不完整')
        if not all(f['actual_contact_safe'] and f['selected_horizon_safe'] for f in episode['frames']):
            raise ValueError('约束控制出现非安全声明')
        for cert in ledger['certificates']:
            n=len(cert['controls'])
            if not(0<n<=80 and cert['start_step']+n<=600 and cert['contact_safe'] and cert['planning_margin_safe']):
                raise ValueError('到达证书预算或安全声明不符')
    if report['zero_recovery']!=recovery_statistics(source/'control',report['rows']):raise ValueError('补查统计不符')
    paired=compare(baseline_rows(config['indices']), report['rows'])
    qualified=bool(config['purpose']=='formal' and paired['samples']==300 and paired['successes']>=210 and paired['formal_joint_benefit'])
    if paired!=report['comparison'] or report['candidate_qualified']!=qualified or report['production_qualified'] is not False:
        raise ValueError('配对指标/条件实验资格不一致')
    return report


def validate_pilot(source, audit_path, teacher):
    source=Path(source).resolve();report=inspect(source);audit=read(audit_path)
    if (report['mode']!='smoke' or report['indices']!=SMOKE or not report['comparison']['formal_joint_benefit']
            or audit['status']!='completed' or audit['source_report_sha256']!=_sha256(source/'report.json')
            or audit['indices']!=SMOKE or audit['code']!=execution_identity(FILES)
            or not audit['all_states_commands_and_reference_candidates_exact']):
        raise ValueError('固定短检/独立审计未通过，不准备正式')
    validate_teacher(read(source/'config.json'),teacher)
    if audit.get('teacher_report_sha256')!=_sha256(Path(teacher)/'report.json'):raise ValueError('独立审计未覆盖教师')
    return report


def prepare(path, *, pilot=None, audit=None, teacher=None):
    path=Path(path).resolve()
    if path.exists():raise FileExistsError('新配置禁止覆盖')
    inspect_baseline(SOURCE);prior=read(SOURCE/'config.json');inputs=dict(prior['inputs'])
    inputs.update({str(SOURCE/name):_sha256(SOURCE/name) for name in ('config.json','report.json','artifacts.json')})
    inputs.update(negative_evidence())
    if pilot is not None:
        if audit is None or teacher is None:raise ValueError('正式配置必须有独立审计和教师验收')
        report=validate_pilot(pilot,audit,teacher)
        if report['identity']['code']!=execution_identity(FILES) or report['identity']['inputs']!=inputs:
            raise ValueError('短检与当前候选身份不一致')
        for p in (Path(pilot)/'report.json',Path(pilot)/'artifacts.json',Path(audit),Path(teacher)/'report.json',Path(teacher)/'artifacts.json'):
            inputs[str(p.resolve())]=_sha256(p)
    config={'kind':'reference_control_candidate_v2','purpose':'formal' if pilot else 'smoke',
            'source':str(SOURCE),'inputs':inputs,'code':execution_identity(FILES),'runtime':runtime_identity(),
            'data':prior['data'],'checkpoint':prior['checkpoint'],'indices':list(range(300)) if pilot else SMOKE,
            'budget_seconds':28800 if pilot else 180,'pilot':str(Path(pilot).resolve()) if pilot else None,
            'audit':str(Path(audit).resolve()) if audit else None,
            'teacher':str(Path(teacher).resolve()) if teacher else None,**FIXED}
    path.parent.mkdir(parents=True,exist_ok=True);atomic_write_json(path,config)
    return config


def precheck(path):
    c=read(path)
    if c['kind']!='reference_control_candidate_v2' or Path(c['source']).resolve()!=SOURCE:
        raise ValueError('补查配置类型/来源不符')
    verify_files(c['inputs'])
    if c['code']!=execution_identity(FILES) or c['runtime']!=runtime_identity() or any(c[k]!=v for k,v in FIXED.items()):
        raise ValueError('执行代码/环境/固定参数漂移')
    previous=read(SOURCE/'config.json');required=dict(previous['inputs'])
    required.update({str(SOURCE/name):_sha256(SOURCE/name) for name in ('config.json','report.json','artifacts.json')})
    required.update(negative_evidence())
    if c['data']!=previous['data'] or c['checkpoint']!=previous['checkpoint'] or any(c['inputs'].get(k)!=v for k,v in required.items()):
        raise ValueError('输入路径/身份不完整')
    check_artifacts(SOURCE)
    if c['purpose']=='smoke':
        if c['indices']!=SMOKE or c['budget_seconds']!=180:raise ValueError('短检范围/预算漂移')
    elif c['purpose']=='formal':
        if c['indices']!=list(range(300)) or c['budget_seconds']!=28800:raise ValueError('正式分母/预算漂移')
        validate_pilot(c['pilot'],c['audit'],c['teacher'])
        for p in (Path(c['pilot'])/'report.json',Path(c['pilot'])/'artifacts.json',Path(c['audit']),Path(c['teacher'])/'report.json',Path(c['teacher'])/'artifacts.json'):
            if c['inputs'].get(str(p.resolve()))!=_sha256(p):raise ValueError('正式前置身份缺失')
    else:raise ValueError('未知用途')
    return c


def run(path, *, formal=False):
    c=precheck(path)
    if formal!=(c['purpose']=='formal'):raise ValueError('正式/短检配置不能混用')
    validate_teacher(c,c['teacher'] if formal else Path(path).resolve().parent/'teacher')
    torch.set_num_threads(1);out=Path(path).resolve().parent/('results' if formal else 'smoke')
    out.mkdir(parents=True,exist_ok=False);folder=out/'control';folder.mkdir();atomic_write_json(out/'config.json',c)
    report={'kind':c['kind'],'status':'running','mode':c['purpose'],'indices':c['indices'],
            'identity':{'code':c['code'],'inputs':c['inputs'],'config_sha256':_sha256(Path(path))},
            'candidate_qualified':False,'production_qualified':False,'geometry_source':c['geometry_source'],
            'training_performed':False,'automatic_deployment':False}
    atomic_write_json(out/'report.json',report);budget=VerificationBudget(c['budget_seconds'])
    vehicle=VehicleConfig(**matrix.load_dataset_manifest(Path(c['data']))['vehicle_model'])
    try:
        spec=matrix.ExperimentSpec(name='reference_control_candidate',bev_source='sensor',trajectory_source='network',executor='mpc_vehicle',
                                   description='GT-assisted LT25 with one safe-consumed-reference-derived feasible lower-cost candidate after original zero; no teacher at runtime')
        with feasible_reference_scope(folder,vehicle,{},budget) as (rows,ledgers):
            raw=matrix.run_validation_experiment(spec,data_path=c['data'],checkpoint_path=c['checkpoint'],output_path=folder/'raw-report.json',
                                                 indices=c['indices'],samples=0,max_steps=600,replan_every=10,control_seed=0,
                                                 progress=lambda n,total,b:print(f'[reference-candidate {n}/{total}] #{b.result.meta["dataset_index"]} {b.result.failure or "success"}',flush=True))
        if raw['reconstruct_failures'] or len(rows)!=len(c['indices']) or len(ledgers)!=len(rows):
            raise ValueError('实际/参考保持账本分母不完整')
        precheck(path);paired=compare(baseline_rows(c['indices']),rows)
        qualified=bool(formal and paired['samples']==300 and paired['successes']>=210 and paired['formal_joint_benefit'])
        report.update(status='completed',rows=rows,comparison=paired,candidate_qualified=qualified,
                      elapsed_seconds=time.perf_counter()-budget.started,
                      zero_recovery=recovery_statistics(folder,rows),
                      retention={'attempts':sum(l['attempts'] for l in ledgers),'certificates':sum(len(l['certificates']) for l in ledgers),
                                 'held_queries':sum(l['held_queries'] for l in ledgers)})
        atomic_write_json(out/'report.json',report)
        (out/'summary.md').write_text(f'# 合法参考窗口控制候选 v2\n\n状态completed；模式{c["purpose"]}。\n\nLT25配对及LT24保持：{paired}\n\n保持统计：{report["retention"]}\n\n补查统计：{report["zero_recovery"]}\n\n仍为GT几何辅助条件实验，production_qualified=false；按车位原容差、无新参考/专家/训练。\n',encoding='utf-8')
        atomic_write_json(out/'artifacts.json',{p.relative_to(out).as_posix():_sha256(p) for p in out.rglob('*') if p.is_file()})
        return report
    except BaseException as exc:
        report.update(status='incomplete',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-budget.started)
        atomic_write_json(out/'report.json',report);raise
