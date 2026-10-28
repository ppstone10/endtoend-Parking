"""独立重建原零解及当前参考候选；控制单元与真实NN短检分别审计。"""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np

from dataset import DatasetGenerator
from experiments import validation_matrix as matrix
from interfaces import VehicleState, ControlCmd, Trajectory
from planner import RectangleFootprintCollisionChecker
from runtime.termination import TerminalChecker
from sim import VehicleConfig, DifferentialDriveModel
from training.reporting import atomic_write_json
from scripts.diagnose_closed_loop_failures import _sha256
from verification.identity import execution_identity
from verification.reference_retention import array_digest
from verification.terminal_capture import TerminalCaptureMPC
from verification.reference_study import inspect, read, FILES
from verification.reference_teacher import validate_teacher, INDICES
from verification.planning_audit import sequence_safe, verify_choice, verify_certificate
from controller import MPCController


def verify_reference_choice(frame, selected, window, state, vehicle, goal, terminal, checker, margin, model):
    context=frame.get('zero_recovery_context');record=frame.get('reference_candidate')
    if context is not None:
        previous=dict(frame);expected=selected
        if record is not None:
            previous['selected']=record['previous_selection'];previous['selected_cost']=record['previous_cost']
            expected=np.zeros((10,2))
        verify_choice(context,previous,expected,window,state,vehicle,goal,terminal,checker,margin,model)
    elif record is not None or frame['selected']=='expanded_safe_sample' or (np.all(selected==0) and frame['intervened']):
        raise ValueError('原补查上下文缺失')
    if record is None:
        if context is not None and np.all(selected==0) and frame['intervened']:raise ValueError('应推导参考候选的零解缺少记录')
        return 0,0
    # 不调用新增推导函数；按原模型在实际窗口逐步独立重建。
    current=deepcopy(state);pairs=[]
    for target in window:
        speed=float(np.clip(np.dot(target[:2]-[current.x,current.y],[np.cos(current.yaw),np.sin(current.yaw)])/.1,-vehicle.max_v,vehicle.max_v))
        turn=float(np.clip(np.arctan2(np.sin(target[2]-current.yaw),np.cos(target[2]-current.yaw))/.1,-vehicle.max_omega,vehicle.max_omega))
        pairs.append([speed,turn]);current=model.step(current,ControlCmd(speed,turn),.1)
    candidate=np.asarray(pairs).reshape(-1);safe=sequence_safe(candidate,state,model,margin)
    cost_function=MPCController(dt=.1,horizon=10,seed=0,**vehicle.mpc_kwargs())
    raw=float(cost_function._batch_cost(candidate[None,:],window,state)[0]) if safe else None
    cost=raw if raw is not None and np.isfinite(raw) else None
    accepted=bool(safe and cost is not None and cost<record['previous_cost'])
    reason='accepted' if accepted else 'unsafe' if not safe else 'nonfinite_cost' if cost is None else 'not_better'
    if (not np.array_equal(candidate.reshape(10,2),record['candidate']) or safe!=record['horizon_safe']
            or cost!=record['candidate_cost'] or accepted!=record['accepted'] or reason!=record['reason']):
        raise ValueError('参考候选推导/扫掠/原成本/接受条件不一致')
    expected=candidate.reshape(10,2) if accepted else np.zeros((10,2))
    label='reference_inverse_candidate' if accepted else record['previous_selection']
    expected_cost=cost if accepted else record['previous_cost']
    if not np.array_equal(expected,selected) or frame['selected']!=label or frame['selected_cost']!=expected_cost:
        raise ValueError('参考候选实际选择未复现')
    return 1,int(accepted)


def audit_teacher(config, source, vehicle, data, manifest, budget_check):
    report=validate_teacher(config,source);source=Path(source);rows=[]
    for result in report['rows']:
        budget_check();index=result['dataset_index']
        task=matrix.reconstruct_dataset_task(data['task_meta'][index],root_seed=manifest['seed'],vehicle=vehicle)
        terminal=TerminalChecker(task.tol_pos,task.tol_yaw)
        params=dict(vehicle_length=vehicle.length,vehicle_width=vehicle.width,resolution=vehicle.collision_check_resolution)
        checker=RectangleFootprintCollisionChecker(task.task.scene.env,collision_margin=0.,**params)
        margin=RectangleFootprintCollisionChecker(task.task.scene.env,collision_margin=.2,**params)
        model=DifferentialDriveModel(**vehicle.vehicle_model_kwargs());frames=read(source/f'episode-{index:04d}.json')['frames'];attempts=accepts=0
        with np.load(source/f'episode-{index:04d}.npz',allow_pickle=False) as tape:
            current=VehicleState.from_array(tape['initial_state']);reference=Trajectory(tape['reference'],float(tape['reference_dt']))
            if not margin.check_trajectory(current,reference)[0]:raise ValueError('教师原参考不满足余量')
            for step,(before,after,command,selected,window,frame) in enumerate(zip(tape['states_before'],tape['states_after'],tape['commands'],tape['selected_control_sequences'],tape['control_windows'],frames,strict=True)):
                budget_check()
                if not np.array_equal(current.to_array(),before) or not np.array_equal(selected[0],command):raise ValueError('教师前态/首动作不一致')
                a,b=verify_reference_choice(frame,selected,window,current,vehicle,task.goal,terminal,checker,margin,model);attempts+=a;accepts+=b
                proposed=model.step(current,ControlCmd(*map(float,command)),.1)
                if (not np.array_equal(proposed.to_array(),after) or not sequence_safe(selected,current,model,margin)
                        or not checker.swept_segment_free(before[:3],after[:3]) or not margin.swept_segment_free(before[:3],after[:3])):raise ValueError('教师实际转移/H10/余量不符')
                if terminal.reached(proposed,task.goal)!=(step==result['steps']-1):raise ValueError('教师原首次到达不符')
                current=proposed
        if attempts!=result['candidate_attempts'] or accepts!=result['candidate_accepts']:raise ValueError('教师候选统计不符')
        rows.append({'dataset_index':index,'steps':result['steps'],'candidate_attempts':attempts,'candidate_accepts':accepts})
    return rows


def audit(source, output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('独立审计输出禁止覆盖')
    report=inspect(source)
    if report['mode']!='smoke':raise ValueError('当前独立审计只接受固定短检，不自动扩大正式全量')
    config=read(source/'config.json');data=DatasetGenerator.load(config['data'])
    manifest=matrix.load_dataset_manifest(Path(config['data']));vehicle=VehicleConfig(**manifest['vehicle_model'])
    model=DifferentialDriveModel(**vehicle.vehicle_model_kwargs());started=time.perf_counter();rows=[]
    def budget_check():
        if time.perf_counter()-started>120:raise TimeoutError('独立控制单元及14例审计120秒预算耗尽')
    for result in report['rows']:
        budget_check()
        index=result['dataset_index'];task=matrix.reconstruct_dataset_task(data['task_meta'][index],root_seed=manifest['seed'],vehicle=vehicle)
        terminal=TerminalChecker(task.tol_pos,task.tol_yaw)
        params=dict(vehicle_length=vehicle.length,vehicle_width=vehicle.width,resolution=vehicle.collision_check_resolution)
        checker=RectangleFootprintCollisionChecker(task.task.scene.env,collision_margin=0.,**params)
        margin=RectangleFootprintCollisionChecker(task.task.scene.env,collision_margin=.2,**params)
        folder=source/'control';ledger=read(folder/f'retention-{index:04d}.json');record=read(folder/f'episode-{index:04d}.json')
        if len(ledger['certificates'])>1:raise ValueError('当前协议不接受多个耗尽/重新保持证书')
        certificates={cert['start_step']:cert for cert in ledger['certificates']};violations=0;choices=0;expanded=0;attempts=0;accepts=0
        with np.load(folder/f'episode-{index:04d}.npz',allow_pickle=False) as tape,np.load(folder/f'retention-{index:04d}.npz',allow_pickle=False) as raw:
            if set(raw.files)!={q['raw_key'] for q in ledger['queries']}:raise ValueError('原始参考槽不完整')
            for n,query in enumerate(ledger['queries']):
                original=raw[query['raw_key']]
                if array_digest(original)!=query['raw_sha256']:raise ValueError('原始推理参考摘要不符')
                if query['certificate'] is None:
                    selected=original
                else:
                    cert=ledger['certificates'][query['certificate']];selected=np.array(cert['reference'],dtype=cert['reference_dtype'])
                    if not np.array_equal(raw[cert['raw_key']],selected):raise ValueError('证书不是某次原始网络参考')
                if not np.array_equal(tape[f'plan_{n:04d}'],selected) or array_digest(selected)!=query['used_sha256']:
                    raise ValueError('实际使用参考与保留账本不符')
            current=VehicleState.from_array(tape['initial_state'])
            for step,(before,after,command,frame) in enumerate(zip(tape['states_before'],tape['states_after'],tape['commands'],record['frames'],strict=True)):
                if not np.array_equal(current.to_array(),before):raise ValueError('实际前态不连续')
                if step in certificates:
                    cert=certificates[step];n=len(cert['controls'])
                    if step+n!=result['steps'] or n>min(80,600-step):raise ValueError('保持跨越原预算或未按证书到达')
                    if not np.array_equal(tape['commands'][step:step+n],cert['controls']):raise ValueError('实际未执行证书动作')
                    if not np.array_equal(tape['states_after'][step:step+n],cert['states_after']):raise ValueError('实际后态未保持证书')
                    verify_certificate(cert,current,vehicle,task.goal,terminal,checker,margin)
                selected=tape['selected_control_sequences'][step]
                if not np.array_equal(selected[0],command) or not sequence_safe(selected,current,model,checker) or not sequence_safe(selected,current,model,margin):
                    raise ValueError('真实首动作/完整时域物理安全不符')
                budget_check()
                a,b=verify_reference_choice(frame,selected,tape['control_windows'][step],current,vehicle,task.goal,terminal,checker,margin,model)
                attempts+=a;accepts+=b;choices+='zero_recovery_context' in frame;expanded+=frame['selected']=='expanded_safe_sample'
                if frame['selected']=='terminal_rotation':
                    cert=frame['terminal_certificate'];end=before[:3].copy();end[2]+=cert['signed_rotation']
                    if not margin.swept_segment_free(before[:3],end):raise ValueError('终端完整角程余量不符')
                proposed=model.step(current,ControlCmd(*map(float,command)),.1)
                if not np.array_equal(proposed.to_array(),after) or not checker.swept_segment_free(before[:3],after[:3]):
                    raise ValueError('实际动作/物理接触未复现')
                if terminal.reached(proposed,task.goal)!=(result['success'] and step==result['steps']-1):raise ValueError('真实原终端首次到达不符')
                safe=margin.swept_segment_free(before[:3],after[:3]);violations+=not safe
                if not safe:raise ValueError('统一余量候选真实执行侵入规划余量')
                if safe!=frame['planning_margin_safe']:raise ValueError('实际余量声明不符')
                current=proposed
            if terminal.reached(current,task.goal)!=result['success'] or violations!=result['planning_margin_violations']:
                raise ValueError('原终端/余量统计不符')
        rows.append({'dataset_index':index,'steps':result['steps'],'certificates':len(certificates),
                     'held_queries':ledger['held_queries'],'planning_margin_violations':violations,
                     'zero_recovery_invocations':choices,'expanded_choices':expanded,
                     'reference_candidate_attempts':attempts,'reference_candidate_accepts':accepts})
    teacher=source.parent/'teacher'
    teacher_rows=audit_teacher(config,teacher,vehicle,data,manifest,budget_check)
    payload={'teacher_report_sha256':_sha256(teacher/'report.json'),'teacher_episodes':teacher_rows,'status':'completed','source_report_sha256':_sha256(source/'report.json'),'code':execution_identity(FILES),
             'indices':config['indices'],'episodes':rows,'all_states_commands_and_reference_candidates_exact':True,
             'elapsed_seconds':time.perf_counter()-started,'production_qualified':False,
             'limits':'固定14真实NN短检及六安全教师全部真实转移/H10/原车位终端/新增候选独立重建；干净旧控制器重建原400样本与原零解，独立原模型/成本重建每个接受或拒绝的参考候选。无补查CEM及失败私有前瞻未全量重算；教师不作为NN闭环收益，非生产或泛化证据。'}
    output.mkdir(parents=True,exist_ok=False);atomic_write_json(output/'report.json',payload)
    return payload
