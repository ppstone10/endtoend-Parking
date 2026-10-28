"""重建原400样本、规划余量补查/跳过裁决及固定短检实际转移。"""
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
from verification.margin_study import inspect, read, FILES


def sequence_safe(sequence, state, model, checker):
    current = deepcopy(state)
    if not checker.pose_free(current.x,current.y,current.yaw):return False
    for pair in np.asarray(sequence).reshape(10,2):
        after=model.step(current,ControlCmd(*map(float,pair)),.1)
        if not checker.swept_segment_free(current.to_array()[:3],after.to_array()[:3]):return False
        current=after
    return True


def verify_choice(context, frame, selected, window, state, vehicle, goal, terminal, checker, margin, model):
    """只用干净旧求解器重建400样本，再独立裁决补查首个可行解。"""
    if (context['state']!=state.to_array().tolist() or context['goal']!=[goal.x,goal.y,goal.yaw]
            or context['tol_pos']!=terminal.tol_pos or context['tol_yaw']!=terminal.tol_yaw
            or not np.array_equal(context['reference_window'],window)):
        raise ValueError('补查上下文与实际状态/目标/窗口不符')
    controller=TerminalCaptureMPC(contact_checker=checker,dt=.1,horizon=10,seed=0,**vehicle.mpc_kwargs())
    controller.bind_terminal(goal,terminal);controller._rng.bit_generator.state=deepcopy(context['rng_state'])
    bank=[];cost_function=controller._batch_cost
    def collect(samples,ref,current):
        costs=cost_function(samples,ref,current)
        if len(samples)>1:bank.append((samples.copy(),costs.copy()))
        return costs
    controller._batch_cost=collect
    original=controller._solve(np.array(context['shifted_mean']),window,state)
    prior=controller.last_safety_audit
    if (not np.all(original==0) or not prior['intervened']
            or prior['selected']!=context['original_selection'] or prior['selected_cost']!=context['original_cost']):
        raise ValueError('原零解/原成本/原选择未独立复现')
    samples=np.concatenate([a for a,b in bank]);costs=np.concatenate([b for a,b in bank])
    if len(samples)!=400:raise ValueError('同次原400样本未重建')
    chosen=original;rank=None;checked=0;chosen_cost=prior['selected_cost'];label=prior['selected']
    outside=not margin.pose_free(state.x,state.y,state.yaw)
    if frame['margin_recovery_skip']!=('already_inside_margin' if outside else None):raise ValueError('已侵入状态跳过声明不符')
    order=[] if outside else np.argsort(costs,kind='stable')[16:]
    for offset,ordinal in enumerate(order,start=17):
        if not np.isfinite(costs[ordinal]) or costs[ordinal]>=prior['selected_cost']:break
        checked+=1
        if sequence_safe(samples[ordinal],state,model,checker) and sequence_safe(samples[ordinal],state,model,margin):
            chosen=samples[ordinal];rank=offset;chosen_cost=float(costs[ordinal]);label='expanded_margin_sample';break
    if (not np.array_equal(chosen.reshape(10,2),selected) or frame['selected']!=label
            or frame['margin_recovery_rank']!=rank or frame['margin_recovery_checked']!=checked
            or frame['selected_cost']!=chosen_cost or not sequence_safe(chosen,state,model,checker)):
        raise ValueError('独立补查排名/成本/序列/扫掠不一致')
    return rank


def verify_certificate(certificate, state, vehicle, goal, terminal, checker, margin):
    """重建证书物理转移和原终端；CEM补查由实际对应帧另行重建。"""
    if (certificate['goal']!=[goal.x,goal.y,goal.yaw] or certificate['tol_pos']!=terminal.tol_pos
            or certificate['tol_yaw']!=terminal.tol_yaw or certificate['control_dt']!=.1):
        raise ValueError('证书原目标/容差/时序不符')
    reference=Trajectory(np.array(certificate['reference']),dt=certificate['reference_dt'])
    if array_digest(reference.points)!=certificate['reference_sha256']:raise ValueError('证书参考身份不符')
    model=DifferentialDriveModel(**vehicle.vehicle_model_kwargs());current=deepcopy(state)
    for k,(before,after,command) in enumerate(zip(certificate['states_before'],certificate['states_after'],certificate['controls'],strict=True)):
        if not np.array_equal(current.to_array(),before):raise ValueError('证书前态不连续')
        proposed=model.step(current,ControlCmd(*map(float,command)),.1)
        if (not np.array_equal(proposed.to_array(),after)
                or not checker.swept_segment_free(before[:3],after[:3])
                or not margin.swept_segment_free(before[:3],after[:3])):raise ValueError('证书转移/实体/余量不符')
        current=proposed
        if terminal.reached(current,goal)!=(k==len(certificate['controls'])-1):raise ValueError('证书首次原终端时刻不符')


def audit(source, output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('独立审计输出禁止覆盖')
    report=inspect(source)
    if report['mode']!='smoke':raise ValueError('当前独立审计只接受固定短检，不自动扩大正式全量')
    config=read(source/'config.json');data=DatasetGenerator.load(config['data'])
    manifest=matrix.load_dataset_manifest(Path(config['data']));vehicle=VehicleConfig(**manifest['vehicle_model'])
    model=DifferentialDriveModel(**vehicle.vehicle_model_kwargs());started=time.perf_counter();rows=[]
    for result in report['rows']:
        if time.perf_counter()-started>90:raise TimeoutError('独立短审计90秒预算耗尽')
        index=result['dataset_index'];task=matrix.reconstruct_dataset_task(data['task_meta'][index],root_seed=manifest['seed'],vehicle=vehicle)
        terminal=TerminalChecker(task.tol_pos,task.tol_yaw)
        params=dict(vehicle_length=vehicle.length,vehicle_width=vehicle.width,resolution=vehicle.collision_check_resolution)
        checker=RectangleFootprintCollisionChecker(task.task.scene.env,collision_margin=0.,**params)
        margin=RectangleFootprintCollisionChecker(task.task.scene.env,collision_margin=.2,**params)
        folder=source/'control';ledger=read(folder/f'retention-{index:04d}.json');record=read(folder/f'episode-{index:04d}.json')
        if len(ledger['certificates'])>1:raise ValueError('当前协议不接受多个耗尽/重新保持证书')
        certificates={cert['start_step']:cert for cert in ledger['certificates']};violations=0;choices=0;expanded=0
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
                if not np.array_equal(selected[0],command) or not sequence_safe(selected,current,model,checker):
                    raise ValueError('真实首动作/完整时域物理安全不符')
                context=frame.get('margin_recovery_context')
                if context is not None:
                    rank=verify_choice(context,frame,selected,tape['control_windows'][step],current,vehicle,task.goal,terminal,checker,margin,model)
                    choices+=1;expanded+=rank is not None
                elif frame['selected']=='expanded_margin_sample' or (np.all(selected==0) and frame['intervened']):
                    raise ValueError('应补查的实际帧缺少重建上下文')
                proposed=model.step(current,ControlCmd(*map(float,command)),.1)
                if not np.array_equal(proposed.to_array(),after) or not checker.swept_segment_free(before[:3],after[:3]):
                    raise ValueError('实际动作/物理接触未复现')
                safe=margin.swept_segment_free(before[:3],after[:3]);violations+=not safe
                if safe!=frame['planning_margin_safe']:raise ValueError('实际余量声明不符')
                current=proposed
            if terminal.reached(current,task.goal)!=result['success'] or violations!=result['planning_margin_violations']:
                raise ValueError('原终端/余量统计不符')
        rows.append({'dataset_index':index,'steps':result['steps'],'certificates':len(certificates),
                     'held_queries':ledger['held_queries'],'planning_margin_violations':violations,
                     'margin_recovery_invocations':choices,'expanded_choices':expanded})
    payload={'status':'completed','source_report_sha256':_sha256(source/'report.json'),'code':execution_identity(FILES),
             'indices':config['indices'],'episodes':rows,'all_states_commands_and_margin_choices_exact':True,
             'elapsed_seconds':time.perf_counter()-started,'production_qualified':False,
             'limits':'固定短检全部实际转移/时域实体扫掠/参考账本/证书物理与余量/原终端核验；每个实际补查帧用干净旧控制器重建原400样本、零解、稳定首个实体及0.2m安全选择与已侵入跳过。无补查CEM及私有失败尝试未全量重算，非生产或泛化证据。'}
    output.mkdir(parents=True,exist_ok=False);atomic_write_json(output/'report.json',payload)
    return payload
