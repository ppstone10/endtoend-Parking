"""用干净控制器重建参考保持证书，并独立检查全部短检实际转移。"""
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
from verification.retention_study import inspect, read, FILES


def verify_certificate(certificate, state, vehicle, goal, terminal, checker, margin):
    """不调用certify_reference；重建同一次CEM/RNG上下文再逐步求解。"""
    if (certificate['goal'] != [goal.x,goal.y,goal.yaw] or certificate['tol_pos'] != terminal.tol_pos
            or certificate['tol_yaw'] != terminal.tol_yaw or certificate['control_dt'] != .1):
        raise ValueError('证书目标/原车位容差/时序不符')
    controller=TerminalCaptureMPC(contact_checker=checker,dt=.1,horizon=10,seed=0,**vehicle.mpc_kwargs())
    controller.bind_terminal(goal,terminal)
    context=certificate['context'];controller._last_u=np.array(context['last_u'])
    controller._rng.bit_generator.state=deepcopy(context['rng_state'])
    reference=Trajectory(np.array(certificate['reference']),dt=certificate['reference_dt'])
    if array_digest(reference.points)!=certificate['reference_sha256']:
        raise ValueError('证书参考内容身份不一致')
    if context['reference_is_progress']:
        controller._progress_traj=reference
        controller._progress_idx=context['progress_idx']
        controller._aligned_cache=None if context['aligned_cache'] is None else np.array(context['aligned_cache'])
    current=deepcopy(state)
    for k,(before,after,expected) in enumerate(zip(certificate['states_before'],certificate['states_after'],certificate['controls'],strict=True)):
        if not np.array_equal(current.to_array(),before):raise ValueError('证书前态未复现')
        command=controller.compute(reference,current)
        if not np.array_equal([command.v,command.omega],expected):raise ValueError('独立CEM证书动作未复现')
        proposed=controller.vehicle_model.step(current,command,.1)
        if (not np.array_equal(proposed.to_array(),after)
                or not checker.swept_segment_free(current.to_array()[:3],proposed.to_array()[:3])
                or not margin.swept_segment_free(current.to_array()[:3],proposed.to_array()[:3])):
            raise ValueError('证书后态/实体/余量未复现')
        current=proposed
        if terminal.reached(current,goal)!=(k==len(certificate['controls'])-1):
            raise ValueError('证书原终端首次到达时刻不符')


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
        certificates={cert['start_step']:cert for cert in ledger['certificates']};violations=0
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
                proposed=model.step(current,ControlCmd(*map(float,command)),.1)
                if not np.array_equal(proposed.to_array(),after) or not checker.swept_segment_free(before[:3],after[:3]):
                    raise ValueError('实际动作/物理接触未复现')
                safe=margin.swept_segment_free(before[:3],after[:3]);violations+=not safe
                if safe!=frame['planning_margin_safe']:raise ValueError('实际余量声明不符')
                current=proposed
            if terminal.reached(current,task.goal)!=result['success'] or violations!=result['planning_margin_violations']:
                raise ValueError('原终端/余量统计不符')
        rows.append({'dataset_index':index,'steps':result['steps'],'certificates':len(certificates),
                     'held_queries':ledger['held_queries'],'planning_margin_violations':violations})
    payload={'status':'completed','source_report_sha256':_sha256(source/'report.json'),'code':execution_identity(FILES),
             'indices':config['indices'],'episodes':rows,'all_states_commands_and_certificates_exact':True,
             'elapsed_seconds':time.perf_counter()-started,'production_qualified':False,
             'limits':'固定短检所有实际转移及参考保持证书的CEM/控制/RNG/原车位终端/实体与余量独立重建；非证书CEM未全量复算，非生产或泛化证据。'}
    output.mkdir(parents=True,exist_ok=False);atomic_write_json(output/'report.json',payload)
    return payload
