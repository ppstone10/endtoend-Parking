"""固定六例已存安全参考的控制单元验收；不接真实网络回合。"""
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from dataset import DatasetGenerator
from experiments import validation_matrix as matrix
from interfaces import VehicleState, Trajectory, ControlCmd
from runtime import ClosedLoopEngine, ExpertSource, TerminalChecker
from sim import VehicleConfig, DifferentialDriveModel
from planner import RectangleFootprintCollisionChecker
from training.reporting import atomic_write_json
from scripts.diagnose_closed_loop_failures import _sha256
from verification.identity import verify_files
from verification.terminal_study import read
from verification.reference_candidate import ReferenceCandidateMPC

INDICES = [131,132,139,161,279,296]


def run_teacher(config_path, config):
    if config['purpose']!='smoke':raise ValueError('教师控制单元只接受短检配置')
    verify_files(config['code']);verify_files(config['inputs'])
    out=Path(config_path).resolve().parent/'teacher';out.mkdir(parents=True,exist_ok=False)
    atomic_write_json(out/'config.json',config)
    rows=[];started=time.perf_counter()
    report={'status':'running','kind':'stored_safe_reference_control_unit','indices':INDICES,
            'code':config['code'],'inputs':config['inputs'],'source_config':str(Path(config_path).resolve()),
            'source_config_sha256':_sha256(Path(config_path)),'network_inference':False,
            'training_performed':False,'production_qualified':False,'rows':rows}
    atomic_write_json(out/'report.json',report)
    try:
        data=DatasetGenerator.load(config['data']);manifest=matrix.load_dataset_manifest(Path(config['data']))
        vehicle=VehicleConfig(**manifest['vehicle_model'])
        for index in INDICES:
            restored=matrix.reconstruct_dataset_task(data['task_meta'][index],root_seed=manifest['seed'],vehicle=vehicle)
            start=VehicleState.from_array(data['states'][index]);terminal=TerminalChecker(restored.tol_pos,restored.tol_yaw)
            points=data['trajs'][index,data['masks'][index]>0];reference=Trajectory(points.copy(),float(data['dt'][0]))
            params=dict(vehicle_length=vehicle.length,vehicle_width=vehicle.width,resolution=vehicle.collision_check_resolution)
            physical=RectangleFootprintCollisionChecker(restored.task.scene.env,collision_margin=0.,**params)
            margin=RectangleFootprintCollisionChecker(restored.task.scene.env,collision_margin=.2,**params)
            if not np.array_equal(start.to_array(),restored.task.start.to_array()) or not margin.check_trajectory(start,reference)[0]:
                raise ValueError('教师初态/完整参考余量不符')
            if not terminal.reached(VehicleState(*map(float,points[-1])),restored.goal):raise ValueError('教师原末点不达标')
            controller=ReferenceCandidateMPC(contact_checker=physical,dt=.1,horizon=10,seed=index,**vehicle.mpc_kwargs())
            controller.bind_terminal(restored.goal,terminal);frames=[];compute=controller.compute
            def observed(trajectory,state):
                if time.perf_counter()-started>90:raise TimeoutError('固定六教师90秒预算耗尽')
                command=compute(trajectory,state)
                frames.append({'before':state.to_array(),'command':[command.v,command.omega],
                               'sequence':controller._last_u.copy().reshape(10,2),
                               'window':controller._reference_points(controller._aligned_cache,controller._progress_idx).copy(),
                               'audit':dict(controller.last_safety_audit)})
                return command
            controller.compute=observed;model=DifferentialDriveModel(**vehicle.vehicle_model_kwargs())
            source=ExpertSource(SimpleNamespace(plan=lambda state,goal:reference))
            engine=ClosedLoopEngine(model,controller,source,terminal,env=restored.task.scene.env,
                                   vehicle_length=vehicle.length,vehicle_width=vehicle.width,replan_every=10,
                                   max_steps=600,meta={'dataset_index':index},collision_checker=physical)
            result=engine.run(start,restored.goal);after=np.asarray([s.to_array() for s in result.record.states]);violations=0
            for frame,target in zip(frames,after,strict=True):
                actual=model.step(VehicleState.from_array(frame['before']),ControlCmd(*frame['command']),.1)
                if not np.array_equal(actual.to_array(),target) or not physical.swept_segment_free(frame['before'][:3],target[:3]):raise ValueError('教师真实状态/接触不符')
                violations+=not margin.swept_segment_free(frame['before'][:3],target[:3])
            row={'dataset_index':index,'success':result.success,'collision':result.collision,'failure':result.failure,
                 'steps':result.steps,'final_pos_err':result.final_pos_err,'final_yaw_err':result.final_yaw_err,
                 'planning_margin_violations':int(violations),
                 'candidate_attempts':sum('reference_candidate' in f['audit'] for f in frames),
                 'candidate_accepts':sum(f['audit'].get('reference_candidate',{}).get('accepted',False) for f in frames)}
            commands=np.asarray([f['command'] for f in frames])
            np.savez_compressed(out/f'episode-{index:04d}.npz',initial_state=start.to_array(),goal=np.array([restored.goal.x,restored.goal.y,restored.goal.yaw]),
                                reference=reference.points,reference_dt=reference.dt,states_before=np.asarray([f['before'] for f in frames]),
                                states_after=after,commands=commands,selected_control_sequences=np.asarray([f['sequence'] for f in frames]),
                                control_windows=np.asarray([f['window'] for f in frames]))
            atomic_write_json(out/f'episode-{index:04d}.json',{'summary':row,'frames':[f['audit'] for f in frames]})
            rows.append(row);atomic_write_json(out/'report.json',report);print(f'[reference-teacher] #{index} {result.failure or "success"} {result.steps}',flush=True)
        report.update(status='completed',qualified=all(r['success'] and not r['collision'] and r['planning_margin_violations']==0 for r in rows),
                      elapsed_seconds=time.perf_counter()-started)
        atomic_write_json(out/'report.json',report)
        atomic_write_json(out/'artifacts.json',{p.name:_sha256(p) for p in out.iterdir() if p.is_file()})
        return report
    except BaseException as exc:
        report.update(status='incomplete',error=f'{type(exc).__name__}: {exc}');atomic_write_json(out/'report.json',report);raise


def validate_teacher(config, source):
    source=Path(source).resolve();report=read(source/'report.json');artifacts=read(source/'artifacts.json')
    required={'config.json','report.json',*(f'episode-{i:04d}.{ext}' for i in INDICES for ext in ('json','npz'))}
    if not required<=set(artifacts):raise ValueError('教师控制单元产物不完整')
    for name,digest in artifacts.items():
        path=(source/name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or _sha256(path)!=digest:raise ValueError('教师产物身份漂移')
    if (report['status']!='completed' or not report['qualified'] or report['indices']!=INDICES
            or [r['dataset_index'] for r in report['rows']]!=INDICES or report['code']!=config['code']
            or any(config['inputs'].get(k)!=v for k,v in report['inputs'].items())
            or _sha256(Path(report['source_config']))!=report['source_config_sha256']
            or not all(r['success'] and not r['collision'] and r['planning_margin_violations']==0 for r in report['rows'])):
        raise ValueError('固定安全教师六例未全部合格或身份不符')
    return report
