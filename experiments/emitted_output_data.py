"""Frozen training-only eight-task cohort and original-MPC teacher gate.

Owns provenance and teacher recordings, not optimization or deployment.
"""
import json
from pathlib import Path

import numpy as np
import torch

from dataset import DatasetGenerator
from experiments.executable_behavior import run_recorded_branch
from experiments.recovery_execution import execution_summary, record_arrays
from experiments.reference_compensation import reference_internal_safe
from interfaces import Trajectory, VehicleState
from runtime import NetworkSource
from scripts.build_selection_dataset import collect_metadata_ids
from scripts.diagnose_reference_execution import code_identity
from scripts.preflight_behavior_repair import build_case
from scripts.prepare_recovery_balance_ab import file_digest, metadata_only, runtime_identity
from sim import VehicleConfig
from training.data import prepare_batches
from training.reporting import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'runs/verification/surround-target-rebuild-lineage-20261027'
INPUTS = {
    'train': DATA_ROOT / 'train/train-balanced.npz',
    'manifest': DATA_ROOT / 'train/manifest.json',
    'parent': ROOT / 'runs/training/lt31-terminal-hold-20261027/control/last.pt',
    'recipe': ROOT / 'configs/training/net-v1-terminal-hold-control.yaml',
    'val': DATA_ROOT / 'val/val.npz',
    'lt33': ROOT / 'runs/verification/lt33-independent-review-20261028/report.json',
    'lt32': ROOT / 'runs/verification/strict-footprint-formal-20261028/results/report.json',
}
# Fixed before reading any new outputs. Seven represented scenes plus a reverse S1.
INDICES = [0, 433, 860, 1290, 1720, 1975, 2315, 2]
RECIPE = dict(seeds=[20261028, 20261029], updates=200, checkpoints=[0, 50, 100, 200],
              batch_size=8, learning_rate=1e-5, teacher_forcing=.2, joint_weight=.1,
              temperature=.25, gradient_clip=1., frames_per_task=4,
              max_steps=600, replan_every=10, control_seed=0)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_digest(path))


def execution_identity():
    paths = [ROOT / 'scripts/train_emitted_output_joint.py',
             ROOT / 'scripts/preflight_behavior_repair.py',
             ROOT / 'scripts/build_selection_dataset.py']
    return {**code_identity(), **{str(p): file_digest(p) for p in paths}}


def regression_cohort(lt33, lt32, metadata):
    lost = lt33['lost_lt32_successes']; gained = lt33['gained_successes']
    native = [r['dataset_index'] for r in lt32['rows']['native'] if r['success']]
    residual = [183, 186, 207, 292, 296]
    if (lt33['status'] != 'completed' or len(lost) != 22 or len(gained) != 8
            or len(native) != 93 or len(metadata) != 300):
        raise ValueError('LT33/LT32 regression denominators changed')
    groups = dict(regressions=lost, recoveries=gained, residual=residual, native_success=native)
    union = sorted(set().union(*groups.values()))
    ids = [metadata[i]['task_id'] for i in union]
    if len(set(ids)) != len(ids):
        raise ValueError('validation task identities duplicated')
    return dict(groups=groups, indices=union, task_ids=ids, unique_tasks=len(ids),
                evaluation_only=True, independent_unseen_test=False)


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('prepared directory exists; do not overwrite')
    identities = {key: file_identity(path) for key, path in INPUTS.items()}
    if (identities['train']['sha256'] != '137b13a63f90188a227f9b8e6b64a36b604c37a71a08775db331c5ec00fb15e2'
            or identities['parent']['sha256'] != '359bfb7eb827acc5dadaf74a31c496b4b8a07346d8eee91ee9b3a8495fc83c9c'):
        raise ValueError('LT31 parent or training source changed')
    metadata = metadata_only(INPUTS['train']); val_meta = metadata_only(INPUTS['val'])
    chosen = [metadata[i] for i in INDICES]
    if (len(metadata) != 2918 or len({m['task_id'] for m in chosen}) != 8
            or any(m.get('recovery') for m in chosen)
            or len({m['scene_name'] for m in chosen}) != 7
            or {m['difficulty']['maneuver'] for m in chosen} != {'forward', 'reverse'}
            or collect_metadata_ids(metadata) & collect_metadata_ids(val_meta)):
        raise ValueError('training source coverage/identity/isolation invalid')
    output.mkdir(parents=True)
    # Decompress the large BEV once and persist only the preselected eight rows.
    with np.load(INPUTS['train'], allow_pickle=False) as archive:
        subset = {k: archive[k][INDICES] if k in {'bevs','goals','states','trajs','masks','task_meta'}
                  else archive[k] for k in archive.files}
    np.savez_compressed(output / 'cohort.npz', **subset)
    atomic_write_json(output / 'source-manifest.json', read_json(INPUTS['manifest']))
    config = dict(schema='emitted_joint_v1', inputs=identities, code=execution_identity(),
                  runtime=runtime_identity(), recipe=RECIPE, indices=INDICES,
                  sources=[dict(index=i, task_id=m['task_id'], scene_name=m['scene_name'],
                                maneuver=m['difficulty']['maneuver']) for i,m in zip(INDICES,chosen)],
                  regression=regression_cohort(read_json(INPUTS['lt33']), read_json(INPUTS['lt32']), val_meta),
                  cohort=file_identity(output / 'cohort.npz'),
                  manifest=file_identity(output / 'source-manifest.json'),
                  candidate_qualified=False, automatic_training=False)
    atomic_write_json(output / 'config.json', config)
    return config


def precheck(prepared):
    prepared = Path(prepared).resolve()
    config = read_json(prepared / 'config.json')
    if (config['schema'] != 'emitted_joint_v1' or config['recipe'] != RECIPE
            or config['indices'] != INDICES or config['code'] != execution_identity()
            or config['runtime'] != runtime_identity()
            or Path(config['runtime']['executable']).resolve() != Path('D:/conda/envs/endtoend-parking/python.exe')):
        raise ValueError('frozen code/runtime/recipe drift; never edit hashes to bypass')
    for item in [*config['inputs'].values(), config['cohort'], config['manifest']]:
        if file_identity(item['path']) != item:
            raise ValueError(f'frozen input drift: {item["path"]}')
    return config


def load_cohort(prepared):
    data = DatasetGenerator.load(Path(prepared) / 'cohort.npz')
    manifest = read_json(Path(prepared) / 'source-manifest.json')
    return data, manifest, VehicleConfig(**manifest['vehicle_model'])


def case_at(data, manifest, vehicle, row):
    case = build_case(data, manifest, row, vehicle)
    case['index'] = INDICES[row]  # Original controller seed, never subset ordinal.
    return case


def initial_samples(data):
    batch = prepare_batches(data, horizon=320, batch_size=8)[0]
    samples = []
    for row, index in enumerate(INDICES):
        goal = data['task_meta'][row]['dataset']['selected_goal']
        samples.append(dict(zip(('bev','goal','state','points','mask'), (x[row].clone() for x in batch))))
        samples[-1].update(source_index=index, frame_index=0, pose=data['states'][row,:3].copy(),
                           tolerances=torch.tensor([goal['tol_pos'], goal['tol_yaw']], dtype=torch.float64))
    return samples


def pose_pass(point, goal, tol_pos, tol_yaw):
    delta = np.asarray(point) - np.asarray(goal)
    pos = float(np.linalg.norm(delta[:2]))
    yaw = float(abs(np.arctan2(np.sin(delta[2]), np.cos(delta[2]))))
    return dict(position_m=pos, yaw_rad=yaw, joint_passed=pos <= tol_pos and yaw <= tol_yaw)


def reference_audit(points, pose, case):
    global_points = NetworkSource._to_global(np.asarray(points), VehicleState(*pose, 0., 0.))
    trajectory = Trajectory(global_points, .2)
    margin = case['planner']._collision_checker
    contact = case['engine_kwargs']['collision_checker']
    def safe(checker):
        return (reference_internal_safe(trajectory, checker)
                and checker.swept_segment_free(np.asarray(pose), global_points[0]))
    return dict(contact_safe=bool(safe(contact)), margin_safe=bool(safe(margin)))


def teacher_gate(rows):
    if len(rows) != 8 or [r['dataset_index'] for r in rows] != INDICES:
        raise ValueError('teacher denominator/order changed; no task substitutions')
    failed = [r['dataset_index'] for r in rows if not r['passed']]
    return dict(passed=not failed, failed_indices=failed,
                decision='ready_for_paired_fit' if not failed else 'stop_fix_labels_or_control_no_training')


def run_teacher(prepared, destination, budget, *, rows_to_run=8):
    """Formal gate uses all eight; a one-case smoke never grants gate qualification."""
    if rows_to_run not in (1, 8):
        raise ValueError('only a one-case smoke or the complete eight-task gate is allowed')
    config = precheck(prepared)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', formal=rows_to_run == 8, rows=[], gate=None,
                  config_sha256=file_digest(Path(prepared) / 'config.json'), training_performed=False)
    atomic_write_json(destination / 'report.json', report)
    try:
        data, manifest, vehicle = load_cohort(prepared)
        originals = initial_samples(data); selected = []
        for row in range(rows_to_run):
            budget.check()
            case = case_at(data, manifest, vehicle, row)
            source = originals[row]; length = int(source['mask'].sum())
            full = np.array(data['trajs'][row,:length], copy=True)
            print(f'TEACHER #{case["index"]} {case["scene_name"]} BEGIN', flush=True)
            label = pose_pass(full[-1], data['goals'][row], *source['tolerances'].tolist())
            safety = reference_audit(source['points'][:length].numpy(), source['pose'], case)
            result, engine, frames = run_recorded_branch(
                model=None, pipeline=case['pipeline'], planner=case['planner'], start=case['start'],
                goal=case['goal'], vehicle=vehicle, engine_kwargs=case['engine_kwargs'],
                index=case['index'], budget=budget, fixed_trajectory=Trajectory(full, .2))
            execution = execution_summary(result, engine.margin_violations)
            states, commands = record_arrays(result)
            picked = np.unique(np.linspace(0, len(frames)-1, min(4,len(frames)), dtype=int)) if frames else []
            frame_checks = []; task_samples = [source]
            for f in picked:
                frame = frames[f]
                if not frame['mask'].any():
                    raise ValueError('teacher frame was never consumed')
                n = frame['length']
                p = frame['points'][:n]
                terminal = pose_pass(p[-1], frame['goal'], *source['tolerances'].tolist())
                audit = reference_audit(p, frame['pose'], case)
                frame_checks.append(dict(frame_index=int(f), offset=frame['offset'], length=n,
                                         terminal=terminal, **audit))
                if f == 0:
                    np.testing.assert_array_equal(frame['bev'], source['bev'].numpy())
                    np.testing.assert_allclose(frame['goal'], source['goal'].numpy(), atol=1e-6, rtol=0)
                    continue
                target = torch.zeros(320,3); target[:n] = torch.from_numpy(p)
                task_samples.append(dict(bev=torch.from_numpy(frame['bev']).float(),
                    goal=torch.from_numpy(frame['goal']).float(), state=torch.from_numpy(frame['state']).float(),
                    points=target, mask=(torch.arange(320)<n).float(), tolerances=source['tolerances'].clone(),
                    pose=np.asarray(frame['pose']), source_index=case['index'], frame_index=int(f)))
            passed = bool(label['joint_passed'] and all(safety.values()) and execution['reached_with_margin']
                          and frame_checks and all(c['terminal']['joint_passed'] and c['contact_safe']
                                                  and c['margin_safe'] for c in frame_checks))
            record = dict(dataset_index=case['index'], task_id=case['task_id'], scene_name=case['scene_name'],
                          label=label, reference=safety, execution=execution, frames=frame_checks, passed=passed)
            report['rows'].append(record); selected.extend(task_samples)
            np.savez_compressed(destination / f'teacher-{case["index"]}.npz', states=states, commands=commands,
                reference=full, dt=.2, progress_trace=np.asarray(engine.mpc.execution_trace),
                planning_margin=np.asarray(engine.margin_violations),
                frame_bevs=np.stack([f['bev'] for f in frames]),
                frame_goals=np.stack([f['goal'] for f in frames]),
                frame_states=np.stack([f['state'] for f in frames]),
                frame_poses=np.array([f['pose'] for f in frames]),
                frame_points=np.stack([f['points'] for f in frames]),
                frame_lengths=np.array([f['length'] for f in frames]), picked=np.asarray(picked))
            print(f'TEACHER #{case["index"]} passed={passed} steps={result.steps}', flush=True)
            atomic_write_json(destination / 'report.json', report)
        precheck(prepared)
        if rows_to_run == 8:
            report['gate'] = teacher_gate(report['rows'])
        report.update(status='completed', selected_samples=len(selected))
        # Failed teacher data is retained for diagnosis, never authorized for fitting.
        torch.save(selected, destination / 'samples.pt')
        atomic_write_json(destination / 'report.json', report)
        artifacts = {str(p.resolve()):file_digest(p) for p in destination.iterdir() if p.is_file()}
        atomic_write_json(destination / 'artifacts.json', artifacts)
        return report
    except BaseException as exc:
        report.update(status='incomplete', error=str(exc))
        atomic_write_json(destination / 'report.json', report)
        raise


def load_authorized_samples(prepared, teacher):
    precheck(prepared)
    teacher = Path(teacher).resolve()
    artifacts = read_json(teacher / 'artifacts.json')
    required = [teacher / name for name in ('report.json', 'samples.pt')]
    required += [teacher / f'teacher-{i}.npz' for i in INDICES]
    if set(artifacts) != {str(p) for p in required}:
        raise ValueError('teacher artifacts incomplete')
    for path, digest in artifacts.items():
        if file_digest(path) != digest:
            raise ValueError('teacher artifacts drift')
    report = read_json(teacher / 'report.json')
    if (report['status'] != 'completed' or not report['formal']
            or report['config_sha256'] != file_digest(Path(prepared) / 'config.json')
            or report['gate'] != teacher_gate(report['rows']) or not report['gate']['passed']):
        raise ValueError('teacher gate not passed: training forbidden')
    samples = torch.load(teacher / 'samples.pt', weights_only=False, map_location='cpu')
    if ({s['source_index'] for s in samples} != set(INDICES)
            or len(samples) != report['selected_samples']):
        raise ValueError('supervision denominator changed')
    return samples
