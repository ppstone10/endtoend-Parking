"""只重建相机target；不重规划、不重采LiDAR、不按task_id去重。"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from dataset import build_task_components
from experiments.closed_loop_evaluation import load_dataset_manifest, reconstruct_dataset_task
from interfaces.bev import CAMERA_PROJECTION_CURRENT, CAMERA_PROJECTION_LEGACY
from dataset.target_bev import rasterize_goal_target
from sim import VehicleConfig
from training.reporting import atomic_write_json


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def load_original_tasks(path, manifest):
    """原任务归档是身份权威，不能用恢复标签重新采样场景。"""
    original = Path(path).resolve()
    original_manifest = load_dataset_manifest(original)
    if any(original_manifest.get(key) != manifest.get(key) for key in ('seed', 'vehicle_model')):
        raise ValueError('原任务归档manifest的seed/车辆与重建来源不一致')
    with np.load(original, allow_pickle=False) as archive:
        if int(archive['schema_version']) != 2:
            raise ValueError('原任务归档需要schema v2')
        return [json.loads(str(value)) for value in archive['task_meta']]


def resolve_rebuild_task(metadata, *, root_seed, vehicle, original_tasks=None):
    """认证恢复谱系，仅在内存中区分原任务机动和恢复监督机动。"""
    normalized = metadata
    recovery = metadata.get('recovery')
    original_maneuver = metadata['difficulty']['maneuver']
    source_index = None
    if recovery and original_tasks is not None:
        source_index = recovery.get('source_dataset_index')
        if (type(source_index) is not int or not 0 <= source_index < len(original_tasks)):
            raise ValueError('恢复source_dataset_index无效')
        parent = original_tasks[source_index]
        dataset = metadata.get('dataset', {})
        audit = dataset.get('maneuver_audit', {})
        if (parent.get('recovery') or recovery.get('schema_version') != 1
                or recovery.get('source_task_id') != parent.get('task_id')
                or metadata.get('task_id') != parent.get('task_id')
                or recovery.get('label_policy') != 'expert_replan_from_learner_state'
                or dataset.get('source') != 'closed_loop_recovery'
                or dataset.get('goal_policy') != 'original_selected_goal'
                or metadata['difficulty']['maneuver'] not in ('forward', 'reverse')
                or audit.get('requested_maneuver') != metadata['difficulty']['maneuver']
                or audit.get('consistent') is not True):
            raise ValueError('恢复来源ID、生成协议或机动审计不一致')
        original_maneuver = parent['difficulty']['maneuver']
        normalized = deepcopy(metadata)
        normalized['difficulty']['maneuver'] = original_maneuver
        # 只允许生产者明确改写的标签，任何坐标/事件/采样配置漂移仍失败。
        keys = ('schema_version', 'scene_name', 'task_type', 'task_id', 'seed',
                'difficulty', 'start', 'goal', 'candidate_goals', 'dynamic_event', 'noise_profile')
        if (any(normalized.get(key) != parent.get(key) for key in keys)
                or dataset.get('selected_goal') != parent.get('dataset', {}).get('selected_goal')
                or dataset.get('vehicle_model') != parent.get('dataset', {}).get('vehicle_model')):
            raise ValueError('恢复谱系存在机动标签以外的原任务漂移')
    task = reconstruct_dataset_task(normalized, root_seed=root_seed, vehicle=vehicle)
    lineage = {'original_task_maneuver': original_maneuver,
               'recovery_label_maneuver': metadata['difficulty']['maneuver'] if recovery else None,
               'source_dataset_index': source_index,
               'recovery_label_differs': bool(recovery and original_maneuver != metadata['difficulty']['maneuver'])}
    return task, lineage


def load_prefix(path, limit):
    """短预检只解压BEV前缀，正式运行保留完整原数组形状与dtype。"""
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files if name != 'bevs'}
    count = len(data['states'])
    if not count:
        raise ValueError('不能重建空数据集')
    selected = min(limit, count) if limit else count
    with zipfile.ZipFile(path) as archive, archive.open('bevs.npy') as stream:
        version = np.lib.format.read_magic(stream)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = reader(stream)
        if fortran or shape[0] != count:
            raise ValueError('需要C顺序、样本分母一致的BEV数组')
        size = int(np.prod(shape[1:]))*selected*dtype.itemsize
        values = stream.read(size)
        if len(values) != size:
            raise ValueError('BEV数组不完整')
        data['bevs'] = np.frombuffer(values, dtype=dtype).reshape((selected, *shape[1:])).copy()
    for name in ('states', 'goals', 'task_meta', 'trajs', 'masks'):
        if len(data[name]) != count:
            raise ValueError(f'{name}样本分母不一致')
        data[name] = data[name][:selected]
    return data, count


def repair_camera_targets(data_path, output_dir, *, limit=0, budget_seconds=3600, manifest_path=None,
                          original_data_path=None):
    """新目录写副本；任一失败停止，失败目录保留报告但不得作为成功产物使用。"""
    if limit < 0 or not np.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError('limit必须非负，budget_seconds必须为有限正数')
    source, output = Path(data_path).resolve(), Path(output_dir).resolve()
    if output == source.parent or output.exists():
        raise FileExistsError('只允许尚不存在的新输出目录，禁止覆盖源或旧产物')
    manifest_path = Path(manifest_path).resolve() if manifest_path else source.parent/'manifest.json'
    if manifest_path.name != 'manifest.json':
        raise ValueError('显式manifest路径必须指向manifest.json')
    source_manifest = load_dataset_manifest(manifest_path.parent/source.name)
    vehicle = VehicleConfig(**source_manifest['vehicle_model'])
    original_data_path = Path(original_data_path).resolve() if original_data_path else None
    original_tasks = load_original_tasks(original_data_path, source_manifest) if original_data_path else None
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    def check_budget():
        if time.perf_counter()-started > budget_seconds:
            raise TimeoutError('相机target重建预算耗尽；保留源数据，换新目录重跑')
    report = {'status': 'running', 'source': str(source), 'output': str(output),
              'source_sha256': file_sha256(source),
              'source_manifest': str(manifest_path),
              'source_manifest_sha256': file_sha256(manifest_path),
              'original_data': str(original_data_path) if original_data_path else None,
              'original_data_sha256': file_sha256(original_data_path) if original_data_path else None,
              'original_manifest_sha256': file_sha256(original_data_path.parent/'manifest.json') if original_data_path else None,
              'camera_projection': CAMERA_PROJECTION_CURRENT, 'production_qualified': False}
    atomic_write_json(output/'report.json', report)
    try:
        data, total = load_prefix(source, limit)
        if int(data['schema_version']) != 2:
            raise ValueError('需要schema v2数据集与完整任务元数据')
        metadata = json.loads(str(data['bev_meta']))
        old_version = metadata.get('camera_projection', CAMERA_PROJECTION_LEGACY)
        if old_version != CAMERA_PROJECTION_LEGACY:
            raise ValueError(f'需要旧image输入，实际camera_projection={old_version}')
        if not np.isfinite(data['states']).all() or not np.isfinite(data['goals']).all():
            raise ValueError('状态和目标必须有限')
        target_index = metadata['channels'].index('target')
        other = [i for i in range(len(metadata['channels'])) if i != target_index]
        before_other, after_other = hashlib.sha256(), hashlib.sha256()
        groups = defaultdict(lambda: {'samples': 0, 'old_zero_hit': 0, 'new_zero_hit': 0,
                                      'old_iou_sum': 0., 'new_iou_sum': 0.})
        rows = []
        for i, raw_meta in enumerate(data['task_meta']):
            check_budget()
            task_meta = json.loads(str(raw_meta))
            restored, lineage = resolve_rebuild_task(task_meta, root_seed=source_manifest['seed'], vehicle=vehicle,
                                                     original_tasks=original_tasks)
            goal = np.asarray([restored.goal.x, restored.goal.y, restored.goal.yaw])
            if not np.allclose(data['goals'][i], goal, atol=1e-10, rtol=0):
                raise ValueError(f'#{i}存储目标与selected_goal不一致')
            _, pipeline = build_task_components(restored.task, vehicle)
            if (metadata['resolution'] != pipeline.bev_config.resolution
                    or tuple(metadata['extent']) != pipeline.bev_config.extent
                    or list(data['bevs'].shape[1:]) != metadata['shape']):
                raise ValueError(f'#{i}BEV网格与任务配置不一致')
            pose = data['states'][i, :3]
            pipeline.set_target_goals([restored.goal])
            frame = pipeline.camera_sensor.capture(*map(float, pose))
            new = pipeline.camera2bev.to_bev(frame, *pose).data[0]
            truth = rasterize_goal_target(restored.goal, vehicle.length, vehicle.width, *pose, pipeline.bev_config)>0
            old = data['bevs'][i, target_index]>0
            before_other.update(np.ascontiguousarray(data['bevs'][i, other]).tobytes())
            dx, dy = goal[:2]-pose[:2]
            bearing = float(np.arctan2(dy, dx)-pose[2])
            bearing = float(np.arctan2(np.sin(bearing), np.cos(bearing)))
            direction = ('front' if abs(bearing)<=np.pi/4 else 'rear' if abs(bearing)>=3*np.pi/4
                         else 'left' if bearing>0 else 'right')
            row = {'index': i, 'task_id': task_meta['task_id'], 'direction': direction,
                   'lineage': lineage,
                   'truth_cells': int(truth.sum()), 'changed': not np.array_equal(data['bevs'][i, target_index], new)}
            for name, mask in [('old', old), ('new', new>0)]:
                hit = int((mask&truth).sum())
                row[name+'_intersection'] = hit
                row[name+'_iou'] = float(hit/max(1, (mask|truth).sum())) if truth.any() else None
            group = groups[f"{task_meta['scene_name']}|{task_meta['difficulty']['noise_level']}|{direction}"]
            group['samples'] += 1
            if truth.any():
                for name in ('old', 'new'):
                    group[name+'_zero_hit'] += int(row[name+'_intersection']==0)
                    group[name+'_iou_sum'] += row[name+'_iou']
            data['bevs'][i, target_index] = new
            after_other.update(np.ascontiguousarray(data['bevs'][i, other]).tobytes())
            rows.append(row)
            if (i+1)%100 == 0:
                print(f'[camera-target {i+1}/{len(data["states"])}]', flush=True)
        if before_other.digest() != after_other.digest():
            raise ValueError('非target通道发生变化')
        metadata['camera_projection'] = CAMERA_PROJECTION_CURRENT
        data['bev_meta'] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
        check_budget()
        destination = output/source.name
        np.savez_compressed(destination, **data)
        atomic_write_json(output/'manifest.json', {
            'schema_version': 1, 'seed': source_manifest['seed'], 'vehicle_model': vehicle.to_metadata(),
            'camera_projection': CAMERA_PROJECTION_CURRENT, 'samples': len(rows),
            'partial': len(rows)!=total, 'source': report['source'], 'source_sha256': report['source_sha256']})
        code_root = Path(__file__).resolve().parents[1]
        report.update(status='completed', total_source_samples=total, rebuilt=len(rows), partial=len(rows)!=total,
                      recovery_labels_different=sum(row['lineage']['recovery_label_differs'] for row in rows),
                      previous_camera_projection=old_version, rows=rows, groups=dict(groups),
                      other_channels_sha256=before_other.hexdigest(), labels_and_task_arrays_copied=True,
                      output_data=str(destination), output_sha256=file_sha256(destination),
                      elapsed_seconds=time.perf_counter()-started,
                      code_sha256={str(p.relative_to(code_root)): file_sha256(p) for p in (
                          Path(__file__), code_root/'sim/sensor_camera.py', code_root/'sensor2bev/camera_bev.py',
                          code_root/'interfaces/sensor.py', code_root/'dataset/components.py')})
        atomic_write_json(output/'report.json', report)
        atomic_write_json(output/'artifacts.json', {p.name: file_sha256(p) for p in output.iterdir()
                                                   if p.is_file() and p.name!='artifacts.json'})
        return report
    except BaseException as exc:
        report.update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(output/'report.json', report)
        raise


def inspect_camera_target_repair(output_dir):
    """独立读取源与副本，逐样本验证保留项，不依赖报告中的自述通过标记。"""
    output = Path(output_dir).resolve()
    report = json.loads((output/'report.json').read_text(encoding='utf-8'))
    if report.get('status') != 'completed':
        raise ValueError('重建未完成')
    artifacts = json.loads((output/'artifacts.json').read_text(encoding='utf-8'))
    required = {'report.json', 'manifest.json', Path(report['output_data']).name}
    if set(artifacts) != required:
        raise ValueError('登记产物缺失或多余')
    for name, digest in artifacts.items():
        path = (output/name).resolve()
        if path.parent != output or file_sha256(path) != digest:
            raise ValueError('重建产物身份不一致')
    source = Path(report['source'])
    if (file_sha256(source) != report['source_sha256']
            or file_sha256(report['source_manifest']) != report['source_manifest_sha256']):
        raise ValueError('重建源身份漂移')
    original_tasks = None
    if report.get('original_data'):
        original = Path(report['original_data'])
        if (file_sha256(original) != report['original_data_sha256']
                or file_sha256(original.parent/'manifest.json') != report['original_manifest_sha256']):
            raise ValueError('原任务归档或manifest身份漂移')
        source_manifest = load_dataset_manifest(Path(report['source_manifest']).parent/source.name)
        original_tasks = load_original_tasks(original, source_manifest)
        vehicle = VehicleConfig(**source_manifest['vehicle_model'])
    old, total = load_prefix(source, report['rebuilt'])
    with np.load(output/Path(report['output_data']).name, allow_pickle=False) as archive:
        new = {key: archive[key] for key in archive.files}
    if (set(old) != set(new) or len(new['states']) != report['rebuilt']
            or total != report['total_source_samples']
            or report['partial'] != (report['rebuilt']!=total)
            or [row['index'] for row in report['rows']] != list(range(report['rebuilt']))):
        raise ValueError('字段、样本分母或索引不一致')
    for key in old:
        if key not in ('bevs', 'bev_meta'):
            if old[key].dtype != new[key].dtype or not np.array_equal(old[key], new[key]):
                raise ValueError(f'监督或身份字段改变：{key}')
    old_meta, new_meta = json.loads(str(old['bev_meta'])), json.loads(str(new['bev_meta']))
    if new_meta.pop('camera_projection', None) != CAMERA_PROJECTION_CURRENT:
        raise ValueError('新输入未声明修复版本')
    old_meta.pop('camera_projection', None)
    if old_meta != new_meta or old['bevs'].shape != new['bevs'].shape or old['bevs'].dtype != new['bevs'].dtype:
        raise ValueError('BEV网格、通道或dtype改变')
    k = old_meta['channels'].index('target')
    other = [i for i in range(len(old_meta['channels'])) if i != k]
    for i in range(report['rebuilt']):
        if original_tasks is not None:
            _, lineage = resolve_rebuild_task(json.loads(str(old['task_meta'][i])), root_seed=source_manifest['seed'],
                                              vehicle=vehicle, original_tasks=original_tasks)
            if lineage != report['rows'][i].get('lineage'):
                raise ValueError(f'#{i}恢复谱系报告不一致')
        if not np.array_equal(old['bevs'][i, other], new['bevs'][i, other]):
            raise ValueError(f'#{i}非target通道改变')
        if not np.isin(new['bevs'][i, k], [0., 1.]).all():
            raise ValueError(f'#{i}target不是有限二值')
    if original_tasks is not None and report.get('recovery_labels_different') != sum(
            row['lineage']['recovery_label_differs'] for row in report['rows']):
        raise ValueError('恢复机动标签变化分母不一致')
    return {'status': 'passed', 'samples': report['rebuilt'], 'partial': report['partial'],
            'labels_task_order_and_other_channels_exact': True, 'production_qualified': False}
