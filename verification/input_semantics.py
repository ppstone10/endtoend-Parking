"""用训练谱系和实际NPZ输入校正部署声明；不改变权重或输入张量。"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch

from dataset import DatasetGenerator
from dataset.target_bev import rasterize_goal_target
from experiments import validation_matrix as matrix
from experiments.component_analysis import local_goal
from interfaces import BEVConfig, BEVTensor, GoalPose, VehicleState
from scripts.diagnose_closed_loop_failures import _sha256
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


def audit_and_correct(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('声明审计必须使用新目录')
    torch.set_num_threads(1)
    parent = root / 'runs/validation/arch-ab/dep-epoch0007.pt'
    lineage = root / 'runs/training/v40-lineage/net-v1/epoch0007.pt'
    training_report = root / 'runs/training/v40-lineage/net-v1/report.json'
    report = json.loads(training_report.read_text(encoding='utf-8'))
    payload = torch.load(parent, map_location='cpu', weights_only=False)
    origin = torch.load(lineage, map_location='cpu', weights_only=False)
    if payload['model_state'].keys() != origin['model_state'].keys() or not all(
            torch.equal(v, origin['model_state'][k]) for k, v in payload['model_state'].items()):
        raise ValueError('部署模型与声明训练epoch7的权重不一致，不改契约')
    train_path, val_path = Path(report['data']['train']), Path(report['data']['val'])
    train = DatasetGenerator.load(train_path)
    manifest = matrix.load_dataset_manifest(val_path)
    vehicle = VehicleConfig(**manifest['vehicle_model'])
    metadata = train['bev_meta']
    config = BEVConfig(metadata['resolution'], tuple(metadata['extent']))
    target_index = metadata['channels'].index('target')
    geometric_matches = 0
    for state, goal, bev in zip(train['states'], train['goals'], train['bevs'], strict=True):
        truth = rasterize_goal_target(GoalPose(*map(float, goal)), vehicle.length, vehicle.width,
                                      *map(float, state[:3]), config)
        geometric_matches += bool(np.array_equal(truth, bev[target_index]))
    probes = []
    for index in (0, 100, 2399, 2400, 2917):
        restored = matrix.reconstruct_dataset_task(train['task_meta'][index], root_seed=manifest['seed'], vehicle=vehicle)
        _, pipeline = matrix._build_pipeline(matrix.EXPERIMENT_SPECS['E5'], restored.task, vehicle)
        pipeline.set_target_goals([restored.goal])
        actual = pipeline.capture_bev(*map(float, train['states'][index, :3]))
        probes.append({'index': index, 'recovery': bool(train['task_meta'][index].get('recovery')),
                       'observed_target': 'geometry' if pipeline.target_override else 'image',
                       'target_exact': bool(np.array_equal(actual.data[target_index], train['bevs'][index, target_index]))})
    if geometric_matches == len(train['states']) or not all(p['target_exact'] and p['observed_target'] == 'image' for p in probes):
        raise ValueError('实际数据不支持image声明校正，不修改模型副本')
    inputs = {str(p.resolve()): _sha256(p) for p in (parent, lineage, training_report, train_path, val_path)}
    output.mkdir(parents=True, exist_ok=False)
    corrected = dict(payload)
    corrected['model_config'] = {**payload['model_config'], 'target_channel': 'image'}
    corrected['resumable'] = False
    corrected['input_contract_correction'] = {
        'parent': str(parent), 'parent_sha256': _sha256(parent), 'weights_changed': False,
        'old_target': payload['model_config']['target_channel'], 'new_target': 'image',
        'evidence': 'training lineage exact weights; stored BEVs passed directly to training; same image target probes'}
    destination = output / 'deployment-image-contract.pt'
    torch.save(corrected, destination)
    old_model, new_model = load_model_checkpoint(parent), load_model_checkpoint(destination)
    for index in (0, 100, 2400, 2917):
        state = train['states'][index]
        goal = GoalPose(*local_goal(train['goals'][index], state))
        bev = BEVTensor(train['bevs'][index], metadata['resolution'], metadata['extent'], metadata['channels'])
        old = old_model.model.predict(bev, goal, VehicleState.from_array(state))
        new = new_model.model.predict(bev, goal, VehicleState.from_array(state))
        if old.dt != new.dt or not np.array_equal(old.points, new.points):
            raise ValueError('仅声明修改改变了实际预测，拒绝候选副本')
    result = {'status': 'completed', 'inputs': inputs, 'checkpoint': str(destination),
              'checkpoint_sha256': _sha256(destination), 'lineage_weights_exact': True,
              'train_samples': len(train['states']), 'train_target_exact_geometry': geometric_matches,
              'train_target_different_from_geometry': len(train['states']) - geometric_matches,
              'image_target_probes': probes, 'prediction_exact_probes': [0, 100, 2400, 2917],
              'weights_or_runtime_inputs_changed': False,
              'limits': '几何相同的样本不代表几何来源；声明副本对齐现有训练/生产输入，不代表停车收益或默认部署。'}
    atomic_write_json(output / 'report.json', result)
    return result
