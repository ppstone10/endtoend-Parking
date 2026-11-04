"""冻结旧任务几何上的轮式教师校准；不导出训练数据或调用旧控制器。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from dataset.wheeled_audit import load_legacy_source
from experiments.closed_loop_evaluation import reconstruct_dataset_task
from experiments.wheeled_evaluation import build_wheeled_engine
from interfaces.wheeled import WheeledControlCmd, WheeledVehicleState
from planner.wheeled_expert import WheeledPlanningError, WheeledReedsSheppPlanner
from planner.wheeled_safety import WheeledTransitionChecker
from sim.vehicle_config import VehicleConfig
from sim.wheeled_config import load_wheeled_vehicle_config
from sim.wheeled_vehicle import KinematicBicycleModel


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_panel(metadata):
    chosen = {}
    for index, item in enumerate(metadata):
        if "recovery" in item:
            raise ValueError("轮式源任务校准不接受恢复样本")
        key = (item["scene_name"], item["difficulty"]["maneuver"])
        chosen.setdefault(key, index)
    return sorted(chosen.values())


def prepare_config(sources, vehicle_path, *, case_selection="scene_maneuver_first"):
    if case_selection not in {"scene_maneuver_first", "all_train_val"}:
        raise ValueError("未知校准源任务选择方式")
    wheel = load_wheeled_vehicle_config(vehicle_path)
    frozen, cases, all_ids, splits = {}, [], set(), set()
    for source in sources:
        source = Path(source).resolve()
        split = source.stem
        if split not in {"train", "val"} or split in splits:
            raise ValueError("校准只接受各一份train/val，不接受test或重复分片")
        splits.add(split)
        data = load_legacy_source(source)
        ids = [item["task_id"] for item in data["task_meta"]]
        if len(set(ids)) != len(ids) or all_ids.intersection(ids):
            raise ValueError("源任务ID重复或跨分片重叠")
        all_ids.update(ids)
        manifest_path = source.parent/"manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        vehicle = VehicleConfig(**manifest["vehicle_model"])
        if (wheel.length, wheel.width, wheel.collision_margin) != (vehicle.length, vehicle.width, vehicle.collision_margin):
            raise ValueError("新旧外廓/余量不同，不能沿用原任务几何身份")
        if not np.all(data["states"][:, 3:5] == 0):
            raise ValueError("源初态非静止，不能从旧omega推定轮式转角")
        panel = select_panel(data["task_meta"])  # 两种选择均拒绝恢复样本。
        indices = range(len(data["task_meta"])) if case_selection == "all_train_val" else panel
        for index in indices:
            item = data["task_meta"][index]
            cases.append(dict(source=str(source), split=split, index=index, task_id=item["task_id"],
                              scene=item["scene_name"], maneuver=item["difficulty"]["maneuver"]))
        for path in (source, manifest_path):
            frozen[str(path)] = sha256_file(path)
    if splits != {"train", "val"}:
        raise ValueError("正式面板必须同时包含train与val")
    for directory in ("interfaces", "sim", "planner", "controller", "runtime", "experiments",
                      "dataset", "training", "metrics", "sensor2bev", "scripts", "verification"):
        for path in sorted((ROOT/directory).glob("*.py")):
            frozen[str(path.resolve())] = sha256_file(path)
    vehicle_path = Path(vehicle_path).resolve()
    frozen[str(vehicle_path)] = sha256_file(vehicle_path)
    config = dict(schema="wheeled_teacher_calibration_v1", vehicle_path=str(vehicle_path),
                model_identity=wheel.identity, cases=cases, frozen_files=frozen, dt=.1,
                max_steps=600, planner_max_seconds=3., case_max_seconds=120.,
                controller=dict(horizon=8, population=8, iterations=1, seed=7),
                initial_steering_assumption=0., minimum_requested_ratio=.5,
                training_exported=False, production_qualified=False)
    # 历史prepared无字段即固定分层面板，保持其语义及逐字典重建兼容。
    if case_selection == "all_train_val":
        config["case_selection"] = case_selection
    return config


def precheck(config):
    if config.get("schema") != "wheeled_teacher_calibration_v1" or not config.get("cases"):
        raise ValueError("校准配置身份无效")
    for path, expected in config["frozen_files"].items():
        if sha256_file(path) != expected:
            raise ValueError(f"冻结输入或源码漂移：{path}")
    wheel = load_wheeled_vehicle_config(config["vehicle_path"])
    if wheel.identity != config["model_identity"]:
        raise ValueError("轮式车型身份漂移")
    # 独立重建面板与参数，不只依赖用户可改写的哈希字典。
    sources = list(dict.fromkeys(case["source"] for case in config["cases"]))
    rebuilt = prepare_config(sources, config["vehicle_path"],
                             case_selection=config.get("case_selection", "scene_maneuver_first"))
    if rebuilt != config:
        raise ValueError("校准配置语义或分母被修改")
    return wheel


def restore_case(config, case):
    source = Path(case["source"])
    data = load_legacy_source(source)
    metadata = data["task_meta"][case["index"]]
    if metadata["task_id"] != case["task_id"]:
        raise ValueError("校准任务ID漂移")
    manifest = json.loads((source.parent/"manifest.json").read_text(encoding="utf-8"))
    restored = reconstruct_dataset_task(metadata, root_seed=manifest["seed"],
                                         vehicle=VehicleConfig(**manifest["vehicle_model"]))
    original = restored.task.start
    np.testing.assert_allclose(data["states"][case["index"]], original.to_array(), rtol=0, atol=1e-10)
    start = WheeledVehicleState(original.x, original.y, original.yaw, 0, config["initial_steering_assumption"])
    return restored, start


def maneuver_ratio(states, direction, dt):
    speeds = np.asarray(states)[1:, 3]
    total = float(np.abs(speeds).sum()*dt)
    requested = speeds > 0 if direction == "forward" else speeds < 0
    return float(np.abs(speeds[requested]).sum()*dt/total) if total > 0 else 1.


def run_case(config, case, *, max_seconds=None):
    started = perf_counter()
    restored, start = restore_case(config, case)
    model = KinematicBicycleModel(load_wheeled_vehicle_config(config["vehicle_path"]))
    checker = WheeledTransitionChecker(model, restored.task.scene.env, dt=config["dt"])
    planner = WheeledReedsSheppPlanner(model, checker, dt=config["dt"], max_steps=config["max_steps"],
                                       max_seconds=config["planner_max_seconds"])
    record = dict(case=case, model_identity=config["model_identity"], status="completed",
                  geometry_source="simulator_ground_truth", initial_steering_assumption=0.,
                  dynamic_event_applied=False)
    try:
        plan = planner.plan(start, restored.goal)
    except WheeledPlanningError as exc:
        return {**record, "planned": False, "error": str(exc), "passed": False,
                "elapsed_seconds": perf_counter()-started}
    ratio = maneuver_ratio(plan.states, case["maneuver"], config["dt"])
    engine = build_wheeled_engine(model.config, dt=config["dt"], max_steps=config["max_steps"],
                                  max_seconds=max_seconds or config["case_max_seconds"],
                                  transition_checker=checker, tol_pos=restored.tol_pos,
                                  tol_yaw=restored.tol_yaw, **config["controller"])
    result = engine.run(start, restored.goal, lambda s, g: plan.trajectory)
    return {**record, "planned": True, "expert_states": plan.states.tolist(),
            "expert_controls": plan.controls.tolist(), "candidate_word": plan.candidate_word,
            "requested_ratio": ratio, "execution": result,
            "passed": ratio >= config["minimum_requested_ratio"] and result["success"],
            "elapsed_seconds": perf_counter()-started}


def inspect_case(config, case, record):
    if record.get("case") != case or record.get("model_identity") != config["model_identity"]:
        raise ValueError("逐任务身份不匹配")
    if record.get("status") != "completed" or record.get("geometry_source") != "simulator_ground_truth":
        raise ValueError("逐任务完成/几何身份无效")
    restored, start = restore_case(config, case)
    model = KinematicBicycleModel(load_wheeled_vehicle_config(config["vehicle_path"]))
    checker = WheeledTransitionChecker(model, restored.task.scene.env, dt=config["dt"])
    if not record.get("planned"):
        if not record.get("error") or record.get("passed"):
            raise ValueError("规划失败记录不完整")
        return False

    def replay(states, controls):
        states = np.asarray(states, dtype=float)
        controls = np.asarray(controls, dtype=float).reshape(-1, 2)
        if states.shape != (len(controls)+1, 5) or not np.isfinite(states).all():
            raise ValueError("校准状态/控制分母或数值错误")
        if len(controls) > config["max_steps"]:
            raise ValueError("校准超出原控制步预算")
        np.testing.assert_allclose(states[0], start.to_array(), rtol=0, atol=1e-10)
        if not checker.pose_free(start):
            raise ValueError("成功参考/执行初态不安全")
        for before, control, after in zip(states[:-1], controls, states[1:]):
            before = WheeledVehicleState.from_array(before)
            after_state = WheeledVehicleState.from_array(after)
            expected = model.step(before, WheeledControlCmd(*control), config["dt"])
            np.testing.assert_allclose(expected.to_array(), after, rtol=0, atol=1e-10)
            if not checker(before, after_state):
                raise ValueError("校准真实转移不满足GT规划余量")
        return states

    teacher = replay(record["expert_states"], record["expert_controls"])
    target = restored.goal
    teacher_yaw = abs(np.arctan2(np.sin(teacher[-1, 2]-target.yaw), np.cos(teacher[-1, 2]-target.yaw)))
    if np.linalg.norm(teacher[-1, :2]-[target.x, target.y]) > 1e-7 or teacher_yaw > 1e-7 or teacher[-1, 3] != 0:
        raise ValueError("专家终态未满足同一严格契约")
    execution = record["execution"]
    if execution.get("model_identity") != config["model_identity"]:
        raise ValueError("执行车型身份不匹配")
    actual = replay(execution["states"], execution["controls"])
    yaw = abs(np.arctan2(np.sin(actual[-1, 2]-target.yaw), np.cos(actual[-1, 2]-target.yaw)))
    reached = np.linalg.norm(actual[-1, :2]-[target.x, target.y]) < restored.tol_pos and yaw < restored.tol_yaw
    # 安全停止/资源到限等失败不能仅凭末态碰巧达标改写成成功。
    success = bool(reached and execution["failure"] is None)
    ratio = maneuver_ratio(teacher, case["maneuver"], config["dt"])
    passed = success and ratio >= config["minimum_requested_ratio"]
    if execution["success"] != success or record["passed"] != passed or abs(record["requested_ratio"]-ratio) > 1e-12:
        raise ValueError("校准汇总与实际状态/请求机动不一致")
    return passed
