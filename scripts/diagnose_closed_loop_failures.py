"""按已完成 E5 基线回放失败，记录真实初态和 MPC 实际控制窗口。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import validation_matrix
from interfaces import VehicleState
from runtime.engine import ClosedLoopEngine
from training.reporting import atomic_write_json


def select_failure_indices(report: dict, scenes: set[str] | None = None) -> list[int]:
    return [
        int(row["meta"]["dataset_index"])
        for row in report["episodes"]
        if not row["success"] and (not scenes or row["meta"]["scene_name"] in scenes)
    ]


def first_blocked_index(checker, start: np.ndarray, points: np.ndarray) -> int | None:
    previous = np.asarray(start, dtype=np.float64)
    for index, point in enumerate(points):
        if not checker.swept_segment_free(previous[:3], point[:3]):
            return index
        previous = point
    return None


def analyze_record(start, record, control_windows, contact_checker, planning_checker) -> dict:
    """窗口不安全是候选解释；整条参考的远端风险单独记录。"""
    if len(control_windows) != len(record.states):
        raise ValueError("实际控制窗口与推进后状态必须一一对应")
    initial = start.to_array()
    before = initial
    first_unsafe = None
    collision = None
    last_planning_safe = None
    for step, (state, window, plan, hit) in enumerate(zip(
        record.states, control_windows, record.plans, record.collisions, strict=True
    )):
        after = state.to_array()
        if planning_checker.pose_free(*before[:3]):
            last_planning_safe = {"step": step, "state": before.tolist()}
        blocked = first_blocked_index(contact_checker, before, window)
        if blocked is not None and first_unsafe is None:
            first_unsafe = {"step": step, "blocked_index": blocked, "before_state": before.tolist()}
        if hit:
            collision = {
                "step": step,
                "before_state": before.tolist(),
                "after_state": after.tolist(),
                "before_pose_free": bool(contact_checker.pose_free(*before[:3])),
                "after_pose_free": bool(contact_checker.pose_free(*after[:3])),
                "actual_transition_safe": bool(contact_checker.swept_segment_free(before[:3], after[:3])),
                "control_window_blocked_index": blocked,
                # 从碰撞前态检查仍在生效的参考；不冒充发出时或实际执行前缀检查。
                "active_full_plan_blocked_from_before": first_blocked_index(contact_checker, before, plan),
                "nearest_plan_point_distance_m": float(np.min(np.linalg.norm(plan[:, :2] - after[:2], axis=1))),
            }
            break
        before = after
    initial_free = bool(contact_checker.pose_free(*initial[:3]))
    if not initial_free:
        mechanism = "initial_pose_collision"
    elif collision is None:
        mechanism = "no_collision"
    elif collision["control_window_blocked_index"] is not None:
        mechanism = "unsafe_control_window_collision"
    else:
        mechanism = "safe_control_window_execution_collision"
    return {
        "initial_state": initial.tolist(),
        "initial_pose_free": initial_free,
        "initial_planning_pose_free": bool(planning_checker.pose_free(*initial[:3])),
        "steps": len(record.states),
        "first_unsafe_control_window": first_unsafe,
        "last_planning_safe_before_collision_or_end": last_planning_safe,
        "collision": collision,
        "candidate_mechanism": mechanism,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_replay(destination, engine, start, goal, result, windows, plans, bevs, render):
    record = result.record
    arrays = {
        "initial_state": start.to_array(),
        "goal": np.asarray([goal.x, goal.y, goal.yaw]),
        "states_after": np.asarray([state.to_array() for state in record.states]).reshape(-1, 5),
        "commands": np.asarray([[cmd.v, cmd.omega] for cmd in record.cmds]).reshape(-1, 2),
        "collisions": np.asarray(record.collisions, dtype=bool),
        "control_windows": np.asarray(windows),
        "plan_issue_steps": np.asarray([row[0] for row in plans], dtype=np.int64),
        "plan_issue_states": np.asarray([row[1] for row in plans]),
        "control_dt": np.asarray(engine.mpc.dt),
    }
    for ordinal, (_, _, points) in enumerate(plans):
        arrays[f"plan_{ordinal:04d}"] = points
    if bevs:
        arrays["initial_bev"] = bevs[0]
        arrays["last_issue_bev"] = bevs[-1]
    np.savez_compressed(destination.with_suffix(".npz"), **arrays)
    if render and record.states:
        import matplotlib.pyplot as plt
        from viz.traj_render import render_episode
        figure = render_episode(
            record, engine.env, goal, engine.vehicle_length, engine.vehicle_width,
            title=f"#{engine.meta['dataset_index']} {result.failure or 'success'}",
            out_path=str(destination),
        )
        plt.close(figure)


def run_diagnosis(baseline_path: Path, output: Path, *, scenes=None, render=False) -> dict:
    baseline_path = baseline_path.resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("status") != "completed" or baseline["experiment"]["name"] != "E5":
        raise ValueError("诊断要求已完成的无干预 E5 基线报告")
    if baseline["experiment"]["safety_mode"] != "none":
        raise ValueError("诊断不能把专家干预回合当作自主基线")
    protocol = baseline["protocol"]
    indices = select_failure_indices(baseline, scenes)
    if not indices:
        raise ValueError("筛选后没有失败任务")
    output = output.resolve()
    if (output / "report.json").exists():
        raise FileExistsError("诊断报告已存在，请使用新的输出目录")
    output.mkdir(parents=True, exist_ok=True)
    data_path, model_path = Path(protocol["data"]), Path(protocol["checkpoint"])
    identity = {
        "baseline": str(baseline_path), "data": str(data_path), "model": str(model_path),
        "data_sha256": _sha256(data_path), "model_sha256": _sha256(model_path),
        "indices": indices, "control_seed": protocol["control_seed"],
        "replan_every": protocol["replan_every"], "max_steps": protocol["max_steps"],
    }
    atomic_write_json(output / "identity.json", identity)
    baseline_episodes = {row["meta"]["dataset_index"]: row for row in baseline["episodes"]}
    diagnostics = []

    class AuditedEngine(ClosedLoopEngine):
        def run(self, start, goal):
            windows, issued_plans, bevs = [], [], []
            compute = self.mpc.compute
            predict = self.source.next_trajectory
            pipeline = self.source.sensor_pipeline
            capture = pipeline.capture_bev

            def audited_compute(trajectory, state):
                command = compute(trajectory, state)
                # 只读 compute 刚生成的缓存和索引，不再次求解或推进索引。
                window = self.mpc._reference_points(self.mpc._aligned_cache, self.mpc._progress_idx)
                windows.append(np.array(window, copy=True))
                return command

            def audited_predict(state):
                trajectory, elapsed = predict(state)
                issued_plans.append((len(windows), state.to_array(), np.array(trajectory.points, copy=True)))
                return trajectory, elapsed

            def audited_capture(*args, **kwargs):
                bev = capture(*args, **kwargs)
                if not bevs:
                    bevs.append(np.array(bev.data, copy=True))
                if len(bevs) == 1:
                    bevs.append(np.array(bev.data, copy=True))
                else:
                    bevs[-1] = np.array(bev.data, copy=True)
                return bev

            self.mpc.compute = audited_compute
            self.source.next_trajectory = audited_predict
            pipeline.capture_bev = audited_capture
            try:
                result = super().run(start, goal)
            finally:
                self.mpc.compute = compute
                self.source.next_trajectory = predict
                pipeline.capture_bev = capture
            index = int(self.meta["dataset_index"])
            row = analyze_record(
                start, result.record, windows, self.collision_checker, self._planning_checker
            )
            row.update({"dataset_index": index, "scene_name": self.meta["scene_name"], "result": result.to_dict()})
            expected = baseline_episodes[index]
            row["baseline_match"] = all(
                expected[key] == getattr(result, key) for key in ("success", "collision", "failure", "steps")
            ) and bool(np.isclose(expected["final_pos_err"], result.final_pos_err, atol=1e-9, rtol=0))
            destination = output / f"episode-{index:04d}"
            _save_replay(destination, self, start, goal, result, windows, issued_plans, bevs, render)
            atomic_write_json(destination.with_suffix(".json"), row)
            diagnostics.append(row)
            return result

    # 复用评测组件，规划余量检查器来自同一个任务规划器；挂接仅限本次调用。
    original_engine = validation_matrix.ClosedLoopEngine
    original_build_source = validation_matrix._build_source
    planning_checkers = []

    def build_source(*args, **kwargs):
        planning_checkers.append(args[1]._collision_checker)
        return original_build_source(*args, **kwargs)

    class TaskAuditedEngine(AuditedEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._planning_checker = planning_checkers.pop()

    validation_matrix.ClosedLoopEngine = TaskAuditedEngine
    validation_matrix._build_source = build_source
    try:
        report = validation_matrix.run_validation_experiment(
            "E5", data_path=data_path, checkpoint_path=model_path,
            indices=indices, samples=0, control_seed=protocol["control_seed"],
            replan_every=protocol["replan_every"], max_steps=protocol["max_steps"],
            output_path=output / "replayed-baseline.json",
            progress=lambda ordinal, total, bundle: print(
                f"[{ordinal}/{total}] #{bundle.result.meta['dataset_index']} "
                f"{bundle.result.failure} match={diagnostics[-1]['baseline_match']} "
                f"{diagnostics[-1]['candidate_mechanism']}", flush=True,
            ),
        )
    finally:
        validation_matrix.ClosedLoopEngine = original_engine
        validation_matrix._build_source = original_build_source
    payload = {
        "schema_version": 1, "identity": identity, "overall": report["overall"],
        "baseline_mismatches": [row["dataset_index"] for row in diagnostics if not row["baseline_match"]],
        "candidate_mechanisms": dict(Counter(row["candidate_mechanism"] for row in diagnostics)),
        "episodes": diagnostics,
    }
    atomic_write_json(output / "report.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenes", default="", help="逗号分隔；默认回放全部失败")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    result = run_diagnosis(args.baseline_report, args.output, scenes=set(filter(None, args.scenes.split(","))), render=args.render)
    print(json.dumps({"mechanisms": result["candidate_mechanisms"], "baseline_mismatches": result["baseline_mismatches"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
