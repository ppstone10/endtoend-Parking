"""闭环五组对照实验套件入口：复现协议冻结 + 按组合批量执行 + 汇总对照。

与 ``scripts/run_validation_experiment.py``（单组合）配套：

- ``--freeze-protocol``：在当前代码与数据集身份下计算可复原索引集，写入协议 JSON；
  后续所有实验共用该索引集，保证"同数据、同索引、同 seed、同重规划周期"。
- ``--experiment``：可重复指定组合（E1/E1r/E2/E2a/E3/E4/E5/E5h/E5s），按顺序执行，
  每个组合写 ``<output>/<name>/report.json``。
- ``--summary``：把本次执行的组合汇总为对照表 ``<output>/summary.json`` 与
  ``<output>/summary.md``（成功率/碰撞率/终点误差/滚动一致性指标并列）。

用法：

    # 1) 冻结协议（可复原索引集）
    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/run_validation_suite.py \
        --freeze-protocol --output runs/validation

    # 2) 冒烟：五组合各 2 条
    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/run_validation_suite.py \
        --experiment E1 --experiment E2 --experiment E3 --experiment E4 --experiment E5 \
        --smoke 2 --output runs/validation/smoke
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dataset import DatasetGenerator
from experiments.closed_loop_evaluation import (
    load_dataset_manifest,
    reconstruct_dataset_task,
)
from experiments.validation_matrix import (
    EXPERIMENT_SPECS,
    run_validation_experiment,
)
from sim import VehicleConfig
from training.reporting import atomic_write_json

# 供内部函数在解析阶段报错用；main 里会替换为真正的解析器。
_PARSER: argparse.ArgumentParser = argparse.ArgumentParser(add_help=False)

# 阶段 0.2 网络选型的候选权重。全部候选都在同一份 tracked_pivot_v7_3000
# 数据身份上训练（train 2400 条，v8/v9/v10/v11 系列从 v7 best.pt 初始化），
# 因此可在统一协议下直接对比；selection 判据与结论见 docs 计划文档 §5 阶段 0.2。
NETWORK_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("v7-flow-v2", "runs/training/v7-flow-v2/net-v1/deployment.pt"),
    ("v7-flow-v3", "runs/training/v7-flow-v3/net-v1/deployment.pt"),
    ("tuning-lr1e3", "runs/training/tuning-v7/net-v1-cumulative-lr1e3/deployment.pt"),
    ("tuning-lr3e4", "runs/training/tuning-v7/net-v1-cumulative-lr3e4/deployment.pt"),
    ("v8-safety-v2", "runs/training/v8-safety-v2/net-v1/deployment.pt"),
    ("v9-safety-v1", "runs/training/v9-safety-v1/net-v1/deployment.pt"),
    ("v9b-w025", "runs/training/v9b-collision-w025/net-v1/deployment.pt"),
    ("v9b-w100", "runs/training/v9b-collision-w100/net-v1/deployment.pt"),
    ("v9c-w0", "runs/training/v9c-collision-w0/net-v1/deployment.pt"),
    ("v9c-tf", "runs/training/v9c-tfcurriculum/net-v1/deployment.pt"),
    ("v9d-v7data", "runs/training/v9d-v7data/net-v1/deployment.pt"),
    ("v10-endpoint", "runs/training/v10-endpoint/net-v1/deployment.pt"),
    ("v10-endpoint-fix", "runs/training/v10-endpoint-fix/net-v1/deployment.pt"),
    ("v11-goal-exempt", "runs/training/v11-goal-exempt/net-v1/deployment.pt"),
)


def _reconstructible_indices(data_path: Path) -> tuple[list[int], list[dict]]:
    """返回当前代码几何下可复原的索引集与失败清单。"""
    data = DatasetGenerator.load(data_path)
    metadata = data.get("task_meta")
    if int(data.get("schema_version", -1)) != 2 or not isinstance(metadata, list):
        raise ValueError("协议冻结要求 schema v2 数据集与 task_meta")
    manifest = load_dataset_manifest(data_path)
    vehicle = VehicleConfig(**manifest["vehicle_model"])
    ok: list[int] = []
    failures: list[dict] = []
    for index, item in enumerate(metadata):
        try:
            reconstruct_dataset_task(item, root_seed=int(manifest["seed"]), vehicle=vehicle)
            ok.append(index)
        except ValueError as exc:
            failures.append(
                {
                    "dataset_index": index,
                    "scene_name": item.get("scene_name"),
                    "task_id": item.get("task_id"),
                    "reason": str(exc),
                }
            )
    return ok, failures


def _freeze_protocol(args: argparse.Namespace, output: Path) -> dict:
    data_path = Path(args.data).resolve()
    started = time.perf_counter()
    indices, failures = _reconstructible_indices(data_path)
    if not indices:
        raise ValueError("没有可复原样本；当前代码几何与数据集身份不一致")
    protocol = {
        "schema_version": 1,
        "kind": "closed_loop_validation_protocol",
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(data_path),
        "model": str(Path(args.model).resolve()) if args.model else None,
        "network_candidates": [
            {"label": label, "path": path}
            for label, path in NETWORK_CANDIDATES
            if Path(path).exists()
        ],
        "seed": args.seed,
        "selection": args.selection,
        "max_steps": args.max_steps,
        "replan_every": args.replan_every,
        "requested_samples": args.samples,
        "indices": indices,
        "n_indices": len(indices),
        "n_unreconstructible": len(failures),
        "reconstruct_failures": failures,
        "scene_histogram": _histogram(indices, data_path),
        "index_scene_names": _index_scene_names(indices, data_path),
        "elapsed_sec": time.perf_counter() - started,
    }
    atomic_write_json(output / "protocol.json", protocol)
    return protocol


def _histogram(indices: list[int], data_path: Path) -> dict:
    data = DatasetGenerator.load(data_path)
    metadata = data["task_meta"]
    counts: dict[str, int] = {}
    for index in indices:
        scene = str(metadata[index].get("scene_name"))
        counts[scene] = counts.get(scene, 0) + 1
    return dict(sorted(counts.items()))


def _index_scene_names(indices: list[int], data_path: Path) -> dict[str, str]:
    """索引 → 场景名映射，供按场景分层抽样使用。"""
    data = DatasetGenerator.load(data_path)
    metadata = data["task_meta"]
    return {str(index): str(metadata[index].get("scene_name")) for index in indices}


def _load_protocol(output: Path) -> dict:
    """在输出目录或其上级目录中查找已冻结的协议文件。

    冒烟/分批执行常把产物写到子目录（如 ``runs/validation/smoke``），
    此时复用上一级 ``runs/validation/protocol.json``，保证索引集一致。
    """
    for candidate in (output, *list(output.parents)[:2]):
        path = candidate / "protocol.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(
        f"在 {output} 及其上级目录找不到 protocol.json；先运行 --freeze-protocol"
    )


def _load_indices(path: str) -> list[int]:
    """从协议/索引 JSON 读取显式索引列表（``--indices-file``）。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("indices", "selected_indices", "evaluated_indices"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise ValueError(f"索引文件 {path} 缺少 indices/selected_indices 字段")
    if not isinstance(payload, list) or not all(isinstance(value, int) for value in payload):
        raise ValueError(f"索引文件 {path} 必须给出整数索引列表")
    if not payload:
        raise ValueError(f"索引文件 {path} 为空")
    return [int(value) for value in payload]


def _select_indices(protocol: dict, *, smoke: int, screen: int = 0) -> list[int]:
    """从协议索引集中取执行子集。

    smoke > 0 取前 N 条（冒烟）；screen > 0 按场景轮询取 N 条（跨候选权重筛选，
    保证每个场景都有代表）；两者都为 0 时用全部协议索引（正式跑批）。
    """
    indices = [int(v) for v in protocol["indices"]]
    if smoke > 0:
        return indices[:smoke]
    if screen > 0 and screen < len(indices):
        return _stratified_subset(indices, protocol, screen)
    return indices


def _stratified_subset(indices: list[int], protocol: dict, count: int) -> list[int]:
    """按场景轮询抽样，返回排序后的索引列表。"""
    scenes = _scene_of_index(protocol)
    groups: dict[str, list[int]] = {}
    for index in indices:
        groups.setdefault(scenes.get(index, "unknown"), []).append(index)
    selected: list[int] = []
    round_index = 0
    while len(selected) < count:
        added = False
        for scene in sorted(groups):
            values = groups[scene]
            if round_index < len(values):
                selected.append(values[round_index])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        round_index += 1
    return sorted(selected)


def _scene_of_index(protocol: dict) -> dict[int, str]:
    """读取协议里的索引→场景映射（冻结时写入）。"""
    mapping = protocol.get("index_scene_names")
    if not mapping:
        return {}
    return {int(key): str(value) for key, value in mapping.items()}


def _parse_model_overrides(values: list[str], parser: argparse.ArgumentParser) -> dict[str, str]:
    """解析 ``LABEL=PATH`` 形式的候选权重覆写；无覆写时返回空表。"""
    overrides: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            parser.error(f"--model-override 需要 LABEL=PATH 形式，收到 {item!r}")
        label, path = item.split("=", 1)
        label, path = label.strip(), path.strip()
        if not label or not path:
            parser.error(f"--model-override 的标签与路径都不能为空：{item!r}")
        if label in overrides:
            parser.error(f"--model-override 标签重复：{label}")
        if not Path(path).exists():
            parser.error(f"--model-override 的权重不存在：{path}")
        overrides[label] = path
    return overrides


def _screening_candidates(protocol: dict) -> list[dict]:
    """协议里记录的候选权重清单（阶段 0.2 网络选型）。"""
    return list(protocol.get("network_candidates") or [])


def _run_suite(args: argparse.Namespace, output: Path) -> dict[str, dict]:
    protocol = _load_protocol(output)
    indices = (
        _load_indices(args.indices_file)
        if args.indices_file
        else _select_indices(protocol, smoke=args.smoke, screen=args.screen_samples)
    )
    data_path = protocol["data"]
    overrides = _parse_model_overrides(args.model_override, _PARSER)
    if args.screen_candidates:
        overrides = {
            candidate["label"]: candidate["path"]
            for candidate in _screening_candidates(protocol)
        }
        if not overrides:
            raise ValueError("协议里没有 network_candidates；先 --freeze-protocol 且数据/代码就绪")
    runs: dict[str, dict] = {}
    for name in args.experiment:
        if name not in EXPERIMENT_SPECS:
            raise ValueError(f"未知实验组合：{name}；可选 {sorted(EXPERIMENT_SPECS)}")
        spec = EXPERIMENT_SPECS[name]
        needs_model = spec.trajectory_source == "network"
        default_model = args.model or protocol.get("model")
        if needs_model and not default_model and not overrides:
            raise ValueError(f"{name} 需要 --model")
        for label, model in (overrides or {None: default_model}).items():
            if needs_model and not model:
                raise ValueError(f"{name} 需要 --model")
            key = name if label is None else f"{name}@{label}"
            destination = output / name if label is None else output / f"{name}@{label}"
            destination.mkdir(parents=True, exist_ok=True)
            print(
                f"=== 开始 {key}（{spec.description}）索引 {len(indices)} 条"
                f"{'' if label is None else f'，权重 {label}'} ===",
                flush=True,
            )

            def progress(ordinal, total, bundle, _key=key):
                result = bundle.result
                cycles = bundle.cycles
                print(
                    f"[{_key} {ordinal}/{total}] #{result.meta['dataset_index']} "
                    f"{'成功' if result.success else f'失败({result.failure})'} "
                    f"位置 {result.final_pos_err:.2f}m "
                    f"航向 {np.degrees(result.final_yaw_err):.1f}° "
                    f"步数 {result.steps} 周期 {cycles.get('cycles', 0)} "
                    f"发散 {cycles.get('divergence_events', 0)} "
                    f"drift {cycles.get('drift_slope')}",
                    flush=True,
                )

            filter_kwargs = {}
            if args.filter_margin is not None:
                filter_kwargs["filter_required_margin"] = args.filter_margin
            if args.filter_max_offset is not None:
                filter_kwargs["filter_max_offset"] = args.filter_max_offset
            if args.filter_max_rounds is not None:
                filter_kwargs["filter_max_rounds"] = args.filter_max_rounds
            report = run_validation_experiment(
                spec,
                data_path=data_path,
                checkpoint_path=model if needs_model else None,
                output_path=destination / "report.json",
                samples=args.samples,
                selection=protocol.get("selection", "stratified"),
                max_steps=args.max_steps,
                replan_every=args.replan_every,
                control_seed=args.seed,
                indices=indices,
                ideal_steps_per_point=args.ideal_steps_per_point,
                ideal_point_spacing_m=args.ideal_point_spacing,
                progress=progress,
                **filter_kwargs,
            )
            if label is not None:
                report["model_label"] = label
            runs[key] = report
    return runs


def _summary_rows(reports: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for name, report in reports.items():
        overall = report["overall"]
        rows.append(
            {
                "experiment": name,
                "model_label": report.get("model_label"),
                "model_name": report["protocol"].get("model_name"),
                "description": report["experiment"]["description"],
                "bev_source": report["experiment"]["bev_source"],
                "trajectory_source": report["experiment"]["trajectory_source"],
                "executor": report["experiment"]["executor"],
                "safety_mode": report["experiment"]["safety_mode"],
                "episodes": overall["evaluated_samples"],
                "success_rate": overall["success_rate"],
                "collision_rate": overall["collision_rate"],
                "final_pos_err_mean": overall["final_pos_err_mean"],
                "final_yaw_err_deg_mean": float(
                    np.degrees(overall["final_yaw_err_mean"])
                ),
                "tracking_rms_mean": overall["tracking_rms_mean"],
                "path_length_mean": overall.get("path_length_mean"),
                "time_dist_ratio_mean": overall.get("time_dist_ratio_mean"),
                "cycles_drift_slope_mean": overall.get("cycles_drift_slope_mean"),
                "cycles_divergence_events_mean": overall.get(
                    "cycles_divergence_events_mean"
                ),
                "cycles_divergence_rate": overall.get("cycles_divergence_rate"),
                "cycles_goal_approach_monotonic_ratio_mean": overall.get(
                    "cycles_goal_approach_monotonic_ratio_mean"
                ),
                "cycles_heading_divergence_events_mean": overall.get(
                    "cycles_heading_divergence_events_mean"
                ),
                "cycles_heading_approach_monotonic_ratio_mean": overall.get(
                    "cycles_heading_approach_monotonic_ratio_mean"
                ),
                "cycles_heading_drift_slope_deg_mean": overall.get(
                    "cycles_heading_drift_slope_deg_mean"
                ),
                "cycles_yaw_err_end_deg_mean": overall.get(
                    "cycles_yaw_err_end_deg_mean"
                ),
                "cycles_pred_dist_to_traj_m_mean": overall.get(
                    "cycles_pred_dist_to_traj_m_mean"
                ),
                "ideal_executor_advance_failures": overall.get(
                    "ideal_executor_advance_failures"
                ),
                "failures": overall.get("failures"),
                "elapsed_sec": overall.get("elapsed_sec"),
            }
        )
    return rows


def _experiment_cell(row: dict) -> str:
    """对照表实验列：``实验@候选标签``（权威字段为 report["model_label"]）。"""
    if row.get("model_label") and "@" not in row["experiment"]:
        return f"{row['experiment']}@{row['model_label']}"
    return row["experiment"]


def _write_markdown(rows: list[dict], output: Path) -> None:
    header = (
        "| 实验 | 说明 | BEV | 轨迹源 | 执行 | 成功 | 碰撞 | 终点位置(m) | "
        "终点航向(°) | 滚动末航向(°) | 航向发散/样本 | 跟踪RMS(m) | drift | "
        "发散/样本 | 失败分类 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for row in rows:
        lines.append(
            "| {experiment} | {description} | {bev_source} | {trajectory_source} | "
            "{executor} | {success:.1%} | {collision:.1%} | {pos:.2f} | {yaw:.1f} | "
            "{roll_yaw} | {yaw_div} | {rms:.3f} | {drift} | {div} | {failures} |".format(
                experiment=_experiment_cell(row),
                description=row["description"],
                bev_source=row["bev_source"],
                trajectory_source=row["trajectory_source"],
                executor=row["executor"],
                success=row["success_rate"],
                collision=row["collision_rate"],
                pos=row["final_pos_err_mean"],
                yaw=row["final_yaw_err_deg_mean"],
                roll_yaw=(
                    "—"
                    if row.get("cycles_yaw_err_end_deg_mean") is None
                    else f"{row['cycles_yaw_err_end_deg_mean']:.1f}"
                ),
                yaw_div=(
                    "—"
                    if row.get("cycles_heading_divergence_events_mean") is None
                    else f"{row['cycles_heading_divergence_events_mean']:.2f}"
                ),
                rms=row["tracking_rms_mean"],
                drift=(
                    "—"
                    if row["cycles_drift_slope_mean"] is None
                    else f"{row['cycles_drift_slope_mean']:+.4f}"
                ),
                div=(
                    "—"
                    if row["cycles_divergence_events_mean"] is None
                    else f"{row['cycles_divergence_events_mean']:.2f}"
                ),
                failures=row["failures"],
            )
        )
    (output / "summary.md").write_text(
        "# 闭环验证对照表\n\n" + header + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    global _PARSER
    _PARSER = parser
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help=f"实验组合，可重复；可选 {sorted(EXPERIMENT_SPECS)}",
    )
    parser.add_argument("--data", default="data/task_dataset/tracked_pivot_v7_3000/val.npz")
    parser.add_argument("--model", default="runs/training/v7-flow-v3/net-v1/deployment.pt")
    parser.add_argument(
        "--model-override",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="同一实验组合对多个候选权重各跑一次，产物写入 <output>/<实验>@<LABEL>",
    )
    parser.add_argument(
        "--screen-candidates",
        action="store_true",
        help="按协议里的 network_candidates 逐个跑（阶段 0.2 网络选型）",
    )
    parser.add_argument("--samples", type=int, default=34)
    parser.add_argument("--selection", choices=["stratified", "head"], default="stratified")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--replan-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", type=int, default=0, help=">0 时只用协议前 N 条索引冒烟")
    parser.add_argument(
        "--screen-samples",
        type=int,
        default=0,
        help=">0 时按场景轮询取 N 条索引（跨候选权重筛选用）",
    )
    parser.add_argument(
        "--indices-file",
        default="",
        help="显式索引列表 JSON（含 indices 字段）；用于历史基线口径对齐",
    )
    parser.add_argument("--ideal-steps-per-point", type=float, default=None)
    parser.add_argument(
        "--ideal-point-spacing",
        type=float,
        default=None,
        help="理想执行的标称点距（米）；默认 0.5，用于推进速度敏感性对照",
    )
    parser.add_argument(
        "--filter-margin",
        type=float,
        default=None,
        help="轨迹级几何过滤要求的净空（米）；不指定用 experiments.validation_matrix 的默认值",
    )
    parser.add_argument("--filter-max-offset", type=float, default=None)
    parser.add_argument("--filter-max-rounds", type=int, default=None)
    parser.add_argument("--output", default="runs/validation")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.freeze_protocol:
        protocol = _freeze_protocol(args, output)
        print(
            f"协议已冻结：{output / 'protocol.json'}；可复原 {protocol['n_indices']} 条，"
            f"不可复原 {protocol['n_unreconstructible']} 条；场景分布 {protocol['scene_histogram']}"
        )
        print(
            f"候选权重 {len(protocol['network_candidates'])} 个："
            f"{[item['label'] for item in protocol['network_candidates']]}"
        )
        if not args.experiment:
            return
    if not args.experiment:
        parser.error("需要 --experiment 或 --freeze-protocol")
    reports = _run_suite(args, output)
    rows = _summary_rows(reports)
    atomic_write_json(output / "summary.json", {"schema_version": 1, "rows": rows})
    _write_markdown(rows, output)
    print(f"汇总：{output / 'summary.json'}；对照表：{output / 'summary.md'}")
    for row in rows:
        print(
            f"  {_experiment_cell(row):22s} 成功 {row['success_rate']:.1%} "
            f"碰撞 {row['collision_rate']:.1%} 终点 {row['final_pos_err_mean']:.2f}m "
            f"航向 {row['final_yaw_err_deg_mean']:.1f}° 失败 {row['failures']}"
        )


if __name__ == "__main__":
    main()
