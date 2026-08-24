"""NPZ 数据集统计与 BEV/专家轨迹叠加抽检。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import textwrap
from typing import Any

import numpy as np

from sim.vehicle_config import MINING_DRILL_RIG, VehicleConfig

from .feasibility import summarize_trajectory_feasibility

from .maneuver import (
    audit_maneuver_consistency,
    summarize_maneuver_consistency,
    trajectory_segment_directions,
)


def summarize_dataset(
    data: dict[str, Any], *, vehicle_config: VehicleConfig = MINING_DRILL_RIG
) -> dict[str, Any]:
    """统计样本数、轨迹长度、倒车距离比例与任务分层数量。"""
    trajs = np.asarray(data["trajs"])
    masks = np.asarray(data["masks"])
    if trajs.shape[:2] != masks.shape:
        raise ValueError("trajs 与 masks 形状不一致")

    lengths: list[float] = []
    reverse_distance = 0.0
    total_distance = 0.0
    for trajectory, mask in zip(trajs, masks):
        count = int(np.count_nonzero(mask))
        points = trajectory[:count]
        if len(points) < 2:
            lengths.append(0.0)
            continue
        delta = np.diff(points[:, :2], axis=0)
        segment_length = np.linalg.norm(delta, axis=1)
        heading = points[:-1, 2]
        signed_progress = delta[:, 0] * np.cos(heading) + delta[:, 1] * np.sin(heading)
        length = float(segment_length.sum())
        lengths.append(length)
        total_distance += length
        reverse_distance += float(segment_length[signed_progress < 0.0].sum())

    metadata = data.get("task_meta")
    if metadata is not None and (
        not isinstance(metadata, list) or len(metadata) != len(trajs)
    ):
        raise ValueError("task_meta 数量必须与样本数量一致")
    countable_metadata = metadata or []
    return {
        "sample_count": int(trajs.shape[0]),
        "trajectory_length_m": _distribution(lengths),
        "reverse_distance_ratio": (
            reverse_distance / total_distance if total_distance > 0.0 else 0.0
        ),
        "scene_counts": _metadata_counts(countable_metadata, "scene_name"),
        "task_type_counts": _metadata_counts(countable_metadata, "task_type"),
        "noise_level_counts": _noise_counts(countable_metadata),
        "maneuver_consistency": summarize_maneuver_consistency(
            trajs, masks, metadata
        ),
        "trajectory_feasibility": summarize_trajectory_feasibility(
            trajs,
            masks,
            dt=data.get("dt", np.asarray([1.0], dtype=np.float64)),
            metadata=metadata,
            vehicle_config=vehicle_config,
            states=data.get("states"),
            goals=data.get("goals"),
        ),
    }


def select_representative_indices(data: dict[str, Any], count: int) -> list[int]:
    """优先覆盖不同任务类型，再从剩余样本中均匀补齐索引。"""
    if count < 0:
        raise ValueError("样本数量不能为负")
    total = int(np.asarray(data["trajs"]).shape[0])
    limit = min(count, total)
    if limit == 0:
        return []

    metadata = data.get("task_meta")
    if metadata is None:
        return _evenly_spaced_indices(total, limit)
    if not isinstance(metadata, list) or len(metadata) != total:
        raise ValueError("task_meta 数量必须与样本数量一致")

    by_task_type: dict[str, list[int]] = {}
    for index, item in enumerate(metadata):
        if not isinstance(item, dict):
            raise ValueError("task_meta 每一项必须是字典")
        task_type = str(item.get("task_type", "unknown"))
        by_task_type.setdefault(task_type, []).append(index)

    selected: list[int] = []
    for task_type in sorted(by_task_type):
        candidates = by_task_type[task_type]
        selected.append(candidates[len(candidates) // 2])
        if len(selected) == limit:
            return selected

    selected_set = set(selected)
    remaining = [index for index in range(total) if index not in selected_set]
    needed = limit - len(selected)
    positions = _evenly_spaced_indices(len(remaining), needed)
    selected.extend(remaining[position] for position in positions)
    return selected


def render_sample_overlay(data: dict[str, Any], index: int, path: str | Path) -> None:
    """保存包含钻机中心轨迹、中间外廓与三类门禁的专家验收图。"""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    bev_meta = data.get("bev_meta")
    if not isinstance(bev_meta, dict):
        raise ValueError("绘制叠加图需要 schema v2 bev_meta")
    bevs = np.asarray(data["bevs"])
    if not 0 <= index < len(bevs):
        raise IndexError("样本索引越界")
    channels = list(bev_meta["channels"])
    front, back, left, right = (float(value) for value in bev_meta["extent"])
    occupancy = bevs[index, channels.index("occupancy")]

    goals = data.get("goals")
    if goals is None or len(goals) != len(bevs):
        raise ValueError("绘制验收图需要与样本对齐的 goals")
    metadata = data.get("task_meta")
    if metadata is not None and (
        not isinstance(metadata, list) or len(metadata) != len(bevs)
    ):
        raise ValueError("task_meta 数量必须与样本数量一致")

    count = int(np.count_nonzero(data["masks"][index]))
    if count == 0:
        raise ValueError("无法绘制空专家轨迹")
    trajectory = np.asarray(data["trajs"][index, :count], dtype=np.float64)
    state = np.asarray(data["states"][index], dtype=np.float64)
    goal = np.asarray(goals[index], dtype=np.float64)
    local = _to_local(trajectory, float(state[0]), float(state[1]), float(state[2]))
    goal_local = _to_local(
        goal[np.newaxis, :], float(state[0]), float(state[1]), float(state[2])
    )[0]
    directions, segment_lengths = trajectory_segment_directions(trajectory)
    path_length = float(segment_lengths.sum())
    reverse_distance = float(segment_lengths[directions < 0].sum())
    reverse_ratio = reverse_distance / path_length if path_length > 0.0 else 0.0
    final_position_error = float(np.linalg.norm(trajectory[-1, :2] - goal[:2]))
    final_yaw_error = abs(_wrap_angle(float(trajectory[-1, 2] - goal[2])))
    item_meta = metadata[index] if metadata is not None else {}
    vehicle_config, model_confirmed = _vehicle_config_from_metadata(item_meta)
    feasibility_meta = item_meta.get("dataset", {}).get("feasibility_audit")
    if not isinstance(feasibility_meta, dict):
        feasibility_meta = None
    requested_maneuver = item_meta.get("difficulty", {}).get("maneuver")
    maneuver_audit = None
    if requested_maneuver is not None:
        maneuver_audit = audit_maneuver_consistency(
            trajectory, requested_maneuver
        )

    figure, (axis, info_axis) = plt.subplots(
        1,
        2,
        figsize=(12.6, 8.2),
        gridspec_kw={"width_ratios": [3.3, 1.45]},
        constrained_layout=True,
    )
    axis.imshow(
        occupancy,
        cmap="Greys",
        origin="upper",
        extent=(-right, left, -back, front),
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        zorder=0,
    )
    if "target" in channels:
        target = np.ma.masked_where(
            bevs[index, channels.index("target")] <= 0.0,
            bevs[index, channels.index("target")],
        )
        axis.imshow(
            target,
            cmap="Oranges",
            alpha=0.72,
            origin="upper",
            extent=(-right, left, -back, front),
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            zorder=1,
        )

    plot_points = local[:, [1, 0]]
    if len(plot_points) > 1:
        segments = np.stack((plot_points[:-1], plot_points[1:]), axis=1)
        for direction, color in ((1, "#1976D2"), (-1, "#8E44AD")):
            selected_segments = segments[directions == direction]
            if len(selected_segments):
                axis.add_collection(
                    LineCollection(
                        selected_segments,
                        colors=color,
                        linewidths=2.6,
                        zorder=4,
                    )
                )
        _draw_direction_arrows(axis, plot_points, segment_lengths, directions)
    axis.scatter(
        plot_points[:, 0],
        plot_points[:, 1],
        s=12,
        facecolor="#FFFFFF",
        edgecolor="#34495E",
        linewidth=0.55,
        alpha=0.82,
        zorder=5,
    )

    yaw_delta = np.abs(_wrap_angles(np.diff(trajectory[:, 2])))
    pivot_segments = (segment_lengths <= 1e-6) & (yaw_delta > 1e-6)
    pivot_point_indices = np.unique(
        np.concatenate(
            (
                np.flatnonzero(pivot_segments),
                np.flatnonzero(pivot_segments) + 1,
            )
        )
    )
    if len(pivot_point_indices):
        axis.scatter(
            plot_points[pivot_point_indices, 0],
            plot_points[pivot_point_indices, 1],
            s=34,
            facecolor="#F39C12",
            edgecolor="#7D3C98",
            linewidth=1.0,
            marker="o",
            zorder=7,
        )

    moving_segment_indices = np.flatnonzero(segment_lengths > 1e-6)
    switch_indices = (
        moving_segment_indices[1:][
            directions[moving_segment_indices[1:]]
            != directions[moving_segment_indices[:-1]]
        ]
        if len(moving_segment_indices) > 1
        else np.array([], dtype=int)
    )
    if len(switch_indices):
        axis.scatter(
            plot_points[switch_indices, 0],
            plot_points[switch_indices, 1],
            s=68,
            facecolor="#F7DC6F",
            edgecolor="#C0392B",
            linewidth=1.5,
            marker="D",
            zorder=7,
        )

    for pose_index in _intermediate_pose_indices(
        trajectory, segment_lengths, switch_indices, pivot_segments
    ):
        _draw_vehicle_pose(
            axis,
            local[pose_index],
            vehicle_config=vehicle_config,
            facecolor="#5DADE2",
            edgecolor="#2471A3",
            label=None,
            alpha=0.12,
            linewidth=0.9,
            draw_heading=True,
        )

    _draw_vehicle_pose(
        axis,
        np.array([0.0, 0.0, 0.0]),
        vehicle_config=vehicle_config,
        facecolor="#2ECC71",
        edgecolor="#087F23",
        label="START",
    )
    _draw_vehicle_pose(
        axis,
        goal_local,
        vehicle_config=vehicle_config,
        facecolor="#FFB347",
        edgecolor="#D35400",
        label="GOAL",
    )
    axis.scatter(
        [plot_points[-1, 0]],
        [plot_points[-1, 1]],
        color="#17202A",
        marker="x",
        s=70,
        linewidth=2.0,
        zorder=8,
    )

    scene_name = str(item_meta.get("scene_name", "unknown"))
    task_type = str(item_meta.get("task_type", "unknown"))
    axis.set(
        xlabel="Left / m",
        ylabel="Forward / m",
        title=f"Expert parking evidence\n{scene_name} · {task_type} · sample {index}",
    )
    _set_evidence_limits(
        axis,
        plot_points,
        goal_local,
        vehicle_config=vehicle_config,
        x_bounds=(-right, left),
        y_bounds=(-back, front),
    )
    axis.set_aspect("equal")
    axis.grid(color="#7F8C8D", alpha=0.25, linewidth=0.7)
    legend_handles = [
        Patch(facecolor="#2ECC71", edgecolor="#087F23", label="Start drill-rig pose"),
        Patch(facecolor="#FFB347", edgecolor="#D35400", label="Goal drill-rig pose"),
        Line2D([0], [0], color="#1976D2", linewidth=2.6, label="Forward path"),
        Line2D([0], [0], color="#8E44AD", linewidth=2.6, label="Reverse path"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#FFFFFF",
            markeredgecolor="#34495E",
            label="Center trajectory point",
        ),
        Patch(
            facecolor="#5DADE2",
            edgecolor="#2471A3",
            alpha=0.18,
            label="Intermediate drill-rig pose",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#F39C12",
            markeredgecolor="#7D3C98",
            label="In-place rotation",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="#F7DC6F",
            markeredgecolor="#C0392B",
            label="Direction change",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#17202A",
            linestyle="none",
            label="Trajectory end",
        ),
    ]
    _render_info_panel(
        info_axis,
        item_meta,
        point_count=count,
        path_length=path_length,
        reverse_ratio=reverse_ratio,
        switch_count=len(switch_indices),
        requested_distance_ratio=(
            None
            if maneuver_audit is None
            else maneuver_audit.requested_distance_ratio
        ),
        maneuver_consistent=(
            None if maneuver_audit is None else maneuver_audit.consistent
        ),
        feasibility_audit=feasibility_meta,
        vehicle_config=vehicle_config,
        model_confirmed=model_confirmed,
        final_position_error=final_position_error,
        final_yaw_error=final_yaw_error,
    )
    info_axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.115),
        ncol=2,
        fontsize=7.6,
        framealpha=0.95,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _evenly_spaced_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    indices = np.linspace(0, total - 1, min(count, total), dtype=int).tolist()
    return list(dict.fromkeys(indices))


def _vehicle_config_from_metadata(
    metadata: dict[str, Any],
) -> tuple[VehicleConfig, bool]:
    payload = metadata.get("dataset", {}).get("vehicle_model")
    if isinstance(payload, dict):
        try:
            return VehicleConfig(**payload), True
        except (TypeError, ValueError):
            pass
    return MINING_DRILL_RIG, False


def _intermediate_pose_indices(
    points: np.ndarray,
    segment_lengths: np.ndarray,
    switch_indices: np.ndarray,
    pivot_segments: np.ndarray,
    *,
    spacing_m: float = 3.0,
    maximum: int = 14,
) -> list[int]:
    """选择等距、换向和原地旋转证据位姿，避免把整图涂满外廓。"""
    if len(points) < 3:
        return []
    selected = {int(index) for index in np.asarray(switch_indices, dtype=int)}
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] > spacing_m:
        for target in np.arange(spacing_m, cumulative[-1], spacing_m):
            selected.add(int(np.searchsorted(cumulative, target, side="left")))

    pivot_indices = np.flatnonzero(pivot_segments)
    if len(pivot_indices):
        run_start = 0
        for position in range(1, len(pivot_indices) + 1):
            run_ended = (
                position == len(pivot_indices)
                or pivot_indices[position] != pivot_indices[position - 1] + 1
            )
            if not run_ended:
                continue
            run = pivot_indices[run_start:position]
            selected.update(
                {
                    int(run[0]),
                    int((run[0] + run[-1] + 1) // 2),
                    int(run[-1] + 1),
                }
            )
            run_start = position

    selected.discard(0)
    selected.discard(len(points) - 1)
    ordered = sorted(index for index in selected if 0 < index < len(points) - 1)
    if len(ordered) <= maximum:
        return ordered
    positions = _evenly_spaced_indices(len(ordered), maximum)
    return [ordered[position] for position in positions]


def _vehicle_polygon(
    pose: np.ndarray, vehicle_config: VehicleConfig = MINING_DRILL_RIG
) -> np.ndarray:
    """返回绘图坐标（left, forward）中的履带钻机四角。"""
    half_length = vehicle_config.length / 2.0
    half_width = vehicle_config.width / 2.0
    corners = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )
    yaw = float(pose[2])
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
        dtype=np.float64,
    )
    forward_left = corners @ rotation.T + np.asarray(pose[:2], dtype=np.float64)
    return forward_left[:, [1, 0]]


def _draw_vehicle_pose(
    axis,
    pose,
    *,
    vehicle_config: VehicleConfig,
    facecolor,
    edgecolor,
    label,
    alpha: float = 0.68,
    linewidth: float = 2.2,
    draw_heading: bool = True,
) -> None:
    from matplotlib.patches import Polygon

    center_left, center_forward = float(pose[1]), float(pose[0])
    axis.add_patch(
        Polygon(
            _vehicle_polygon(np.asarray(pose), vehicle_config),
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            zorder=6,
        )
    )
    if draw_heading:
        arrow_length = vehicle_config.length * 0.48
        axis.annotate(
            "",
            xy=(
                center_left + arrow_length * np.sin(float(pose[2])),
                center_forward + arrow_length * np.cos(float(pose[2])),
            ),
            xytext=(center_left, center_forward),
            arrowprops={"arrowstyle": "-|>", "color": edgecolor, "lw": linewidth},
            zorder=8,
        )
    if label:
        axis.text(
            center_left,
            center_forward,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="#FFFFFF",
            zorder=9,
        )


def _set_evidence_limits(
    axis,
    plot_points,
    goal_pose,
    *,
    vehicle_config: VehicleConfig,
    x_bounds,
    y_bounds,
) -> None:
    footprint_points = np.vstack(
        (
            _vehicle_polygon(np.array([0.0, 0.0, 0.0]), vehicle_config),
            _vehicle_polygon(np.asarray(goal_pose), vehicle_config),
        )
    )
    evidence = np.vstack((plot_points, footprint_points))
    minimum = evidence.min(axis=0)
    maximum = evidence.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    margin = np.maximum(3.0, span * 0.14)
    x_min = max(float(x_bounds[0]), float(minimum[0] - margin[0]))
    x_max = min(float(x_bounds[1]), float(maximum[0] + margin[0]))
    y_min = max(float(y_bounds[0]), float(minimum[1] - margin[1]))
    y_max = min(float(y_bounds[1]), float(maximum[1] + margin[1]))
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)


def _draw_direction_arrows(axis, plot_points, segment_lengths, directions) -> None:
    usable = np.flatnonzero(segment_lengths > 1e-6)
    if not len(usable):
        return
    positions = _evenly_spaced_indices(len(usable), min(7, len(usable)))
    for position in positions:
        index = int(usable[position])
        vector = plot_points[index + 1] - plot_points[index]
        vector /= np.linalg.norm(vector)
        midpoint = (plot_points[index + 1] + plot_points[index]) / 2.0
        arrow_length = 0.85
        color = "#0D47A1" if directions[index] > 0 else "#5B2C6F"
        axis.annotate(
            "",
            xy=midpoint + vector * arrow_length / 2.0,
            xytext=midpoint - vector * arrow_length / 2.0,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5},
            zorder=5,
        )


def _render_info_panel(
    axis,
    metadata: dict[str, Any],
    *,
    point_count: int,
    path_length: float,
    reverse_ratio: float,
    switch_count: int,
    requested_distance_ratio: float | None,
    maneuver_consistent: bool | None,
    feasibility_audit: dict[str, Any] | None,
    vehicle_config: VehicleConfig,
    model_confirmed: bool,
    final_position_error: float,
    final_yaw_error: float,
) -> None:
    difficulty = metadata.get("difficulty", {})
    dataset_meta = metadata.get("dataset", {})
    selected_goal = dataset_meta.get("selected_goal", metadata.get("goal", {}))
    dynamic_event = metadata.get("dynamic_event")
    if isinstance(dynamic_event, dict):
        trigger = dynamic_event.get("trigger", {})
        action = str(dynamic_event.get("action", "unknown"))
        action = {"add_circle_obstacle": "circle obstacle"}.get(action, action)
        event_text = f"{action} @ {float(trigger.get('value', 0.0)):.0%} path"
    else:
        event_text = "none"
    tolerance_position = selected_goal.get("tol_pos")
    tolerance_yaw = selected_goal.get("tol_yaw")
    goal_pass = (
        tolerance_position is not None
        and tolerance_yaw is not None
        and final_position_error <= float(tolerance_position)
        and final_yaw_error <= float(tolerance_yaw)
    )
    maneuver_status = (
        "N/A"
        if maneuver_consistent is None
        else ("PASS" if maneuver_consistent else "FAIL")
    )
    feasibility_status = (
        "N/A"
        if feasibility_audit is None
        else ("PASS" if feasibility_audit.get("feasible") is True else "FAIL")
    )
    all_pass = (
        goal_pass
        and maneuver_consistent is True
        and feasibility_status == "PASS"
        and model_confirmed
    )
    status_text = (
        f"GOAL: {'PASS' if goal_pass else 'REVIEW'} | "
        f"MANEUVER: {maneuver_status} | FEASIBILITY: {feasibility_status}"
    )
    status_color = "#148F77" if all_pass else "#C0392B"

    task_id = textwrap.fill(str(metadata.get("task_id", "unknown")), width=27)
    details = [
        "TASK",
        f"ID\n{task_id}",
        f"Scene             {metadata.get('scene_name', 'unknown')}",
        f"Type              {metadata.get('task_type', 'unknown')}",
        f"Maneuver          {difficulty.get('maneuver', 'unknown')}",
        f"Noise             {difficulty.get('noise_level', 'unknown')}",
        f"Adjacent trucks   {difficulty.get('adjacent_occupancy', 'unknown')}",
        f"Goal kind         {selected_goal.get('kind', 'unknown')}",
        f"Goal spot         {selected_goal.get('spot_id', 'unknown')}",
        f"Goal policy       {dataset_meta.get('goal_policy', 'unknown')}",
        f"Dynamic event     {event_text}",
        "",
        "DRILL RIG MODEL",
        f"Name              {vehicle_config.name}",
        f"Version           {vehicle_config.model_version}",
        f"Envelope          {vehicle_config.length:.2f} x {vehicle_config.width:.2f} m",
        f"Planning v        {vehicle_config.plan_v:.2f} m/s",
        f"Planning omega    {vehicle_config.plan_max_omega:.2f} rad/s",
        f"Model metadata    {'confirmed' if model_confirmed else 'missing/fallback'}",
        "",
        "TRAJECTORY",
        f"Points            {point_count}",
        f"Path length       {path_length:.2f} m",
        f"Reverse distance  {reverse_ratio:.1%}",
        "Requested share  "
        + (
            "unknown"
            if requested_distance_ratio is None
            else f"{requested_distance_ratio:.1%}"
        ),
        f"Maneuver check    {maneuver_status}",
        f"Direction changes {switch_count}",
        f"Pivot segments    {_audit_value(feasibility_audit, 'pivot_segment_count', 'unknown')}",
        f"Max speed         {_audit_number(feasibility_audit, 'max_linear_speed_mps', 'm/s')}",
        f"Max yaw rate      {_audit_number(feasibility_audit, 'max_angular_speed_radps', 'rad/s')}",
        f"Lateral residual  {_audit_number(feasibility_audit, 'max_lateral_residual_m', 'm')}",
        f"Feasibility       {feasibility_status}",
        "",
        "FINAL ERROR",
        f"Position          {final_position_error:.3f} m",
        f"Heading           {np.degrees(final_yaw_error):.2f} deg",
    ]
    axis.set_facecolor("#F7F9F9")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#D5D8DC")
    axis.text(
        0.06,
        0.96,
        "\n".join(details),
        transform=axis.transAxes,
        ha="left",
        va="top",
        family="monospace",
        fontsize=8.35,
        linespacing=1.2,
        color="#17202A",
    )
    axis.text(
        0.5,
        0.055,
        status_text,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#FFFFFF",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": status_color,
            "edgecolor": "none",
        },
    )


def _audit_value(
    audit: dict[str, Any] | None, key: str, fallback: str
) -> str:
    if audit is None or key not in audit:
        return fallback
    return str(audit[key])


def _audit_number(
    audit: dict[str, Any] | None, key: str, unit: str
) -> str:
    if audit is None or key not in audit:
        return "unknown"
    try:
        return f"{float(audit[key]):.3f} {unit}"
    except (TypeError, ValueError):
        return "invalid"


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def _metadata_counts(metadata: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(item.get(key, "unknown")) for item in metadata)
    return dict(sorted(counts.items()))


def _noise_counts(metadata: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(item.get("difficulty", {}).get("noise_level", "unknown"))
        for item in metadata
    )
    return dict(sorted(counts.items()))


def _to_local(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    local = np.empty_like(points, dtype=np.float64)
    dx = points[:, 0] - x
    dy = points[:, 1] - y
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    local[:, 0] = cos_yaw * dx + sin_yaw * dy
    local[:, 1] = -sin_yaw * dx + cos_yaw * dy
    yaw_delta = points[:, 2] - yaw
    local[:, 2] = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
    return local


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _wrap_angles(angles: np.ndarray) -> np.ndarray:
    values = np.asarray(angles, dtype=np.float64)
    return np.arctan2(np.sin(values), np.cos(values))
