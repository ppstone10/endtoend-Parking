"""NPZ 数据集统计与 BEV/专家轨迹叠加抽检。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import textwrap
from typing import Any

import numpy as np

from sim.vehicle_config import MINING_TRUCK


def summarize_dataset(data: dict[str, Any]) -> dict[str, Any]:
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

    metadata = data.get("task_meta") or []
    return {
        "sample_count": int(trajs.shape[0]),
        "trajectory_length_m": _distribution(lengths),
        "reverse_distance_ratio": (
            reverse_distance / total_distance if total_distance > 0.0 else 0.0
        ),
        "scene_counts": _metadata_counts(metadata, "scene_name"),
        "task_type_counts": _metadata_counts(metadata, "task_type"),
        "noise_level_counts": _noise_counts(metadata),
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
    """保存包含矿卡起终位姿、行驶方向和到位误差的专家轨迹验收图。"""
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
    directions, segment_lengths = _segment_directions(trajectory)
    path_length = float(segment_lengths.sum())
    reverse_distance = float(segment_lengths[directions < 0].sum())
    reverse_ratio = reverse_distance / path_length if path_length > 0.0 else 0.0
    final_position_error = float(np.linalg.norm(trajectory[-1, :2] - goal[:2]))
    final_yaw_error = abs(_wrap_angle(float(trajectory[-1, 2] - goal[2])))
    item_meta = metadata[index] if metadata is not None else {}

    figure, (axis, info_axis) = plt.subplots(
        1,
        2,
        figsize=(11.5, 7.0),
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

    switch_indices = np.flatnonzero(directions[1:] != directions[:-1]) + 1
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

    _draw_vehicle_pose(
        axis,
        np.array([0.0, 0.0, 0.0]),
        facecolor="#2ECC71",
        edgecolor="#087F23",
        label="START",
    )
    _draw_vehicle_pose(
        axis,
        goal_local,
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
        x_bounds=(-right, left),
        y_bounds=(-back, front),
    )
    axis.set_aspect("equal")
    axis.grid(color="#7F8C8D", alpha=0.25, linewidth=0.7)
    legend_handles = [
        Patch(facecolor="#2ECC71", edgecolor="#087F23", label="Start truck pose"),
        Patch(facecolor="#FFB347", edgecolor="#D35400", label="Goal truck pose"),
        Line2D([0], [0], color="#1976D2", linewidth=2.6, label="Forward path"),
        Line2D([0], [0], color="#8E44AD", linewidth=2.6, label="Reverse path"),
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


def _segment_directions(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(trajectory) < 2:
        return np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float64)
    delta = np.diff(trajectory[:, :2], axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    heading = trajectory[:-1, 2]
    signed_progress = delta[:, 0] * np.cos(heading) + delta[:, 1] * np.sin(heading)
    directions = np.where(signed_progress < -1e-9, -1, 1).astype(np.int8)
    return directions, lengths


def _vehicle_polygon(pose: np.ndarray) -> np.ndarray:
    """返回绘图坐标（left, forward）中的矿卡四角。"""
    half_length = MINING_TRUCK.length / 2.0
    half_width = MINING_TRUCK.width / 2.0
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


def _draw_vehicle_pose(axis, pose, *, facecolor, edgecolor, label) -> None:
    from matplotlib.patches import Polygon

    center_left, center_forward = float(pose[1]), float(pose[0])
    axis.add_patch(
        Polygon(
            _vehicle_polygon(np.asarray(pose)),
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=2.2,
            alpha=0.68,
            zorder=6,
        )
    )
    arrow_length = MINING_TRUCK.length * 0.48
    axis.annotate(
        "",
        xy=(
            center_left + arrow_length * np.sin(float(pose[2])),
            center_forward + arrow_length * np.cos(float(pose[2])),
        ),
        xytext=(center_left, center_forward),
        arrowprops={"arrowstyle": "-|>", "color": edgecolor, "lw": 2.2},
        zorder=8,
    )
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


def _set_evidence_limits(axis, plot_points, goal_pose, *, x_bounds, y_bounds) -> None:
    footprint_points = np.vstack(
        (
            _vehicle_polygon(np.array([0.0, 0.0, 0.0])),
            _vehicle_polygon(np.asarray(goal_pose)),
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
    status_text = "GOAL CHECK: PASS" if goal_pass else "GOAL CHECK: REVIEW"
    status_color = "#148F77" if goal_pass else "#C0392B"

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
        "TRAJECTORY",
        f"Points            {point_count}",
        f"Path length       {path_length:.2f} m",
        f"Reverse distance  {reverse_ratio:.1%}",
        f"Direction changes {switch_count}",
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
        fontsize=9.2,
        linespacing=1.32,
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
