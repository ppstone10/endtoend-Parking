"""闭环五组对照实验的汇总记分卡与决策树归因。

把 ``scripts/run_validation_suite.py`` 产出的各组报告汇总成一张对照表，并按
``docs/closed_loop_five_experiment_plan.md`` §4 的决策树给出归因结论与未决项。

判据（计划 §2.3，先验阈值）：

- E1/E4：成功率 ≥98%、碰撞 ≤1%、跟踪 RMS <0.05m；
- E2/E2a/E3/E5：纯网络成功率 ≥70%、碰撞 ≤10%；
- E4 与 E1 的成功率差 ≤5pt 且碰撞差 ≤2pt；
- E5 与 E4 的差值体现网络净贡献，用于区分"感知问题"与"规划问题"。

模块只做汇总与判定，不重新跑实验、不改报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from metrics.rollout import linear_slope


@dataclass(frozen=True)
class Gate:
    """一条判据：指标名 + 比较方向 + 阈值 + 说明。"""

    metric: str
    comparison: str  # ">=" 或 "<="
    threshold: float
    description: str

    def evaluate(self, value: float | None) -> bool | None:
        if value is None:
            return None
        if self.comparison == ">=":
            return value >= self.threshold
        return value <= self.threshold

    def to_dict(self, value: float | None) -> dict[str, Any]:
        passed = self.evaluate(value)
        return {
            "metric": self.metric,
            "comparison": self.comparison,
            "threshold": self.threshold,
            "value": value,
            "passed": passed,
            "description": self.description,
        }


CONTROL_GATES: tuple[Gate, ...] = (
    Gate("success_rate", ">=", 0.98, "E1 控制链地基成功率"),
    Gate("collision_rate", "<=", 0.01, "E1 控制链地基碰撞率"),
    Gate("tracking_rms_mean", "<=", 0.05, "E1 跟踪 RMS（控制层健康）"),
)

PLANNING_GATES: tuple[Gate, ...] = (
    Gate("success_rate", ">=", 0.70, "纯网络闭环成功率准入线"),
    Gate("collision_rate", "<=", 0.10, "纯网络碰撞率准入线"),
)

PERCEPTION_GATES: tuple[Gate, ...] = (
    Gate("success_rate_delta", ">=", -0.05, "E4 与 E1 成功率差（≥ -5pt）"),
    Gate("collision_rate_delta", "<=", 0.02, "E4 与 E1 碰撞率差（≤ 2pt）"),
)

__all__ = [
    "CONTROL_GATES",
    "PERCEPTION_GATES",
    "PLANNING_GATES",
    "Gate",
    "build_scorecard",
    "format_markdown",
    "load_reports",
]


def load_reports(directory: str | Path) -> dict[str, dict]:
    """读取目录下各 ``<实验>/report.json``（``<实验>@<权重>`` 也支持）。"""
    root = Path(directory)
    reports: dict[str, dict] = {}
    for path in sorted(root.glob("*/report.json")):
        import json

        reports[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    return reports


def _metric(report: dict, key: str) -> float | None:
    value = report.get("overall", {}).get(key)
    if value is None:
        return None
    return float(value)


def _episode_yaw_series(report: dict) -> list[float]:
    """逐样本"重规划末航向误差"序列（按样本顺序）。"""
    series: list[float] = []
    for episode in report.get("episodes", []):
        value = episode.get("cycles", {}).get("yaw_err_end_deg")
        if value is not None:
            series.append(float(value))
    return series


def build_scorecard(reports: dict[str, dict]) -> dict[str, Any]:
    """由各组报告生成记分卡与决策树结论。"""
    rows: list[dict[str, Any]] = []
    for name, report in reports.items():
        overall = report["overall"]
        experiment = report["experiment"]
        yaw_series = _episode_yaw_series(report)
        rows.append(
            {
                "key": name,
                "experiment": experiment["name"],
                "bev_source": experiment["bev_source"],
                "trajectory_source": experiment["trajectory_source"],
                "executor": experiment["executor"],
                "safety_mode": experiment["safety_mode"],
                "model_label": report.get("model_label"),
                "episodes": overall.get("evaluated_samples"),
                "success_rate": overall.get("success_rate"),
                "collision_rate": overall.get("collision_rate"),
                "final_pos_err_mean": overall.get("final_pos_err_mean"),
                "final_yaw_err_deg_mean": (
                    None
                    if overall.get("final_yaw_err_mean") is None
                    else float(np.degrees(overall["final_yaw_err_mean"]))
                ),
                "tracking_rms_mean": overall.get("tracking_rms_mean"),
                "time_dist_ratio_mean": overall.get("time_dist_ratio_mean"),
                "failures": overall.get("failures"),
                "cycles_divergence_events_mean": overall.get(
                    "cycles_divergence_events_mean"
                ),
                "cycles_heading_divergence_events_mean": overall.get(
                    "cycles_heading_divergence_events_mean"
                ),
                "cycles_yaw_err_end_deg_mean": overall.get("cycles_yaw_err_end_deg_mean"),
                "cycles_drift_slope_mean": overall.get("cycles_drift_slope_mean"),
                "cycles_heading_drift_slope_deg_mean": overall.get(
                    "cycles_heading_drift_slope_deg_mean"
                ),
                "yaw_error_slope_deg_per_sample": (
                    None
                    if (slope := linear_slope(yaw_series)) is None
                    else float(slope)
                ),
                "reconstruct_failures": overall.get("reconstruct_failures"),
            }
        )

    by_experiment = {row["experiment"]: row for row in rows}
    gates: dict[str, list[dict[str, Any]]] = {}
    for name, gate_set in (
        ("E1", CONTROL_GATES),
        ("E3", PLANNING_GATES),
        ("E5", PLANNING_GATES),
    ):
        row = by_experiment.get(name)
        if row is None:
            continue
        gates[name] = [gate.to_dict(row.get(gate.metric)) for gate in gate_set]

    deltas: dict[str, float | None] = {}
    e1 = by_experiment.get("E1")
    e4 = by_experiment.get("E4")
    if e1 is not None and e4 is not None:
        deltas["E4_minus_E1_success"] = _safe_delta(
            e4.get("success_rate"), e1.get("success_rate")
        )
        deltas["E4_minus_E1_collision"] = _safe_delta(
            e4.get("collision_rate"), e1.get("collision_rate")
        )
        gates["E4_vs_E1"] = [
            Gate(
                "E4_minus_E1_success", ">=", -0.05, "E4 与 E1 成功率差（≥ -5pt）"
            ).to_dict(deltas["E4_minus_E1_success"]),
            Gate(
                "E4_minus_E1_collision", "<=", 0.02, "E4 与 E1 碰撞率差（≤ 2pt）"
            ).to_dict(deltas["E4_minus_E1_collision"]),
        ]
    e5 = by_experiment.get("E5")
    if e4 is not None and e5 is not None:
        deltas["E5_minus_E4_success"] = _safe_delta(
            e5.get("success_rate"), e4.get("success_rate")
        )

    return {
        "schema_version": 1,
        "rows": rows,
        "deltas": deltas,
        "gates": gates,
        "decision": _decision(by_experiment, reports),
        "limitations": _limitations(reports),
    }


def _safe_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _all_pass(gates: list[dict[str, Any]]) -> bool | None:
    results = [gate["passed"] for gate in gates if gate["passed"] is not None]
    if not results:
        return None
    return all(results)


def _decision(by_experiment: dict[str, dict], reports: dict[str, dict]) -> dict[str, Any]:
    """按计划 §4 决策树给出落点。"""
    e1 = by_experiment.get("E1")
    e2_keys = [key for key, report in reports.items() if report["experiment"]["name"] == "E2"]
    e3 = by_experiment.get("E3")
    e5 = by_experiment.get("E5")

    if e1 is None or e1.get("success_rate") is None:
        return {
            "branch": "undetermined",
            "reason": "缺少 E1 控制链地基结果，无法下移决策树",
        }
    control_ok = _all_pass(
        [gate.to_dict(e1.get(gate.metric)) for gate in CONTROL_GATES]
    )
    if control_ok is False:
        return {
            "branch": "control_layer",
            "reason": (
                "E1（GT 环境 + 传统规划 + MPC + 矿卡）未达控制链判据，"
                "失败归属控制层或车辆模型，其余实验结论暂不可用"
            ),
        }

    network_rows = [row for row in (e3, e5) if row is not None]
    planning_ok = None
    if network_rows:
        planning_ok = all(
            _all_pass([gate.to_dict(row.get(gate.metric)) for gate in PLANNING_GATES])
            for row in network_rows
        )

    e2_summary = None
    for key in e2_keys:
        report = reports[key]
        e2_summary = {
            "key": key,
            "success_rate": report["overall"].get("success_rate"),
            "collision_rate": report["overall"].get("collision_rate"),
            "yaw_err_end_deg": report["overall"].get("cycles_yaw_err_end_deg_mean"),
        }

    if planning_ok is True:
        return {
            "branch": "pass",
            "reason": "E3/E5 均达到纯网络准入线，按第一标准可交付；E2 结果用于解释裕度",
            "e2": e2_summary,
        }
    if e3 is not None and e5 is not None and e3.get("success_rate") is not None:
        if e5["success_rate"] < e3["success_rate"] - 0.05:
            return {
                "branch": "perception_layer",
                "reason": (
                    f"E5（传感器 BEV）{e5['success_rate']:.1%} 明显低于 E3（GT BEV）"
                    f"{e3['success_rate']:.1%}，BEV 误差是主要损失来源，先回感知层"
                ),
                "e2": e2_summary,
            }
    return {
        "branch": "planning_layer",
        "reason": (
            "控制层达标而网络闭环未达准入线：失败归属轨迹规划层（网络滚动质量），"
            "按计划应走训练侧（DAgger 完整化 / 近端终点与航向监督），不再调 MPC 参数"
        ),
        "e2": e2_summary,
    }


def _limitations(reports: dict[str, dict]) -> list[str]:
    """从报告中提取必须随结论一起声明的限制。"""
    notes: list[str] = []
    for key, report in reports.items():
        failures = report["overall"].get("reconstruct_failures") or 0
        if failures:
            notes.append(
                f"{key}：有 {failures} 条样本因当前代码几何与数据集身份不一致而跳过"
                "（S3/S5/S7/S8），这些场景的结论在 v8 数据重训前不成立"
            )
    episodes = {
        report["overall"].get("evaluated_samples") for report in reports.values()
    }
    if episodes and max(episodes) < 100:
        notes.append(
            f"样本量偏小（最大 {max(episodes)} 条），成功率置信区间宽，结论只作方向性判断"
        )
    return notes


def format_markdown(scorecard: dict[str, Any]) -> str:
    """把记分卡渲染为 Markdown。"""
    header = (
        "| 组 | BEV | 轨迹源 | 执行 | 权重 | 样本 | 成功 | 碰撞 | 终点位置(m) | "
        "终点航向(°) | 滚动末航向(°) | 跟踪RMS(m) | 失败分类 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for row in scorecard["rows"]:
        lines.append(
            "| {experiment} | {bev} | {traj} | {exec} | {model} | {n} | {succ} | "
            "{coll} | {pos} | {yaw} | {roll_yaw} | {rms} | {failures} |".format(
                experiment=row["experiment"],
                bev=row["bev_source"],
                traj=row["trajectory_source"],
                exec=row["executor"],
                model=row["model_label"] or "—",
                n=row["episodes"],
                succ=_pct(row["success_rate"]),
                coll=_pct(row["collision_rate"]),
                pos=_num(row["final_pos_err_mean"]),
                yaw=_num(row["final_yaw_err_deg_mean"]),
                roll_yaw=_num(row["cycles_yaw_err_end_deg_mean"]),
                rms=_num(row["tracking_rms_mean"]),
                failures=row["failures"],
            )
        )
    parts = ["# 闭环五组对照实验记分卡", "", header + "\n".join(lines), ""]
    parts.append("## 判据核对")
    for name, gate_list in sorted(scorecard["gates"].items()):
        parts.append(f"### {name}")
        for gate in gate_list:
            parts.append(
                "  - {metric} {comparison} {threshold}：{value} → {passed}".format(
                    metric=gate["metric"],
                    comparison=gate["comparison"],
                    threshold=gate["threshold"],
                    value=_num(gate["value"]),
                    passed={True: "通过", False: "不通过", None: "缺数据"}[gate["passed"]],
                )
            )
    parts.append("")
    parts.append("## 决策树落点")
    decision = scorecard["decision"]
    parts.append(f"- 分支：**{decision['branch']}**")
    parts.append(f"- 依据：{decision['reason']}")
    if decision.get("e2"):
        parts.append(f"- E2（理想执行）对照：{decision['e2']}")
    if scorecard["deltas"]:
        parts.append("")
        parts.append("## 配对差值")
        for key, value in sorted(scorecard["deltas"].items()):
            parts.append(f"- {key}：{_pct(value)}")
    if scorecard["limitations"]:
        parts.append("")
        parts.append("## 限制")
        for note in scorecard["limitations"]:
            parts.append(f"- {note}")
    return "\n".join(parts) + "\n"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"
