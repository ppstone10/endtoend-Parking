"""联合比较独立选型报告；只判定，不改模型或部署权重。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.comparison import PLANNING_GATES
from training.reporting import atomic_write_json


def assess_candidate_metrics(baseline: dict, candidate: dict) -> dict:
    gates = [gate.to_dict(float(candidate[gate.metric])) for gate in PLANNING_GATES]
    passed = all(gate["passed"] for gate in gates)
    success_delta = candidate["success_rate"] - baseline["success_rate"]
    collision_delta = candidate["collision_rate"] - baseline["collision_rate"]
    nondegrading = success_delta >= 0 and collision_delta <= 0
    if passed:
        status = "selection_gate_passed_requires_main_protocol"
    elif nondegrading and (success_delta > 0 or collision_delta < 0):
        status = "improves_baseline_not_admitted"
    elif success_delta == 0 and collision_delta == 0:
        status = "unchanged_not_admitted"
    else:
        status = "tradeoff_or_regression_not_admitted"
    return {"status": status, "gates": gates, "success_delta": success_delta, "collision_delta": collision_delta,
            "failures": candidate.get("failures", {}), "deployment_modified": False}


def compare_reports(baseline: dict, candidate: dict) -> dict:
    for report in (baseline, candidate):
        if report.get("probe", {}).get("kind") == "diagnostic_controller_counterfactual" or report.get("protocol", {}).get("controller_candidate") not in (None, "baseline"):
            raise ValueError("实验控制器反事实不属于固定控制协议，不能作为模型选型准入")
        if report.get("status") != "completed" or report["experiment"]["name"] != "E5" or report["experiment"]["safety_mode"] != "none":
            raise ValueError("联合选型只接受已完成的无干预 E5")
        if report["overall"]["reconstruct_failures"] != 0:
            raise ValueError("任务复原失败，不能缩小分母比较")
        indices = report["protocol"]["evaluated_indices"]
        if len(indices) != len(set(indices)) or len(indices) != report["overall"]["evaluated_samples"]:
            raise ValueError("任务索引存在重复或数量不一致")
    for key in ("data", "evaluated_indices", "max_steps", "replan_every", "control_seed", "mpc_dt"):
        if baseline["protocol"][key] != candidate["protocol"][key]:
            raise ValueError(f"选型协议不一致：{key}")
    return assess_candidate_metrics(baseline["overall"], candidate["overall"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    decision = compare_reports(baseline, candidate)
    decision.update({"baseline": str(args.baseline.resolve()), "candidate": str(args.candidate.resolve())})
    atomic_write_json(args.output, decision)
    print(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
