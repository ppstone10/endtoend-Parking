"""轮式显式短检入口；不读取旧数据、不调用网络或履带终端接管。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.wheeled_evaluation import build_wheeled_engine
from interfaces import GoalPose
from interfaces.wheeled import WheeledControlCmd, WheeledTrajectory, WheeledVehicleState
from sim.wheeled_config import load_wheeled_vehicle_config
from sim.wheeled_vehicle import KinematicBicycleModel


def tracking_smoke(config, case):
    """固定短轨迹来自基础模型，仅检验控制/执行，不冒充专家规划或正式评测。"""
    speed, delta = {"straight": (.5, 0.), "reverse": (-.5, 0.), "arc": (.5, .3)}[case]
    model = KinematicBicycleModel(config)
    state = WheeledVehicleState(0, 0, 0)
    poses = [[state.x, state.y, state.yaw]]
    for command in [WheeledControlCmd(speed, delta)]*45 + [WheeledControlCmd(0, delta)]*12:
        state = model.step(state, command, .1)
        poses.append([state.x, state.y, state.yaw])
    trajectory = WheeledTrajectory(np.asarray(poses), .1, config.identity)
    engine = build_wheeled_engine(config, max_steps=100, horizon=8, population=12, iterations=2, seed=7)
    result = engine.run(WheeledVehicleState(0, 0, 0), GoalPose(*poses[-1]), lambda s, g: trajectory)
    return dict(case=case, success=result["success"], failure=result["failure"], **result["summary"])


def expert_smoke(config, case):
    """固定三例GT几何专家+MPC短检，沿用项目600步与原位姿门槛。"""
    from planner.wheeled_expert import WheeledReedsSheppPlanner
    from planner.wheeled_safety import WheeledTransitionChecker
    from sim import ParkingEnvironment
    model = KinematicBicycleModel(config)
    checker = WheeledTransitionChecker(model, ParkingEnvironment(world_size=80), dt=.1)
    planner = WheeledReedsSheppPlanner(model, checker, dt=.1)
    start = WheeledVehicleState(0, 0, 0)
    goal = {"straight": GoalPose(2, 0, 0), "reverse": GoalPose(-2, 0, 0),
            "arc": GoalPose(3, 2, .4)}[case]
    plan = planner.plan(start, goal)
    if len(plan.controls) > 600:
        raise ValueError("短检专家参考超过原600步预算")
    engine = build_wheeled_engine(config, max_steps=600, transition_checker=checker,
                                  horizon=8, population=8, iterations=1, seed=7)
    result = engine.run(start, goal, lambda s, g: plan.trajectory)
    return dict(case=case, success=result["success"], failure=result["failure"],
                expert_controls=len(plan.controls), candidate_word=plan.candidate_word,
                geometry_source="simulator_ground_truth", **result["summary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vehicles/wheeled_research_v1.json")
    parser.add_argument("--mode", choices=("tracking", "expert"), default="tracking")
    parser.add_argument("--case", choices=("straight", "reverse", "arc", "all"), default="all")
    args = parser.parse_args()
    config = load_wheeled_vehicle_config(args.config)
    cases = ("straight", "reverse", "arc") if args.case == "all" else (args.case,)
    smoke = tracking_smoke if args.mode == "tracking" else expert_smoke
    rows = [smoke(config, case) for case in cases]
    print(json.dumps(dict(kind=f"wheeled_{args.mode}_smoke", formal=False,
                          model_identity=config.identity, parameter_source=config.parameter_source,
                          rows=rows), ensure_ascii=False, indent=2))
    return 0 if all(row["success"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
