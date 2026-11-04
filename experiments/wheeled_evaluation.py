"""轮式组合根；旧NPZ评测与旧训练/恢复入口不隐式切换车型。"""

import math

from controller.wheeled_mpc import WheeledMPCController
from runtime.wheeled_execution import WheeledClosedLoopEngine, WheeledExecutor
from sim.wheeled_vehicle import KinematicBicycleModel


def build_wheeled_engine(config, *, dt=.1, max_steps=600, replan_every=0,
                         transition_checker=None, tol_pos=.3, tol_yaw=math.radians(10),
                         max_seconds=None, **controller_options):
    """供显式轮式短验证和后续数据入口共用，核对模型身份由执行器负责。"""
    if transition_checker is not None:
        if abs(getattr(transition_checker, "dt", dt)-dt) > 1e-12:
            raise ValueError("轮式运行时与几何检查dt不一致")
        checker_model = getattr(transition_checker, "model", None)
        if checker_model is not None and checker_model.config.identity != config.identity:
            raise ValueError("轮式运行时与几何检查车型身份不一致")
    model = KinematicBicycleModel(config)
    controller = WheeledMPCController(model, dt=dt, transition_checker=transition_checker,
                                      **controller_options)
    return WheeledClosedLoopEngine(WheeledExecutor(controller, model), max_steps=max_steps,
                                   replan_every=replan_every, transition_checker=transition_checker,
                                   tol_pos=tol_pos, tol_yaw=tol_yaw, max_seconds=max_seconds)
