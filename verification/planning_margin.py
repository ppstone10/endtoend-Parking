"""用同一规划余量检查器约束完整控制输出路径；仅显式实验使用。"""
from contextlib import contextmanager
from planner import RectangleFootprintCollisionChecker
from verification.zero_recovery import ZeroRecoveryMPC
from verification import terminal_capture
from verification.reference_retention import retention_scope


class PlanningMarginMPC(ZeroRecoveryMPC):
    def __init__(self, *, contact_checker, **kwargs):
        self.physical_checker=contact_checker
        planning_checker=RectangleFootprintCollisionChecker(
            contact_checker.env,vehicle_length=2*contact_checker.half_length,
            vehicle_width=2*contact_checker.half_width,collision_margin=.2,
            resolution=contact_checker.resolution)
        super().__init__(contact_checker=planning_checker,**kwargs)


@contextmanager
def planning_margin_scope(output,vehicle,expected,budget):
    original=terminal_capture.TerminalCaptureMPC
    terminal_capture.TerminalCaptureMPC=PlanningMarginMPC
    try:
        with retention_scope(output,vehicle,expected,budget) as result:yield result
    finally:
        terminal_capture.TerminalCaptureMPC=original
