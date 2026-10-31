"""REF-UNGATE-001: one-switch ablation of the frozen LT37 controller."""
from scripts.runtime_refinement_mpc import RefinedCEMController, PROFILES

PROFILE = 'ungated'
OPTIONS = {**PROFILES['combined'], 'progress': False}


class UngatedRefinedCEMController(RefinedCEMController):
    """Use original clock progress; retain the complete frozen combined solve."""

    def __init__(self, **kwargs):
        super().__init__(profile='combined', **kwargs)
        self.options = OPTIONS.copy()
        self.profile = PROFILE
