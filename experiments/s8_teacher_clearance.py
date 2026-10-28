"""仅离线教师使用的固定净空规划器副本；执行审计仍由原case拥有。"""

from copy import deepcopy

import numpy as np

from planner.collision import RectangleFootprintCollisionChecker


def clearance_teacher_planner(original):
    """独立复制原0.2m规划器，只有教师规划外廓增加到0.3m。"""
    if not np.isclose(original.collision_margin,.2,rtol=0.,atol=1e-12):
        raise ValueError('教师对照要求原规划余量固定0.2m')
    teacher=deepcopy(original)
    teacher.collision_margin=.3
    teacher._collision_checker=RectangleFootprintCollisionChecker(teacher.env,
        vehicle_length=teacher.vehicle_length,vehicle_width=teacher.vehicle_width,
        collision_margin=.3,resolution=teacher.collision_check_resolution)
    if teacher.vehicle_model_metadata:
        teacher.vehicle_model_metadata['collision_margin']=.3
    return teacher
