#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""机械臂轨迹规划抽象（关节空间 + 笛卡尔/任务空间），便于接入 OMPL、MoveIt、自研 IK+插值等。"""

from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Planner(Protocol):
    """同一机械臂的规划接口：关节空间与笛卡尔空间各一条入口，输出均为关节路标 (T, dof)。

    - 仅支持其中一种时：另一方法可 ``raise NotImplementedError``；`pose_dim` 在无笛卡尔能力时建议为 ``0``。
    - 两种都支持时：`pose_dim` 为单步位姿长度（常见 6：xyz+rpy，或 7：xyz+四元数）。
    """

    @property
    def dof(self) -> int:
        ...

    @property
    def pose_dim(self) -> int:
        """笛卡尔位姿向量长度；仅实现关节规划时可为 ``0``。"""
        ...

    def plan_joint(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        *,
        timeout: float = 5.0,
        interpolate_steps: int = 100,
        simplify: bool = True,
    ) -> np.ndarray:
        """start/goal: shape (dof,)；返回 waypoints (T, dof)，float64。"""
        ...

    def plan_cartesian(
        self,
        start_pose: np.ndarray,
        goal_pose: np.ndarray,
        *,
        timeout: float = 5.0,
        interpolate_steps: int = 100,
        simplify: bool = True,
    ) -> np.ndarray:
        """start_pose/goal_pose: shape (pose_dim,)；返回关节路标 (T, dof)，float64。"""
        ...


ValidityFn = Callable[[np.ndarray], bool]
"""自定义碰撞/限位检查：输入一维关节向量 (dof,)，可自由闭包捕获 MuJoCo 模型。"""

PoseValidityFn = Callable[[np.ndarray], bool]
"""任务空间检查：输入一维位姿向量 (pose_dim,)（与具体实现的 `pose_dim` 一致）。"""
