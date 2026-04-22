#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Franka Panda 单臂 7 关节近似限位（弧度），与 MuJoCo 常用 Panda 模型一致量级。"""

from __future__ import annotations

import numpy as np

# 参考 URDF 常见 Panda 关节范围（可按需微调）
PANDA_ARM_JOINT_LOW = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float64,
)
PANDA_ARM_JOINT_HIGH = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float64,
)


def default_panda_arm_bounds() -> tuple[np.ndarray, np.ndarray]:
    return PANDA_ARM_JOINT_LOW.copy(), PANDA_ARM_JOINT_HIGH.copy()
