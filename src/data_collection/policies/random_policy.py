#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
随机策略：用于数据采集的简单探索策略
"""

import numpy as np

from ...simulation.loader import DEFAULT_Q0, qpos_to_ctrl


class RandomPolicy:
    """在关节限制内随机采样目标位置"""

    # Panda 关节限制（弧度）
    JOINT_LIMITS = np.array([
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
        [0.0, 0.04],   # 夹爪
        [0.0, 0.04],
    ])

    def __init__(self, nu: int = 9):
        """
        Args:
            nu: 控制维度（actuator 数量）
        """
        self.nu = nu

    def __call__(self, obs: dict) -> np.ndarray:
        """obs -> action (ctrl)"""
        qpos = obs["qpos"]
        nq = min(len(qpos), len(self.JOINT_LIMITS))
        # 在限制内随机采样
        low = self.JOINT_LIMITS[:nq, 0]
        high = self.JOINT_LIMITS[:nq, 1]
        target = np.random.uniform(low, high)
        return qpos_to_ctrl(target.tolist(), self.nu)
