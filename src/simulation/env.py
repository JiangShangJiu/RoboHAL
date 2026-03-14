#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Panda 仿真环境封装

提供 reset()、step()、get_obs() 等接口，供数据采集和可视化复用。
"""

import numpy as np

from .loader import load_panda, DEFAULT_Q0, qpos_to_ctrl


class PandaEnv:
    """Franka Panda MuJoCo 仿真环境"""

    def __init__(
        self,
        scene: str | None = None,
        model_root=None,
        dt: float = 0.002,
    ):
        """
        Args:
            scene: 场景名或路径，如 scene.xml / mjx_single_cube.xml / scenes/table.xml
            model_root: 模型根目录，None 则自动查找
            dt: 仿真步长
        """
        self.model, self.data = load_panda(scene, model_root)
        self.dt = dt
        self.model.opt.timestep = dt
        self._mujoco = __import__("mujoco")
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu

    def reset(self, qpos: np.ndarray | None = None) -> dict:
        """
        重置环境

        Args:
            qpos: 初始关节位置，None 则使用 DEFAULT_Q0

        Returns:
            obs: 当前观测
        """
        self._mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            nq = min(self.nq, len(qpos))
            self.data.qpos[:nq] = qpos[:nq]
        else:
            nq = min(self.nq, len(DEFAULT_Q0))
            self.data.qpos[:nq] = [DEFAULT_Q0[i] for i in range(nq)]
        self.data.ctrl[:] = qpos_to_ctrl(self.data.qpos[:self.nq].tolist(), self.nu)
        self._mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        执行一步仿真

        Args:
            action: 控制量，长度 nu（关节位置或 ctrl）

        Returns:
            obs, reward, done, info
        """
        self.data.ctrl[:] = action[: self.nu]
        self._mujoco.mj_step(self.model, self.data)
        obs = self.get_obs()
        reward = 0.0  # 可扩展
        done = False
        info = {}
        return obs, reward, done, info

    def get_obs(self) -> dict:
        """获取当前观测"""
        self._mujoco.mj_forward(self.model, self.data)
        return {
            "qpos": self.data.qpos[: self.nq].copy(),
            "qvel": self.data.qvel[: self.nv].copy(),
            "time": float(self.data.time),
        }

    def get_hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """获取末端执行器（hand）位姿"""
        self._mujoco.mj_forward(self.model, self.data)
        hand_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, "hand"
        )
        if hand_id >= 0:
            return (
                self.data.xpos[hand_id].copy(),
                self.data.xquat[hand_id].copy(),
            )
        return np.zeros(3), np.array([1, 0, 0, 0])
