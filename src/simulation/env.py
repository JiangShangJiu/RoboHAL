#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MuJoCo 仿真环境：reset / step / get_obs。"""

import numpy as np

from .loader import DEFAULT_Q0, default_robot, load_robot, qpos_to_ctrl


class SimEnv:
    """通用仿真环境（多机器人）。"""

    def __init__(
        self,
        scene: str | None = None,
        robot: str | None = None,
        dt: float = 0.002,
    ):
        self.robot = robot or default_robot()
        self.model, self.data = load_robot(scene=scene, robot=self.robot)
        self.dt = dt
        self.model.opt.timestep = dt
        self._mujoco = __import__("mujoco")
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu

    def reset(self, qpos: np.ndarray | None = None) -> dict:
        self._mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            nq = min(self.nq, len(qpos))
            self.data.qpos[:nq] = qpos[:nq]
            self.data.ctrl[:] = qpos_to_ctrl(self.data.qpos[: self.nq].tolist(), self.nu)
        elif self.robot == "panda":
            nq = min(self.nq, len(DEFAULT_Q0))
            self.data.qpos[:nq] = [DEFAULT_Q0[i] for i in range(nq)]
            self.data.ctrl[:] = qpos_to_ctrl(self.data.qpos[: self.nq].tolist(), self.nu)
        else:
            self.data.ctrl[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        self.data.ctrl[:] = action[: self.nu]
        self._mujoco.mj_step(self.model, self.data)
        return self.get_obs(), 0.0, False, {}

    def get_obs(self) -> dict:
        self._mujoco.mj_forward(self.model, self.data)
        return {
            "qpos": self.data.qpos[: self.nq].copy(),
            "qvel": self.data.qvel[: self.nv].copy(),
            "time": float(self.data.time),
        }

    def get_hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
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
