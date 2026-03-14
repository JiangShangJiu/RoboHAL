#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轨迹管理器：统一录制与回放

提供 record()、save()、load()、replay() 接口。
"""

import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..simulation import PandaEnv
from ..simulation.loader import qpos_to_ctrl
from .storage import save_trajectories, load_trajectories


class TrajectoryManager:
    """统一的数据录制与回放管理"""

    def __init__(self, env: PandaEnv | None = None, scene: str | None = None):
        """
        Args:
            env: 仿真环境，None 则新建
            scene: 场景名，如 scene.xml / mjx_single_cube.xml
        """
        self.env = env or PandaEnv(scene=scene)
        self._episodes: list[list[dict]] = []

    def record(
        self,
        policy: Callable[[dict], np.ndarray],
        num_episodes: int = 10,
        max_steps_per_episode: int = 1000,
        qpos_init: np.ndarray | None = None,
    ) -> list[list[dict]]:
        """
        录制轨迹

        Args:
            policy: 策略函数 obs -> action
            num_episodes: 轨迹条数
            max_steps_per_episode: 每条最大步数
            qpos_init: 初始关节位置（可选）

        Returns:
            episodes: 录制的轨迹列表
        """
        self._episodes = []
        for _ in range(num_episodes):
            obs = self.env.reset(qpos_init)
            ep = []
            for _ in range(max_steps_per_episode):
                action = policy(obs)
                next_obs, reward, done, info = self.env.step(action)
                ep.append({
                    "obs": {k: np.array(v) if isinstance(v, (list, np.ndarray)) else v
                           for k, v in obs.items()},
                    "action": np.array(action, dtype=np.float64),
                    "next_obs": {k: np.array(v) if isinstance(v, (list, np.ndarray)) else v
                                 for k, v in next_obs.items()},
                    "reward": reward,
                    "done": done,
                })
                obs = next_obs
                if done:
                    break
            self._episodes.append(ep)
        return self._episodes

    def save(self, path: str | Path, format: str = "hdf5") -> None:
        """保存已录制的轨迹"""
        if not self._episodes:
            raise RuntimeError("没有可保存的轨迹，请先调用 record()")
        save_trajectories(self._episodes, path, format=format)

    def load(self, path: str | Path) -> list[list[dict]]:
        """从文件加载轨迹"""
        self._episodes = load_trajectories(path)
        return self._episodes

    @property
    def episodes(self) -> list[list[dict]]:
        """当前持有的轨迹"""
        return self._episodes

    def replay(
        self,
        episode_idx: int = 0,
        slowdown: float = 1.0,
    ) -> None:
        """
        回放指定轨迹

        Args:
            episode_idx: 轨迹索引
            slowdown: 回放减速倍数（>1 更慢）
        """
        import mujoco
        import mujoco.viewer

        if not self._episodes:
            raise RuntimeError("没有可回放的轨迹，请先调用 record() 或 load()")
        if episode_idx >= len(self._episodes):
            raise IndexError(f"episode_idx {episode_idx} >= {len(self._episodes)}")

        ep = self._episodes[episode_idx]
        qpos_seq = np.array([t["obs"]["qpos"] for t in ep])
        model, data = self.env.model, self.env.data
        nu = model.nu
        dt = float(model.opt.timestep)
        step_duration = dt * slowdown

        with mujoco.viewer.launch_passive(model, data) as viewer:
            for i in range(len(qpos_seq)):
                if not viewer.is_running():
                    break
                qpos = qpos_seq[i]
                nq = min(model.nq, len(qpos))
                data.qpos[:nq] = qpos[:nq]
                data.ctrl[:] = qpos_to_ctrl(qpos.tolist(), nu)
                mujoco.mj_forward(model, data)
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(step_duration)
