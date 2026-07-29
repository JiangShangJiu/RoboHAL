#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轨迹回放

从 data/ 或 outputs/ 加载轨迹，用 MuJoCo viewer 回放。
"""

from pathlib import Path

import numpy as np

from ..simulation import SimEnv
from ..simulation.loader import qpos_to_ctrl
from .viewer import launch_viewer


def replay_trajectory(
    qpos_sequence: np.ndarray,
    env: SimEnv | None = None,
    dt: float = 0.002,
    slowdown: float = 1.0,
) -> None:
    """
    回放关节轨迹

    Args:
        qpos_sequence: (T, nq) 关节位置序列
        env: 仿真环境，None 则新建
        dt: 仿真步长
        slowdown: 回放减速倍数（>1 则更慢）
    """
    import time
    import mujoco
    import mujoco.viewer  # mujoco 3.x 需显式导入

    if env is None:
        env = SimEnv(dt=dt)
    model, data = env.model, env.data
    nu = model.nu

    def sync_cb(m, d):
        pass  # 由下方循环控制

    step_duration = dt * slowdown
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(len(qpos_sequence)):
            if not viewer.is_running():
                break
            qpos = qpos_sequence[i]
            nq = min(model.nq, len(qpos))
            data.qpos[:nq] = qpos[:nq]
            data.ctrl[:] = qpos_to_ctrl(qpos.tolist(), nu)
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(step_duration)


def load_and_replay(
    path: str | Path,
    episode_idx: int = 0,
) -> None:
    """
    从 HDF5 文件加载轨迹并回放

    Args:
        path: 轨迹文件路径
        episode_idx: 要回放的轨迹索引
    """
    from ..data_collection.storage import load_trajectories

    episodes = load_trajectories(path)
    if episode_idx >= len(episodes):
        raise IndexError(f"episode_idx {episode_idx} >= num_episodes {len(episodes)}")
    ep = episodes[episode_idx]
    qpos_seq = np.array([t["obs"]["qpos"] for t in ep])
    replay_trajectory(qpos_seq)
