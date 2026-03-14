#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轨迹存储

支持 HDF5 格式保存/加载轨迹数据。
"""

from pathlib import Path
from typing import Any

import numpy as np


def save_trajectories(
    episodes: list[list[dict]],
    path: str | Path,
    format: str = "hdf5",
) -> None:
    """
    保存轨迹到文件

    Args:
        episodes: 多条轨迹，每条为 [{"obs", "action", "next_obs", "reward", "done"}, ...]
        path: 输出路径
        format: "hdf5" 或 "npz"
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "hdf5":
        import h5py
        with h5py.File(path, "w") as f:
            for i, ep in enumerate(episodes):
                grp = f.create_group(f"episode_{i}")
                obs_list = [t["obs"]["qpos"] for t in ep]
                action_list = [t["action"] for t in ep]
                grp.create_dataset("qpos", data=np.array(obs_list))
                grp.create_dataset("action", data=np.array(action_list))
    elif format == "npz":
        # 简单格式：每条轨迹单独保存为 npz
        for i, ep in enumerate(episodes):
            qpos = np.array([t["obs"]["qpos"] for t in ep])
            action = np.array([t["action"] for t in ep])
            np.savez(path.parent / f"{path.stem}_ep{i}.npz", qpos=qpos, action=action)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_trajectories(path: str | Path) -> list[dict]:
    """
    从 HDF5 文件加载轨迹

    Returns:
        episodes: 与 save 时结构一致
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    import h5py
    episodes = []
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            if key.startswith("episode_"):
                grp = f[key]
                qpos = np.array(grp["qpos"])
                action = np.array(grp["action"])
                ep = []
                for j in range(len(qpos)):
                    ep.append({
                        "obs": {"qpos": qpos[j], "qvel": np.zeros_like(qpos[j])},
                        "action": action[j],
                        "next_obs": {"qpos": qpos[j] if j == len(qpos) - 1 else qpos[j + 1], "qvel": np.zeros_like(qpos[j])},
                        "reward": 0.0,
                        "done": j == len(qpos) - 1,
                    })
                episodes.append(ep)
    return episodes
