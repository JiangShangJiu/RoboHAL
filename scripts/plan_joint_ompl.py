#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 OMPL 在关节空间规划一条轨迹，并保存为与本项目兼容的 HDF5（可选 MuJoCo 回放）。

依赖: pip install -e ".[planning]"

示例:
    python scripts/plan_joint_ompl.py --goal 0,0,0,-1.5,0,1.5,0.7 -o data/ompl_plan.h5
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_collection.storage import save_trajectories  # noqa: E402
from src.planning import PandaOMPLJointPlanner, arm7_to_mujoco_qpos  # noqa: E402
from src.simulation import PandaEnv  # noqa: E402
from src.simulation.loader import DEFAULT_Q0, qpos_to_ctrl  # noqa: E402


def _parse_vec(s: str, n: int) -> np.ndarray:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != n:
        raise argparse.ArgumentTypeError(f"需要 {n} 个用逗号分隔的数，得到 {len(parts)} 个")
    return np.array(parts, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="OMPL 关节空间规划并导出 HDF5")
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="起点 7 关节，逗号分隔；默认使用 loader.DEFAULT_Q0 前 7 维",
    )
    parser.add_argument(
        "--goal",
        type=str,
        required=True,
        help="终点 7 关节，逗号分隔",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/ompl_plan.h5",
        help="输出 HDF5 路径",
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=100, help="插值后路标数")
    parser.add_argument(
        "--planner", type=str, default="RRTConnect", help="OMPL 规划器名称"
    )
    parser.add_argument(
        "--no-simplify", action="store_true", help="跳过 OMPL 路径简化"
    )
    args = parser.parse_args()

    start7 = (
        _parse_vec(args.start, 7)
        if args.start
        else np.array(DEFAULT_Q0[:7], dtype=np.float64)
    )
    goal7 = _parse_vec(args.goal, 7)

    planner = PandaOMPLJointPlanner(planner=args.planner)
    arm_path = planner.plan_joint(
        start7,
        goal7,
        timeout=args.timeout,
        interpolate_steps=args.steps,
        simplify=not args.no_simplify,
    )

    nu = PandaEnv().nu
    episodes = []
    ep = []
    for j, row in enumerate(arm_path):
        qpos = arm7_to_mujoco_qpos(row)
        if j + 1 < len(arm_path):
            qnext = arm7_to_mujoco_qpos(arm_path[j + 1])
        else:
            qnext = qpos
        action = np.array(qpos_to_ctrl(qpos.tolist(), nu), dtype=np.float64)
        ep.append({
            "obs": {"qpos": qpos, "qvel": np.zeros_like(qpos)},
            "action": action,
            "next_obs": {"qpos": qnext, "qvel": np.zeros_like(qpos)},
            "reward": 0.0,
            "done": False,
        })
    if ep:
        ep[-1]["done"] = True
    episodes.append(ep)

    out = Path(args.output)
    save_trajectories(episodes, out, format="hdf5")
    print(f"已保存 {len(arm_path)} 个路标到 {out.resolve()}")
    print(f"回放: python scripts/replay_trajectory.py {out} --episode 0")


if __name__ == "__main__":
    main()
    # 部分环境下 OMPL Python 绑定在解释器退出析构时会触发堆损坏；任务已完成则直接退出。
    os._exit(0)
