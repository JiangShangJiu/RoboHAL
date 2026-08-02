#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
展示仿真环境

启动 MuJoCo viewer。空格切换：保持静止 / 随机探索。

用法:
    python scripts/show_env.py
    python scripts/show_env.py --scene kitchen_lite.xml
    python scripts/show_env.py --robot pal_tiago_dual
    python scripts/show_env.py --list-robots
    python scripts/show_env.py --list-scenes
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import numpy as np

from src.simulation import SimEnv, list_available_robots, list_available_scenes
from src.simulation.loader import DEFAULT_Q0, qpos_to_ctrl
from src.data_collection.policies import RandomPolicy


def _init_pose(model: mujoco.MjModel, data: mujoco.MjData, robot: str) -> None:
    """按机器人类型设置安全初始位姿（禁止把 Panda q0 写进 freejoint/移动底盘）。"""
    mujoco.mj_resetData(model, data)
    if robot == "panda":
        nq = min(model.nq, len(DEFAULT_Q0))
        data.qpos[:nq] = DEFAULT_Q0[:nq]
        data.ctrl[:] = qpos_to_ctrl(DEFAULT_Q0, model.nu)
    else:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description="展示仿真环境")
    parser.add_argument("--scene", "-s", type=str, default=None, help="场景名，如 empty.xml / kitchen_lite.xml")
    parser.add_argument("--robot", "-r", type=str, default=None, help="机器人名，默认 panda")
    parser.add_argument("--list-robots", action="store_true", help="列出可用机器人后退出")
    parser.add_argument("--list-scenes", action="store_true", help="列出可用场景后退出")
    args = parser.parse_args()

    if args.list_robots:
        print("可用机器人:")
        for name in list_available_robots():
            print(f"  {name}")
        return

    if args.list_scenes:
        scenes = list_available_scenes(args.robot)
        print("可用场景:")
        for name, path in scenes:
            print(f"  {name}")
        return

    env = SimEnv(scene=args.scene, robot=args.robot)
    model, data = env.model, env.data
    nu = model.nu
    dt = float(model.opt.timestep)
    robot = env.robot

    _init_pose(model, data, robot)

    policy = RandomPolicy(nu=nu)
    paused = True  # 默认静止，便于观察环境

    def key_callback(keycode):
        nonlocal paused
        if keycode == 32:  # 空格
            paused = not paused

    if robot == "pal_tiago_dual":
        lookat = [0.5, -0.4, 0.6]
        distance = 5.0
    elif robot == "robot_soccer_kit":
        lookat = [0.0, 0.0, 0.4]
        distance = 2.5
    else:
        lookat = [0.55, 0, 0.5]
        distance = 2.2

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = lookat
        viewer.cam.distance = distance
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20

        print(f"仿真环境 (robot={robot}, scene: {args.scene or 'empty.xml'})")
        print("  空格: 切换 静止/随机探索")
        print("  ESC: 退出")

        while viewer.is_running():
            step_start = time.time()
            if paused:
                data.ctrl[:] = 0.0
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
            else:
                obs = env.get_obs()
                action = policy(obs)
                data.ctrl[:] = np.asarray(action, dtype=np.float64)[:nu]
                mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.time() - step_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
