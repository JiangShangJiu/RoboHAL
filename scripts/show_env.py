#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
展示数据采集仿真环境

启动 Panda MuJoCo 仿真，显示数据采集时使用的场景。
可按空格键切换：保持静止 / 随机探索。

用法:
    python scripts/show_env.py
    python scripts/show_env.py --scene mjx_single_cube.xml
    python scripts/show_env.py --list-scenes
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer

from src.simulation import PandaEnv, list_available_scenes
from src.simulation.loader import DEFAULT_Q0, qpos_to_ctrl
from src.data_collection.policies import RandomPolicy


def main():
    parser = argparse.ArgumentParser(description="展示仿真环境")
    parser.add_argument("--scene", "-s", type=str, default=None, help="场景名，如 scene.xml / mjx_single_cube.xml")
    parser.add_argument("--list-scenes", action="store_true", help="列出可用场景后退出")
    args = parser.parse_args()

    if args.list_scenes:
        scenes = list_available_scenes()
        print("可用场景:")
        for name, path in scenes:
            print(f"  {name}")
        return

    env = PandaEnv(scene=args.scene)
    model, data = env.model, env.data
    nu = model.nu
    dt = float(model.opt.timestep)

    # 初始位姿
    nq = min(model.nq, len(DEFAULT_Q0))
    data.qpos[:nq] = [DEFAULT_Q0[i] for i in range(nq)]
    data.ctrl[:] = qpos_to_ctrl(DEFAULT_Q0, nu)
    mujoco.mj_forward(model, data)

    policy = RandomPolicy(nu=nu)
    paused = True  # 默认静止，便于观察环境

    def key_callback(keycode):
        nonlocal paused
        if keycode == 32:  # 空格
            paused = not paused

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0.3, 0, 0.4]
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20

        print(f"数据采集仿真环境 (scene: {args.scene or 'scene.xml'})")
        print("  空格: 切换 静止/随机探索")
        print("  ESC: 退出")

        while viewer.is_running():
            step_start = time.time()
            if not paused:
                obs = env.get_obs()
                action = policy(obs)
                data.ctrl[:] = action
            mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.time() - step_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
