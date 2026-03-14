#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据采集入口

用法:
    python scripts/collect_data.py --episodes 10 --steps 500 --output data/demo.h5
    python scripts/collect_data.py --scene mjx_single_cube.xml
    python scripts/collect_data.py --list-scenes
"""

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_collection import TrajectoryManager
from src.data_collection.policies import RandomPolicy
from src.simulation import list_available_scenes


def main():
    parser = argparse.ArgumentParser(description="采集 Panda 仿真轨迹数据")
    parser.add_argument("--scene", "-s", type=str, default=None, help="场景名")
    parser.add_argument("--list-scenes", action="store_true", help="列出可用场景")
    parser.add_argument("--episodes", type=int, default=5, help="采集轨迹条数")
    parser.add_argument("--steps", type=int, default=500, help="每条轨迹最大步数")
    parser.add_argument("--output", type=str, default="data/demo.h5", help="输出文件路径")
    args = parser.parse_args()

    if args.list_scenes:
        for name, path in list_available_scenes():
            print(name)
        return

    mgr = TrajectoryManager(scene=args.scene)
    policy = RandomPolicy(nu=mgr.env.nu)

    print(f"采集 {args.episodes} 条轨迹，每条最多 {args.steps} 步...")
    mgr.record(policy, num_episodes=args.episodes, max_steps_per_episode=args.steps)
    total_steps = sum(len(ep) for ep in mgr.episodes)
    print(f"共采集 {total_steps} 个 transition")

    out_path = Path(args.output)
    mgr.save(out_path, format="hdf5")
    print(f"已保存到 {out_path}")


if __name__ == "__main__":
    main()
