#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轨迹回放脚本

用法:
    python scripts/replay_trajectory.py data/demo.h5
    python scripts/replay_trajectory.py data/demo.h5 --episode 2 --slowdown 2
    python scripts/replay_trajectory.py data/demo.h5 --scene mjx_single_cube.xml
"""

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_collection import TrajectoryManager


def main():
    parser = argparse.ArgumentParser(description="回放采集的轨迹")
    parser.add_argument("path", type=str, help="轨迹文件路径 (HDF5)")
    parser.add_argument("--scene", "-s", type=str, default=None, help="回放时使用的场景")
    parser.add_argument("--episode", type=int, default=0, help="轨迹索引")
    parser.add_argument("--slowdown", type=float, default=1.0, help="回放减速倍数")
    args = parser.parse_args()

    mgr = TrajectoryManager(scene=args.scene)
    mgr.load(args.path)
    mgr.replay(episode_idx=args.episode, slowdown=args.slowdown)


if __name__ == "__main__":
    main()
