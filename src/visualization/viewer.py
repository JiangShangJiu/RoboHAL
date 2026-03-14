#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MuJoCo viewer 封装

提供统一的被动/主动 viewer 接口。
"""

import numpy as np


def launch_viewer(model, data, sync_callback=None):
    """
    启动 MuJoCo 被动 viewer

    Args:
        model: mujoco.MjModel
        data: mujoco.MjData
        sync_callback: 每帧回调 (model, data) -> None，用于更新 data
    """
    import mujoco
    import mujoco.viewer  # mujoco 3.x 需显式导入
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if sync_callback:
                sync_callback(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()
