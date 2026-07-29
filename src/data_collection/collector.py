#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据采集器

在仿真中执行策略，记录 (obs, action, next_obs, reward, done) 等。
"""

import numpy as np
from typing import Callable

from ..simulation import SimEnv


class DataCollector:
    """采集仿真轨迹数据"""

    def __init__(self, env: SimEnv, policy: Callable[[dict], np.ndarray]):
        """
        Args:
            env: 仿真环境
            policy: 策略函数 obs -> action
        """
        self.env = env
        self.policy = policy

    def collect_episode(
        self,
        max_steps: int = 1000,
        qpos_init: np.ndarray | None = None,
    ) -> list[dict]:
        """
        采集一条轨迹

        Args:
            max_steps: 最大步数
            qpos_init: 初始关节位置

        Returns:
            transitions: [{"obs", "action", "next_obs", "reward", "done"}, ...]
        """
        obs = self.env.reset(qpos_init)
        transitions = []
        for _ in range(max_steps):
            action = self.policy(obs)
            next_obs, reward, done, info = self.env.step(action)
            transitions.append({
                "obs": obs.copy(),
                "action": np.array(action, dtype=np.float64),
                "next_obs": next_obs.copy(),
                "reward": reward,
                "done": done,
            })
            obs = next_obs
            if done:
                break
        return transitions

    def collect(
        self,
        num_episodes: int = 10,
        max_steps_per_episode: int = 1000,
    ) -> list[list[dict]]:
        """
        采集多条轨迹

        Returns:
            episodes: 每条轨迹的 transitions 列表
        """
        episodes = []
        for _ in range(num_episodes):
            ep = self.collect_episode(max_steps=max_steps_per_episode)
            episodes.append(ep)
        return episodes
