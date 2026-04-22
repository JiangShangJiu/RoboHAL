#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 OMPL 的关节空间几何规划（无 ROS）。

需在环境中安装可选依赖：pip install 'ompl>=1.6' 或 pip install -e ".[planning]"
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .limits import default_panda_arm_bounds
from .protocols import Planner, ValidityFn

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError as e:  # pragma: no cover
    ob = None  # type: ignore[misc, assignment]
    og = None  # type: ignore[misc, assignment]
    _OMPL_IMPORT_ERROR = e
else:
    _OMPL_IMPORT_ERROR = None


def _require_ompl() -> None:
    if _OMPL_IMPORT_ERROR is not None:
        raise ImportError(
            "未安装 OMPL Python 绑定。请执行: pip install 'ompl>=1.6' "
            "或 pip install -e \".[planning]\""
        ) from _OMPL_IMPORT_ERROR


def _make_planner(si, name: str):
    _require_ompl()
    planners = {
        "RRTConnect": og.RRTConnect,
        "RRTstar": og.RRTstar,
        "BITstar": og.BITstar,
        "PRM": og.PRM,
        "LBKPIECE1": og.LBKPIECE1,
        "EST": og.EST,
    }
    cls = planners.get(name)
    if cls is None:
        raise ValueError(
            f"未知规划器: {name}。可选: {', '.join(sorted(planners))}"
        )
    return cls(si)


class OMPLJointPlanner(Planner):
    """
    在 RealVectorStateSpace 上调用 OMPL 的关节空间规划（任意维数，由 joint_low/high 决定）。

    - 仅实现 `plan_joint`；`pose_dim == 0`，`plan_cartesian` 未实现。
    - validity: 可选；为 None 时仅检查关节上下界（与 OMPL bounds 一致）。
    - 后续若要把 MuJoCo 碰撞接入，传入 validity 即可，无需改 OMPL 本体。
    - Panda 默认限位见 PandaOMPLJointPlanner。
    """

    def __init__(
        self,
        joint_low: np.ndarray,
        joint_high: np.ndarray,
        *,
        validity: Optional[ValidityFn] = None,
        planner: str = "RRTConnect",
    ) -> None:
        _require_ompl()
        low = np.asarray(joint_low, dtype=np.float64).reshape(-1)
        high = np.asarray(joint_high, dtype=np.float64).reshape(-1)
        if low.shape != high.shape:
            raise ValueError("joint_low / joint_high 长度须一致")
        dof = int(low.shape[0])
        if dof < 1:
            raise ValueError("至少 1 个关节")
        if np.any(low >= high):
            raise ValueError("关节下限须严格小于上限")

        self._dof = dof
        self._low = low
        self._high = high
        self._validity = validity
        self._planner_name = planner

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def pose_dim(self) -> int:
        return 0

    def _state_valid(self, state) -> bool:
        q = np.array([float(state[i]) for i in range(self._dof)], dtype=np.float64)
        if self._validity is not None:
            return bool(self._validity(q))
        return True

    def plan_cartesian(
        self,
        start_pose: np.ndarray,
        goal_pose: np.ndarray,
        *,
        timeout: float = 5.0,
        interpolate_steps: int = 100,
        simplify: bool = True,
    ) -> np.ndarray:
        del start_pose, goal_pose, timeout, interpolate_steps, simplify
        raise NotImplementedError(
            "OMPLJointPlanner 仅支持关节空间，请使用 plan_joint(start, goal, ...)。"
        )

    def plan_joint(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        *,
        timeout: float = 5.0,
        interpolate_steps: int = 100,
        simplify: bool = True,
    ) -> np.ndarray:
        _require_ompl()
        start = np.asarray(start, dtype=np.float64).reshape(-1)
        goal = np.asarray(goal, dtype=np.float64).reshape(-1)
        if start.shape != (self._dof,) or goal.shape != (self._dof,):
            raise ValueError(f"start/goal 须为 ({self._dof},) 向量")

        space = ob.RealVectorStateSpace(self._dof)
        bounds = ob.RealVectorBounds(self._dof)
        for i in range(self._dof):
            bounds.setLow(i, float(self._low[i]))
            bounds.setHigh(i, float(self._high[i]))
        space.setBounds(bounds)

        ss = og.SimpleSetup(space)
        ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(self._state_valid)
        )

        s0 = ob.State(space)
        g0 = ob.State(space)
        for i in range(self._dof):
            s0[i] = float(start[i])
            g0[i] = float(goal[i])
        ss.setStartAndGoalStates(s0, g0)

        ss.setPlanner(_make_planner(ss.getSpaceInformation(), self._planner_name))

        solved = ss.solve(float(timeout))
        if not solved:
            raise RuntimeError(
                f"OMPL 在 {timeout}s 内未找到可行路径（规划器={self._planner_name}）"
            )

        if simplify:
            ss.simplifySolution()

        path = ss.getSolutionPath()
        n_interp = max(2, int(interpolate_steps))
        path.interpolate(n_interp)

        out = np.zeros((path.getStateCount(), self._dof), dtype=np.float64)
        for t in range(path.getStateCount()):
            st = path.getState(t)
            for i in range(self._dof):
                out[t, i] = float(st[i])
        return out


class PandaOMPLJointPlanner(OMPLJointPlanner):
    """Panda 7-DOF 臂：默认关节界为 limits.default_panda_arm_bounds。"""

    def __init__(
        self,
        *,
        validity: Optional[ValidityFn] = None,
        planner: str = "RRTConnect",
    ) -> None:
        low, high = default_panda_arm_bounds()
        super().__init__(low, high, validity=validity, planner=planner)


def arm7_to_mujoco_qpos(
    arm7: np.ndarray,
    finger: float = 0.04,
) -> np.ndarray:
    """
    将 7 维臂关节扩展为 MuJoCo Panda 常见 9 维 qpos（末 2 维为夹爪开合）。
    """
    q = np.asarray(arm7, dtype=np.float64).reshape(-1)
    if q.shape[0] != 7:
        raise ValueError("arm7 须为长度 7")
    return np.concatenate([q, np.full(2, finger, dtype=np.float64)])
