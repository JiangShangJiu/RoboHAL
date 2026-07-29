#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MuJoCo 模型 / 场景加载器。

  assets/robots/<robot>/     机器人本体（XML + mesh）
  assets/scenes/<robot>/     该机器人可用场景
"""

from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assets_root() -> Path:
    return _project_root() / "assets"


def default_robot() -> str:
    return os.environ.get("ROBOARENA_ROBOT", "panda")


def find_robot_root(robot: str | None = None) -> Path:
    """查找 assets/robots/<robot>。"""
    name = robot or default_robot()
    root = _assets_root() / "robots" / name
    if root.is_dir():
        return root.resolve()
    raise FileNotFoundError(
        f"未找到机器人目录: assets/robots/{name}。"
        f"可用: {list_available_robots()}"
    )


def list_available_robots() -> list[str]:
    robots_dir = _assets_root() / "robots"
    if not robots_dir.is_dir():
        return []
    return sorted(p.name for p in robots_dir.iterdir() if p.is_dir())


def list_available_scenes(robot: str | None = None) -> list[tuple[str, Path]]:
    """返回 [(panda/empty.xml, Path), ...]。"""
    name = robot or default_robot()
    scenes: list[tuple[str, Path]] = []
    scenes_dir = _assets_root() / "scenes" / name
    if scenes_dir.is_dir():
        for p in sorted(scenes_dir.glob("*.xml")):
            scenes.append((f"{name}/{p.name}", p.resolve()))
    return scenes


def resolve_scene(
    scene: str | None = None,
    robot: str | None = None,
) -> Path:
    """
    解析场景路径。

    scene 支持: empty.xml / kitchen_lite.xml / panda/empty.xml / 绝对路径
    """
    name = robot or default_robot()
    scene = scene or os.environ.get("ROBOARENA_SCENE", "empty.xml")

    p = Path(scene)
    if p.is_absolute() and p.exists():
        return p.resolve()

    rel = scene.removeprefix("scenes/")
    candidates: list[Path] = [
        _assets_root() / "scenes" / rel,
        _assets_root() / "scenes" / name / Path(rel).name,
        _assets_root() / "scenes" / name / scene,
    ]
    if not p.is_absolute():
        candidates.append((_project_root() / scene).resolve())
        candidates.append(Path(scene).resolve())

    for c in candidates:
        if c.exists():
            return c.resolve()

    for disp, path in list_available_scenes(name):
        if (
            disp == scene
            or disp == rel
            or disp.endswith(f"/{scene}")
            or Path(disp).name == Path(scene).name
        ):
            return path

    available = [n for n, _ in list_available_scenes(name)]
    raise FileNotFoundError(
        f"场景不存在: {scene}（robot={name}）。可用: {available}"
    )


def load_robot(
    scene: str | None = None,
    robot: str | None = None,
):
    """
    加载场景，返回 (MjModel, MjData)。

    MuJoCo 跨目录 include 时 meshdir 不可靠，故在临时目录合并 XML。
    """
    import re
    import tempfile

    import mujoco

    name = robot or default_robot()
    xml_path = resolve_scene(scene, robot=name)
    robot_root = find_robot_root(name)
    mesh_abs = str((robot_root / "assets").resolve())
    include_prefix = f"../../robots/{name}/"

    with tempfile.TemporaryDirectory(prefix="roboarena_mjcf_") as td:
        td_path = Path(td)

        for src in robot_root.glob("*.xml"):
            text = src.read_text(encoding="utf-8")
            for old in ('meshdir="assets"', 'meshdir="./assets/"', 'meshdir="assets/"'):
                text = text.replace(old, f'meshdir="{mesh_abs}"')
            (td_path / src.name).write_text(text, encoding="utf-8")

        scenes_dir = xml_path.parent
        for src in scenes_dir.glob("*.xml"):
            text = src.read_text(encoding="utf-8")
            text = text.replace(include_prefix, "")
            text = re.sub(
                rf'file="(?:\.\./)*robots/{re.escape(name)}/([^"]+)"',
                r'file="\1"',
                text,
            )
            (td_path / src.name).write_text(text, encoding="utf-8")

        model = mujoco.MjModel.from_xml_path(str(td_path / xml_path.name))
        data = mujoco.MjData(model)
        return model, data


# Panda 默认初始关节位置（7 臂 + 2 夹爪）
DEFAULT_Q0 = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785, 0.04, 0.04]

GRIPPER_CTRL_SCALE = 255.0 / 0.04


def qpos_to_ctrl(qpos: list[float], nu: int) -> list[float]:
    """qpos → ctrl。Panda(nu==8) 时夹爪映射到 0–255；其余按前 nu 维对齐。"""
    if nu == 8:
        ctrl = list(qpos[:7])
        if len(qpos) >= 8:
            ctrl.append(qpos[7] * GRIPPER_CTRL_SCALE)
        else:
            ctrl.append(0.0)
        return ctrl[:nu]

    q = [float(x) for x in qpos[:nu]]
    if len(q) < nu:
        q.extend([0.0] * (nu - len(q)))
    return q
