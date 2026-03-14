#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MuJoCo Panda 模型加载器

模型路径优先顺序：
1. 环境变量 ROBOARENA_PANDA_PATH
2. 项目 assets/franka_emika_panda

场景切换：
- 默认 scene.xml，可通过 ROBOARENA_SCENE 或 --scene 指定
- 支持 assets/franka_emika_panda/*.xml 及 assets/scenes/*.xml
"""

import os
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_model_root() -> Path:
    """查找 Franka Panda 模型目录"""
    project_root = _project_root()
    candidates = []
    if os.environ.get("ROBOARENA_PANDA_PATH"):
        candidates.append(Path(os.environ["ROBOARENA_PANDA_PATH"]))
    candidates.append(project_root / "assets" / "franka_emika_panda")
    for root in candidates:
        if root.is_dir():
            return root
    raise FileNotFoundError(
        "未找到 Franka Panda 模型目录。"
        "请确保 assets/franka_emika_panda 存在，或设置环境变量 ROBOARENA_PANDA_PATH"
    )


def list_available_scenes() -> list[tuple[str, Path]]:
    """
    列出可用场景

    Returns:
        [(显示名, 绝对路径), ...]
        包含 assets/franka_emika_panda/*.xml 和 assets/scenes/*.xml
    """
    project_root = _project_root()
    scenes: list[tuple[str, Path]] = []

    # 模型目录下的 xml
    try:
        model_root = find_model_root()
        for p in sorted(model_root.glob("*.xml")):
            scenes.append((p.name, p.resolve()))
    except FileNotFoundError:
        pass

    # assets/scenes/ 下的自定义场景
    scenes_dir = project_root / "assets" / "scenes"
    if scenes_dir.is_dir():
        for p in sorted(scenes_dir.glob("*.xml")):
            scenes.append((f"scenes/{p.name}", p.resolve()))

    return scenes


def resolve_scene(scene: str | None = None) -> Path:
    """
    解析场景路径

    Args:
        scene: 场景名或路径。None 则用 ROBOARENA_SCENE 或 "scene.xml"

    Returns:
        场景文件的绝对路径
    """
    scene = scene or os.environ.get("ROBOARENA_SCENE", "scene.xml")
    project_root = _project_root()

    # 绝对路径
    p = Path(scene)
    if p.is_absolute() and p.exists():
        return p

    # 相对路径：先查 model_root，再查 assets/scenes
    model_root = find_model_root()
    candidates = [
        model_root / scene,
        model_root / scene.split("/")[-1],  # scenes/xxx.xml -> xxx.xml in model_root
        project_root / "assets" / "scenes" / scene.split("/")[-1],
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    # 按显示名匹配 list_available_scenes
    for name, path in list_available_scenes():
        if name == scene or name.endswith(f"/{scene}") or name == f"scenes/{scene}":
            return path

    raise FileNotFoundError(f"场景不存在: {scene}。可用: {[n for n, _ in list_available_scenes()]}")


def load_panda(
    model_file: str | None = None,
    model_root: Path | str | None = None,
):
    """
    加载 Franka Panda MuJoCo 模型

    Args:
        model_file: 场景名或路径，如 scene.xml / scenes/table.xml。None 用默认
        model_root: 模型根目录，None 则自动查找（仅当 model_file 为相对文件名时使用）

    Returns:
        model: mujoco.MjModel
        data: mujoco.MjData
    """
    import mujoco

    if model_file is None:
        model_file = os.environ.get("ROBOARENA_SCENE", "scene.xml")

    if model_root is not None:
        root = Path(model_root)
        xml_path = root / model_file
        if not xml_path.exists():
            xml_path = root / model_file.split("/")[-1]
    else:
        xml_path = resolve_scene(model_file)

    if not xml_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


# Panda 默认初始关节位置（7 臂 + 2 夹爪）
DEFAULT_Q0 = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785, 0.04, 0.04]

# Panda 夹爪 ctrl 映射：关节位置 0~0.04 对应 ctrl 0~255
GRIPPER_CTRL_SCALE = 255.0 / 0.04


def qpos_to_ctrl(qpos: list[float], nu: int) -> list[float]:
    """将 qpos 转为 Panda 的 ctrl：前 7 个直接对应，第 8 个（夹爪）需映射到 0-255"""
    ctrl = list(qpos[:7])
    if nu >= 8 and len(qpos) >= 8:
        ctrl.append(qpos[7] * GRIPPER_CTRL_SCALE)
    return ctrl[:nu]
