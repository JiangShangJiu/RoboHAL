#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 Panda(nohand) + Allegro(right) 重新生成 panda_allegro.xml。"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROBOTS = HERE.parent
ARM_XML = ROBOTS / "panda" / "panda_nohand.xml"
HAND_XML = ROBOTS / "allegro" / "right_hand.xml"
OUT_XML = HERE / "panda_allegro.xml"
MESH_DIR = HERE / "assets"

# menagerie 独立场景里 palm 的展示姿态；挂法兰前必须清掉
_PALM_DISPLAY_QUAT = 'quat="0 1 0 1"'

# Allegro base_link 约在 palm 局部 z∈[-0.07, 0.06]；
# 沿 attachment +Z 至少外移 ~0.07，背面才离开法兰，再留一点间隙。
_PALM_POS = (0.0, 0.0, 0.09)
_PALM_QUAT = (1.0, 0.0, 0.0, 0.0)


def main() -> None:
    if not ARM_XML.exists():
        raise FileNotFoundError(ARM_XML)
    if not HAND_XML.exists():
        raise FileNotFoundError(HAND_XML)

    MESH_DIR.mkdir(parents=True, exist_ok=True)
    for src in (ROBOTS / "panda" / "assets", ROBOTS / "allegro" / "assets"):
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, MESH_DIR / f.name)

    arm = mujoco.MjSpec.from_file(str(ARM_XML.resolve()))

    hand_xml = HAND_XML.read_text(encoding="utf-8")
    mesh_abs = str((ROBOTS / "allegro" / "assets").resolve())
    hand_xml = hand_xml.replace('meshdir="assets"', f'meshdir="{mesh_abs}"', 1)
    if _PALM_DISPLAY_QUAT not in hand_xml:
        raise RuntimeError("未找到 palm 展示 quat，请检查 allegro/right_hand.xml")
    pos_s = " ".join(str(x) for x in _PALM_POS)
    quat_s = " ".join(str(x) for x in _PALM_QUAT)
    hand_xml = hand_xml.replace(
        f'<body name="palm" {_PALM_DISPLAY_QUAT}',
        f'<body name="palm" pos="{pos_s}" quat="{quat_s}"',
        1,
    )
    hand = mujoco.MjSpec.from_string(hand_xml)

    palm = hand.body("palm")
    print(f"palm before attach: pos={list(palm.pos)} quat={list(palm.quat)}")

    site = next(s for s in arm.sites if s.name == "attachment_site")
    arm.attach(hand, prefix="allegro/", site=site)
    model = arm.compile()

    xml = arm.to_xml()
    xml = xml.replace('<mujoco model="panda nohand">', '<mujoco model="panda_allegro">', 1)
    xml = xml.replace('meshdir="assets/"', 'meshdir="assets"', 1)
    xml = re.sub(r'meshdir="[^"]*allegro/assets"', 'meshdir="assets"', xml)

    # 腕部与手掌/指根接触排除，避免残余间隙抖动
    excludes = [
        ("link7", "allegro/palm"),
        ("link6", "allegro/palm"),
        ("link7", "allegro/ff_base"),
        ("link7", "allegro/mf_base"),
        ("link7", "allegro/rf_base"),
        ("link7", "allegro/th_base"),
    ]
    contact_extra = "\n".join(
        f'    <exclude body1="{a}" body2="{b}"/>' for a, b in excludes
    )
    if "</contact>" in xml:
        xml = xml.replace("</contact>", contact_extra + "\n  </contact>", 1)
    else:
        xml = xml.replace(
            "</mujoco>",
            f"  <contact>\n{contact_extra}\n  </contact>\n</mujoco>",
            1,
        )

    arm_q = "0 0 0 -1.57079 0 1.57079 -0.7853"
    hand_q = " ".join(["0"] * (model.nq - 7))
    qpos = f"{arm_q} {hand_q}"
    new_key = f'<key name="home" qpos="{qpos}" ctrl="{qpos}"/>'
    xml, n = re.subn(r'<key name="home"[^/]*/>', new_key, xml, count=1)
    if n != 1:
        raise RuntimeError("未能更新 home keyframe")

    OUT_XML.write_text(xml, encoding="utf-8")
    check = mujoco.MjModel.from_xml_path(str(OUT_XML))
    palm_line = re.search(r'<body name="allegro/palm"[^>]*>', xml)
    print(f"wrote {OUT_XML}  nq={check.nq} nu={check.nu}")
    print(f"palm tag: {palm_line.group(0) if palm_line else None}")

    d = mujoco.MjData(check)
    d.qpos[:7] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
    mujoco.mj_forward(check, d)
    pid = mujoco.mj_name2id(check, mujoco.mjtObj.mjOBJ_BODY, "allegro/palm")
    l7 = mujoco.mj_name2id(check, mujoco.mjtObj.mjOBJ_BODY, "link7")
    tips = np.mean(
        [
            d.xpos[mujoco.mj_name2id(check, mujoco.mjtObj.mjOBJ_BODY, n)]
            for n in ("allegro/ff_tip", "allegro/mf_tip", "allegro/rf_tip")
        ],
        axis=0,
    )
    print("link7", np.round(d.xpos[l7], 3))
    print("palm ", np.round(d.xpos[pid], 3))
    print("palm->tips", np.round(tips - d.xpos[pid], 3))
    print("|palm-link7|", round(float(np.linalg.norm(d.xpos[pid] - d.xpos[l7])), 3))


if __name__ == "__main__":
    main()
