#!/usr/bin/env python3
"""Install a locally supplied Cruzr S2 MJCF/URDF and its referenced assets.

The upstream USD submodule is not anonymously accessible. This script deliberately
does not substitute another robot or claim to convert arbitrary Isaac Sim USDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET


UPSTREAM = "https://github.com/fiveages-sim/ubtech-usds"
UPSTREAM_REVISION = "c8219fde6affb8ed85de45faf422565b0a50bee4"
PROJECT_ASSETS = Path(__file__).resolve().parents[1] / "assets"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_reference(reference: str, base: Path, packages: dict[str, Path]) -> Path:
    if reference.startswith("package://"):
        package, _, relative = reference[len("package://") :].partition("/")
        if package not in packages:
            raise ValueError(
                f"Unresolved ROS package {package!r}; pass --package {package}=/path/to/package"
            )
        path = packages[package] / relative
    elif "://" in reference:
        raise ValueError(f"Only local files and explicit package:// mappings are supported: {reference}")
    else:
        path = base / reference
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Missing model dependency: {path}")
    if path.stat().st_size < 1024 and path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(f"Git LFS pointer instead of model data: {path}; fetch the LFS assets first")
    return path


def _load_spec(source: Path, packages: dict[str, Path]):
    import mujoco

    if source.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}:
        raise ValueError(
            "USD import is not implemented by this script. Supply a verified MJCF or URDF "
            "with meshes and license; see assets/CRUZR_S2.md."
        )
    root = ET.parse(source).getroot()
    if root.tag == "mujoco":
        spec = mujoco.MjSpec.from_file(str(source))
    elif root.tag == "robot":
        for joint in root.findall("joint"):
            if joint.find("mimic") is not None or joint.get("type") in {"floating", "planar"}:
                raise ValueError(
                    f"URDF joint {joint.get('name')!r} uses mimic/floating/planar semantics. "
                    "Convert and validate these constraints in MJCF before importing."
                )
        for element in root.findall(".//mesh") + root.findall(".//texture"):
            if "filename" in element.attrib:
                element.set("filename", str(_resolve_reference(element.get("filename", ""), source.parent, packages)))
        extension = root.find("mujoco")
        if extension is None:
            extension = ET.SubElement(root, "mujoco")
        compiler = extension.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(extension, "compiler")
        # Preserve visual geometry, file paths and fixed link names for hardware frames.
        compiler.set("strippath", "false")
        compiler.set("discardvisual", "false")
        compiler.set("fusestatic", "false")
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)
        compiler.attrib.pop("assetdir", None)
        spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    else:
        raise ValueError(f"Expected MJCF <mujoco> or URDF <robot>, got <{root.tag}>")
    spec.compile()
    return spec, root.tag


def _package_model(spec, source: Path, destination: Path) -> list[dict[str, str]]:
    """Flatten includes using MuJoCo, then copy files to collision-free local names."""
    root = ET.fromstring(spec.to_xml())
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    source_base = Path(spec.modelfiledir) if spec.modelfiledir else source.parent
    directories = {
        "mesh": Path(spec.meshdir),
        "texture": Path(spec.texturedir),
    }
    dependencies = []
    for element in root.iter():
        for attribute, reference in list(element.attrib.items()):
            if attribute != "file" and not (element.tag == "texture" and attribute.startswith("file")):
                continue
            if element.tag not in {"mesh", "texture", "hfield"}:
                raise ValueError(f"Unsupported external <{element.tag}> dependency; flatten it to MJCF first")
            base = source_base / directories.get(element.tag, Path())
            path = _resolve_reference(reference, base, {})
            digest = _digest(path)
            relative = Path("assets") / f"{digest[:16]}_{path.name}"
            target = destination / relative
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
            element.set(attribute, relative.as_posix())
            dependencies.append({"source_file": path.name, "installed_file": relative.as_posix(), "sha256": digest})
    for attribute in ("meshdir", "texturedir", "assetdir"):
        compiler.attrib.pop(attribute, None)
    compiler.set("strippath", "false")
    ET.indent(root)
    ET.ElementTree(root).write(destination / "cruzr_s2.xml", encoding="utf-8", xml_declaration=True)
    return dependencies


def import_model(source: Path, licenses: list[Path], assets_dir: Path, packages: dict[str, Path]) -> Path:
    import mujoco
    import numpy as np

    source = _resolve_reference(str(source.resolve()), source.parent, {})
    for license_path in licenses:
        if not license_path.is_file() or not license_path.read_bytes().strip():
            raise ValueError(f"License/notice file is missing or empty: {license_path}")
    if not licenses:
        raise ValueError("Supply the actual model license with --license; the parent repository license is insufficient")
    robot_dir = assets_dir / "robots" / "cruzr_s2"
    scene_dir = assets_dir / "scenes" / "cruzr_s2"
    if robot_dir.exists() or scene_dir.exists():
        raise ValueError(f"Refusing to overwrite an existing Cruzr S2 installation under {assets_dir}")

    spec, source_format = _load_spec(source, packages)
    # Stage on the destination filesystem and publish only after validation succeeds.
    assets_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cruzr-import-", dir=assets_dir) as staging:
        staged_robot = Path(staging) / "robot"
        staged_robot.mkdir()
        dependencies = _package_model(spec, source, staged_robot)
        model_path = staged_robot / "cruzr_s2.xml"
        model = mujoco.MjModel.from_xml_path(str(model_path))
        scalar_joints = [
            index for index in range(model.njnt)
            if model.jnt_type[index] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ]
        if not scalar_joints:
            raise ValueError("The model has no hinge/slide joints to expose as hardware")
        joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in scalar_joints]
        if any(name is None for name in joint_names):
            raise ValueError("All hinge/slide joints must have names for the hardware interface")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        if not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc)):
            raise ValueError("The imported model has a non-finite initial state")

        notices = []
        for index, path in enumerate(licenses):
            relative = f"LICENSE_{index + 1}_{path.name}"
            shutil.copy2(path, staged_robot / relative)
            notices.append({"file": relative, "sha256": _digest(path)})
        metadata = {
            "robot": "cruzr_s2",
            "identity": "User-supplied model; this importer cannot verify manufacturer identity or physical calibration",
            "upstream_reference": UPSTREAM,
            "upstream_expected_revision": UPSTREAM_REVISION,
            "source_file": source.name,
            "source_sha256": _digest(source),
            "source_format": "URDF" if source_format == "robot" else "MJCF",
            "mujoco_version": mujoco.__version__,
            "licenses": notices,
            "dependencies": dependencies,
            "joints": joint_names,
            "validation": "Compiled packaged MJCF and checked finite initial forward dynamics; no hardware calibration inferred",
            "urdf_note": "URDF roots are fixed unless the source MJCF defines a mobile base. URDF drive limits, sensors and vendor interfaces require separate configuration.",
        }
        (staged_robot / "provenance.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staged_scene = Path(staging) / "scene"
        staged_scene.mkdir()
        (staged_scene / "empty.xml").write_text(
            '<mujoco model="cruzr_s2 scene">\n'
            '  <include file="../../robots/cruzr_s2/cruzr_s2.xml"/>\n'
            '</mujoco>\n', encoding="utf-8",
        )
        robot_dir.parent.mkdir(parents=True, exist_ok=True)
        scene_dir.parent.mkdir(parents=True, exist_ok=True)
        # Prevent a concurrent importer from overwriting an already-published model.
        if robot_dir.exists() or scene_dir.exists():
            raise ValueError("A Cruzr S2 installation appeared during import; refusing to overwrite it")
        staged_robot.rename(robot_dir)
        try:
            staged_scene.rename(scene_dir)
        except OSError:
            shutil.rmtree(robot_dir)
            raise
    return scene_dir / "empty.xml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local, verified Cruzr S2 MJCF or URDF")
    parser.add_argument("--license", type=Path, action="append", required=True, help="Model license/notice; repeat for multiple files")
    parser.add_argument("--package", action="append", default=[], metavar="NAME=PATH", help="Resolve a URDF package:// prefix to a local ROS package")
    parser.add_argument("--assets-dir", type=Path, default=PROJECT_ASSETS, help="Destination assets directory")
    args = parser.parse_args()
    try:
        packages = {}
        for mapping in args.package:
            name, separator, path = mapping.partition("=")
            if not separator or not name or not path:
                raise ValueError(f"Invalid --package mapping: {mapping!r}; use NAME=/path/to/package")
            packages[name] = Path(path).expanduser().resolve()
        scene = import_model(args.source.expanduser(), [p.expanduser() for p in args.license], args.assets_dir.expanduser().resolve(), packages)
    except Exception as error:
        print(f"Cruzr S2 import failed: {error}", file=sys.stderr)
        return 1
    print(f"Installed verified-loadable model: {scene}")
    print("Review provenance.json and configure drive limits/base mobility before physical validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
