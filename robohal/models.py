"""Load MJCF scenes without depending on the old simulation package.

Assets live in ``assets/{robots,scenes}`` in a source checkout. Set
``ROBOHAL_ASSETS`` to use an external assets directory. Include files and
models use native MuJoCo path semantics first. For legacy bundled models,
a fallback resolves asset directories relative to their declaring document.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

if TYPE_CHECKING:
    import mujoco


def _assets_root() -> Path:
    configured = os.environ.get("ROBOHAL_ASSETS")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"ROBOHAL_ASSETS is not a directory: {root}")
        return root
    return Path(__file__).resolve().parents[1] / "assets"


def _robot_name(robot: str) -> str:
    if not robot or robot in {".", ".."} or "/" in robot or "\\" in robot:
        raise ValueError(f"Invalid robot name: {robot!r}")
    return robot


def available_robots() -> list[str]:
    """Return robot names with at least one registered XML scene.

    This only inspects filenames, so listing assets needs no rendering context
    or model compilation. Standalone components without a scene are excluded.
    """
    scenes = _assets_root() / "scenes"
    if not scenes.is_dir():
        return []
    return sorted(
        path.name
        for path in scenes.iterdir()
        if path.is_dir() and any(scene.is_file() for scene in path.glob("*.xml"))
    )


def available_scenes(robot: str) -> list[str]:
    """Return scene filenames for a robot, such as ``empty.xml``."""
    root = _assets_root() / "scenes" / _robot_name(robot)
    return sorted(path.name for path in root.glob("*.xml") if path.is_file())


def load_robot(robot: str = "panda", scene: str = "empty.xml") -> mujoco.MjModel:
    """Compile a scene from the configured asset collection."""
    root = (_assets_root() / "scenes" / _robot_name(robot)).resolve()
    relative = Path(scene)
    # Also accept the labels used by the previous asset browser.
    if relative.parts[:1] == ("scenes",):
        relative = Path(*relative.parts[1:])
    if relative.parts[:1] == (robot,):
        relative = Path(*relative.parts[1:])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Scene must be inside its robot directory; use load_mjcf for other paths")
    if not path.is_file():
        raise FileNotFoundError(
            f"Scene {scene!r} for robot {robot!r} not found. "
            f"Available scenes: {available_scenes(robot)}; robots: {available_robots()}"
        )
    return load_mjcf(path)


def load_mjcf(path: str | os.PathLike[str]) -> mujoco.MjModel:
    """Compile MJCF, resolving nested includes and local asset directories.

    Native MuJoCo loading is attempted first to preserve standard compiler
    semantics. A legacy compatibility pass follows only if native loading
    fails, allowing included robot files to declare their own asset directory.
    XML copies exist only for the duration of compilation. Meshes and textures
    are read in place, and no source asset is modified. Duplicate basenames in
    different include directories do not collide.
    """
    import mujoco

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MJCF file not found: {source}")

    try:
        return mujoco.MjModel.from_xml_path(str(source))
    except ValueError:
        # Older assets use per-document meshdir conventions that the native
        # include reader cannot consistently resolve. Valid models never pass
        # through this compatibility layer.
        pass

    with tempfile.TemporaryDirectory(prefix="robohal_mjcf_") as directory:
        destination = Path(directory)
        counter = 0

        def rewrite(
            document: Path,
            inherited: dict[str, str],
            ancestors: tuple[Path, ...],
        ) -> Path:
            nonlocal counter
            document = document.resolve()
            if document in ancestors:
                chain = " -> ".join(str(item) for item in (*ancestors, document))
                raise ValueError(f"Circular MJCF include: {chain}")
            if not document.is_file():
                raise FileNotFoundError(f"MJCF include not found: {document}")
            tree = ET.parse(document)
            root = tree.getroot()
            settings = inherited.copy()
            for compiler in root.findall("compiler"):
                # assetdir is shorthand; explicit type-specific directories win.
                if "assetdir" in compiler.attrib:
                    assetdir = str((document.parent / compiler.attrib["assetdir"]).resolve())
                    settings.update(meshdir=assetdir, texturedir=assetdir)
                for key in ("meshdir", "texturedir"):
                    if key in compiler.attrib:
                        settings[key] = str((document.parent / compiler.attrib[key]).resolve())
                if "strippath" in compiler.attrib:
                    settings["strippath"] = compiler.attrib["strippath"]
                for key in ("assetdir", "meshdir", "texturedir"):
                    compiler.attrib.pop(key, None)
                # All rewritten paths are absolute and must retain directories.
                if "strippath" in compiler.attrib:
                    compiler.set("strippath", "false")

            for element in root.iter():
                if element.tag == "include":
                    include = element.get("file")
                    if not include:
                        raise ValueError(f"MJCF include has no file in {document}")
                    target = (document.parent / include).resolve()
                    normalized = rewrite(target, settings, (*ancestors, document))
                    element.set("file", str(normalized))
                elif element.tag in {"mesh", "skin", "texture", "hfield", "model"}:
                    directory_key = "texturedir" if element.tag == "texture" else "meshdir"
                    # model references have their own MJCF-relative paths.
                    base = document.parent if element.tag == "model" else Path(
                        settings.get(directory_key, str(document.parent))
                    )
                    attributes = ["file"]
                    if element.tag == "texture":
                        attributes.extend("file" + side for side in ("left", "right", "up", "down", "front", "back"))
                    for attribute in attributes:
                        filename = element.get(attribute)
                        if not filename:
                            continue
                        if settings.get("strippath") == "true":
                            filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
                        target = (base / filename).resolve()
                        if element.tag == "model" and target.suffix.lower() == ".xml":
                            if "name" not in element.attrib:
                                element.set("name", target.stem)
                            target = rewrite(target, {}, (*ancestors, document))
                        element.set(attribute, str(target))

            counter += 1
            output = destination / f"{counter}_{document.name}"
            tree.write(output, encoding="utf-8", xml_declaration=True)
            return output

        normalized = rewrite(source, {}, ())
        return mujoco.MjModel.from_xml_path(str(normalized))
