#!/usr/bin/env python3
"""Assemble the Windows x64 MXU release package."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKING_DIR = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from configure import configure_ocr_model  # noqa: E402


COPY_IGNORE = shutil.ignore_patterns(
    "*.pdb",
    "*.PDB",
    "*.pyc",
    "*.pyo",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
)
TOP_LEVEL_RUNTIME_DIRS = ("cache", "config", "debug", "logs")
ALLOWED_STAGING_ENTRIES = ("deps", "python")
PROJECT_FILES = ("README.md", "LICENSE", "CONTACT", "requirements.txt")
THIRD_PARTY_LICENSES = ("LICENSE-MaaFramework", "LICENSE-MaaCommonAssets")
REQUIRED_MAAFW_FILES = (
    "MaaFramework.dll",
    "MaaToolkit.dll",
    "MaaAdbControlUnit.dll",
    "MaaWin32ControlUnit.dll",
    "MaaAgentClient.dll",
    "MaaAgentServer.dll",
)
REQUIRED_OCR_FILES = ("det.onnx", "rec.onnx", "keys.txt")
REQUIRED_AGENT_BINARY_DIRS = ("maatouch", "minitouch")
PROJECT_DIR_PREFIX = "{PROJECT_DIR}/"


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def _require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def _copy_tree(source: Path, destination: Path) -> None:
    _require_dir(source)
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=COPY_IGNORE,
    )


def _strip_project_dir(path: str) -> str:
    return path.removeprefix(PROJECT_DIR_PREFIX)


def transform_interface(interface: dict[str, Any], version: str) -> dict[str, Any]:
    """Return an MXU-compatible copy of a project interface."""
    if not version:
        raise ValueError("version must not be empty")

    transformed = json.loads(json.dumps(interface))
    transformed["version"] = version
    transformed["title"] = "星塔助手 MXU"
    transformed["icon"] = "Assets/logo.ico"
    transformed["license"] = "LICENSE"
    transformed.pop("custom_title", None)
    transformed.pop("mirrorchyan_rid", None)
    transformed.pop("mirrorchyan_multiplatform", None)

    agent = transformed.get("agent")
    if not isinstance(agent, dict):
        raise ValueError("interface.agent must be an object")
    agent["child_exec"] = "./python/python.exe"
    agent["child_args"] = ["-u", "./agent/main.py"]
    agent.pop("timeout", None)

    resources = transformed.get("resource")
    if not isinstance(resources, list):
        raise ValueError("interface.resource must be an array")
    for resource in resources:
        paths = resource.get("path") if isinstance(resource, dict) else None
        if not isinstance(paths, list) or not all(
            isinstance(path, str) for path in paths
        ):
            raise ValueError("each resource.path must be an array of strings")
        resource["path"] = [_strip_project_dir(path) for path in paths]

    controllers = transformed.get("controller")
    if not isinstance(controllers, list):
        raise ValueError("interface.controller must be an array")
    for controller in controllers:
        if not isinstance(controller, dict):
            raise ValueError("each controller must be an object")
        if "attach_resource_path" not in controller:
            continue
        paths = controller["attach_resource_path"]
        if not isinstance(paths, list) or not all(
            isinstance(path, str) for path in paths
        ):
            raise ValueError(
                "each controller.attach_resource_path must be an array of strings"
            )
        controller["attach_resource_path"] = [
            _strip_project_dir(path) for path in paths
        ]

    return transformed


def _copy_project_files(working_dir: Path, install_dir: Path) -> None:
    for name in PROJECT_FILES:
        source = working_dir / name
        _require_file(source)
        shutil.copy2(source, install_dir / name)


def _copy_mxu(mxu_dir: Path, install_dir: Path) -> None:
    mxu_executable = mxu_dir / "mxu.exe"
    mxu_license = mxu_dir / "LICENSE"
    _require_file(mxu_executable)
    _require_file(mxu_license)
    shutil.copy2(mxu_executable, install_dir / "mxu.exe")
    shutil.copy2(mxu_license, install_dir / "LICENSE-MXU")


def _copy_maafw(deps_dir: Path, install_dir: Path) -> None:
    maafw_dir = install_dir / "maafw"
    _copy_tree(deps_dir / "bin", maafw_dir)
    _copy_tree(
        deps_dir / "share" / "MaaAgentBinary",
        maafw_dir / "MaaAgentBinary",
    )
    maafw_license = deps_dir / "LICENSE-MaaFramework"
    _require_file(maafw_license)
    shutil.copy2(maafw_license, install_dir / maafw_license.name)


def _copy_project_payload(working_dir: Path, install_dir: Path, version: str) -> None:
    # Keep using the shared OCR import behavior before copying project resources.
    configure_ocr_model()

    assets_dir = working_dir / "assets"
    _copy_tree(assets_dir / "resource", install_dir / "resource")
    _copy_tree(working_dir / "agent", install_dir / "agent")

    logo = assets_dir / "logo.ico"
    source_interface = assets_dir / "interface.json"
    _require_file(logo)
    _require_file(source_interface)

    destination_assets = install_dir / "Assets"
    destination_assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(logo, destination_assets / "logo.ico")

    common_assets_license = assets_dir / "MaaCommonAssets" / "LICENSE"
    _require_file(common_assets_license)
    shutil.copy2(
        common_assets_license,
        install_dir / "LICENSE-MaaCommonAssets",
    )

    with source_interface.open(encoding="utf-8") as file:
        interface = json.load(file)
    transformed = transform_interface(interface, version)
    with (install_dir / "interface.json").open("w", encoding="utf-8") as file:
        json.dump(transformed, file, ensure_ascii=False, indent=4)
        file.write("\n")

    _copy_project_files(working_dir, install_dir)


def remove_build_artifacts(install_dir: Path) -> None:
    """Remove files that must not be shipped, without touching package code."""
    for path in sorted(install_dir.rglob("*"), reverse=True):
        if path.is_file() and path.suffix.lower() in {".pdb", ".pyc", ".pyo"}:
            path.unlink()
        elif path.is_dir() and path.name in {
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
        }:
            shutil.rmtree(path)


def _package_path(install_dir: Path, value: str, label: str) -> Path:
    if not value or Path(value).is_absolute() or "\\" in value:
        raise ValueError(f"{label} must be a non-empty relative POSIX path: {value!r}")
    path = install_dir.joinpath(*Path(value).parts)
    try:
        path.resolve().relative_to(install_dir.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the package: {value!r}") from error
    return path


def validate_package(install_dir: Path) -> dict[str, Any]:
    """Statically validate the assembled package and return its interface."""
    _require_dir(install_dir)
    required_files = (
        "mxu.exe",
        "interface.json",
        "Assets/logo.ico",
        "python/python.exe",
        "agent/main.py",
        "LICENSE",
        "LICENSE-MXU",
        *THIRD_PARTY_LICENSES,
        "README.md",
        "CONTACT",
        "requirements.txt",
    )
    for relative_path in required_files:
        _require_file(install_dir / relative_path)

    for name in REQUIRED_MAAFW_FILES:
        _require_file(install_dir / "maafw" / name)
    maa_agent_binary = install_dir / "maafw" / "MaaAgentBinary"
    _require_dir(maa_agent_binary)
    for name in REQUIRED_AGENT_BINARY_DIRS:
        binary_dir = maa_agent_binary / name
        _require_dir(binary_dir)
        if not any(path.is_file() for path in binary_dir.rglob("*")):
            raise ValueError(f"maafw/MaaAgentBinary/{name} is empty")

    ocr_dir = install_dir / "resource" / "base" / "model" / "ocr"
    for name in REQUIRED_OCR_FILES:
        _require_file(ocr_dir / name)

    wheels_dir = install_dir / "deps"
    _require_dir(wheels_dir)
    if not any(
        wheel.name.lower().startswith("maafw-") for wheel in wheels_dir.glob("*.whl")
    ):
        raise ValueError("install/deps does not contain a maafw wheel")

    forbidden = [
        path
        for path in install_dir.rglob("*")
        if (path.is_file() and path.suffix.lower() in {".pdb", ".pyc", ".pyo"})
        or (path.is_dir() and path.name == "__pycache__")
    ]
    if forbidden:
        raise ValueError(f"forbidden build artifact remains: {forbidden[0]}")
    for name in TOP_LEVEL_RUNTIME_DIRS:
        if (install_dir / name).exists():
            raise ValueError(f"runtime directory must not be shipped: {name}")

    interface_path = install_dir / "interface.json"
    with interface_path.open(encoding="utf-8") as file:
        interface = json.load(file)
    if not isinstance(interface, dict):
        raise ValueError("interface.json root must be an object")

    expected_values = {
        "name": "MaaStellaSora",
        "title": "星塔助手 MXU",
        "icon": "Assets/logo.ico",
        "license": "LICENSE",
    }
    for key, expected in expected_values.items():
        if interface.get(key) != expected:
            raise ValueError(f"interface.{key} must be {expected!r}")
    if not isinstance(interface.get("version"), str) or not interface["version"]:
        raise ValueError("interface.version must be a non-empty string")

    serialized = json.dumps(interface, ensure_ascii=False)
    if PROJECT_DIR_PREFIX in serialized:
        raise ValueError(f"interface.json still contains {PROJECT_DIR_PREFIX}")
    for forbidden_key in (
        "custom_title",
        "mirrorchyan_rid",
        "mirrorchyan_multiplatform",
    ):
        if forbidden_key in interface:
            raise ValueError(f"interface.json still contains {forbidden_key}")

    agent = interface.get("agent")
    if not isinstance(agent, dict) or "timeout" in agent:
        raise ValueError("interface.agent is invalid or still contains timeout")
    if agent.get("child_exec") != "./python/python.exe":
        raise ValueError("interface.agent.child_exec is not MXU-compatible")
    if agent.get("child_args") != ["-u", "./agent/main.py"]:
        raise ValueError("interface.agent.child_args is not MXU-compatible")

    referenced_files: list[tuple[str, str]] = [
        (str(interface.get("icon", "")), "interface.icon"),
        (str(interface.get("license", "")), "interface.license"),
        (agent["child_exec"].removeprefix("./"), "interface.agent.child_exec"),
    ]
    child_args = agent["child_args"]
    referenced_files.append(
        (child_args[-1].removeprefix("./"), "interface.agent.child_args")
    )

    imports = interface.get("import")
    if not isinstance(imports, list) or not all(
        isinstance(path, str) for path in imports
    ):
        raise ValueError("interface.import must be an array of strings")
    referenced_files.extend((path, "interface.import") for path in imports)

    resources = interface.get("resource")
    if not isinstance(resources, list):
        raise ValueError("interface.resource must be an array")
    referenced_dirs: list[tuple[str, str]] = []
    for resource in resources:
        paths = resource.get("path") if isinstance(resource, dict) else None
        if not isinstance(paths, list) or not all(
            isinstance(path, str) for path in paths
        ):
            raise ValueError("each resource.path must be an array of strings")
        referenced_dirs.extend((path, "interface.resource.path") for path in paths)

    controllers = interface.get("controller")
    if not isinstance(controllers, list):
        raise ValueError("interface.controller must be an array")
    for controller in controllers:
        if not isinstance(controller, dict):
            raise ValueError("each controller must be an object")
        if "attach_resource_path" not in controller:
            continue
        paths = controller["attach_resource_path"]
        if not isinstance(paths, list) or not all(
            isinstance(path, str) for path in paths
        ):
            raise ValueError(
                "each controller.attach_resource_path must be an array of strings"
            )
        referenced_dirs.extend(
            (path, "interface.controller.attach_resource_path") for path in paths
        )

    for relative_path, label in referenced_files:
        referenced = _package_path(install_dir, relative_path, label)
        if not referenced.is_file():
            raise FileNotFoundError(f"{label} references missing file: {relative_path}")
    for relative_path, label in referenced_dirs:
        referenced = _package_path(install_dir, relative_path, label)
        if not referenced.is_dir():
            raise FileNotFoundError(
                f"{label} references missing directory: {relative_path}"
            )

    return interface


def validate_staging_paths(
    working_dir: Path, install_dir: Path, deps_dir: Path, mxu_dir: Path
) -> None:
    """Restrict cleanup and merged copies to a fresh package staging directory."""
    if install_dir == working_dir or not install_dir.is_relative_to(working_dir):
        raise ValueError("install_dir must be a strict child of working_dir")

    for label, source_dir in (("deps_dir", deps_dir), ("mxu_dir", mxu_dir)):
        if (
            install_dir == source_dir
            or install_dir in source_dir.parents
            or source_dir in install_dir.parents
        ):
            raise ValueError(f"install_dir must not overlap {label}")

    if install_dir.exists():
        unexpected = sorted(
            path.name
            for path in install_dir.iterdir()
            if path.name not in ALLOWED_STAGING_ENTRIES
        )
        if unexpected:
            raise ValueError(
                "install_dir must be a fresh staging directory containing only "
                f"{ALLOWED_STAGING_ENTRIES}: {unexpected}"
            )


def build_package(
    version: str,
    *,
    working_dir: Path = WORKING_DIR,
    install_dir: Path | None = None,
    deps_dir: Path | None = None,
    mxu_dir: Path | None = None,
) -> Path:
    """Assemble and validate an MXU package in an existing CI staging directory."""
    working_dir = working_dir.resolve()
    install_dir = (install_dir or working_dir / "install").resolve()
    deps_dir = (deps_dir or working_dir / "deps").resolve()
    mxu_dir = (mxu_dir or working_dir / "MXU").resolve()

    validate_staging_paths(working_dir, install_dir, deps_dir, mxu_dir)
    install_dir.mkdir(parents=True, exist_ok=True)
    _copy_mxu(mxu_dir, install_dir)
    _copy_maafw(deps_dir, install_dir)
    _copy_project_payload(working_dir, install_dir, version)
    remove_build_artifacts(install_dir)
    validate_package(install_dir)
    return install_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="project release tag, for example v1.2.3")
    parser.add_argument("--install-dir", type=Path, default=WORKING_DIR / "install")
    parser.add_argument("--deps-dir", type=Path, default=WORKING_DIR / "deps")
    parser.add_argument("--mxu-dir", type=Path, default=WORKING_DIR / "MXU")
    args = parser.parse_args()

    output = build_package(
        args.version,
        install_dir=args.install_dir,
        deps_dir=args.deps_dir,
        mxu_dir=args.mxu_dir,
    )
    print(f"MXU package assembled and validated: {output}")


if __name__ == "__main__":
    main()
