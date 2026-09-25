"""准备 MFAAvalonia 发行包的项目文件与 MaaFramework 运行库。"""

import argparse
from pathlib import Path
import shutil
import sys
import json

SCRIPT_DIR = Path(__file__).resolve().parent
WORKING_DIR = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from configure import configure_ocr_model
from package_common import (
    COPY_IGNORE, copy_project_files, copy_tree, remove_build_artifacts, require_file,
)
from resource_layout import copy_resources, validate_staging_directory


def install_deps(working_dir: Path, install_path: Path, platform_tag: str) -> None:
    """安装 MaaFramework 依赖到对应架构路径

    Args:
        platform_tag: 平台标签，如 win-x64, linux-arm64, osx-arm64
    """
    if not platform_tag:
        raise ValueError("platform_tag is required")
    if platform_tag.startswith("win-"):
        require_file(working_dir / "deps" / "bin" / "MaaWin32ControlUnit.dll")

    excluded = shutil.ignore_patterns(
        "*MaaDbgControlUnit*", "*MaaThriftControlUnit*", "*MaaRpc*", "*MaaHttp*",
    )
    shutil.copytree(
        working_dir / "deps" / "bin",
        install_path / "runtimes" / platform_tag / "native",
        ignore=lambda directory, names: set(excluded(directory, names))
        | set(COPY_IGNORE(directory, names)),
        dirs_exist_ok=True,
    )
    copy_tree(
        working_dir / "deps" / "share" / "MaaAgentBinary",
        install_path / "MaaAgentBinary",
    )


def install_resource(working_dir: Path, install_path: Path) -> None:
    copy_resources(
        working_dir / "assets" / "resource",
        install_path / "resource",
        ignore=COPY_IGNORE,
    )
    configure_ocr_model(
        working_dir / "assets", install_path / "resource" / "base" / "model" / "ocr"
    )
    copy_tree(
        working_dir / "assets" / "interface",
        install_path / "interface",
    )
    assets_dir = install_path / "Assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(working_dir / "assets" / "logo.ico", assets_dir)


def transform_interface(interface: dict, version: str, platform_tag: str) -> dict:
    """根据目标平台生成 MFA 界面声明，保留客户端扩展字段。"""
    if not version:
        raise ValueError("version must not be empty")
    child_exec = {
        "win": "./python/python.exe",
        "osx": "./python/bin/python3",
        "linux": "python3",
    }.get(platform_tag.split("-", 1)[0])
    if child_exec is None or platform_tag not in {
        f"{system}-{arch}"
        for system in ("win", "osx", "linux")
        for arch in ("x64", "arm64")
    }:
        raise ValueError(f"Unsupported platform_tag: {platform_tag}")

    transformed = json.loads(json.dumps(interface))
    transformed["version"] = version
    transformed["custom_title"] = f"星塔助手{version}"
    transformed["agent"]["child_exec"] = child_exec
    transformed["agent"]["child_args"] = ["-u", "./agent/main.py"]
    return transformed


def build_package(
    version: str,
    platform_tag: str,
    *,
    working_dir: Path = WORKING_DIR,
    install_dir: Path | None = None,
) -> Path:
    """向干净暂存目录安装项目内容，供工作流随后合入 MFA 客户端。"""
    working_dir = working_dir.resolve()
    install_path = (install_dir or working_dir / "install").resolve()
    validate_staging_directory(working_dir, install_path)

    with (working_dir / "assets" / "interface.json").open(encoding="utf-8") as file:
        interface = transform_interface(json.load(file), version, platform_tag)

    install_path.mkdir(parents=True, exist_ok=True)
    install_deps(working_dir, install_path, platform_tag)
    install_resource(working_dir, install_path)
    copy_project_files(working_dir, install_path)
    copy_tree(working_dir / "agent", install_path / "agent")
    remove_build_artifacts(install_path)
    if platform_tag.startswith("win-"):
        (install_path / "用管理员身份运行喵.txt").touch()
    with (install_path / "interface.json").open("w", encoding="utf-8") as file:
        json.dump(interface, file, ensure_ascii=False, indent=4)
        file.write("\n")
    return install_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("platform_tag")
    parser.add_argument("--working-dir", type=Path, default=WORKING_DIR)
    parser.add_argument("--install-dir", type=Path)
    args = parser.parse_args()
    output = build_package(
        args.version, args.platform_tag,
        working_dir=args.working_dir, install_dir=args.install_dir,
    )
    print(f"Install to {output} successfully.")


if __name__ == "__main__":
    main()
