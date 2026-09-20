"""为尚未发布到 PyPI 的 MaaFramework 版本准备同版 Python wheel。"""

import argparse
from pathlib import Path, PurePosixPath
import json
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.error import HTTPError
from urllib.request import urlopen
import zipfile

from resolve_releases import maafw_tag_to_pep440


WHEEL_PLATFORMS = {
    "win_amd64", "win_arm64",
    "manylinux2014_x86_64", "manylinux2014_aarch64",
    "macosx_13_0_x86_64", "macosx_13_0_arm64",
}


def prepare_maafw_wheel(tag: str, sdk_dir: Path, deps_dir: Path, platform_tag: str) -> None:
    """优先使用 PyPI；缺少该版本时从同一标签源码和 SDK 构建。"""
    version = maafw_tag_to_pep440(tag)
    deps_dir = deps_dir.resolve()
    deps_dir.mkdir(parents=True, exist_ok=True)
    if platform_tag not in WHEEL_PLATFORMS:
        raise ValueError(f"不支持的 MaaFramework wheel 平台: {platform_tag}")
    try:
        with urlopen(f"https://pypi.org/pypi/maafw/{version}/json", timeout=30) as response:
            release = json.load(response)
        for asset in release["urls"]:
            name = asset["filename"]
            if name.endswith(".whl") and not asset.get("yanked"):
                platforms = name[:-4].rsplit("-", 1)[-1].split(".")
                if platform_tag in platforms or "any" in platforms:
                    return
    except HTTPError as exc:
        if exc.code != 404:
            raise

    sdk_bin = sdk_dir.resolve() / "bin"
    library = (
        "MaaFramework.dll" if platform_tag.startswith("win_") else
        "libMaaFramework.dylib" if platform_tag.startswith("macosx_") else
        "libMaaFramework.so"
    )
    if not (sdk_bin / library).is_file():
        raise FileNotFoundError(sdk_bin / library)

    print(f"PyPI 缺少 maafw {version} 的 {platform_tag} wheel，使用 {tag} 源码和当前 SDK 构建。")
    with tempfile.TemporaryDirectory(prefix="maafw-", dir=deps_dir.parent) as temp:
        work = Path(temp)
        archive_path = work / "source.zip"
        with urlopen(
            f"https://codeload.github.com/MaaXYZ/MaaFramework/zip/refs/tags/{tag}",
            timeout=30,
        ) as response, archive_path.open("wb") as output:
            shutil.copyfileobj(response, output)

        source = work / "binding"
        source.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                parts = PurePosixPath(member.filename).parts
                if member.is_dir() or len(parts) < 2:
                    continue
                relative = PurePosixPath(*parts[1:])
                if relative.is_relative_to("source/binding/Python"):
                    relative = relative.relative_to("source/binding/Python")
                elif str(relative) not in {"README.md", "README_en.md", "LICENSE.md"}:
                    continue
                target = (source / relative).resolve()
                if not target.is_relative_to(source):
                    raise ValueError(f"源码归档路径越界: {member.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))

        # 上游以占位版本发布源码；原生库复制规则与其 pip_pack 工具一致。
        project = source / "pyproject.toml"
        text, count = re.subn(
            r'^version = "0"$', f'version = "{version}"',
            project.read_text(encoding="utf-8"), count=1, flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError("MaaFramework Python 源码的版本字段不符合预期")
        project.write_text(
            text
            + '\n[tool.hatch.build.targets.wheel.force-include]\n"maa/bin" = "maa/bin"\n'
            + '\n[tool.hatch.build.hooks.custom]\npath = "ci_wheel_hook.py"\n',
            encoding="utf-8",
        )
        shutil.copytree(
            sdk_bin, source / "maa" / "bin",
            ignore=shutil.ignore_patterns("MaaPiCli*", "MaaNode*"),
        )
        # 在上游声明的隔离构建环境中设置目标平台，支持 Linux 交叉打包。
        (source / "ci_wheel_hook.py").write_text(
            "from hatchling.builders.hooks.plugin.interface import BuildHookInterface\n\n"
            "class CustomBuildHook(BuildHookInterface):\n"
            "    def initialize(self, version, build_data):\n"
            f"        build_data['tag'] = {json.dumps(f'py3-none-{platform_tag}')}\n"
            "        build_data['pure_python'] = False\n",
            encoding="utf-8",
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir",
             str(deps_dir), str(source)],
            check=True, timeout=300,
        )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maafw-tag", required=True)
    parser.add_argument("--platform-tag", required=True, choices=sorted(WHEEL_PLATFORMS))
    parser.add_argument("--sdk-dir", type=Path, default=root / "deps")
    parser.add_argument("--deps-dir", type=Path, default=root / "install" / "deps")
    args = parser.parse_args()
    prepare_maafw_wheel(args.maafw_tag, args.sdk_dir, args.deps_dir, args.platform_tag)


if __name__ == "__main__":
    main()
