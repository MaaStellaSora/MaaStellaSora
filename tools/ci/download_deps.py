#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载Python依赖到deps目录的脚本
自动检测当前平台并下载对应架构的wheel文件
"""

import os
import sys
import subprocess
import argparse
import platform
import re
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


MAAFW_VERSION_PATTERN = re.compile(
    r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:(?:a|b|rc)(?:0|[1-9]\d*))?$"
)


def get_platform_tag():
    """自动检测当前平台并返回对应的平台标签"""
    os_type = platform.system()
    os_arch = platform.machine()

    print(f"检测到操作系统: {os_type}, 架构: {os_arch}")

    if os_type == "Windows":
        # 在Windows ARM64环境中，platform.machine()可能错误返回AMD64
        # 我们需要检查处理器标识符来确定真实架构
        processor_identifier = os.environ.get("PROCESSOR_IDENTIFIER", "")

        # 检查是否为ARM64处理器
        if "ARMv8" in processor_identifier or "ARM64" in processor_identifier:
            print(f"检测到ARM64处理器: {processor_identifier}")
            os_arch = "ARM64"

        # 映射platform.machine()到pip的平台标签
        arch_mapping = {
            "AMD64": "win_amd64",
            "x86_64": "win_amd64",
            "ARM64": "win_arm64",
            "aarch64": "win_arm64",
        }
        platform_tag = arch_mapping.get(os_arch, f"win_{os_arch.lower()}")

    elif os_type == "Darwin":  # macOS
        # 映射platform.machine()到pip的平台标签
        arch_mapping = {
            "x86_64": "macosx_10_9_x86_64",
            "arm64": "macosx_11_0_arm64",
            "aarch64": "macosx_11_0_arm64",
        }
        platform_tag = arch_mapping.get(os_arch, f"macosx_10_9_{os_arch}")

    elif os_type == "Linux":
        # 映射platform.machine()到pip的平台标签
        arch_mapping = {
            "x86_64": "manylinux2014_x86_64",
            "aarch64": "manylinux2014_aarch64",
            "arm64": "manylinux2014_aarch64",
        }
        platform_tag = arch_mapping.get(os_arch, f"manylinux2014_{os_arch}")

    else:
        raise ValueError(f"不支持的操作系统: {os_type}")

    print(f"使用平台标签: {platform_tag}")
    return platform_tag


def build_download_command(
    deps_path, platform_tag=None, *, python_version=None, maafw_version=None
):
    """构造 pip download 命令。"""
    requirements_file = Path("requirements.txt")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "-r",
        str(requirements_file),
        "-d",
        str(deps_path),
        "--only-binary=:all:",
    ]
    if platform_tag:
        cmd.extend(["--platform", platform_tag])
    if python_version:
        cmd.extend(["--python-version", python_version])
    if maafw_version:
        if not MAAFW_VERSION_PATTERN.fullmatch(maafw_version):
            raise ValueError(f"无效的 MaaFramework Python 版本: {maafw_version}")
        cmd.extend(["--pre", f"maafw=={maafw_version}"])
    return cmd


def validate_maafw_wheel(deps_path, expected_version):
    """确认依赖目录只包含目标版本的 maafw wheel。"""
    maafw_wheels = []
    for wheel in Path(deps_path).glob("*.whl"):
        parts = wheel.name[:-4].split("-")
        if len(parts) >= 2 and re.sub(r"[-_.]+", "-", parts[0]).lower() == "maafw":
            maafw_wheels.append((wheel, parts[1].lower()))

    expected = expected_version.lower()
    if len(maafw_wheels) != 1 or maafw_wheels[0][1] != expected:
        found = [wheel.name for wheel, _ in maafw_wheels]
        raise RuntimeError(
            f"maafw wheel 必须唯一且版本为 {expected_version}，实际为: {found or '无'}"
        )


def download_dependencies(
    deps_dir,
    platform_tag,
    *,
    python_version=None,
    maafw_version=None,
    allow_native_fallback=True,
):
    """下载依赖到指定目录"""
    # 创建deps目录
    deps_path = Path(deps_dir)
    deps_path.mkdir(parents=True, exist_ok=True)

    print(f"开始下载平台 {platform_tag} 的依赖到 {deps_dir}")

    # 从requirements.txt读取依赖
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("错误: requirements.txt 文件不存在")
        return False

    # 首先尝试下载平台特定的wheel文件
    try:
        cmd = build_download_command(
            deps_path,
            platform_tag,
            python_version=python_version,
            maafw_version=maafw_version,
        )

        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        if result.stderr:
            print("警告信息:")
            print(result.stderr)

        if maafw_version:
            validate_maafw_wheel(deps_path, maafw_version)

        # 列出下载的文件
        whl_files = list(deps_path.glob("*.whl"))
        print(f"\n下载的wheel文件 ({len(whl_files)} 个):")
        for whl_file in whl_files:
            print(f"  {whl_file.name}")

        print(f"依赖下载完成到: {deps_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"平台特定下载失败: {e}")
        if allow_native_fallback and e.stderr and (
            "Could not find a version" in e.stderr
            or "No matching distribution" in e.stderr
        ):
            print("某些包可能不支持当前平台，尝试通用下载策略...")

            # 回退到通用下载策略（不指定平台）
            try:
                cmd_fallback = build_download_command(
                    deps_path,
                    python_version=python_version,
                    maafw_version=maafw_version,
                )

                print(f"执行回退命令: {' '.join(cmd_fallback)}")
                result = subprocess.run(
                    cmd_fallback, check=True, capture_output=True, text=True
                )
                print(result.stdout)

                if result.stderr:
                    print("警告信息:")
                    print(result.stderr)

                if maafw_version:
                    validate_maafw_wheel(deps_path, maafw_version)

                # 列出下载的文件
                whl_files = list(deps_path.glob("*.whl"))
                print(f"\n下载的wheel文件 ({len(whl_files)} 个):")
                for whl_file in whl_files:
                    print(f"  {whl_file.name}")

                print(f"通用策略下载完成到: {deps_path}")
                return True

            except subprocess.CalledProcessError as e2:
                print(f"通用策略也失败: {e2}")
                if e2.stdout:
                    print("stdout:", e2.stdout)
                if e2.stderr:
                    print("stderr:", e2.stderr)
                return False
        else:
            if e.stdout:
                print("stdout:", e.stdout)
            if e.stderr:
                print("stderr:", e.stderr)
            return False


def main():
    parser = argparse.ArgumentParser(description="下载Python依赖到deps目录")
    parser.add_argument("--deps-dir", default="deps", help="依赖下载目录 (默认: deps)")
    parser.add_argument(
        "--platform-tag",
        help="显式指定 pip 平台标签；交叉构建时必须传入，例如 manylinux2014_aarch64",
    )
    parser.add_argument(
        "--python-version",
        help="显式指定目标 Python 版本，例如 3.12",
    )
    parser.add_argument(
        "--maafw-version",
        help="锁定 maafw wheel 的 PEP 440 版本，例如 5.13.0b5",
    )

    args = parser.parse_args()

    try:
        # 交叉构建使用显式平台；原生构建继续自动检测。
        platform_tag = args.platform_tag or get_platform_tag()
        if args.platform_tag:
            print(f"使用显式平台标签: {platform_tag}")

        # 下载依赖
        success = download_dependencies(
            args.deps_dir,
            platform_tag,
            python_version=args.python_version,
            maafw_version=args.maafw_version,
            allow_native_fallback=not bool(args.platform_tag),
        )

        if success:
            print("✅ 依赖下载成功")
            sys.exit(0)
        else:
            print("❌ 依赖下载失败")
            sys.exit(1)

    except Exception as e:
        print(f"❌ 脚本执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
