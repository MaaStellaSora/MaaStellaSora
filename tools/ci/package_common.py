"""MFA 与 MXU 共用的发行文件复制和校验。"""

import shutil
from pathlib import Path


COPY_IGNORE = shutil.ignore_patterns(
    "*.pdb", "*.PDB", "*.pyc", "*.pyo", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".mypy_cache",
)
PROJECT_FILES = {
    "README.md": "README.md",
    "LICENSE": "LICENSE",
    "assets/CONTACT": "CONTACT",
    "assets/requirements.txt": "requirements.txt",
}
REQUIRED_OCR_FILES = ("det.onnx", "rec.onnx", "keys.txt")


def require_file(path: Path) -> None:
    """校验发行所需文件存在。"""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_dir(path: Path) -> None:
    """校验发行所需目录存在。"""
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def copy_tree(source: Path, destination: Path) -> None:
    """复制发行内容并过滤构建缓存。"""
    require_dir(source)
    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=COPY_IGNORE)


def copy_project_files(working_dir: Path, install_dir: Path) -> None:
    """复制说明、许可证和运行依赖清单。"""
    for source, destination in PROJECT_FILES.items():
        path = working_dir / source
        require_file(path)
        shutil.copy2(path, install_dir / destination)


def remove_build_artifacts(install_dir: Path) -> None:
    """清理已校验的发行暂存目录中的构建缓存。"""
    install_dir = install_dir.resolve()
    for path in sorted(install_dir.rglob("*"), reverse=True):
        if not path.resolve().is_relative_to(install_dir):
            raise ValueError(f"build artifact path escapes staging directory: {path}")
        if path.is_file() and path.suffix.lower() in {".pdb", ".pyc", ".pyo"}:
            path.unlink()
        elif path.is_dir() and path.name in {
            "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
        }:
            shutil.rmtree(path)
