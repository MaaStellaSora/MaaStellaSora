"""各平台发行包共用的文件复制和校验。"""

import shutil
from pathlib import Path


COPY_IGNORE = shutil.ignore_patterns(
    "*.pdb", "*.PDB", "*.pyc", "*.pyo", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "dev_config.json",
)
PROJECT_FILES = {
    "README.md": "README.md",
    "LICENSE": "LICENSE",
    "assets/CONTACT": "CONTACT",
    "assets/requirements.txt": "requirements.txt",
}
REQUIRED_OCR_FILES = ("det.onnx", "rec.onnx", "keys.txt")
NEURAL_TASK = "interface/tasks/climb_tower_neural.json"


def payload_ignore(platform_tag: str | None = None):
    """Unsupported releases keep the original tasks and omit neural resources."""
    def ignored(directory, names):
        result = set(COPY_IGNORE(directory, names))
        if platform_tag is not None and platform_tag != "win-x64":
            result.update(name for name in names if name.startswith("climb_tower_neural") and name.endswith(".json"))
            result.update({"ClimbTower_neural"} & set(names))
        return result
    return ignored


def configure_agent(interface: dict, platform_tag: str) -> None:
    """Select the portable package entry point without changing the ordinary Agent."""
    supported = platform_tag == "win-x64"
    interface["agent"]["child_args"] = ["-u", "./agent/neural_main.py" if supported else "./agent/main.py"]
    if not supported:
        interface["import"] = [path for path in interface["import"] if path != NEURAL_TASK]


def copy_agent_files(working_dir: Path, install_dir: Path, platform_tag: str) -> None:
    """Copy runtime code and the fixed model; keep development tools out of releases."""
    source = working_dir / "agent"
    require_dir(source)

    def ignored(directory, names):
        result = set(COPY_IGNORE(directory, names))
        result.update({"tests", "benchmarks"} & set(names))
        relative = Path(directory).relative_to(source).as_posix()
        if platform_tag != "win-x64":
            if relative == ".":
                result.update(name for name in names if name.startswith("neural_") and name.endswith(".py"))
            elif relative == "custom/action":
                result.update({"climb_tower_neural"} & set(names))
        return result

    shutil.copytree(source, install_dir / "agent", dirs_exist_ok=True, ignore=ignored)


def require_file(path: Path) -> None:
    """校验发行所需文件存在。"""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_dir(path: Path) -> None:
    """校验发行所需目录存在。"""
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def copy_tree(source: Path, destination: Path, *, platform_tag: str | None = None) -> None:
    """复制发行内容并过滤构建缓存。"""
    require_dir(source)
    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=payload_ignore(platform_tag))


def copy_project_files(working_dir: Path, install_dir: Path, *, platform_tag: str = "") -> None:
    """复制说明、许可证和运行依赖清单。"""
    for source, destination in PROJECT_FILES.items():
        path = working_dir / source
        require_file(path)
        shutil.copy2(path, install_dir / destination)
    if platform_tag == "win-x64":
        # The distributed list is standalone; its include refers to source assets.
        optional = working_dir / "assets" / "requirements-win-x64.txt"
        base = (working_dir / "assets" / "requirements.txt").read_text(encoding="utf-8")
        (install_dir / "requirements.txt").write_text(
            optional.read_text(encoding="utf-8").replace("-r requirements.txt", base.rstrip()),
            encoding="utf-8",
        )


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
