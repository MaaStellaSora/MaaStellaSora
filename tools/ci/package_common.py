"""各平台发行包共用的文件复制和校验。"""

import json
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
NEURAL_MODULE = Path("custom/action/climb_tower_neural")


def _neural_checkpoint_path(agent_dir: Path) -> Path:
    module_dir = agent_dir / NEURAL_MODULE
    metadata_path = module_dir / "model.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Neural model metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid neural model metadata: {metadata_path}")
    checkpoint_name = metadata.get("checkpoint")
    if not isinstance(checkpoint_name, str) or not checkpoint_name:
        raise ValueError(f"Invalid checkpoint path in {metadata_path}")

    checkpoint = Path(checkpoint_name)
    if checkpoint.is_absolute():
        raise ValueError(f"Neural checkpoint path must be relative: {checkpoint_name}")
    checkpoint_path = (module_dir / checkpoint).resolve()
    try:
        checkpoint_path.relative_to(module_dir.resolve())
    except ValueError as error:
        raise ValueError(
            f"Neural checkpoint path escapes its module: {checkpoint_name}"
        ) from error
    require_file(checkpoint_path)
    return checkpoint_path


def copy_agent_files(
    working_dir: Path, install_dir: Path, *, include_neural_model: bool = True
) -> None:
    """Copy runtime code, optionally omitting the large neural checkpoint."""
    source = working_dir / "agent"
    require_dir(source)
    omitted_checkpoint = (
        _neural_checkpoint_path(source) if not include_neural_model else None
    )

    def ignored(directory, names):
        result = set(COPY_IGNORE(directory, names))
        result.update({"tests", "benchmarks"} & set(names))
        if omitted_checkpoint and Path(directory).resolve() == omitted_checkpoint.parent:
            result.add(omitted_checkpoint.name)
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


def copy_tree(source: Path, destination: Path) -> None:
    """复制发行内容并过滤构建缓存。"""
    require_dir(source)
    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=COPY_IGNORE)


def copy_project_files(
    working_dir: Path, install_dir: Path, *, platform_tag: str = ""
) -> None:
    """复制说明、许可证和运行依赖清单。"""
    for source, destination in PROJECT_FILES.items():
        path = working_dir / source
        require_file(path)
        shutil.copy2(path, install_dir / destination)
    if platform_tag == "win-x64":
        extra_requirements = working_dir / "assets" / "requirements-win-x64.txt"
        base_requirements = (
            working_dir / "assets" / "requirements.txt"
        ).read_text(encoding="utf-8")
        (install_dir / "requirements.txt").write_text(
            extra_requirements.read_text(encoding="utf-8").replace(
                "-r requirements.txt", base_requirements.rstrip()
            ),
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
