"""将源码资源布局转换为兼容既有客户端的发布布局。"""

import shutil
from pathlib import Path


ALLOWED_STAGING_ENTRIES = ("deps", "python")
PIPELINE_PATHS = {
    "common/base.json": "base.json",
    "daily/login.json": "login.json",
    "daily/friend.json": "friend.json",
    "daily/grant.json": "grant.json",
    "daily/mail.json": "mail.json",
    "daily/quest.json": "quest.json",
    "daily/shop.json": "shop.json",
    "daily/talk.json": "talk.json",
    "daily/task.json": "task.json",
    "combat/fight.json": "fight.json",
    "combat/operation.json": "operation.json",
    "combat/proving_grounds.json": "proving_grounds.json",
    "activity/activity.json": "activity.json",
    "activity/activity_challenge.json": "activity_challenge.json",
    "invite/invite.json": "invite.json",
}


def validate_staging_directory(working_dir: Path, install_dir: Path) -> None:
    """只允许在预先准备了 Python 和依赖的干净目录中打包。"""
    working_dir = working_dir.resolve()
    install_dir = install_dir.resolve()
    if install_dir == working_dir or not install_dir.is_relative_to(working_dir):
        raise ValueError("install_dir must be a strict child of working_dir")
    if install_dir.exists():
        unexpected = sorted(
            path.name
            for path in install_dir.iterdir()
            if path.name not in ALLOWED_STAGING_ENTRIES or not path.is_dir()
        )
        if unexpected:
            raise ValueError(
                "install_dir must be a fresh staging directory containing only "
                f"{ALLOWED_STAGING_ENTRIES}: {unexpected}"
            )


def copy_resources(source: Path, destination: Path, *, ignore=None) -> None:
    """保留资源内容，并将分组后的 pipeline 复制到原发布路径。"""
    pipelines = set(source.glob("*/pipeline"))
    mapped_files = {}
    destinations = set()
    for pipeline in sorted(pipelines):
        for path in sorted(pipeline.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(pipeline).as_posix()
            target = PIPELINE_PATHS.get(relative)
            if target is None:
                if relative.split("/", 1)[0] not in {"common", "climb_tower"}:
                    raise ValueError(f"unmapped pipeline file: {path}")
                target = relative
            target_path = destination / pipeline.relative_to(source) / target
            key = target_path.as_posix().casefold()
            if key in destinations or (
                target != relative and (pipeline / target).exists()
            ):
                raise ValueError(f"pipeline destination conflict: {target_path}")
            destinations.add(key)
            if target != relative:
                mapped_files[path] = target_path

    def ignore_grouped_files(directory, names):
        ignored = set(ignore(directory, names)) if ignore else set()
        directory = Path(directory)
        if directory in pipelines:
            ignored.update({"daily", "combat", "activity", "invite"} & set(names))
        ignored.update(name for name in names if directory / name in mapped_files)
        return ignored

    shutil.copytree(
        source, destination, dirs_exist_ok=True, ignore=ignore_grouped_files
    )
    for path, target in mapped_files.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
