#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析双 GUI 发布所需的最新上游 Release。"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from http.client import HTTPException, IncompleteRead
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GITHUB_API = "https://api.github.com"
PAGE_SIZE = 30
MAX_REQUEST_ATTEMPTS = 3

PROJECTS = {
    "maafw": {
        "repository": "MaaXYZ/MaaFramework",
        "assets": {
            "maafw_win_x86_64_asset": "MAA-win-x86_64-*.zip",
            "maafw_win_aarch64_asset": "MAA-win-aarch64-*.zip",
            "maafw_linux_x86_64_asset": "MAA-linux-x86_64-*.zip",
            "maafw_linux_aarch64_asset": "MAA-linux-aarch64-*.zip",
            "maafw_macos_x86_64_asset": "MAA-macos-x86_64-*.zip",
            "maafw_macos_aarch64_asset": "MAA-macos-aarch64-*.zip",
        },
    },
    "mfa": {
        "repository": "MaaXYZ/MFAAvalonia",
        "assets": {
            "mfa_win_x64_asset": "MFAAvalonia-*-win-x64.zip",
            "mfa_win_arm64_asset": "MFAAvalonia-*-win-arm64.zip",
            "mfa_linux_x64_asset": "MFAAvalonia-*-linux-x64.tar.gz",
            "mfa_linux_arm64_asset": "MFAAvalonia-*-linux-arm64.tar.gz",
            "mfa_osx_x64_asset": "MFAAvalonia-*-osx-x64.tar.gz",
            "mfa_osx_arm64_asset": "MFAAvalonia-*-osx-arm64.tar.gz",
        },
    },
    "mxu": {
        "repository": "MistEO/MXU",
        "assets": {
            "mxu_win_x86_64_asset": "MXU-win-x86_64-*.zip",
        },
    },
}

MAAFW_TAG_PATTERN = re.compile(
    r"^v(?P<version>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))"
    r"(?:-(?P<stage>alpha|beta|rc)\.(?P<number>0|[1-9]\d*))?$"
)


class ReleaseResolutionError(RuntimeError):
    """上游 Release 数据不满足发布约束。"""


def parse_published_at(value: Any) -> datetime:
    """将 GitHub 的 ISO 8601 发布时间转换为 UTC datetime。"""
    if not isinstance(value, str) or not value:
        raise ReleaseResolutionError("Release 缺少 published_at")

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReleaseResolutionError(f"无效的 published_at: {value}") from exc

    if parsed.tzinfo is None:
        raise ReleaseResolutionError(f"published_at 缺少时区: {value}")
    return parsed.astimezone(timezone.utc)


def select_latest_release(releases: list[dict[str, Any]]) -> dict[str, Any]:
    """选择发布时间最新的非草稿 Release，发布时间相同时取较大的 ID。"""
    candidates: list[tuple[datetime, int, dict[str, Any]]] = []
    for release in releases:
        if release.get("draft"):
            continue
        try:
            release_id = int(release["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReleaseResolutionError("Release 缺少有效的 id") from exc
        candidates.append(
            (parse_published_at(release.get("published_at")), release_id, release)
        )

    if not candidates:
        raise ReleaseResolutionError("没有可用的非草稿 Release")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def select_asset(release: dict[str, Any], pattern: str) -> str:
    """按大小写敏感 glob 唯一匹配 Release 资产。"""
    asset_names = [
        asset.get("name")
        for asset in release.get("assets", [])
        if isinstance(asset, dict) and isinstance(asset.get("name"), str)
    ]
    matches = [name for name in asset_names if fnmatch.fnmatchcase(name, pattern)]
    if len(matches) != 1:
        tag = release.get("tag_name", "<unknown>")
        raise ReleaseResolutionError(
            f"Release {tag} 的资产规则 {pattern!r} 应唯一匹配，实际匹配 {len(matches)} 个: "
            f"{matches or '无'}"
        )
    return matches[0]


def maafw_tag_to_pep440(tag: str) -> str:
    """将 MaaFramework Release tag 严格转换为 PEP 440 版本。"""
    match = MAAFW_TAG_PATTERN.fullmatch(tag)
    if not match:
        raise ReleaseResolutionError(f"无法转换 MaaFramework tag: {tag}")

    version = match.group("version")
    stage = match.group("stage")
    if not stage:
        return version

    stage_map = {"alpha": "a", "beta": "b", "rc": "rc"}
    return f"{version}{stage_map[stage]}{match.group('number')}"


def github_get_json(url: str, token: str | None) -> Any:
    """读取 GitHub REST API JSON。"""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "MaaStellaSora-release-resolver",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers)
    for attempt in range(MAX_REQUEST_ATTEMPTS):
        try:
            with urlopen(  # noqa: S310 - URL 由固定 GitHub API 和仓库常量组成
                request, timeout=30
            ) as response:
                return json.load(response)
        except HTTPError:
            raise
        except (IncompleteRead, URLError, TimeoutError, json.JSONDecodeError):
            if attempt + 1 == MAX_REQUEST_ATTEMPTS:
                raise
            time.sleep(2**attempt)

    raise AssertionError("unreachable")


def list_releases(repository: str, token: str | None) -> list[dict[str, Any]]:
    """分页读取仓库的全部 Releases，包含预览版。"""
    releases: list[dict[str, Any]] = []
    page = 1
    while True:
        url = (
            f"{GITHUB_API}/repos/{repository}/releases"
            f"?per_page={PAGE_SIZE}&page={page}"
        )
        payload = github_get_json(url, token)
        if not isinstance(payload, list):
            raise ReleaseResolutionError(f"{repository} Releases API 返回了非列表数据")
        if not all(isinstance(item, dict) for item in payload):
            raise ReleaseResolutionError(f"{repository} Releases API 包含无效条目")
        releases.extend(payload)
        if len(payload) < PAGE_SIZE:
            return releases
        page += 1


def resolve_project(
    repository: str, asset_patterns: dict[str, str], token: str | None
) -> tuple[dict[str, Any], dict[str, str]]:
    """解析一个仓库的最新 Release 及其必需资产。"""
    release = select_latest_release(list_releases(repository, token))
    if not isinstance(release.get("tag_name"), str) or not release["tag_name"]:
        raise ReleaseResolutionError(f"{repository} 最新 Release 缺少 tag_name")
    assets = {
        output_name: select_asset(release, pattern)
        for output_name, pattern in asset_patterns.items()
    }
    return release, assets


def resolve_all(token: str | None) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """解析三个上游项目，生成 workflow outputs 和摘要数据。"""
    outputs: dict[str, str] = {}
    summary_rows: list[dict[str, Any]] = []

    for key, config in PROJECTS.items():
        release, assets = resolve_project(
            config["repository"], config["assets"], token
        )
        tag = release["tag_name"]
        outputs[f"{key}_tag"] = tag
        outputs.update(assets)
        summary_rows.append(
            {
                "repository": config["repository"],
                "tag": tag,
                "published_at": release["published_at"],
                "assets": list(assets.values()),
            }
        )

    outputs["maafw_python_version"] = maafw_tag_to_pep440(outputs["maafw_tag"])
    return outputs, summary_rows


def append_github_outputs(path: Path, outputs: dict[str, str]) -> None:
    """追加简单的 GitHub Actions step outputs。"""
    with path.open("a", encoding="utf-8", newline="\n") as output_file:
        for key, value in outputs.items():
            if "\n" in value or "\r" in value:
                raise ReleaseResolutionError(f"output {key} 含有换行符")
            output_file.write(f"{key}={value}\n")


def append_github_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """记录本次构建实际选中的上游版本和资产。"""
    lines = [
        "## 上游发布版本",
        "",
        "| 仓库 | Tag | 发布时间 | 必需资产 |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        assets = "<br>".join(f"`{name}`" for name in row["assets"])
        lines.append(
            f"| `{row['repository']}` | `{row['tag']}` | "
            f"`{row['published_at']}` | {assets} |"
        )
    lines.append("")
    with path.open("a", encoding="utf-8", newline="\n") as summary_file:
        summary_file.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="解析双 GUI 构建使用的上游 Release")
    parser.add_argument(
        "--github-output",
        default=os.environ.get("GITHUB_OUTPUT"),
        help="GitHub Actions output 文件，默认读取 GITHUB_OUTPUT",
    )
    parser.add_argument(
        "--github-summary",
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="GitHub Actions summary 文件，默认读取 GITHUB_STEP_SUMMARY",
    )
    parser.add_argument(
        "--github-token",
        default=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    try:
        outputs, summary_rows = resolve_all(args.github_token)
        for key, value in outputs.items():
            print(f"{key}={value}")
        if args.github_output:
            append_github_outputs(Path(args.github_output), outputs)
        if args.github_summary:
            append_github_summary(Path(args.github_summary), summary_rows)
    except (
        ReleaseResolutionError,
        HTTPError,
        URLError,
        HTTPException,
        OSError,
    ) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
