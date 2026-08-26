#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

REVIEW_MARKER = "<!-- maa-pr-audit-review -->"
AUDIT_CHECK_NAMES = {
    "Static checks",
    "Dependency review",
    "CodeQL / python",
    "CodeQL / actions",
}
RESULT_LABELS = {
    "success": "✅ 通过",
    "failure": "❌ 未通过",
    "cancelled": "⚪ 已取消",
    "skipped": "⏭️ 已跳过",
    "timed_out": "⌛ 已超时",
    "action_required": "⚠️ 需要处理",
}
BLOCKING_RESULTS = {"failure", "timed_out", "action_required"}


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class GitHubClient:
    def __init__(self, token: str, api_url: str) -> None:
        self._token = token
        self._api_url = api_url.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            f"{self._api_url}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "User-Agent": "MaaStellaSora-PR-Audit",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"GitHub API {method} {path} returned {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"GitHub API {method} {path} could not be reached: {error.reason}"
            ) from error

        if not content:
            return None
        return json.loads(content)

    def list_all(self, path: str) -> list[Any]:
        items: list[Any] = []
        separator = "&" if "?" in path else "?"
        page = 1
        while True:
            result = self.request(
                "GET", f"{path}{separator}per_page=100&page={page}"
            )
            if not isinstance(result, list):
                raise TypeError(f"Expected a list from GitHub API path: {path}")
            items.extend(result)
            if len(result) < 100:
                return items
            page += 1


def clean_text(value: Any, limit: int = 300) -> str:
    text = " ".join(str(value or "").split()).replace("@", "@\u200b")
    if len(text) > limit:
        return f"{text[: limit - 1]}…"
    return text


def format_annotation(annotation: dict[str, Any]) -> str:
    path = clean_text(annotation.get("path"), 160).replace("`", "'") or "未知文件"
    line = annotation.get("start_line")
    location = f"`{path}:{line}`" if line else f"`{path}`"
    title = clean_text(annotation.get("title"), 120)
    message = clean_text(annotation.get("message"))
    detail = " — ".join(part for part in (title, message) if part)
    return f"- {location}：{detail or '请查看对应检查注解。'}"


def collect_annotations(
    client: GitHubClient, repository: str, check_sha: str
) -> list[str]:
    result = client.request(
        "GET", f"/repos/{repository}/commits/{check_sha}/check-runs?per_page=100"
    )
    check_runs = result.get("check_runs", []) if isinstance(result, dict) else []
    findings: list[str] = []
    seen: set[tuple[Any, ...]] = set()

    for check_run in check_runs:
        if check_run.get("name") not in AUDIT_CHECK_NAMES:
            continue
        annotations = client.list_all(
            f"/repos/{repository}/check-runs/{check_run['id']}/annotations"
        )
        for annotation in annotations:
            if annotation.get("annotation_level") not in {"failure", "warning"}:
                continue
            key = (
                annotation.get("path"),
                annotation.get("start_line"),
                annotation.get("title"),
                annotation.get("message"),
            )
            if key in seen:
                continue
            seen.add(key)
            findings.append(format_annotation(annotation))
            if len(findings) == 10:
                return findings

    return findings


def build_review_body(
    *,
    repository: str,
    pull_number: str,
    head_sha: str,
    run_id: str,
    server_url: str,
    changed_files: int,
    additions: int,
    deletions: int,
    results: list[tuple[str, str]],
    findings: list[str],
) -> str:
    result_rows = "\n".join(
        f"| {name} | {RESULT_LABELS.get(result, f'⚪ {result}')} |"
        for name, result in results
    )
    has_blocker = any(result in BLOCKING_RESULTS for _, result in results)
    incomplete = any(result not in {"success", "failure"} for _, result in results)

    if has_blocker or findings:
        conclusion = "⚠️ 发现需要处理的高置信度问题，请先查看下方建议和对应检查注解。"
        recommendation = "优先修复未通过的检查；完成后推送新提交，机器人会重新审查。"
    elif incomplete:
        conclusion = "⏳ 部分检查未完整结束，请在全部完成后再决定是否合入。"
        recommendation = "请打开 Checks 页面确认取消、跳过或超时的具体原因。"
    else:
        conclusion = "✅ 自动审计未发现高置信度阻断问题。"
        recommendation = "可以进入人工复核；仍需重点确认业务逻辑、交互体验和需求符合度。"

    finding_section = ""
    if findings:
        finding_section = "\n### 高置信度发现\n\n" + "\n".join(findings) + "\n"

    checks_url = f"{server_url}/{repository}/pull/{pull_number}/checks"
    run_url = f"{server_url}/{repository}/actions/runs/{run_id}"
    return f"""{REVIEW_MARKER}
## PR 自动审计概览

`github-actions[bot]` 已审查提交 `{head_sha[:7]}`。本次变更涉及 **{changed_files}** 个文件，新增 **{additions}** 行、删除 **{deletions}** 行。

| 检查 | 结果 |
| --- | --- |
{result_rows}

### 审计结论

{conclusion}
{finding_section}
### 建议

{recommendation}

[查看全部 PR 检查]({checks_url}) · [查看本次审计日志]({run_url})

---

此 Review 仅汇总确定性、高置信度自动检查，不会自动批准、请求变更或合并 PR；业务逻辑仍需维护者人工审查。
"""


def main() -> int:
    token = require_env("GH_TOKEN")
    repository = require_env("GITHUB_REPOSITORY")
    pull_number = require_env("PR_NUMBER")
    head_sha = require_env("HEAD_SHA")
    check_sha = require_env("CHECK_SHA")
    run_id = require_env("GITHUB_RUN_ID")
    server_url = require_env("GITHUB_SERVER_URL")
    api_url = require_env("GITHUB_API_URL")
    results = [
        ("静态检查", require_env("STATIC_RESULT")),
        ("依赖审查", require_env("DEPENDENCY_RESULT")),
        ("CodeQL", require_env("CODEQL_RESULT")),
    ]

    client = GitHubClient(token, api_url)
    reviews = client.list_all(f"/repos/{repository}/pulls/{pull_number}/reviews")
    if any(
        review.get("commit_id") == head_sha
        and REVIEW_MARKER in (review.get("body") or "")
        for review in reviews
    ):
        print(f"Review for commit {head_sha[:7]} already exists; nothing to do.")
        return 0

    pull = client.request("GET", f"/repos/{repository}/pulls/{pull_number}")
    findings = collect_annotations(client, repository, check_sha)
    body = build_review_body(
        repository=repository,
        pull_number=pull_number,
        head_sha=head_sha,
        run_id=run_id,
        server_url=server_url,
        changed_files=int(pull["changed_files"]),
        additions=int(pull["additions"]),
        deletions=int(pull["deletions"]),
        results=results,
        findings=findings,
    )
    review = client.request(
        "POST",
        f"/repos/{repository}/pulls/{pull_number}/reviews",
        {"body": body, "commit_id": head_sha, "event": "COMMENT"},
    )
    print(f"Created PR review: {review.get('html_url', review.get('id'))}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, TypeError) as error:
        print(f"::error::{error}", file=sys.stderr)
        raise SystemExit(1) from error
