# PR 自动审计

Sourcery 提供 AI Review，GitHub Actions 提供确定性检查；最终结论仍由维护者判断。

| 检查 | 作用 |
| --- | --- |
| Sourcery | 中文 Review、审查指南和行内建议 |
| Static checks | Ruff、actionlint、zizmor |
| CodeQL / python | Python 安全审计 |
| Dependency review | 阻止引入中危及以上漏洞依赖 |
| Resource checks | 使用 MaaFw 验证各服务区及 Windows 叠加资源 |
| Install | 构建 Windows、Linux、macOS 的 x64/ARM64 安装包 |
| Dependabot | 每周维护依赖 |

## 触发策略

- Python、工作流或依赖配置变更运行 `Static checks`、`Dependency review` 和 `CodeQL / python`。
- 资源变更运行 `Resource checks`，始终使用包含预发行版的最新版 MaaFw。
- PR 仅在安装工作流、`tools/ci/**`、`requirements.txt` 或 `assets/interface.json` 变化时构建六个平台。
- 正式标签始终构建全部六个平台。
- Sourcery 审查指向 `main` 的 PR，但不作为必需检查。

## Fork PR

首次外部贡献者的 GitHub Actions 保留人工批准；Sourcery 可先提供 Review，且不得自动批准工作流或向 fork 提供写令牌和 secrets。
