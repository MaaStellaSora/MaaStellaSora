# PR 自动审计

本仓库使用 GitHub 原生能力和仓库内的 GitHub Actions 自动审计 PR。所有检查均由
`github-actions[bot]` 或 `dependabot[bot]` 运行，不需要注册第三方平台、安装额外 GitHub App、
配置第三方 API Key 或再次登录其他服务。

| 检查 | 作用 | PR 结果 |
| --- | --- | --- |
| GitHub Actions | 编排确定性代码质量与安全检查 | Checks 页面结果和 PR 行内注解 |
| CodeQL | 审计 Python 与 GitHub Actions 的安全数据流 | PR 注解和 Security 告警 |
| Dependency Review | 检测 PR 新增或升级的已知漏洞依赖 | 中危及以上漏洞阻止合并 |
| Ruff | 快速检测 Python 语法、未定义名称等确定性错误 | 失败即阻止合并 |
| actionlint | 校验 GitHub Actions 语法和表达式 | 失败即阻止合并 |
| zizmor | 检查 Actions 表达式注入、权限和供应链风险 | 高危且高置信问题阻止合并 |
| Dependabot | 维护 Python、npm 与 GitHub Actions 依赖 | 每周创建分组升级 PR |

## 仓库管理员的一次性设置

1. 在当前 GitHub 仓库的 `Settings > Advanced Security` 中确认 Dependency graph、Dependabot alerts 和 Dependabot security updates 已启用。不要同时启用 CodeQL default setup，本仓库的 `pr_audit.yml` 已使用 advanced setup。
2. `PR Audit` 首次在 `main` 成功运行后，在 `Settings > Rules > Rulesets` 为 `main` 添加分支规则，要求 PR 才能合并，并将以下状态检查设为必需：
   - `Static checks`
   - `Dependency review`
   - `CodeQL / python`
   - `CodeQL / actions`
3. 在同一 ruleset 开启 `Require code scanning results`，将 CodeQL 设为必需工具。建议初始阈值使用 `Security alerts: High or higher`，稳定运行一段时间后再评估是否收紧为 `Medium or higher`。
4. 建议同时开启“合并前分支必须为最新”和“所有对话必须解决”。如果维护者经常从 fork 提交 PR，不要给 fork 工作流开放写权限或 secrets。

`PR Audit` 也会在每次推送到 `main` 以及每周定时运行，以发现未经过依赖变更 PR 的新漏洞。Action 引用全部锁定到完整提交 SHA，由 Dependabot 提交升级 PR，评审后再更新。
