# PR 自动审计

本仓库使用互补的自动审计层，避免把安全、依赖和业务逻辑问题都交给同一种工具：

| 检查 | 作用 | PR 结果 |
| --- | --- | --- |
| CodeRabbit | 审查业务逻辑、回归风险和可维护性 | 中文摘要、逐行评论、必要时请求修改 |
| CodeQL | 审计 Python 与 GitHub Actions 的安全数据流 | PR 注解和 Security 告警 |
| Dependency Review | 检测 PR 新增或升级的已知漏洞依赖 | 中危及以上漏洞阻止合并 |
| Ruff | 快速检测 Python 语法、未定义名称等确定性错误 | 失败即阻止合并 |
| actionlint | 校验 GitHub Actions 语法和表达式 | 失败即阻止合并 |
| zizmor | 检查 Actions 表达式注入、权限和供应链风险 | 高危且高置信问题阻止合并 |
| Dependabot | 维护 Python、npm 与 GitHub Actions 依赖 | 每周创建分组升级 PR |

## 仓库管理员的一次性设置

1. 在 [CodeRabbit](https://app.coderabbit.ai/) 使用 GitHub 登录，将 GitHub App 安装到 `MaaStellaSora/MaaStellaSora`。根目录的 `.coderabbit.yaml` 会自动生效；公开仓库使用 OSS 免费方案。
2. 在 `Settings > Advanced Security` 中确认 Dependency graph、Dependabot alerts 和 Dependabot security updates 已启用。不要同时启用 CodeQL default setup，本仓库的 `pr_audit.yml` 已使用 advanced setup。
3. `PR Audit` 首次在 `main` 成功运行后，在 `Settings > Rules > Rulesets` 为 `main` 添加分支规则，要求 PR 才能合并，并将以下状态检查设为必需：
   - `Static checks`
   - `Dependency review`
   - `CodeQL / python`
   - `CodeQL / actions`
4. 在同一 ruleset 开启 `Require code scanning results`，将 CodeQL 设为必需工具。建议初始阈值使用 `Security alerts: High or higher`，稳定运行一段时间后再评估是否收紧为 `Medium or higher`。
5. 建议同时开启“合并前分支必须为最新”和“所有对话必须解决”。如果维护者经常从 fork 提交 PR，不要给 fork 工作流开放写权限或 secrets。

`PR Audit` 也会在每次推送到 `main` 以及每周定时运行，以发现未经过依赖变更 PR 的新漏洞。Action 引用全部锁定到完整提交 SHA，由 Dependabot 提交升级 PR，评审后再更新。
