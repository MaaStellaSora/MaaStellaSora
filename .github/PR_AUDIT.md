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
| MaaFw Resource | 按项目实际加载顺序校验 base、台/国际/日服及 Windows 叠加资源 | 资源加载失败即阻止合并 |
| Dependabot | 维护 Python、npm 与 GitHub Actions 依赖 | 每周创建分组升级 PR |

## 为什么适合本项目

- `agent/` 与 `tools/ci/` 的 Python 代码由 Ruff 和 CodeQL 互补检查：前者捕获确定性语法与名称错误，后者分析安全数据流。
- 七个 GitHub Actions 工作流由 actionlint、zizmor 和 CodeQL Actions 查询共同审计，覆盖表达式注入、权限和供应链风险。
- `requirements.txt`、`requirements-ci.txt`、`agent/requirements.txt`、`tools/ci/requirements.txt` 和 `package-lock.json` 由 Dependency Review 与 Dependabot 覆盖。
- `assets/interface.json` 定义的资源并非只有 `base`。`check.yml` 会真实调用 MaaFw，分别加载 `base`、`base + tw/en/jp`，并再次覆盖四种 Windows 附加资源组合。
- 资源检查使用固定的 Python 3.12，并在每次运行时安装 MaaFw 最新版本（包含预发行版），以尽早发现上游兼容性变化；Ruff 与 GitHub Actions 则锁定版本并由 Dependabot 提交升级 PR。

## 免费与外部依赖边界

当前仓库是公开仓库，本次 PR 审计与资源检查只使用 `ubuntu-latest` 和 `macos-latest` 标准 GitHub-hosted runner，
不使用 larger runner，也不保存审计 artifact。因此这些检查的 GitHub Actions 运行分钟不收费；CodeQL、
Dependency Review、Dependabot alerts、security updates 和 version updates 对当前公开仓库也无需购买额外产品。

Ruff、actionlint 与 zizmor 均作为开源 GitHub Action 在 runner 内运行，不调用付费 SaaS，
不需要第三方账号、API Key 或仓库 secret。所有 Action 都锁定到完整且不可变的提交 SHA，
再由 Dependabot 提交版本升级 PR，避免跟随可移动 tag 静默更新。

以上免费结论依赖仓库保持公开并继续使用标准 runner。如果仓库改为私有、改用 larger runner、
大量保存 artifact，或引入需要授权的外部服务，必须重新评估 GitHub 计划、Actions 存储与 Code Security 费用。

## 仓库管理员设置

1. 在当前 GitHub 仓库的 `Settings > Advanced Security` 中确认 Dependency graph、Dependabot alerts 和 Dependabot security updates 已启用。不要同时启用 CodeQL default setup，本仓库的 `pr_audit.yml` 已使用 advanced setup。

以下均为可选加固，不影响当前允许有写入权限的维护者直接推送 `main`：

1. `PR Audit` 首次在 `main` 成功运行后，可在 `Settings > Rules > Rulesets` 为 `main` 添加分支规则，要求 PR 才能合并，并将以下状态检查设为必需：
   - `Static checks`
   - `Dependency review`
   - `CodeQL / python`
   - `CodeQL / actions`
   - `Resource checks`
2. 可在同一 ruleset 开启 `Require code scanning results`，将 CodeQL 设为必需工具。建议初始阈值使用 `Security alerts: High or higher`，稳定运行一段时间后再评估是否收紧为 `Medium or higher`。
3. 可同时开启“合并前分支必须为最新”和“所有对话必须解决”。如果维护者经常从 fork 提交 PR，不要给 fork 工作流开放写权限或 secrets。

`PR Audit` 也会在每次推送到 `main` 以及每周定时运行，以发现未经过依赖变更 PR 的新漏洞。Action 引用全部锁定到完整提交 SHA，由 Dependabot 提交升级 PR，评审后再更新；MaaFw 是有意保留的例外，每次资源检查都会解析包含预发行版在内的最新可用版本。
