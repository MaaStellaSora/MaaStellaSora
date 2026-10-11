# Android 构建与发布交接

本文记录 MaaStellaSora Android 构建的背景、工作流接口与关键设计选择，供后续维护者修改 Android 构建或接入发布流程时参考。Android 客户端外壳由 [MaaFwApp](https://github.com/Aliothmoon/MaaFwApp) 提供，本仓库只准备项目资源、Python Agent 和 MaaFramework 运行库，再通过 MaaFwApp 的构建配方生成 APK。

## 目标与边界

- 尽量把 Android 专属逻辑限制在 `.github/workflows/android.yml`，不修改桌面端资源定义和发行布局。
- 独立构建可用于 PR、手动调试和 `install.yml` 的复用调用。
- 所有构建使用调用方解析出的同一个版本号；Android APK 与桌面包在正式发布时使用同一 Release tag。
- 非正式构建使用 MaaFwApp 默认的 debug 签名；只有发布工作流明确判定为 release 时才使用 release 签名。
- CI 做可自动化的 APK / PI 静态检查；模拟器或真机上的安装、授权和任务启动仍需实际验证。

## 文件职责

| 路径 | 职责 |
| --- | --- |
| `.github/workflows/android.yml` | 可复用的 Android 构建入口：准备 Agent 与 PI、生成 MaaFwApp profile、构建和检查 APK、上传 Actions artifact |
| `.github/workflows/install.yml` | 解析版本和发布类型；调用 Android workflow；正式发布时下载 Android artifact 并附加到 GitHub Release |
| `assets/interface.json` | 桌面与 Android 共用的 PI 源入口；包含 GitHub 仓库、资源及导入声明 |
| `assets/welcome_android.md` | Android 专属欢迎内容；暂存后的 PI `welcome` 指向此文件，不改变源 `interface.json` 的桌面行为 |
| `assets/logo.png` | MaaFwApp profile 中配置的 Android 应用图标 |
| `agent/` | 随 PI 打包的 Agent 脚本；Android 运行时由 profile 声明的 Python bundle 启动 |

## 工作流调用与产物流转

`install.yml` 的 `meta` job 负责决定 `tag` 和 `is_release`。Android job 通过本地 reusable workflow 调用传入：

| Android workflow 接口 | 调用值 | 用途 |
| --- | --- | --- |
| `version_name`（必需字符串） | `needs.meta.outputs.tag` | 同时写入 Android PI `interface.json.version` 和 APK `versionName` |
| `release`（布尔值） | `needs.meta.outputs.is_release == 'true'` | 选择 debug 或 release 构建及签名 |
| `ANDROID_SIGNING_KEYSTORE_BASE64` | 仓库签名 secret | release keystore 的 Base64 内容 |
| `ANDROID_SIGNING_KEYSTORE_PASSWORD` | 仓库签名 secret | keystore 密码 |
| `ANDROID_SIGNING_KEY_ALIAS` | 仓库签名 secret | 证书 alias |
| `ANDROID_SIGNING_KEY_PASSWORD` | 仓库签名 secret | 证书密码 |

秘密只在调用方判定为 release 时传递。`android.yml` 仍自行检查四项签名资料并验证 keystore alias；release 缺少密钥时应失败，不能静默降级成 debug 签名包。

构建成功后，Android workflow 上传名称为 `MaaStellaSora-android-arm64-v8a-<版本>` 的 artifact，其中 APK 文件名是 `MaaStellaSora-android-arm64-v8a-<版本>.apk`。`install.yml` 的 release job 等待 Android job 成功，下载同一运行中的所有平台 artifacts，检查该 APK 存在且非空，再通过 GitHub Release action 将 `*.apk` 与桌面压缩包一并发布。

结果如下：

| 运行情形 | Android 结果 |
| --- | --- |
| 被 PR / 手动 / 非 release 事件触发的 `install.yml` 构建 | debug 签名 APK 保留为 Actions artifact；不附加到正式 Release |
| `install.yml` 判断为 release | release 签名 APK；作为正式 Release 资产与桌面版本一起发布 |
| 独立运行 `android.yml` 的 `workflow_dispatch` | 默认 debug 构建；可以提供 `version_name` 覆写版本，但不提供 release 签名入口 |

debug 与 release 使用不同的 Android applicationId 和桌面显示名：

| 构建类型 | MaaFwApp profile 配置 | 最终包名 | 桌面显示名 |
| --- | --- | --- | --- |
| debug | `app.id: maastellasora.debug` | `com.aliothmoon.maafw.maastellasora.debug` | `MaaStellaSora Debug` |
| release | `app.id: maastellasora` | `com.aliothmoon.maafw.maastellasora` | `MaaStellaSora` |

不同包名使 debug 与 release 可同时安装；debug 不会覆盖 release 的应用数据，也不会因签名不同而发生 Android 覆盖安装失败。正式版包名保持稳定，确保后续正式版仍以同一 applicationId 和 release 密钥升级。

向 `install.yml` 增加新的调用参数时，须同时核对 `android.yml` 的 `workflow_call.inputs` / `workflow_call.secrets`。发布资产文件名同时被 artifact 上传、release 前检查和 Release 文件 glob 使用，改名时必须一并调整。

`android.yml` 有独立的 PR 触发器，监听工作流本身、`assets/logo.png` 和 `assets/welcome_android.md`；这些 Android 专属改动会直接构建 Android debug APK。共享资源（例如 `assets/interface/**`、`assets/resource/**`、`agent/**`）由 `install.yml` 的 `pull_request.paths` 触发集成构建，并调用 Android reusable workflow。若同一个 PR 同时修改 Android 专属文件和共享资源，两个 workflow 都会启动，Android 构建可能重复；这是专属快速反馈与共享资源全平台验证之间的取舍。修改任一 workflow 的 PR 路径过滤器时，应确认对应文件仍至少由一个构建入口覆盖。

## Android 打包过程

`android.yml` 的步骤按 MaaFwApp 的 PI 打包模型组织：

1. 检出本仓库和 MaaFwApp，安装 Python 3.12、JDK 17 与 Android SDK。
2. 使用 MaaFwApp 的 `build_agent_bundle.py` 为 `arm64-v8a` 构建 Python Agent bundle；依赖来自 `agent/requirements.txt` 和 Chaquopy Android wheel 索引。
3. 建立独立暂存 PI `android-build/pi/`，复制 `assets/` 与 `agent/`，确保这里的根级存在 `interface.json`。只在暂存副本中更新版本与 Android welcome 字段。
4. 构建前检查 `interface.json` 的 `import`、`resource.path` 和 `controller.attach_resource_path` 都能从 PI 根解析。
5. 生成 MaaFwApp `profile.yaml`：PI 根作为 `assets`，显式列出 include；`app.icon` 指向项目 logo，Agent 运行时指向刚生成的 bundle。
6. 下载并部署 ARM64 MaaFramework runtime，写入 MaaFwApp 所需 `local.properties`，构建所选 APK variant。
7. 检查 APK 包名、版本、签名、原生库和 `assets/pi.zip` 内容；验证通过后按统一名称上传 artifact。

**暂存目录是有意设计，不要直接改写源资源。** MaaFwApp profile 的 `assets` 必须指向包含根级 `interface.json` 的目录；PI 的导入和资源路径相对于该目录解析。`include` 一旦显式提供，就取代 MaaFwApp 的默认 include 列表，所以新增 PI 文件时要确认它已被纳入。修改资源或 Agent 入口时，应同步核对 PI 源定义、暂存检查、profile include 和最终 APK 检查。

Android 的 `welcome` 和版本值均只更改 PI 暂存副本，避免为了 Android UI 修改桌面端配置。图标来自 `assets/logo.png`。若未来某项配置需要跨平台共用，再判断是否应移入源 `interface.json`，不要仅为了 Android 构建方便改变桌面行为。

## 版本号与资源重解包

这里涉及三个不同概念，不要混为一谈：

1. `interface.json.version` 是 PI 元数据版本。Android 构建将其改为调用传入的版本。
2. APK `versionName` 是 MaaFwApp 和 Android 系统显示的应用版本，也由同一 `version_name` 设置。MaaFwApp 的 GitHub 更新检查以应用版本与 Release tag 比较，不是用 PI 版本拼装下载地址。
3. APK `versionCode` 是 Android 整数版本。所核对的 MaaFwApp 构建逻辑由 `git rev-list --count HEAD` 产生；目前 MaaFwApp 没有提供对应的 `build.versionCode` 项目覆写接口。当前 MaaFwApp checkout 使用 `actions/checkout` 默认浅克隆，因此该命令在构建中会得到 `1`。它同时被 MaaFwApp 用作 PI 解包标记，但本项目不负责重新定义其生成规则。

因此，设置 `version_name` 不会递增 `versionCode`。若更新 APK 后应用仍沿用旧的已解包资源，用户可在 MaaFwApp 设置中手动执行“重新解压资源”。这是本项目当前采用的低复杂度处理方式，不应误以为 PI `version` 或 `versionName` 会自动触发解包。若今后决定依赖自动 PI 重解包，应先确认上游支持的版本策略；不要只改 `versionName` 并假设它会影响解包标记。

非 release 版本由 `install.yml` 根据最新 Release tag、日期和提交生成，格式须继续符合 `android.yml` 的版本校验规则，并可被 MaaFwApp 的 SemVer 更新比较接受。正式 release 则以 tag 作为唯一版本输入。不要让 Android workflow 在被 `install.yml` 调用时另行推导版本，否则 APK、PI 与桌面产物会不一致。

## GitHub 更新与 Release 资产

`assets/interface.json` 中的 `github` 指向 `https://github.com/MaaStellaSora/MaaStellaSora`。MaaFwApp 根据该仓库的 GitHub Releases 检查版本，再从对应 Release 的 assets 中按 APK 扩展名和包内 ABI 选择安装包。当前 workflow 只构建 `arm64-v8a`，因此资产名必须包含 MaaFwApp 识别的 ARM64 标记 `arm64-v8a`。

**Actions artifact 不是 GitHub Release asset。** 单独运行 Android workflow 时，APK 可供维护者从 Actions 下载，但 MaaFwApp 的 GitHub 更新器不会从 Actions artifacts 获取 APK。只有 `install.yml` 的 release job 将 APK 附加到对应 GitHub Release 后，应用内 GitHub 更新源才有可下载资产。

当前 Release 附件需要稳定地满足以下配对关系。这里的包名为 release applicationId；debug 包使用独立的 `.debug` 后缀，不会随 Release 发布：

```text
Release tag: vX.Y.Z
APK asset:   MaaStellaSora-android-arm64-v8a-vX.Y.Z.apk
APK versionName: vX.Y.Z
PI version:  vX.Y.Z
```

请保持仓库地址、tag / `version_name`、APK 文件名及 ABI 标记一致。若改为多 ABI 构建，应同时检查 MaaFwApp 的 ABI 选择逻辑、APK 文件名、CI 产物检查和 Release 资产，不要只改 Gradle ABI。

## 签名密钥

正式 release 使用仓库 Actions secrets：

- `ANDROID_SIGNING_KEYSTORE_BASE64`
- `ANDROID_SIGNING_KEYSTORE_PASSWORD`
- `ANDROID_SIGNING_KEY_ALIAS`
- `ANDROID_SIGNING_KEY_PASSWORD`

Base64 secret 应由正式发布使用的 keystore 编码得到。密钥不得加入仓库、artifact、日志或本地构建输出；应在仓库设置中配置并限制有权修改 workflow / secrets 的人员。后续 release 必须使用同一应用签名密钥，否则 Android 不会把新 APK 视为现有安装的可覆盖升级。Debug APK 使用 MaaFwApp 的默认 debug key，只适用于调试和 CI artifact。

正式发版前应确认上述 secrets 已配置，并用 release 构建验证 keystore、alias、签名与设备升级安装。仅检查 `apksigner verify` 只能证明 APK 有有效签名，不能证明它使用了预期的发布证书。

## 产物检查的范围

现有 workflow 在构建后按 variant 检查对应的包名，并检查：

- APK 唯一性、debug / release 对应的包名、预期 `versionName` 和 APK 签名。
- MaaFramework ARM64 必需 `.so` 文件。
- APK 中的 `assets/pi.zip`，及其根级 `interface.json`、PI 版本、Android welcome、导入文件、资源目录文件与 `agent/main.py`。
- 最终 artifact 文件名与 `install.yml` 发布检查要求相符。

这些是可重复的静态检查，不验证 Android 系统安装、Shizuku / root 授权、Python 运行时加载、控制器连接或任务实际执行。大范围改动应另做模拟器或真机冒烟测试，并按 MaaFwApp 日志指引收集运行信息。

## 修改时的维护清单

- 改输入参数：同步核对 `android.yml` 的手动触发与 `workflow_call` 接口，以及 `install.yml` 的调用。
- 改 PI 文件或路径：保持暂存 PI 根级 `interface.json` 正确；同步核对 imports、resources、include 和 APK archive 检查。
- 改版本策略：确认正式 tag、非 release CI 版本、PI version、APK `versionName`、artifact 名和 release 检查一致。
- 改 ABI：同步 Agent bundle、MaaFramework runtime、MaaFwApp profile、APK ABI 检查和资产命名。
- 改签名：确保 debug/release 分支仍明确分离；不得在缺少 release 密钥时成功产出未签名正式包。
- 改上传或下载 artifact：确认 Android 是 `install.yml` release job 的依赖，并验证 release job 能以预期目录结构找到 APK。
- 保持 GitHub Actions 引用固定到完整 commit SHA，并保留注释中的上游版本标签。
- 文档或 workflow 修改后至少检查 YAML 解析、Reusable workflow inputs/secrets 对齐、内嵌脚本语法和 `git diff --check`；真正发版仍需在 GitHub Actions 运行验证。

## 设计参考

- MaaFwApp 资源接入说明：[INTEGRATION.md](https://github.com/Aliothmoon/MaaFwApp/blob/main/INTEGRATION.md)
- MaaFwApp profile 样例：[pi-profile.sample.yaml](https://github.com/Aliothmoon/MaaFwApp/blob/main/pi-profile.sample.yaml)
- MaaFwApp 上游项目：[Aliothmoon/MaaFwApp](https://github.com/Aliothmoon/MaaFwApp)
- 版本生成实现：[GitVersion.kt](https://github.com/Aliothmoon/MaaFwApp/blob/main/build-logic/convention/src/main/kotlin/com/aliothmoon/maafw/gradle/GitVersion.kt)
- Android 构建配置：[AndroidApplicationConventionPlugin.kt](https://github.com/Aliothmoon/MaaFwApp/blob/main/build-logic/convention/src/main/kotlin/com/aliothmoon/maafw/gradle/AndroidApplicationConventionPlugin.kt)
- PI 安装标记与强制重解：[PiInstaller.kt](https://github.com/Aliothmoon/MaaFwApp/blob/main/app/src/main/java/com/aliothmoon/maafw/project/PiInstaller.kt)、[PiInstallCoordinator.kt](https://github.com/Aliothmoon/MaaFwApp/blob/main/app/src/main/java/com/aliothmoon/maafw/project/PiInstallCoordinator.kt)
- MaaFwApp 更新实现：[GitHubUpdate.kt](https://github.com/Aliothmoon/MaaFwApp/blob/main/app/src/main/java/com/aliothmoon/maafw/update/GitHubUpdate.kt)

版本生成实现按 MaaFwApp 提交 `aefe8f217d174673ba98d467a3248258659e042e` 核对。构建工作流目前检出 MaaFwApp 默认分支，而不是固定该提交；这样能跟进上游，但也意味着上游构建接口变化可能影响本项目 CI。若要固定上游版本，应选择一个明确的 MaaFwApp tag 或 commit，并建立后续升级该 pin 的维护流程；无明确升级策略前不要随意 pin 到临时检出的版本。无论是否固定，上游默认分支或 pin 更新后，都应重新核对本文记录的构建参数、版本行为和更新逻辑。
