<!-- markdownlint-disable MD033 MD041 -->

<div align="center">
    <img src="assets/logo.png" alt="StellaSora-Auto-Helper" width="200" />
    <h1>MaaStellaSora</h1>
    <p>星塔助手（MaaStellaSora）提供自动签到、清理日常等功能，由 MaaFramework 强力驱动</p>
</div>

遇到问题请去Issues反馈，或前往QQ交流群进行反馈

QQ交流群：**1063132902**  密码：**星塔旅人**

## 功能

- [x] 登录游戏并签到
- [x] 清理活动
- [x] 赠礼
- [x] 五次邀约
- [x] 领取&发送好友干劲
- [x] 领取委托并重新派遣
- [x] 领取任务
- [x] 自动爬塔
- [x] 自动进行指定关卡
- [x] 真格挑战
- [x] 猎影合围
- [x] 灾变防线（官服 Windows 桌面端 1280×720）

灾变防线会自动选择灾变难度、沿用已有纪录或为两个区域的空位选择纪录，完成命运卡片选择、上下半场战斗、走位进传送门、领奖及返回主页。
请使用「桌面端（前台）」控制器，并保持游戏窗口在前台；走位使用真实键盘输入，会占用操作，鼠标可能被短暂接管。
目前仅支持官服 Windows 桌面端 **1280×720**，不支持安卓、模拟器、PlayCover 或其他分辨率。每期只能挑战一次，且仅从当前屏幕选择纪录，不会下滑搜索，请提前准备可用纪录。

## 安装与使用

### 桌面端

> 星塔助手目前只对比例为16:9的游戏客户端提供支持，如果你的游戏客户端比例不为16:9请自行寻找改分辨率方法或是使用模拟器或安卓版

1. 前往 [GitHub Releases](https://github.com/MaaStellaSora/MaaStellaSora/releases) 下载对应系统的压缩包。
2. 当前构建与发行使用 **MFAAvalonia**：下载 `MaaStellaSora-{系统}-{架构}-vX.Y.Z.zip`（Windows）或对应的 `.tar.gz`（Linux/macOS），支持现有全部发布平台及自动更新。MXU 构建暂时停用，历史版本的可用包以对应 Release 为准。
3. 将压缩包完整解压到独立目录，运行包内的 MFAAvalonia 主程序。
4. 如果需要操控 Windows 版《星塔旅人》，请使用管理员权限运行 GUI；使用 ADB 操控模拟器时可直接启动。

### 安卓端

> 安卓端目前只构建手机与平板用户使用的arm64-v8a架构版本，如果你的设备不是arm64-v8a架构，请使用桌面端或者自行构建

1. 前往 [GitHub Releases](https://github.com/MaaStellaSora/MaaStellaSora/releases) 下载 `MaaStellaSora-android-arm64-v8a-vX.Y.Z.apk`。
2. 安卓端构建与发行使用 **MaaFwApp**，暂时只支持通过GitHub更新。
3. 下载后，直接打开安装，即可使用。

## 参与开发或贡献

选择[贡献方式](docs/CONTRIBUTING.md#贡献方式)，反馈问题时参考[调试截图与日志](docs/CONTRIBUTING.md#调试截图与日志)，完成修改后按[提交 PR](docs/CONTRIBUTING.md#提交-pr)准备说明与验证结果。

参考文档：[项目结构](docs/zh_cn/项目结构.md) · [Pipeline 编写规范](docs/zh_cn/Pipeline编写规范.md) · [AI 工具入口](AGENTS.md)

## 鸣谢

本项目由 **[MaaFramework](https://github.com/MaaXYZ/MaaFramework)** 强力驱动！

本项目部分功能使用 **[MaaPipelineEditor](https://github.com/kqcoxn/MaaPipelineEditor)** 进行辅助编辑

感谢以下开发者对本项目作出的贡献:

[![Contributors](https://contrib.rocks/image?repo=SodaCodeSave/StellaSora-Auto-Helper&max=1000)](https://github.com/SodaCodeSave/StellaSora-Auto-Helper/graphs/contributors)

## 相关项目

- **[MaaFramework](https://github.com/MaaXYZ/MaaFramework)** 基于图像识别的自动化黑盒测试框架
- **[MFAAvalonia](https://github.com/MaaXYZ/MFAAvalonia)** 基于 Avalonia 的通用 GUI，由 MaaFramework 强力驱动
- **[MXU](https://github.com/MistEO/MXU)** 基于 Web 技术的 MaaFramework 通用 GUI
- **[MaaFwApp](https://github.com/MaaXYZ/MaaFwApp)** 基于 Android 的 MaaFramework 通用 GUI
- **[MaaPipelineEditor](https://github.com/kqcoxn/MaaPipelineEditor)** 可视化阅读与构建 Pipeline，功能完备，极致轻量跨平台，提供渐进式本地功能扩展，无缝兼容新旧项目
