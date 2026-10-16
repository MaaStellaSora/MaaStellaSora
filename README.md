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
- [x] 领取基金奖励
- [x] 自动爬塔
- [x] 自动进行指定关卡
- [x] 真格挑战
- [x] 猎影合围
- [x] 魔王之塔
- [x] 灾变防线（官服前台限定测试版）

## 安装与使用

### 桌面端（Windows/Linux/macOS）  / 安卓模拟器
> 星塔助手桌面端目前只对比例为16:9的游戏客户端提供支持，如果你的游戏客户端比例不为16:9请自行寻找改分辨率方法或是使用模拟器

1. 前往 [GitHub Releases](https://github.com/MaaStellaSora/MaaStellaSora/releases) 下载对应系统的压缩包。
2. 当前构建与发行使用 **MFAAvalonia**：下载 `MaaStellaSora-{系统}-{架构}-vX.Y.Z.zip`（Windows）或对应的 `.tar.gz`（Linux/macOS），支持现有全部发布平台及自动更新。MXU 构建暂时停用，历史版本的可用包以对应 Release 为准。
3. 将压缩包完整解压到独立目录，运行包内的 MFAAvalonia 主程序。
4. 使用 ADB 操控模拟器时可直接启动 MFAAvalonia；如果需要操控 Windows 版《星塔旅人》，请使用管理员权限运行 MFAAvalonia。

### 安卓端（手机/平板）
> 安卓端目前只构建手机与平板用户使用的arm64-v8a架构版本，如果你是使用安卓模拟器，请使用上面的**MFAAvalonia**版本
1. 前往 [GitHub Releases](https://github.com/MaaStellaSora/MaaStellaSora/releases) 下载 `MaaStellaSora-android-arm64-v8a-vX.Y.Z.apk`。
2. 安卓端构建与发行使用 **MaaFwApp**，暂时只支持通过GitHub更新。
3. 下载后，直接打开安装，即可使用。
4. 使用前，需要先根据提示，使用内置的**Shizuku**权限管理工具获取权限，然后授权MaaStellaSora才能正常使用。
5. 注意：任何调整手机屏幕颜色的功能（包括但不限于护眼模式、省电模式、特殊色彩模式等）都会影响到识别精度，从而出现问题，请在使用时关闭这些功能。

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
- **[MaaFwApp](https://github.com/Aliothmoon/MaaFwApp)** 基于 Android 的 MaaFramework 通用 GUI
- **[MaaPipelineEditor](https://github.com/kqcoxn/MaaPipelineEditor)** 可视化阅读与构建 Pipeline，功能完备，极致轻量跨平台，提供渐进式本地功能扩展，无缝兼容新旧项目
