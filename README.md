<!-- markdownlint-disable MD033 MD041 -->

<div align="center">
    <img src="assets/logo.png" alt="StellaSora-Auto-Helper" width="200" />
    <h1>MaaStellaSora</h1>
    <p>星塔助手（MaaStellaSora）提供自动签到、清理日常等功能，由 MaaFramework 强力驱动</p>
</div>

遇到问题请去Issues反馈，或前往QQ交流群进行反馈

QQ交流群：**1063132902**  密码：**星塔旅人**

> 项目当前仍处于预览版，可能会遇到部分问题

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
- [ ] 自动刷记录（根据优先度）
- [ ] 更多内容实现中

## 安装与使用

> 星塔助手目前只对比例为16:9的游戏客户端提供支持，如果你的游戏客户端比例不为16:9请自行寻找改分辨率方法或是使用模拟器

1. 前往 [GitHub Releases](https://github.com/MaaStellaSora/MaaStellaSora/releases) 下载对应系统的压缩包。
2. 根据需要选择 GUI：
   - **MFAAvalonia（推荐）**：下载 `MaaStellaSora-{系统}-{架构}-vX.Y.Z.zip`（Windows）或对应的 `.tar.gz`（Linux/macOS），支持现有全部发布平台及自动更新。
   - **MXU**：Windows x64 用户可下载 `MaaStellaSora-mxu-win-amd64-vX.Y.Z.zip`。MXU 当前不支持自动更新，后续版本需重新前往 GitHub Releases 下载。
3. 将压缩包完整解压到任意目录，运行包内的 MFAAvalonia 主程序；MXU 用户运行 `mxu.exe`。
4. 如果需要操控 Windows 版《星塔旅人》，请使用管理员权限运行 GUI；使用 ADB 时可直接启动。

## 鸣谢

本项目由 **[MaaFramework](https://github.com/MaaXYZ/MaaFramework)** 强力驱动！

本项目部分功能使用 **[MaaPipelineEditor](https://github.com/kqcoxn/MaaPipelineEditor)** 进行辅助编辑

感谢以下开发者对本项目作出的贡献:

[![Contributors](https://contrib.rocks/image?repo=SodaCodeSave/StellaSora-Auto-Helper&max=1000)](https://github.com/SodaCodeSave/StellaSora-Auto-Helper/graphs/contributors)

## 相关项目

- **[MaaFramework](https://github.com/MaaXYZ/MaaFramework)** 基于图像识别的自动化黑盒测试框架
- **[MFAAvalonia](https://github.com/MaaXYZ/MFAAvalonia)** 基于 Avalonia 的通用 GUI，由 MaaFramework 强力驱动
- **[MXU](https://github.com/MistEO/MXU)** 基于 Web 技术的 MaaFramework 通用 GUI
- **[MaaPipelineEditor](https://github.com/kqcoxn/MaaPipelineEditor)** 可视化阅读与构建 Pipeline，功能完备，极致轻量跨平台，提供渐进式本地功能扩展，无缝兼容新旧项目
