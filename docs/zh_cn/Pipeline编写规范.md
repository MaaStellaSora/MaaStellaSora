# Pipeline 编写规范

本文约定本项目 Pipeline 的文件组织、字段排列和覆盖写法，适用于 `assets/resource/*/pipeline/`，以及 `assets/interface.json`、`assets/interface/tasks/` 中的 `pipeline_override`。新增和修改节点时使用同一套写法，便于跨模块阅读和审查。

字段含义与执行行为以实际使用版本的 [MaaFramework Pipeline 协议](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/zh_cn/3.1-任务流水线协议.md)和 [Project Interface v2 协议](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/zh_cn/3.3-ProjectInterfaceV2协议.md)为准；本文规定项目内的表达方式。采用新的框架字段时，同时核对发行依赖版本与客户端支持情况。

## 文件组织与命名

- Pipeline 按功能归入 `common/`、`daily/`、`combat/`、`activity/`、`invite/`、`climb_tower/`，具体职责见[项目结构](项目结构.md#目录与入口)。相关流程优先放在已有模块文件中。
- `base` 保存基础流程；`tw`、`en`、`jp` 保存区服差异；`windows` 保存桌面控制器差异。覆盖文件沿用对应基础文件的相对路径，便于对照。
- 节点名采用现有的 `模块_用途` 形式，例如 `邮箱_打开邮箱`、`通用_返回主页`。同一资源包内节点名唯一，跨资源包的同名节点用于覆盖。
- 文件内节点按流程组织，入口及其相关步骤相邻。整理既有文件时保留节点定义顺序，使 diff 集中在实际变化上。
- 图片引用相对于资源包的 `image/` 目录，文件名大小写与实际文件一致。
- 新增界面任务文件时，在 `assets/interface.json` 的 `import` 中登记；新增或移动 Pipeline 文件时，核对 `tools/ci/resource_layout.py` 的发行路径映射。

节点名也是界面 `entry`、`next`、Agent 动态调用和覆盖的引用标识。重命名时同步核对这些引用；任务、选项和 case 的 `name` 还用于配置关联，界面文案优先通过 `label` 表达。

阅读流程时，从界面任务的 `entry` 进入节点图，再检查选项的 `pipeline_override`；遇到 Custom 节点，按 `custom_action` 或 `custom_recognition` 注册名定位 Python 模块。

## v2 节点写法

识别与动作使用对象形式：`type` 在前，`param` 在后。识别参数放在 `recognition.param`，动作参数放在 `action.param`；等待、重复、后继等控制字段放在节点层。

下面是 `daily/mail.json` 中的一个节点，展示识别、动作、后继和提示的排列：

```json
{
    "邮箱_打开邮箱": {
        "recognition": {
            "type": "TemplateMatch",
            "param": {
                "template": [
                    "email.png"
                ]
            }
        },
        "action": {
            "type": "Click"
        },
        "next": [
            "邮箱_一键领取"
        ],
        "focus": {
            "Node.Action.Starting": "正在打开邮箱"
        }
    }
}
```

节点字段按下表分组排列，组内采用表中顺序，只写需要的字段：

| 顺序 | 分组           | 字段                                            |
| ---- | -------------- | ----------------------------------------------- |
| 1    | 控制属性       | `enabled`、`inverse`、`max_hit`、`anchor`       |
| 2    | 识别           | `recognition`                                   |
| 3    | 动作前等待     | `pre_delay`、`pre_wait_freezes`                 |
| 4    | 动作           | `action`                                        |
| 5    | 重复           | `repeat`、`repeat_delay`、`repeat_wait_freezes` |
| 6    | 动作后等待     | `post_delay`、`post_wait_freezes`               |
| 7    | 后继与异常     | `rate_limit`、`timeout`、`next`、`on_error`     |
| 8    | 展示与附加数据 | `focus`、`attach`                               |

这是阅读顺序；实际执行顺序由框架定义。例如等待画面稳定发生在对应的固定延迟之前。新增字段按用途归入相应分组，并同步更新此表。

- 无参数的识别或动作只保留 `type`，省略空 `param`。
- `expected`、`template`、`roi`、`threshold` 等属于识别参数；`target`、`duration` 等动作参数随相应动作类型放置。
- 稳定等待字段使用 `pre_wait_freezes`、`post_wait_freezes`、`repeat_wait_freezes`。
- `attach`、`custom_action_param`、`custom_recognition_param` 的内部结构由使用它们的模块定义，保持对应的数据类型、键名和数组含义。

## 节点引用与执行顺序

普通引用使用节点名字符串；仅包含 `name` 和 `jump_back: true` 的引用写为 `[JumpBack]节点名`。例如 `邮箱_入口` 的后继：

```json
{
    "next": [
        "邮箱_判断是否在主页",
        "[JumpBack]通用_返回主页"
    ]
}
```

含其他属性的引用保留对象形式，以完整表达属性。`JumpBack` 表示子流程结束后回到父节点重新检测后继，具体行为见上游[节点属性](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/zh_cn/3.1-任务流水线协议.md#节点属性)。

`next` 按顺序尝试识别，先命中的节点取得执行机会，因此数组顺序属于流程语义。整理时保留 `next`、`on_error`、模板列表及其他参数数组的原有顺序。文件名和节点定义位置用于组织源码，流程关系通过节点引用表达。

## 区服与界面覆盖

区服覆盖和 `pipeline_override` 都采用相同的 v2 参数层级与字段排列，只声明本层需要改变的字段。同类型的识别或动作仅修改参数时，可以省略沿用的 `type`。

例如英文资源中 `通用_关闭升级界面` 的覆盖，识别类型和其余参数沿用基础节点：

```json
{
    "通用_关闭升级界面": {
        "recognition": {
            "param": {
                "expected": [
                    "Authorization"
                ]
            }
        }
    }
}
```

省略字段表示沿用已加载节点或默认配置。覆盖只声明所需差异字段；`0`、`false`、空数组和空对象分别按所在字段的协议含义处理。识别类型发生变化时，按新类型配置所需参数。

界面中的 `pipeline_override` 以节点名为键，值为节点的局部定义。输入使用 `{输入名}` 占位符，`inputs` 中的 `pipeline_type` 决定替换后的类型，`verify` 表达合法输入范围。例如 `assets/interface.json` 中的快速作战次数以整数传入 `action.param.custom_action_param`，对应覆盖如下：

```json
{
    "pipeline_override": {
        "通用_换算快速作战次数": {
            "action": {
                "type": "Custom",
                "param": {
                    "custom_action": "utool_calc_repeat",
                    "custom_action_param": "{次数}"
                }
            }
        }
    }
}
```

多个选项影响同一节点时，按客户端实际应用顺序核对合并结果，包括嵌套选项和输入替换。`import`、任务选项及 case 数组沿用现有顺序；覆盖 `next` 等数组时提供所需的完整列表。资源叠加顺序见[项目结构](项目结构.md#资源加载)。

## 格式化与修改范围

JSON 文件使用 Tab 缩进，数组布局由 `tools/format/.prettierrc` 和已锁定的 Prettier 插件统一生成。安装、针对改动文件的格式化命令及编辑器配置见[个性化配置](个性化配置.md#格式化与提交检查)。

Prettier 负责缩进、换行等排版；字段顺序、v2 参数层级和覆盖范围在编辑与审查时核对。

字段整理保留参数值与数组顺序。识别文本、阈值、区域、动作、等待时间、重复次数和后继变化按功能修改说明原因并验证。显式时序与识别参数表达现有运行条件，精简这些配置时核对实际行为。公共节点按已有的跨流程复用需求提取，单个模块的逻辑就近维护。

## 提交前验证

1. 检查改动文件的格式、重复键、节点引用和图片路径。编辑器使用 `deps/tools/` 中对应的 Pipeline、Interface 或 Interface import Schema 辅助检查，局部覆盖结合基础节点核对。
2. 使用与目标发行包一致的 MaaFramework 和 Python 绑定执行真实资源加载。受影响的区服同时检查自身组合与 Windows 叠加；公共节点变更覆盖全部八组组合。按[资源加载顺序](项目结构.md#资源加载)调用 `tools/checks/check_resource.py`。
3. 界面修改验证默认选项、相关 case、输入替换及组合覆盖；动态覆盖还验证同一上下文中的连续执行，确认后继和参数随当前输入更新。
4. 流程修改通过客户端实机检查任务启动、关键分支、完成退出，以及受影响的失败和停止路径。记录实际测试的区服、控制器和选项；资源加载成功说明可解析，实机运行用于确认功能。问题定位所需的[截图与日志](../CONTRIBUTING.md#调试截图与日志)按贡献指南收集。
5. 新增或移动资源文件时检查发行包内的路径和加载结果。按贡献指南中的[提交 PR](../CONTRIBUTING.md#提交-pr)说明记录修改目的、实际验证范围及结果，供审查者核对。
