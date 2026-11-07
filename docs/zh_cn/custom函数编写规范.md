# Custom 函数编写规范

本文约定项目中 MaaFramework 自定义动作（CustomAction）和自定义识别（CustomRecognition）的适用场景、模块设计、日志与任务中止检查方式。Custom 函数通过 Python Agent 实现，注册后由 Pipeline 节点调用；节点配置与覆盖写法见 [Pipeline 编写规范](Pipeline编写规范.md)。

## 适用场景与职责划分

新增功能时，优先使用 MaaFramework Pipeline 的识别、动作、后继关系和覆盖能力实现流程控制及简单条件判断。只有当逻辑涉及复杂状态判断、数据处理或多步骤协作，且难以由 Pipeline 清晰表达时，才建议使用 Custom 函数。避免仅为封装简单判断或转发参数而增加 Agent 代码。

Custom 函数应聚焦于 Pipeline 不适合承担的自定义逻辑。尽量让 Pipeline 表达可读的流程结构，让 Python 模块处理必要的复杂计算、状态分析和数据组织；不要把本可由 Pipeline 清晰描述的整个任务流程搬入 Agent。

## 模块组织与扩展

按功能和职责组织模块，沿用 `agent/custom/action/` 与 `agent/custom/reco/` 下的现有目录结构。使用 `AgentServer.custom_action` 或 `AgentServer.custom_recognition` 注册实现，并确保注册名与 Pipeline 中的 `custom_action` 或 `custom_recognition` 引用完全一致。注册名是调用接口，修改时应同步检查所有引用。

模块入口负责适配 MaaFramework 参数、调用内部逻辑并返回接口要求的结果；可复用的业务判断和数据处理应拆分到职责明确的函数或组件中，避免在注册类中堆积过多细节。

当同一功能需要支持多种可替换的处理策略，或预计后续会增加、调整策略时，采用策略模式组织实现：为策略约定一致的输入和输出，将不同策略的实现相互隔离，并由独立的选择逻辑根据配置或运行状态确定具体策略。避免将多种策略及其分支判断持续堆积在 Custom 函数入口中。若功能只有单一且稳定的实现，不必为了使用设计模式而增加额外抽象。

## MaaFramework 接口参考

MaaFramework 的 Agent 编写说明可能不覆盖所有 Python 绑定接口用法。实现前应先确认项目实际使用的 MaaFramework 版本，并查阅对应版本的绑定源码、方法文档字符串和类型定义；也可结合项目中已有的自定义动作与识别实现了解调用方式。不要仅依据其他版本的示例推断接口行为。

重点核对回调方法、参数类型、返回值及上下文操作，并根据实际接口约定处理失败和停止场景。项目中的实现示例包括 `agent/custom/action/` 和 `agent/custom/reco/`；Pipeline 中的调用方式见 [Pipeline 编写规范](Pipeline编写规范.md#节点引用与执行顺序)。

## 任务中止检查

MaaFramework 不会强行中断正在执行的 Custom 函数。用户请求中止任务后，可通过 `context.tasker.stopping` 检查当前任务是否已进入停止状态；Custom 函数应在适当的检查点停止后续工作，并按对应接口约定返回。

停止状态不会自动阻止函数继续运行：例如，任务进入停止状态后，`context.run_recognition` 仍可能正常执行；而 `context.run_task` 会立即返回。因此，涉及批量调用 `context.run_recognition`、长循环或大量计算时，应在开始耗时工作前及循环中的适当位置检查停止状态，避免任务中止后继续执行不必要的工作。无需在每个简单操作前后重复检查，应根据操作耗时和可安全退出的位置设置检查点。

## 日志

使用项目提供的 `utils.logger` 获取模块 logger，并以 `__name__` 作为名称，便于区分日志来源：

```python
from utils import logger as logger_module

logger = logger_module.get_logger(__name__)
```

按日志用途选择级别：

- `debug`：排查问题所需的详细信息，例如中间判断结果。
- `info`：关键流程进展和正常执行结果。
- `warning`：可继续运行但需要关注的异常情况。
- `error`：导致当前操作失败、需要定位的问题。

日志应提供定位问题所需的关键上下文和结果。捕获异常时，不要静默忽略；应按场景记录异常信息，并保留必要的堆栈信息。避免在高频循环中重复输出大量日志，也不要记录与问题定位无关的完整图像或冗长数据。

`info`、`warning` 和 `error` 日志面向用户，消息必须使用 `$` 开头的 i18n 键，并在 `agent/agent_config/i18n/` 下维护对应翻译。`debug` 仅用于开发排查，可使用普通日志消息，无需添加 i18n。不要在功能模块中另行配置日志处理器或绕过项目 logger。

## 提交前检查

1. 确认 Custom 注册名与 Pipeline 引用一致，并检查参数传递、节点覆盖及受影响调用方。
2. 检查正常执行、参数或识别失败、任务中止等相关路径；涉及循环、批量识别或大量计算时，确认停止检查能够及时结束后续工作。
3. 通过客户端验证实际任务流程，并记录测试的区服、控制器、选项和结果。贡献说明与调试信息收集方式见[贡献指南](../CONTRIBUTING.md)。