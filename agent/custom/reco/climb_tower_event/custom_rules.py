from typing import Callable
from maa.context import Context

from .models import EventInfo, Choice


CustomFunction = Callable[[Context, dict, EventInfo, list[Choice]], Choice | None]

CUSTOM_RULE_REGISTRY: dict[str, CustomFunction] = {}

def custom_rule(name: str):
    """注册自定义状态判断函数的装饰器"""
    def decorator(func: CustomFunction):
        CUSTOM_RULE_REGISTRY[name] = func
        return func
    return decorator


@custom_rule("sample")
def test_sample(context: Context, rule: dict, event_info: EventInfo, matched_choices: list[Choice]) -> Choice:
    """测试用"""
    return matched_choices[0]
