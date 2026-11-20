"""Ordered tower pages and their existing navigation responsibilities."""
from typing import NamedTuple


class Page(NamedTuple):
    name: str
    node: str
    modal: bool = False


# These are observations after an action, not alternate task entry routes.
PAGES = (
    Page("purchase_confirm", "星塔_节点_商店_购买协奏音符_检查确认框_agent", True),
    Page("note_detail", "星塔_神经网络_音符详情已打开", True),
    Page("note_confirm", "星塔_点击空白处关闭_agent", True),
    Page("bag", "星塔_背包_核实背包界面_agent"),
    Page("settlement", "星塔_记录界面_读取数值_agent"),
    Page("potential", "星塔_节点_进入潜能选择_agent"),
    Page("event", "星塔_节点_进入对话选择_agent"),
    Page("shop", "星塔_节点_商店_确认处于购物界面_agent"),
    Page("shop_main", "星塔_节点_商店_回到商店层_agent"),
    Page("dialogue", "星塔_节点_进入对话_agent"),
    Page("field", "星塔_神经网络_普通场景已出现"),
)
BY_NAME = {page.name: page for page in PAGES}
BAG_HOSTS = frozenset(("potential", "event", "shop", "shop_main"))
BAG_RETURN_PAGES = BAG_HOSTS | {"dialogue", "field", "note_confirm"}


def classify(context, image, *, candidates=None):
    """Recognize one explicit frame; modal pages always retain precedence."""
    for page in PAGES:
        if context.tasker.stopping:
            return None
        if candidates is not None and not page.modal and page.name not in candidates:
            continue
        result = context.run_recognition(page.node, image)
        if result and result.hit:
            return page.name
    return None
