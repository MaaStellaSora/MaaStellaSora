from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .handler import ShopHandler
from .context import Item

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_shop_pipeline_buyer")


def buy_general(item: Item, handler: ShopHandler) -> bool:
    """执行普通商品购买。

    将调用方传入的 reserve_coin 注入潜能选择节点，防止潜能选择把预留给强化的金币刷光。
    由 pipeline 按需取用，调用方无需判断商品类型。

    Args:
        item: 单个格子信息对象。
        handler: 店铺处理器对象。

    Returns:
        bool: 购买任务成功返回 True。
    """
    # 确保处于购物界面
    handler.interactor.enter_shopping()

    # 购买环节
    data = handler.data
    context = handler.interactor.context

    reserved_coin = data.total_enhancement_cost
    potential_source = "specified_drink" if item.trekker_specified else "drink"
    override: dict[str, Any] = {
        "星塔_节点_商店_购物_购买道具_agent": {
            "action": {"param": {"target": item.price_roi}}
        },
        "星塔_节点_选择潜能_agent": {
            "attach": {
                "reserved_coin": reserved_coin,
                "potential_source": potential_source
            }
        }
    }
    logger.info(f"购买 {item.display_name} * {item.quantity} ({item.price})")
    result = context.run_task("星塔_节点_商店_购物_购买道具_agent", override)

    # 购买后处理环节
    item.bought = True  # 不管购买是否成功，均标记为已购买，防止卡死
    handler.interactor.update_image()
    handler.update_coin()
    if result and result.status.succeeded:
        logger.debug(f"购买 {item.display_name} * {item.quantity} 成功")
        if "melody" in item.internal_name: # 如果是音符要更新数量
            handler.data.melodies[item.internal_name].count += item.quantity
        return True
    else:
        logger.error(f"购买 {item.display_name} 过程出现问题")
        return False


def buy_assist_melody(item: Item, handler: ShopHandler) -> bool:
    """执行协奏音符购买，走单独的协奏音符 pipeline。

    Args:
        item: 单个格子信息对象。
        handler: 店铺处理器对象。

    Returns:
        bool: 购买任务成功返回 True。
    """
    # 确保处于购物界面
    handler.interactor.enter_shopping()

    # 开始尝试购买协奏音符
    override: dict[str, Any] = {
        "星塔_节点_商店_购买协奏音符_agent": {
            "action": {"param": {"target": item.price_roi}}
        },
    }
    run_result = handler.interactor.context.run_task("星塔_节点_商店_购买协奏音符_agent", override)
    if not (run_result and run_result.status.succeeded):
        logger.error(f"点击协奏音符 {item.display_name} 过程出现问题")
        return False

    # 开始验证协奏音符
    passed = True
    image = handler.interactor.context.tasker.controller.post_screencap().wait().get()
    # 验证是否是协奏音符
    reco_detail = handler.interactor.context.run_recognition("星塔_节点_商店_购买协奏音符_核实协奏_agent", image)
    if not (reco_detail and reco_detail.hit):
        logger.info(f"{item.display_name} 不是协奏音符")
        passed = False
    # 验证协奏技能是否解锁
    elif handler.data.params.buy_assist_before_unlock and _is_assist_skill_unlocked(handler):
        logger.info(f"{item.display_name} 相关协奏技能已解锁，无需购买")
        passed = False

    # 如果没有通过验证，关闭确认框
    item.checked = True
    if not passed:
        run_result = handler.interactor.context.run_task("星塔_节点_商店_购买协奏音符_退出购买_agent")
        if run_result and run_result.status.succeeded:
            logger.debug(f"关闭购买协奏音符 {item.display_name} 成功")
        else:
            logger.error(f"关闭购买协奏音符 {item.display_name} 过程出现问题")
        return False

    # 通过验证，确认购买
    logger.info(f"购买协奏音符 {item.display_name} * {item.quantity} ({item.price})")
    run_result = handler.interactor.context.run_task("星塔_节点_商店_购物_购买道具_确认购买_agent")

    # 购买后处理环节，更新数据
    item.bought = True # 进入购买环节之后，不管是否成功都标记为已购买，防止卡死
    handler.interactor.update_image()
    handler.update_coin()
    if run_result and run_result.status.succeeded:
        logger.info(f"购买 {item.display_name} 成功")
        if "melody" in item.internal_name: # 如果是音符要更新数量
            handler.data.melodies[item.internal_name].count += item.quantity
        return True
    else:
        logger.error(f"购买 {item.display_name} 过程出现问题")
        return False

def _is_assist_skill_unlocked(handler: ShopHandler) -> bool:
    """检查协奏技能是否已解锁。

    Args:
        handler: 店铺处理器对象。

    Returns:
        bool: 是否已解锁。
    """
    lv0_melody = (10, 15)
    handler.interactor.update_image()

    # 寻找roi左边边界
    if filtered_boxes := handler.interactor.locate_assist_melody_text():
        x = min([r[0] for r in filtered_boxes])
        w = 863 + 42 - x
        roi = [x, 285, w, 22]
    else:
        roi = [863, 285, 42, 22]

    # 正式核实数量
    if text := handler.interactor.get_assist_melody_count(roi):
        current_melody = int(text[:-2])
        required_melody = int(text[-2:])
        logger.debug(f"识别到的现有音符数量：{current_melody}，协奏技能升级要求数量：{required_melody}")

        # 协奏技能未解锁，需要符合：
        # 1. 升级要求音符数量为 lv0_melody 中的值
        # 2. 且当前音符数量小于升级要求数量
        if required_melody in lv0_melody and current_melody < required_melody:
            return False
        # 目前有一种情况无法应对，当有两种技能分别需要同一种协奏音符且解锁数量分别为10和15时，音符到达10后，对话框提示会从15直接跳到25
        # 但是目前没有去处理这种情况，因为过程会比较复杂，只好统一按照15音符激活lv1去处理
        elif required_melody == 25 and current_melody < 15:
            return False

    if not text:
        logger.warning("未能解析到协奏音符文本，默认按已解锁处理")

    return True

