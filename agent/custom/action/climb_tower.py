from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
import json
import time


@AgentServer.custom_action("shop_action")
class ShopAction(CustomAction):
    """商店动作器"""

    # 格子坐标配置常量
    GRID_ROIS = {
        1: [638, 159, 114, 133],
        2: [791, 157, 114, 133],
        3: [941, 157, 114, 133],
        4: [1094, 162, 114, 133],
        5: [641, 359, 114, 133],
        6: [791, 361, 114, 133],
        7: [943, 360, 114, 133],
        8: [1093, 361, 114, 133],
    }

    # 空白区域坐标常量
    BLANK_AREA = [471, 486, 335, 216]

    # 固定返回按钮坐标（备用）
    FIXED_BACK_BUTTON = [50, 17, 37, 38]

    # 等待时间常量（秒）
    WAIT_SHORT = 0.5
    WAIT_SHORT = 1.0

    def __init__(self):
        super().__init__()
        self._shop_processed = False  # 商店流程已处理标志位
        self._strengthen_processed = False  # 强化流程已处理标志位
        self._last_recognition_results = {}  # 保存识别结果，避免重复识别

    def _success_result(self) -> CustomAction.RunResult:
        """返回成功结果的辅助方法"""
        return CustomAction.RunResult(success=True)

    def _failure_result(self) -> CustomAction.RunResult:
        """返回失败结果的辅助方法"""
        return CustomAction.RunResult(success=False)

    def _calculate_click_coords(self, coords: tuple) -> tuple:
        """计算点击坐标

        Args:
            coords: 坐标信息 (x, y, w, h)

        Returns:
            点击坐标 (click_x, click_y)
        """
        x, y, w, h = coords
        click_x = x + w // 2
        click_y = y + h // 2
        return click_x, click_y

    def _recognize_and_click(
        self,
        context: Context,
        recognize_name: str,
        success_msg: str,
        failure_msg: str,
        fixed_coords: tuple = None,
        img=None,
    ) -> CustomAction.RunResult:
        """通用的识别并点击方法

        Args:
            context: 上下文对象
            recognize_name: 识别名称
            success_msg: 识别成功时的消息
            failure_msg: 识别失败时的消息
            fixed_coords: 如果识别失败，使用的固定坐标 (x, y, w, h)
            img: 如果已经有截图，可以直接传入，避免重复获取

        Returns:
            操作结果
        """
        # 如果没有提供截图，获取最新截图
        if img is None:
            img = context.tasker.controller.post_screencap().wait().get()

        # 执行识别
        reco_result = context.run_recognition(recognize_name, img)

        if reco_result and reco_result.hit and reco_result.best_result:
            # 获取识别到的坐标并执行点击
            box = reco_result.best_result.box
            print(f"{success_msg}，位置: {box}")

            # 计算点击坐标（使用box的中心）
            click_x, click_y = self._calculate_click_coords(box)

            # 执行点击操作
            result = context.tasker.controller.post_click(click_x, click_y).wait()
            print(f"{success_msg}结果: {result}")

            return self._success_result()
        elif fixed_coords:
            # 如果提供了固定坐标，使用固定坐标点击
            print(f"{failure_msg}，使用固定坐标点击")
            click_x, click_y = self._calculate_click_coords(fixed_coords)

            # 执行点击操作
            result = context.tasker.controller.post_click(click_x, click_y).wait()
            print(f"{success_msg}结果: {result}")

            return self._success_result()
        else:
            print(failure_msg)
            return self._failure_result()

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        """执行商店动作"""

        config = argv.custom_action_param
        print(f"商店动作器参数: {config}")

        try:
            # 重置标志位
            self._shop_processed = False
            self._strengthen_processed = False
            print("重置标志位: _shop_processed=False, _strengthen_processed=False")

            # 如果 config 是字符串，尝试解析为 JSON 对象
            if isinstance(config, str):
                shop_config = json.loads(config)
            else:
                shop_config = config
            action_type = shop_config.get("type", "click_grid")

            # 完整商店流程处理
            if action_type == "complete_shop_flow":
                return self._complete_shop_flow(context, argv, shop_config)

            # 默认返回失败
            print(f"Unknown action type: {action_type}")
            return CustomAction.RunResult(success=False)

        except Exception as e:
            print(f"商店动作器错误: {e}")
            return CustomAction.RunResult(success=False)

    def _process_grid(self, context, argv, shop_config):
        """处理商店格子（点击、检查售罄/货币不足）"""
        grid_index = shop_config.get("grid_index", 1)
        print(f"正在点击商店格子 {grid_index}")

        roi = self.GRID_ROIS.get(grid_index, self.GRID_ROIS[1])

        # 计算点击坐标
        click_x, click_y = self._calculate_click_coords(roi)

        # 执行点击操作
        result = context.tasker.controller.post_click(click_x, click_y).wait()
        print(f"点击商店格子 {grid_index} 结果: {result}")

        # 等待界面切换
        time.sleep(self.WAIT_SHORT)  # 等待0.5秒，确保界面切换完成

        # 获取最新截图
        img = context.tasker.controller.post_screencap().wait().get()

        sold_out_result = context.run_recognition("星塔_节点_商店_购物_售罄_agent", img)

        if sold_out_result and sold_out_result.hit:
            print(f"格子 {grid_index} 售罄，处理下一个格子")
            # 售罄，直接处理下一个格子，无需关闭
            return False

        # 2. 检查是否货币不足
        not_enough_result = context.run_recognition(
            "星塔_节点_商店_购物_货币不足_agent", img
        )

        if not_enough_result and not_enough_result.hit:
            print(f"格子 {grid_index} 货币不足，处理下一个格子")
            # 货币不足，直接处理下一个格子，无需关闭
            return False

        # 检查是否进入了物品详情界面，并返回物品类型
        return self._get_item_type(context, img)

    def _is_item_detail(self, context, img):
        """检查是否进入了物品详情界面

        Returns:
            bool: 是否为物品详情界面
        """
        print("检查是否进入了物品详情界面")

        # 1. 识别是否物品详情界面
        item_detail_result = context.run_recognition(
            "星塔_节点_商店_购物_格子主界面_agent", img
        )

        if item_detail_result and item_detail_result.hit:
            print("识别到物品详情界面")
            return True

        print("未识别到物品详情界面")
        return False

    def _get_item_type(self, context, img):
        """获取物品类型（buff或note）"""
        print("识别物品类型")

        # 识别是否有buff相关特征
        buff_result = context.run_recognition(
            "星塔_节点_商店_购物_格子_buff_agent", img
        )

        if buff_result and buff_result.hit:
            print("识别到buff特征")
            return "buff_interface"

        # 识别是否有note相关文字
        note_result = context.run_recognition(
            "星塔_节点_商店_购物_格子_音符_agent", img
        )

        if note_result and note_result.hit:
            print("识别到note特征")
            return "note_interface"

        # 默认返回buff_interface，因为buff更常见
        print("未明确识别到物品类型，返回undef_type")
        return "undef_type"

    def _check_discount(self, context, img, item_type=None):
        """检查是否有优惠

        Args:
            context: 上下文对象
            img: 截图
            item_type: 物品类型，buff_interface或note_interface

        Returns:
            bool: 是否有优惠
        """
        print(f"检查是否有优惠，物品类型: {item_type}")

        # 根据物品类型选择不同的优惠识别节点
        if item_type == "buff_interface":
            # buff使用星塔_节点_商店_购物_格子_buff优惠_agent
            discount_node = "星塔_节点_商店_购物_格子_buff优惠_agent"
            print(f"使用buff优惠识别节点: {discount_node}")
        else:
            # note或其他类型使用原来的优惠识别节点
            discount_node = "星塔_节点_商店_购物_格子_优惠_agent"
            print(f"使用普通优惠识别节点: {discount_node}")

        # 识别是否有优惠
        discount_result = context.run_recognition(discount_node, img)

        if discount_result and discount_result.hit:
            print("识别到优惠")
            return True

        print("未识别到优惠")
        return False

    def _buy_item(self, context, argv, shop_config, img=None):
        """购买商品"""
        print("正在购买商品")

        return self._recognize_and_click(
            context=context,
            recognize_name="星塔_节点_商店_购物_格子_购买_agent",
            success_msg="识别到购买按钮",
            failure_msg="未识别到购买按钮",
            img=img,
        )

    def _close_grid(self, context, argv, shop_config, img=None):
        """关闭商店格子"""
        print("正在关闭商店格子")

        return self._recognize_and_click(
            context=context,
            recognize_name="星塔_节点_商店_购物_格子_关闭_agent",
            success_msg="识别到关闭按钮",
            failure_msg="未识别到关闭按钮",
            img=img,
        )

    def _strengthen_operation(self, context, argv, shop_config, img=None):
        """强化操作"""
        print("正在执行强化操作")

        return self._recognize_and_click(
            context=context,
            recognize_name="星塔_节点_商店_强化_agent",
            success_msg="识别到强化按钮",
            failure_msg="未识别到强化按钮",
            img=img,
        )

    def _refresh_shop(self, context, argv, shop_config):
        """刷新商店"""
        print("正在刷新商店")

        result = self._recognize_and_click(
            context=context,
            recognize_name="星塔_节点_最终商店_点击刷新_agent",
            success_msg="识别到刷新按钮",
            failure_msg="未识别到刷新按钮",
        )

        # 特殊处理：如果刷新操作执行了（识别到了刷新按钮），即使点击失败，也返回成功
        # 因为这可能是因为没有刷新次数了
        if result.success:
            # 刷新后尝试识别三次星塔_节点_最终商店_无法刷新_agent
            print("刷新后尝试识别星塔_节点_最终商店_无法刷新_agent")
            max_attempts = 3
            for attempt in range(max_attempts):
                # 获取最新截图
                img = context.tasker.controller.post_screencap().wait().get()

                # 识别无法刷新节点
                cannot_refresh_result = context.run_recognition(
                    "星塔_节点_最终商店_无法刷新_agent", img
                )

                print(f"第 {attempt + 1} 次识别星塔_节点_最终商店_无法刷新_agent")

                if cannot_refresh_result and cannot_refresh_result.hit:
                    print("识别到无法刷新节点，返回失败结果")
                    return self._failure_result()

                # 等待一段时间后重试
                time.sleep(self.WAIT_SHORT)

            # 三次都没识别到，正常返回
            print("未识别到无法刷新节点，正常返回")
            return result
        else:
            # 只有当根本没识别到刷新按钮时，才返回失败
            return self._failure_result()

    def _click_blank(self, context, argv, shop_config):
        """点击空白处关闭"""
        print("正在点击空白处关闭")

        # 使用M9A方式获取截图并识别
        img = context.tasker.controller.post_screencap().wait().get()

        # 执行识别，查找空白区域
        # 这里我们直接使用固定区域作为空白处，因为空白处没有明显特征
        target = self.BLANK_AREA
        print(f"使用空白区域: {target}")

        # 计算点击坐标（使用区域中心）
        click_x = target[0] + target[2] // 2
        click_y = target[1] + target[3] // 2

        # 执行点击操作
        result = context.tasker.controller.post_click(click_x, click_y).wait()
        print(f"点击空白处关闭结果: {result}")

        return self._success_result()

    def _click_back(self, context, argv, shop_config):
        """点击返回"""
        print("正在点击返回")

        return self._recognize_and_click(
            context=context,
            recognize_name="星塔_节点_商店_返回_agent",
            success_msg="识别到返回按钮",
            failure_msg="未识别到返回按钮",
            fixed_coords=self.FIXED_BACK_BUTTON,
        )

    def _complete_shop_flow(self, context, argv, shop_config):
        """完整商店流程处理"""
        print("正在进行完整商店流程处理")

        try:
            # 获取商店类型
            shop_type = shop_config.get("shop_type", "regular")
            print(f"商店类型: {shop_type}")

            # 初始化可购买格子列表，只在一次流程中初始化一次
            available_grids = None

            # 内部循环处理完整商店流程
            timeout_seconds = 200  # 设置超时时间，防止无限循环
            start_time = time.time()  # 记录开始时间
            iteration = 0
            consecutive_complete_count = 0  # 连续未识别到状态的次数
            max_consecutive_complete = 3  # 最大连续未识别到状态次数

            while (time.time() - start_time) < timeout_seconds:
                iteration += 1
                print(
                    f"商店流程循环第 {iteration} 次，已运行 {time.time() - start_time:.2f} 秒"
                )

                # 获取最新截图
                img = context.tasker.controller.post_screencap().wait().get()

                # 识别当前界面状态
                current_state = self._get_shop_state(context, img)
                print(f"当前商店状态: {current_state}")

                # 检查是否连续未识别到状态
                if current_state == "shop_flow_complete":
                    consecutive_complete_count += 1
                    print(
                        f"连续未识别到状态次数: {consecutive_complete_count}/{max_consecutive_complete}"
                    )
                    if consecutive_complete_count >= max_consecutive_complete:
                        print(
                            f"连续 {max_consecutive_complete} 次未识别到状态，结束流程"
                        )
                        return self._success_result()
                    # 等待一段时间后重试
                    time.sleep(self.WAIT_SHORT)
                    continue
                else:
                    # 识别到有效状态，重置连续未识别计数
                    consecutive_complete_count = 0

                # 根据不同状态执行不同操作
                if current_state in ["shop_shopping"]:
                    # 处理商店购物状态
                    continue_flag = self._handle_shop_shopping_state(context, img)
                    if not continue_flag:
                        break
                    continue

                elif current_state == "blank_close":
                    # 处理空白处关闭状态
                    continue_flag = self._handle_blank_close_state(context, img)
                    if not continue_flag:
                        break
                    continue

                elif current_state == "item_main":
                    # 处理物品主界面状态
                    continue_flag = self._handle_item_main_state(
                        context, argv, shop_config
                    )
                    if not continue_flag:
                        break
                    continue

                elif current_state == "buff_main":
                    # 处理buff选择状态
                    continue_flag = self._handle_buff_main_state(
                        context, argv, shop_config
                    )
                    if not continue_flag:
                        break
                    continue

                elif current_state == "shop_main":
                    # 处理商店主界面状态
                    continue_flag, available_grids = self._handle_shop_main_state(
                        context, argv, shop_config, shop_type, available_grids, img
                    )
                    if not continue_flag:
                        break
                    continue

                elif current_state == "shop_main_processed":
                    # 处理已处理过的商店主界面状态
                    print("处理已处理过的商店主界面，执行返回操作")
                    # 执行返回操作
                    self._click_back(context, argv, shop_config)
                    # 等待返回完成
                    time.sleep(self.WAIT_SHORT)
                    continue

                elif current_state == "strengthen_process":
                    # 处理强化流程状态
                    continue_flag = self._handle_strengthen_process_state(context, img)
                    if not continue_flag:
                        break
                    continue

                elif current_state == "end_strengthen":
                    # 处理结束强化状态
                    print("识别到结束强化，设置_strengthen_processed=True")
                    # 设置强化已处理标志
                    self._strengthen_processed = True
                    continue

                elif current_state == "shop_next_floor":
                    # 处理下一层状态
                    print("识别到下一层按钮，执行点击操作")
                    # 使用之前保存的下一层按钮识别结果
                    next_floor_result = self._last_recognition_results.get(
                        "shop_next_floor_result"
                    )
                    if (
                        next_floor_result
                        and next_floor_result.hit
                        and next_floor_result.best_result
                    ):
                        # 获取识别到的坐标并执行点击
                        box = next_floor_result.best_result.box
                        print(f"识别到下一层按钮，位置: {box}")
                        # 计算点击坐标
                        click_x, click_y = self._calculate_click_coords(box)
                        # 执行点击操作
                        result = context.tasker.controller.post_click(
                            click_x, click_y
                        ).wait()
                        print(f"点击下一层按钮结果: {result}")
                    # 等待界面切换
                    time.sleep(self.WAIT_SHORT)
                    continue

                elif current_state == "final_shop_leave":
                    # 处理最终商店离开星塔状态
                    print("识别到最终商店离开星塔按钮，执行点击操作")
                    # 使用之前保存的最终商店离开星塔按钮识别结果
                    final_leave_result = self._last_recognition_results.get(
                        "final_leave_result"
                    )
                    if (
                        final_leave_result
                        and final_leave_result.hit
                        and final_leave_result.best_result
                    ):
                        # 获取识别到的坐标并执行点击
                        box = final_leave_result.best_result.box
                        print(f"识别到最终商店离开星塔按钮，位置: {box}")
                        # 计算点击坐标
                        click_x, click_y = self._calculate_click_coords(box)
                        # 执行点击操作
                        result = context.tasker.controller.post_click(
                            click_x, click_y
                        ).wait()
                        print(f"点击最终商店离开星塔按钮结果: {result}")
                    # 等待界面切换
                    time.sleep(self.WAIT_SHORT)
                    continue

                elif current_state == "leave_tower":
                    # 处理离开星塔状态
                    print("识别到离开星塔按钮，执行点击操作")
                    # 使用之前保存的离开星塔按钮识别结果
                    leave_result = self._last_recognition_results.get("leave_result")
                    if leave_result and leave_result.hit and leave_result.best_result:
                        # 获取识别到的坐标并执行点击
                        box = leave_result.best_result.box
                        print(f"识别到离开星塔按钮，位置: {box}")
                        # 计算点击坐标
                        click_x, click_y = self._calculate_click_coords(box)
                        # 执行点击操作
                        result = context.tasker.controller.post_click(
                            click_x, click_y
                        ).wait()
                        print(f"点击离开星塔按钮结果: {result}")
                    # 等待界面切换
                    time.sleep(self.WAIT_SHORT)
                    continue

                elif current_state == "not_enough_money_set_strengthen_processed":
                    # 处理货币不足设置强化已处理状态
                    print("识别到货币不足节点，设置_strengthen_processed=True")
                    self._strengthen_processed = True
                    continue

                elif current_state in ["buff_interface", "note_interface"]:
                    # 处理物品详情界面状态
                    continue_flag = self._handle_item_detail_state(
                        context, argv, shop_config, current_state
                    )
                    if not continue_flag:
                        break
                    continue

                elif current_state in ["not_enough_money", "sold_out"]:
                    # 处理货币不足或售罄状态
                    continue_flag = self._handle_not_enough_money_state(
                        context, argv, shop_config, current_state
                    )
                    if not continue_flag:
                        break
                    continue

                else:
                    # 处理未知状态
                    continue_flag = self._handle_unknown_state(
                        context, argv, shop_config, current_state
                    )
                    if not continue_flag:
                        break
                    continue
        except Exception as e:
            print(f"处理完整商店流程时发生错误: {e}")
            import traceback

            traceback.print_exc()
            return self._failure_result()

        # 流程正常结束
        return self._success_result()

    def _handle_shop_shopping_state(self, context, img) -> bool:
        """处理商店购物状态"""
        print("使用之前保存的商店购物按钮识别结果")
        shop_shopping_result = self._last_recognition_results.get(
            "shop_shopping_result"
        )

        if (
            shop_shopping_result
            and shop_shopping_result.hit
            and shop_shopping_result.best_result
        ):
            # 获取识别到的坐标并执行点击
            box = shop_shopping_result.best_result.box
            print(f"识别到商店购物按钮，位置: {box}")

            # 计算点击坐标
            click_x, click_y = self._calculate_click_coords(box)

            # 执行点击操作
            result = context.tasker.controller.post_click(click_x, click_y).wait()
            print(f"成功点击商店购物按钮")

            # 等待界面切换
            time.sleep(self.WAIT_SHORT)  # 等待0.5秒，确保界面切换完成
        else:
            print("未识别到商店购物按钮，可能已经进入商店主界面")

        # 继续循环
        return True

    def _handle_blank_close_state(self, context, img) -> bool:
        """处理空白处关闭状态"""
        print("使用之前保存的空白处关闭识别结果")
        blank_result = self._last_recognition_results.get("blank_result")
        if blank_result and blank_result.hit and blank_result.best_result:
            box = blank_result.best_result.box
            print(f"识别到点击空白处关闭按钮，位置: {box}")

            # 计算点击坐标
            click_x, click_y = self._calculate_click_coords(box)

            # 执行点击操作
            result = context.tasker.controller.post_click(click_x, click_y).wait()
            print(f"成功点击空白处关闭按钮")

            # 等待界面切换
            time.sleep(self.WAIT_SHORT)  # 等待0.5秒，确保界面切换完成

        # 继续循环
        return True

    def _handle_item_main_state(self, context, argv, shop_config) -> bool:
        """处理物品主界面状态"""
        print("关闭物品主界面")
        self._close_grid(context, argv, shop_config)

        # 继续循环
        return True

    def _handle_buff_main_state(self, context, argv, shop_config) -> bool:
        """处理buff选择状态"""
        print("进入buff选择流程")
        buff_result = self._select_buff(context, argv, shop_config)
        if not buff_result.success:
            print("buff选择失败")
        else:
            print("buff选择成功")

        # 继续循环
        return True

    def _handle_shop_main_state(
        self, context, argv, shop_config, shop_type, available_grids, img
    ) -> tuple[bool, list]:
        """处理商店主界面状态

        Returns:
            tuple: (是否继续循环, 更新后的available_grids)
        """

        # 只在第一次进入商店主界面时获取可购买格子列表
        if available_grids is None:
            # 识别可购买的格子，只获取一次
            available_grids = self._get_available_grids(context, img)
            print(f"初始可购买格子列表: {available_grids}")

        if available_grids:
            # 还有可购买的格子，处理第一个
            grid_index = available_grids[0]
            print(f"处理格子: {grid_index}")

            # 处理格子
            click_result = self._process_grid(context, argv, {"grid_index": grid_index})

            # 根据_process_grid的返回值处理格子
            if click_result is False:
                # 格子售罄或货币不足，从列表中移除该格子
                print(f"格子 {grid_index} 售罄或货币不足，从列表中移除")
                available_grids.pop(0)
                # 继续循环，处理下一个格子
                return True, available_grids
            elif click_result in [
                "buff_interface",
                "note_interface",
            ]:  # 处理成功，返回物品类型
                # 成功进入物品详情界面，处理购买逻辑
                item_type = click_result
                print(
                    f"成功进入格子 {grid_index} 的物品详情界面，物品类型: {item_type}"
                )

                # 获取最新截图
                img = context.tasker.controller.post_screencap().wait().get()

                # 检查是否有优惠
                has_discount = self._check_discount(context, img, item_type)

                # 音符类型特殊处理：先判断是否有音符激活节点，再判断优惠
                if item_type == "note_interface":
                    print("处理音符类型物品")
                    # 判断是否有音符激活节点
                    note_activate_result = context.run_recognition(
                        "星塔_节点_商店_购物_格子_音符_激活_agent", img
                    )
                    if note_activate_result and note_activate_result.hit:
                        print("识别到音符激活节点")
                        # 再判断是否有优惠
                        if has_discount:
                            print("识别到优惠，尝试购买商品")
                            buy_result = self._buy_item(context, argv, shop_config, img)
                            if buy_result.success:
                                print("购买成功")
                            else:
                                print("购买失败")
                        else:
                            print("未识别到优惠，跳过购买")
                            # 关闭格子
                            self._close_grid(context, argv, shop_config, img)
                    else:
                        print("未识别到音符激活节点，跳过购买")
                        # 关闭格子
                        self._close_grid(context, argv, shop_config, img)
                else:  # buff类型直接判断优惠
                    print("处理buff类型物品")
                    # 根据优惠情况决定是否购买
                    if has_discount:
                        print("识别到优惠，尝试购买商品")
                        buy_result = self._buy_item(context, argv, shop_config, img)
                        if buy_result.success:
                            print("购买成功")
                        else:
                            print("购买失败")
                    else:
                        print("未识别到优惠，跳过购买")
                        # 关闭格子
                        self._close_grid(context, argv, shop_config, img)

                # 等待界面返回商店主界面
                time.sleep(self.WAIT_SHORT)  # 等待0.5秒，确保界面切换完成

                # 格子处理完成，从列表中移除
                print(f"格子 {grid_index} 处理完成，从列表中移除")
                available_grids.pop(0)
                # 继续循环，处理下一个格子
                return True, available_grids
            else:
                # 不是以上情况，保留格子并continue
                print(
                    f"点击格子 {grid_index} 未成功进入物品详情界面，保留格子待下次处理"
                )
                return True, available_grids
        else:
            # 没有可购买的格子了
            print("所有格子处理完成，继续处理其他状态")

            # 最终商店尝试刷新
            if shop_type == "final":
                print(f"最终商店没有可购买的格子，尝试刷新")
                refresh_result = self._refresh_shop(context, argv, shop_config)
                if refresh_result.success:
                    # 刷新成功，重置格子列表，重新开始处理
                    return True, None
                else:
                    # 刷新失败，继续处理其他状态
                    print("刷新失败，继续处理其他状态")

            # 设置商店已处理标志位
            self._shop_processed = True
            print(f"商店流程已处理，设置_shop_processed=True")

            # 点击空白处关闭，继续处理其他状态
            self._click_blank(context, argv, shop_config)
            # 流程未完成，继续循环
            return True, []

    def _handle_strengthen_process_state(self, context, img) -> bool:
        """处理强化流程状态"""
        # 执行强化操作
        print("使用之前保存的强化按钮识别结果")
        strengthen_result = self._last_recognition_results.get("strengthen_result")

        if (
            strengthen_result
            and strengthen_result.hit
            and strengthen_result.best_result
        ):
            # 获取识别到的坐标并执行点击
            box = strengthen_result.best_result.box
            print(f"识别到强化按钮，位置: {box}")

            # 计算点击坐标
            click_x, click_y = self._calculate_click_coords(box)

            # 执行点击操作
            result = context.tasker.controller.post_click(click_x, click_y).wait()
            print(f"成功点击强化按钮")

            # 等待界面切换
            time.sleep(self.WAIT_SHORT)  # 等待0.5秒，确保界面切换完成

        # 继续循环
        return True

    def _handle_enter_next_state(self) -> bool:
        """处理进入下一层状态"""
        # 进入下一层
        print("识别到进入下一层状态，结束商店流程")
        # 流程完成，退出循环
        return False

    def _handle_item_detail_state(
        self, context, argv, shop_config, current_state
    ) -> bool:
        """处理物品详情界面状态"""
        # 在商品详情界面，尝试购买
        print(f"在{current_state}，尝试购买商品")
        buy_result = self._buy_item(context, argv, shop_config)
        if buy_result.success:
            print("购买成功")
        else:
            print("购买失败")

        # 关闭格子
        self._close_grid(context, argv, shop_config)
        # 继续循环，返回商店主界面
        return True

    def _handle_not_enough_money_state(
        self, context, argv, shop_config, current_state
    ) -> bool:
        """处理货币不足或售罄状态"""
        # 货币不足或售罄，关闭提示
        print(f"遇到{current_state}，关闭提示")
        self._click_blank(context, argv, shop_config)
        # 继续循环，返回商店主界面
        return True

    def _handle_unknown_state(self, context, argv, shop_config, current_state) -> bool:
        """处理未知状态"""
        # 未知状态，点击空白处关闭，结束流程
        print(f"未知状态 {current_state}，结束流程")
        self._click_blank(context, argv, shop_config)
        # 流程完成，退出循环
        return False

    def _check_buff_selection(self, context, img):
        """检查是否需要选择buff"""
        print("检查是否需要选择buff")

        # 识别buff推荐图标
        buff_reco_result = context.run_recognition("星塔_节点_选择buff_推荐_agent", img)

        if buff_reco_result and buff_reco_result.hit:
            print("识别到buff推荐图标，需要选择buff")
            return True

        print("不需要选择buff")
        return False

    def _select_buff(self, context, argv, shop_config):
        """选择buff"""
        print("正在选择buff")

        try:
            # 获取最新截图
            img = context.tasker.controller.post_screencap().wait().get()

            # 识别buff推荐图标
            buff_reco_result = context.run_recognition(
                "星塔_节点_选择buff_推荐_agent", img
            )

            if (
                buff_reco_result
                and buff_reco_result.hit
                and buff_reco_result.best_result
            ):
                # 获取识别到的坐标并执行点击
                box = buff_reco_result.best_result.box
                print(f"识别到buff推荐图标，位置: {box}")

                # 计算点击坐标（使用box的中心）
                click_x = box[0] + box[2] // 2
                click_y = box[1] + box[3] // 2

                # 执行点击操作
                result = context.tasker.controller.post_click(click_x, click_y).wait()
                print(f"成功点击buff推荐图标")

                # 等待一下，然后点击"拿走"按钮
                time.sleep(self.WAIT_SHORT)  # 等待0.5秒，确保界面切换

                # 获取最新截图
                img = context.tasker.controller.post_screencap().wait().get()

                # 识别"拿走"按钮
                take_result = context.run_recognition(
                    "星塔_节点_选择buff_拿走_agent", img
                )

                if take_result and take_result.hit and take_result.best_result:
                    # 获取识别到的坐标并执行点击
                    take_box = take_result.best_result.box
                    print(f"识别到拿走按钮，位置: {take_box}")

                    # 计算点击坐标（使用box的中心）
                    take_x = take_box[0] + take_box[2] // 2
                    take_y = take_box[1] + take_box[3] // 2

                    # 执行点击操作
                    result = context.tasker.controller.post_click(take_x, take_y).wait()
                    print(f"成功点击拿走按钮")
                    return CustomAction.RunResult(success=True)
                else:
                    print("未识别到拿走按钮")
                    return CustomAction.RunResult(success=False)
            else:
                print("未识别到buff推荐图标")
                return CustomAction.RunResult(success=False)

        except Exception as e:
            print(f"选择buff时发生错误: {e}")
            import traceback

            traceback.print_exc()
            return CustomAction.RunResult(success=False)

    def _get_shop_state(self, context, img):
        """获取商店当前状态"""
        print("识别商店当前状态")

        # 清空上一次的识别结果
        self._last_recognition_results.clear()

        # 1. 识别是否在buff选择界面
        buff_reco_result = context.run_recognition("星塔_节点_选择buff_推荐_agent", img)
        self._last_recognition_results["buff_reco_result"] = buff_reco_result
        if buff_reco_result and buff_reco_result.hit:
            print("识别到buff选择界面")
            return "buff_main"

        # 2. 识别是否在物品详情界面
        item_type = self._is_item_detail(context, img)
        self._last_recognition_results["item_type"] = item_type
        if item_type:
            print(f"识别到物品详情界面，类型: {item_type}")
            return "item_main"

        blank_result = context.run_recognition("星塔_点击空白处关闭", img)
        self._last_recognition_results["blank_result"] = blank_result
        if blank_result and blank_result.hit:
            print("识别到点击空白处关闭")
            return "blank_close"

        # 3. 识别是否在商店主界面
        shop_main_result = context.run_recognition("星塔_节点_商店_主界面_agent", img)
        self._last_recognition_results["shop_main_result"] = shop_main_result
        if shop_main_result and shop_main_result.hit:
            print("识别到商店主界面")
            # 如果商店已处理，返回新的状态
            if self._shop_processed:
                print("商店已处理，返回状态 shop_main_processed")
                return "shop_main_processed"
            else:
                return "shop_main"

        # 4. 商店进入、强化、下一层和进入下一层并列判断
        # 先识别商店购物按钮
        shop_shopping_result = context.run_recognition(
            "星塔_节点_商店_商店购物_agent", img
        )
        self._last_recognition_results["shop_shopping_result"] = shop_shopping_result
        if (
            shop_shopping_result
            and shop_shopping_result.hit
            and not self._shop_processed
        ):
            print("识别到商店购物按钮，且未处理过商店")
            return "shop_shopping"

        # 识别结束强化节点
        end_strengthen_result = context.run_recognition(
            "星塔_节点_商店_结束强化_agent", img
        )
        self._last_recognition_results["end_strengthen_result"] = end_strengthen_result
        if (
            end_strengthen_result
            and end_strengthen_result.hit
            and not self._strengthen_processed
        ):
            print("识别到结束强化节点")
            return "end_strengthen"

        # 识别货币不足节点，返回状态（如果商店已处理）
        not_enough_money_result = context.run_recognition(
            "星塔_节点_商店_购物_货币不足_agent", img
        )
        self._last_recognition_results["not_enough_money_result"] = (
            not_enough_money_result
        )
        if not_enough_money_result and not_enough_money_result.hit:
            print("识别到货币不足节点")
            if self._shop_processed:
                print("商店已处理，返回状态用于设置强化已处理标志")
                return "not_enough_money_set_strengthen_processed"

        # 再识别强化按钮
        strengthen_result = context.run_recognition("星塔_节点_商店_强化_agent", img)
        self._last_recognition_results["strengthen_result"] = strengthen_result
        if (
            strengthen_result
            and strengthen_result.hit
            and self._shop_processed
            and not self._strengthen_processed
        ):
            print("识别到强化按钮，且商店已处理，强化未处理")
            return "strengthen_process"

        # 识别下一层按钮
        next_floor_result = context.run_recognition("星塔_节点_商店_下一层_agent", img)
        self._last_recognition_results["shop_next_floor_result"] = next_floor_result
        if next_floor_result and next_floor_result.hit:
            print("识别到下一层按钮")
            return "shop_next_floor"

        # 识别最终商店离开星塔按钮
        final_leave_result = context.run_recognition(
            "星塔_节点_最终商店_离开星塔_agent", img
        )
        self._last_recognition_results["final_leave_result"] = final_leave_result
        if final_leave_result and final_leave_result.hit:
            print("识别到最终商店离开星塔按钮")
            return "final_shop_leave"

        # 识别离开星塔按钮
        leave_result = context.run_recognition("星塔_离开星塔_agent", img)
        self._last_recognition_results["leave_result"] = leave_result
        if leave_result and leave_result.hit:
            print("识别到离开星塔按钮")
            return "leave_tower"

        # 其他情况返回enter_next
        print("未识别到需要处理的状态，返回shop_flow_complete用于结束整个流程")
        return "shop_flow_complete"

    def _get_available_grids(self, context, img):
        """获取可购买的格子列表"""
        print("识别可购买的格子")

        # 格子坐标配置
        grid_rois = {
            1: [638, 159, 114, 133],
            2: [791, 157, 114, 133],
            3: [941, 157, 114, 133],
            4: [1094, 162, 114, 133],
            5: [641, 359, 114, 133],
            6: [791, 361, 114, 133],
            7: [943, 360, 114, 133],
            8: [1093, 361, 114, 133],
        }

        available_grids = []

        # 遍历所有格子，识别是否可购买
        for grid_index, roi in grid_rois.items():
            print(f"正在识别格子 {grid_index}")

            if grid_index > 5:
                available_grids.append(grid_index)
                continue
            grid_main_result = context.run_recognition(
                "星塔_节点_商店_购物_格子_音符_agent",
                img,
                pipeline_override={
                    "星塔_节点_商店_购物_格子_音符_agent": {
                        "recognition": {
                            "param": {
                                "roi": roi,
                            }
                        }
                    }
                },
            )
            self._last_recognition_results["grid_main_result"] = grid_main_result
            if grid_main_result and grid_main_result.hit:
                available_grids.append(grid_index)

        return available_grids
