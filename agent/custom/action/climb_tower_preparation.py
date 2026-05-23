import os
import re
import time
import json
from typing import Optional
from pathlib import Path

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils import logger as logger_module
logger = logger_module.get_logger("climb_tower_preparation")


@AgentServer.custom_action("ascension_preparation")
class AscensionPreparation(CustomAction):
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        """检查并导入预设文件，为爬塔流程做准备

        Args:
            context: 任务上下文。
            argv: 自定义动作参数。

        Returns:
            bool: 成功时返回 True，失败时返回 False。
        """

        # 导入json作业参数
        node_data = context.get_node_data(argv.node_name)
        preset_path = Path(os.path.abspath(__file__)).parent.parent.parent / "presets"
        full_path = ""

        try:
            preset_name = node_data["attach"]["preset_name"]

            if not preset_name:
                logger.debug("未提供预设作业，将使用默认选项")
                return True

            # 读取预设作业
            full_path = (preset_path / preset_name).with_suffix(".json")
            with open(full_path, "r", encoding="utf-8") as f:
                presets = json.load(f)
                priority_list = presets.get("priority_list", [])
                preset_element = presets.get("element", "")
                preset_melodies = presets.get("melodies", [])
                preset_trekker_names = presets.get("trekker_names", [])
                preset_potential_refresh = presets.get("potential_refresh", 0)

        except FileNotFoundError:
            logger.error(f"无法找到预设作业文件：{full_path}")
            logger.error(f"请核实预设作业名字是否正确，或预设作业是否存在等")
            return False
        except json.decoder.JSONDecodeError as e:
            logger.error(f"无法解析作业文件，错误信息：{e}")
            logger.error("请核实json内容的格式是否正确")
            context.tasker.post_stop()
            return False

        err = self._validate_priority_list(priority_list)
        if err:
            logger.error(f"潜能优先级设置校验失败：{err}")
            context.tasker.post_stop()
            return False

        context.override_pipeline({
            "星塔_节点_选择潜能_agent": {
                "attach": {
                    "priority_list": priority_list
                }
            }
        })

        if preset_element:
            logger.info(f"从作业中检测到预设属性：{preset_element}，将覆盖选项中的属性塔选择")
            preset_element = preset_element.lower()
            match preset_element:
                case "aqua" | "ventus" | "水" | "风" | "風":
                    context.override_pipeline({
                        "星塔_属性塔选择_agent": {
                            "recognition": {
                                "param": {
                                    "template": [
                                        "ClimbTower_agent/爬塔_水风__384_271_129_39__334_221_229_139.png"
                                    ]
                                }
                            }
                        }
                    })
                case "ignis" | "umbra" | "火" | "暗" | "闇":
                    context.override_pipeline({
                        "星塔_属性塔选择_agent": {
                            "recognition": {
                                "param": {
                                    "template": [
                                        "ClimbTower_agent/爬塔_火暗__381_404_129_39__331_354_229_139.png"
                                    ]
                                }
                            }
                        }
                    })
                case "terra" | "lux" | "地" | "光":
                    context.override_pipeline({
                        "星塔_属性塔选择_agent": {
                            "recognition": {
                                "param": {
                                    "template": [
                                        "ClimbTower_agent/爬塔_光土__387_137_124_45__337_87_224_145.png"
                                    ]
                                }
                            }
                        }
                    })
                case _:
                    logger.error(f"检测到未知属性：{preset_element}，请核实属性名是否符合文档要求")
                    context.tasker.post_stop()
                    return False

        if preset_trekker_names:
            logger.info(f"从作业中检测到预设队伍：{preset_trekker_names}，爬塔前会自动选择该队伍")
            context.override_pipeline({
                "星塔_编队角色_选择队伍_agent": {
                    "attach": {
                        "trekker_names": preset_trekker_names
                    }
                }
            })

        if preset_potential_refresh:
            logger.info(f"从作业中检测到预设潜能最大刷新次数：{preset_potential_refresh}，将覆盖选项设置")
            context.override_pipeline({
                "星塔_节点_选择潜能_agent": {
                    "attach": {
                        "max_refresh_count": preset_potential_refresh
                    }
                }
            })

        if preset_melodies:
            logger.info(f"从作业中检测到预设音符：{preset_melodies}，爬塔时会买入以上音符")
            node_data = context.get_node_data("星塔_节点_商店_购物_agent")
            shop_attachments = node_data.get("attach", {})
            for melody in preset_melodies:
                melody = melody.lower()
                if melody in shop_attachments:
                    context.override_pipeline({
                        "星塔_节点_商店_购物_agent": {
                            "attach": {
                                melody: True
                            }
                        }
                    })
                else:
                    logger.error(f"导入音符：{melody} 失败，请核实音符名是否符合文档要求")
                    context.tasker.post_stop()
                    return False


        logger.info(f"已导入预设作业：{preset_name}")
        return True

    @staticmethod
    def _validate_priority_list(priority_list: object) -> Optional[str]:
        """宽松校验 priority_list：仅校验关键字段与类型。

        Args:
            priority_list: 预设作业中的 priority_list 字段

        Returns:
            Optional[str]: 校验通过时返回 None，否则返回错误信息
        """
        if not isinstance(priority_list, list):
            return f"priority_list 必须是 list，实际是 {type(priority_list).__name__}"

        def _is_non_negative_int(value: object) -> bool:
            if isinstance(value, bool):
                return False
            return isinstance(value, int) and value >= 0

        def _is_positive_int(value: object) -> bool:
            if isinstance(value, bool):
                return False
            return isinstance(value, int) and value > 0

        def _validate_level_range(level_obj: dict, level_path: str) -> Optional[str]:
            la = level_obj.get("level_at_least")
            lm = level_obj.get("level_at_most")
            if la is not None and not _is_non_negative_int(la):
                return f"{level_path}.level_at_least 必须是非负整数"
            if lm is not None and not _is_non_negative_int(lm):
                return f"{level_path}.level_at_most 必须是非负整数"
            if la is not None and lm is not None and la > lm:
                return f"{level_path}.level_at_least 不能大于 level_at_most"
            return None

        def _validate_condition_item(cond_item: dict, cond_path: str) -> Optional[str]:
            if not isinstance(cond_item, dict):
                return f"{cond_path} 必须是 dict"

            has_count = ("count_at_least" in cond_item) or ("count_at_most" in cond_item)
            if has_count:
                trekker = cond_item.get("trekker")
                if not isinstance(trekker, str) or not trekker.strip():
                    return f"{cond_path}.trekker 必须是非空字符串（数量条件必填）"

                ca = cond_item.get("count_at_least")
                cm = cond_item.get("count_at_most")
                if ca is not None and not _is_non_negative_int(ca):
                    return f"{cond_path}.count_at_least 必须是非负整数"
                if cm is not None and not _is_non_negative_int(cm):
                    return f"{cond_path}.count_at_most 必须是非负整数"
                if ca is not None and cm is not None and ca > cm:
                    return f"{cond_path}.count_at_least 不能大于 count_at_most"

                return _validate_level_range(cond_item, cond_path)

            if "potential" not in cond_item:
                return f"{cond_path} 缺少 potential（等级条件必填）"
            if not isinstance(cond_item["potential"], str) or not cond_item["potential"].strip():
                return f"{cond_path}.potential 必须是非空字符串"

            if "level_at_least" not in cond_item and "level_at_most" not in cond_item:
                return f"{cond_path} 缺少 level_at_least / level_at_most（至少一个）"

            return _validate_level_range(cond_item, cond_path)

        for i, rule in enumerate(priority_list, start=1):
            path = f"priority_list[{i}]"

            if not isinstance(rule, dict):
                return f"{path} 必须是 dict"

            if "potential" not in rule:
                return f"{path} 缺少必填字段 potential"

            potential = rule["potential"]
            if isinstance(potential, str):
                if not potential.strip():
                    return f"{path}.potential 不能为空字符串"
            elif isinstance(potential, list):
                if not potential:
                    return f"{path}.potential 不能为空列表"
                for j, name in enumerate(potential, start=1):
                    if not isinstance(name, str) or not name.strip():
                        return f"{path}.potential[{j}] 必须是非空字符串"
            else:
                return f"{path}.potential 必须是字符串或字符串列表"

            if "trekker" in rule:
                if not isinstance(rule["trekker"], str) or not rule["trekker"].strip():
                    return f"{path}.trekker 必须是非空字符串"

            if "level_span" in rule and not _is_positive_int(rule["level_span"]):
                return f"{path}.level_span 必须是正整数"

            if "max_level" in rule and not _is_positive_int(rule["max_level"]):
                return f"{path}.max_level 必须是正整数"

            if "refresh" in rule and not _is_non_negative_int(rule["refresh"]):
                return f"{path}.refresh 必须是非负整数"

            if "condition" in rule:
                condition = rule["condition"]
                if not isinstance(condition, list):
                    return f"{path}.condition 必须是 list，不能是 {type(condition).__name__}"

                for c_idx, branch in enumerate(condition, start=1):
                    c_path = f"{path}.condition[{c_idx}]"
                    if isinstance(branch, dict):
                        err = _validate_condition_item(branch, c_path)
                        if err:
                            return err
                    elif isinstance(branch, list):
                        if not branch:
                            return f"{c_path}（OR 分支）不能为空列表"
                        for b_idx, item in enumerate(branch, start=1):
                            err = _validate_condition_item(item, f"{c_path}[{b_idx}]")
                            if err:
                                return err
                    else:
                        return f"{c_path} 必须是 dict 或 list"

        return None


@AgentServer.custom_action("select_party")
class SelectParty(CustomAction):

    NAME_ROI = {
        "main": [574, 500, 170, 55],
        "sub1": [235, 466, 170, 55],
        "sub2": [910, 466, 170, 55]
    }

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        """选择队伍

        Args:
            context: 任务上下文。
            argv: 自定义动作参数。

        Returns:
            bool: 是否成功选择队伍。
        """

        node_data = context.get_node_data(argv.node_name)
        if not node_data:
            node_data = {}
        attachment = node_data.get("attach", {})
        trekker_names = attachment.get("trekker_names", {})
        main_trekker_names = trekker_names.get("main", [])
        sub_trekker_names = trekker_names.get("sub", [])

        if not main_trekker_names or not sub_trekker_names:
            return True

        # 数据清洗
        cleaned_main_trekker_names = [re.sub(r'\W', '', name) for name in main_trekker_names]
        cleaned_sub_trekker_names = [re.sub(r'\W', '', name) for name in sub_trekker_names]

        chars_to_remove = "ー"
        table = str.maketrans("", "", chars_to_remove)
        cleaned_main_trekker_names = [name.translate(table) for name in cleaned_main_trekker_names]
        cleaned_sub_trekker_names = [name.translate(table) for name in cleaned_sub_trekker_names]

        for p in range(6):
            image = context.tasker.controller.post_screencap().wait().get()
            logger.debug(f"开始识别第{p+1}个队伍")
            reco_names = []
            for position, roi in self.NAME_ROI.items():
                logger.debug(f"开始识别{position}位置的旅人名称")
                reco_name = self._recognize_trekker_name(context, roi, image)
                cleaned_reco_name = re.sub(r'\W', '', reco_name)
                cleaned_reco_name = cleaned_reco_name.translate(table)
                if "main" in position and cleaned_reco_name in cleaned_main_trekker_names:
                    reco_names.append(reco_name)
                elif "sub" in position and cleaned_reco_name in cleaned_sub_trekker_names:
                    reco_names.append(reco_name)
            if len(reco_names) == 3:
                logger.info(f"成功识别到队伍：{reco_names}")
                return True
            context.tasker.controller.post_click(1245, 345).wait()
            time.sleep(1)

        logger.error("没有识别到作业对应队伍，请检查旅人名称是否正确，或是否有现成作业的编队")
        context.tasker.post_stop()
        return False

    @staticmethod
    def _recognize_trekker_name(context, roi, image=None):
        """
        识别旅人名称

        Args:
            context: 任务上下文。
            roi: 识别区域。
            image: 图片数据。

        Returns:
            str: 识别到的旅人名称。
        """
        if image is None:
            image = context.tasker.controller.post_screencap().wait().get()

        reco_detail = context.run_recognition("星塔_编队角色_识别旅人名称_agent", image, {
            "星塔_编队角色_识别旅人名称_agent": {
                "recognition": {
                    "param": {
                        "roi": roi
                    }
                }
            }
        })
        if reco_detail and reco_detail.hit:
            logger.debug(f"识别到旅人名称：{[[r.text, r.score] for r in reco_detail.filtered_results]}")
            return "".join(r.text for r in reco_detail.filtered_results)

        if reco_detail and reco_detail.all_results:
            logger.debug(f"没有识别到旅人名称")
            logger.debug(f"识别到的结果：{[[r.text, r.score] for r in reco_detail.all_results]}")
        else:
            logger.error(f"识别旅人名称失败")

        return ""
