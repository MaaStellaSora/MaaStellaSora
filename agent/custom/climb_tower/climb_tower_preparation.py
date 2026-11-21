import os
import re
import time
import json
from typing import Optional
from pathlib import Path

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from custom.climb_tower.climb_tower_potential.state import State

from utils import logger as logger_module

logger = logger_module.get_logger(__name__)


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
        # 获取当前节点的attach数据
        preparation_node_data = context.get_node_data(argv.node_name) or {}
        attachments = preparation_node_data.get("attach", {})

        # 获得潜能选择节点的attach数据，以获取潜能模式
        potential_node_data = context.get_node_data("星塔_节点_选择潜能_agent") or {}
        potential_attachments = potential_node_data.get("attach", {})
        handler = potential_attachments.get("handler", "")

        # 如果潜能模式为json时，导入json作业参数
        if handler == "json":
            preset_path = Path(os.path.abspath(__file__)).parent.parent.parent / "presets"
            json_import_result = self._import_json_priority_list(context, preparation_node_data, preset_path)
            if not json_import_result:
                return False

        # 如果潜能模式为preset时，检查刷新阈值能否被转为float类型
        if handler.startswith("preset"):
            threshold_coef_str = potential_attachments.get("threshold_coef_str", "")
            threshold_decay_str = potential_attachments.get("threshold_decay_str", "")
            try:
                float(threshold_coef_str)
                float(threshold_decay_str)
            except ValueError:
                logger.error(
                    "$agent.custom.action.climb_tower_preparation.run.refresh_coefficients_invalid",
                    {
                        "threshold_coef_str": threshold_coef_str,
                        "threshold_decay_str": threshold_decay_str,
                    },
                )
                return False

        # 导入对话选项预案
        event_rules_file = attachments.get("event_rules", "")
        event_rules_path = Path(os.path.abspath(__file__)).parent.parent.parent / "agent_config" / event_rules_file
        event_rules_path = event_rules_path.with_suffix(".json")
        event_rules = self._load_event_rules(event_rules_path)
        if not event_rules: # 不允许空规则
            logger.error(
                "$agent.custom.action.climb_tower_preparation.run.event_rules_load_failed",
                {
                    "event_rules_file": event_rules_file,
                },
            )
            return False
        context.override_pipeline({"星塔_节点_进行对话选择_agent": {"attach": {"rules": event_rules}}})
        logger.info(
            "$agent.custom.action.climb_tower_preparation.run.imported",
            {
                "event_rules_file": event_rules_file,
                "count": len(event_rules),
            },
        )

        # 清除潜能状态
        reset_state = attachments.get("reset_state", False)
        if reset_state:
            State.reset()

        return True

    @staticmethod
    def _load_event_rules(event_rules_path: Path) -> list:
        """加载对话选项规则文件

        Args:
            event_rules_path (Path): 对话选项规则文件路径。

        Returns:
            list: 对话选项规则列表。
        """
        try:
            with open(event_rules_path, "r", encoding="utf-8") as f:
                content = f.read()
                # 因为规则可能作为正则使用，若预案中包含未转义的加号(+)会影响匹配，先做文本层面检查并给出提示
                # if "+" in content:
                #     logger.warning(
                #         f"对话选项规则'{event_rules_path}'中包含加号(+)作为正则元字符，可能需要转义以避免解析或匹配异常")
                event_rules = json.loads(content)
                if not isinstance(event_rules, list):
                    logger.error(
                        "$agent.custom.action.climb_tower_preparation._load_event_rules.event_rules_wrong_type",
                        {
                            "event_rules_path": event_rules_path,
                        },
                    )
                    return []
                if not event_rules:
                    logger.error(
                        "$agent.custom.action.climb_tower_preparation._load_event_rules.event_rules_empty",
                        {
                            "event_rules_path": event_rules_path,
                        },
                    )
                    return []
                return event_rules
        except FileNotFoundError:
            logger.error(
                "$agent.custom.action.climb_tower_preparation._load_event_rules.event_rules_file_not_found",
                {
                    "event_rules_path": event_rules_path,
                },
            )
            logger.error(
                "$agent.custom.action.climb_tower_preparation._load_event_rules.verify_event_rules_filename"
            )
            return []
        except json.decoder.JSONDecodeError as e:
            logger.error(
                "$agent.custom.action.climb_tower_preparation._load_event_rules.event_rules_parse_failed",
                {
                    "e": e,
                },
            )
            logger.error(
                "$agent.custom.action.climb_tower_preparation._load_event_rules.verify_json_format"
            )
            return []
        except OSError as e:
            logger.error(
                "$agent.custom.action.climb_tower_preparation._load_event_rules.file_read_failed",
                {
                    "e": e,
                },
            )
            logger.error(
                "$agent.custom.action.climb_tower_preparation._load_event_rules.verify_file_permissions"
            )
            return []

    def _import_json_priority_list(self, context: Context, node_data: dict, preset_path: Path) -> bool:
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
            logger.error(
                "$agent.custom.action.climb_tower_preparation._import_json_priority_list.preset_job_not_found",
                {
                    "full_path": full_path,
                },
            )
            logger.error(
                "$agent.custom.action.climb_tower_preparation._import_json_priority_list.verify_preset_job_name"
            )
            return False
        except json.decoder.JSONDecodeError as e:
            logger.error(
                "$agent.custom.action.climb_tower_preparation._import_json_priority_list.preset_job_parse_failed",
                {
                    "e": e,
                },
            )
            logger.error(
                "$agent.custom.action.climb_tower_preparation._import_json_priority_list.verify_json_format"
            )
            return False

        err = self._validate_priority_list(priority_list)
        if err:
            logger.error(
                "$agent.custom.action.climb_tower_preparation._import_json_priority_list.potential_priority_invalid",
                {
                    "err": err,
                },
            )
            return False

        context.override_pipeline({
            "星塔_节点_选择潜能_agent": {
                "attach": {
                    "priority_list": priority_list
                }
            }
        })

        if preset_element:
            logger.info(
                "$agent.custom.action.climb_tower_preparation._import_json_priority_list.selected",
                {
                    "preset_element": preset_element,
                },
            )
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
                            },
                            "attach": {
                                "active_melodies": ["melody_of_aqua", "melody_of_ventus"]
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
                            },
                            "attach": {
                                "active_melodies": ["melody_of_ignis", "melody_of_umbra"]
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
                            },
                            "attach": {
                                "active_melodies": ["melody_of_lux", "melody_of_terra"]
                            }
                        }
                    })
                case _:
                    logger.error(
                        "$agent.custom.action.climb_tower_preparation._import_json_priority_list.unknown_element",
                        {
                            "preset_element": preset_element,
                        },
                    )
                    context.tasker.post_stop()
                    return False

        if preset_trekker_names:
            logger.info(
                "$agent.custom.action.climb_tower_preparation._import_json_priority_list.team_selected",
                {
                    "preset_trekker_names": preset_trekker_names,
                },
            )
            context.override_pipeline({
                "星塔_编队角色_选择队伍_agent": {
                    "attach": {
                        "trekker_names": preset_trekker_names
                    }
                }
            })

        if preset_potential_refresh:
            logger.info(
                "$agent.custom.action.climb_tower_preparation._import_json_priority_list.detected",
                {
                    "preset_potential_refresh": preset_potential_refresh,
                },
            )
            context.override_pipeline({
                "星塔_节点_选择潜能_agent": {
                    "attach": {
                        "max_refresh_count": preset_potential_refresh
                    }
                }
            })

        if preset_melodies:
            logger.info(
                "$agent.custom.action.climb_tower_preparation._import_json_priority_list.melody_detected",
                {
                    "preset_melodies": preset_melodies,
                },
            )
            node_data = context.get_node_data("星塔_节点_商店_商店主流程_agent") or {}
            shop_attachments = node_data.get("attach", {})
            for melody in preset_melodies:
                melody = melody.lower()
                if melody in shop_attachments:
                    context.override_pipeline({
                        "星塔_节点_商店_商店主流程_agent": {
                            "attach": {
                                melody: True
                            }
                        }
                    })
                else:
                    logger.error(
                        "$agent.custom.action.climb_tower_preparation._import_json_priority_list.melody_import_failed",
                        {
                            "melody": melody,
                        },
                    )
                    return False

        logger.info(
            "$agent.custom.action.climb_tower_preparation._import_json_priority_list.imported",
            {
                "preset_name": preset_name,
            },
        )
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
            logger.debug(f"开始识别第{p + 1}个队伍")
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
                logger.info(
                    "$agent.custom.action.climb_tower_preparation.run.team_recognized",
                    {
                        "reco_names": reco_names,
                    },
                )
                return True
            context.tasker.controller.post_click(1245, 345).wait()
            time.sleep(1)

        logger.error("$agent.custom.action.climb_tower_preparation.run.preset_team_not_recognized")
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
            logger.error(
                "$agent.custom.action.climb_tower_preparation._recognize_trekker_name.trekker_name_recognition_failed"
            )

        return ""
