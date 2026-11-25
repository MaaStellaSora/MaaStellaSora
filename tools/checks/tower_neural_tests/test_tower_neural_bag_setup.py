"""A later sweep can skip its optional bag visit while the offer is still visible."""
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural import bag_preset_reader as bag, maa_hook, preset_tool


class BagSetupTests(unittest.TestCase):
    def test_unknown_member_compiles_with_corrected_header(self):
        reader = bag._Reader(Mock(), "cn", None)
        count = dict(text="0/16", score=.99, box=[860, 90, 35, 12])
        with (patch.object(reader, "_confirm_display"), patch.object(reader, "_visible_cards", return_value=[]),
              patch.object(reader, "headers", return_value=[dict(box=[840, 90, 12, 12])]),
              patch.object(bag, "_matches", return_value=[]), patch.object(bag, "read_text") as read):
            readings = [("星璃", .70), ("星黎", .99), ("星璃", .88), ("星", .995), ("星黎星", .70)]
            for frame, (text, score) in enumerate(readings, 1):
                reader.frame = frame
                read.return_value = [dict(text=text, score=score, box=[310, 90, 40, 12]), count]
                reader.feed(None)
        member, = reader.members
        self.assertEqual(member["name"], "星黎")
        cards = [dict(name=f"潜能{i}", rarity_class=kind, level=0, target_level=1, name_confirmed=True)
                 for i, kind in enumerate(["core"] * 4 + ["rare"] * 3 + ["common"] * 9)]
        party = [dict(role=1, name=member["name"], cards=cards, obtained_count=0,
                      raw=dict(header_readings=member["header_readings"]))]
        for role, name in [(2, "辅助甲"), (3, "辅助乙")]:
            party.append(dict(role=role, name=name, cards=cards, obtained_count=0,
                              raw=dict(header_readings=[dict(text=name, score=.99)])))
        preset = preset_tool.compile_bag_preset(party, "cn", dict(characters=[], potentials=[]))
        self.assertTrue(preset["party"][0]["local_potential_catalogue"])
        self.assertEqual(preset["bag_party"][0]["name"], "星黎")

    def test_fresh_preset_requires_one_target_across_the_whole_party(self):
        for target_kind in (None, "common", "core"):
            with self.subTest(target_kind=target_kind):
                runtime = maa_hook.ValueModelRuntime(Path(maa_hook.__file__).parent, ROOT)
                preset = dict(party=[dict(name_ja_jp=str(role), targets=[
                    dict(target_level=int(role == 1 and target_kind is not None),
                         rarity_class=target_kind or "common")]) for role in range(3)])
                runtime.notes = Mock(observing=True)
                runtime.collector = Mock()
                runtime.logger = Mock()
                context = Mock()
                context.tasker.stopping = False
                runtime.language = Mock(return_value="jp")
                runtime._bag_task = Mock(return_value=True)
                runtime._returned_from_bag = Mock(return_value=True)
                with (
                    patch("custom.climb_tower_neural.bag_preset_reader.read_bag_party", return_value=[]),
                    patch("custom.climb_tower_neural.preset_tool.compile_bag_preset", return_value=preset),
                    patch.object(maa_hook, "ObservationTracker") as tracker,
                    patch.object(maa_hook, "ValueModel") as model,
                ):
                    self.assertEqual(runtime.initialize_from_bag(context), target_kind is not None)
                if target_kind is None:
                    tracker.assert_not_called()
                    model.assert_not_called()
                    self.assertIsNone(runtime.confirmed_preset)
                    context.tasker.post_stop.assert_called_once()
                    runtime.logger.error.assert_called_once_with(
                        "$agent.custom.action.climb_tower_neural.preset_empty", {})
                    runtime.logger.info.assert_not_called()
                    runtime.collector.bind_preset.assert_not_called()
                    runtime.notes.read_bag_setup.assert_not_called()
                    self.assertEqual(runtime._bag_task.call_count, 1)
                    runtime._returned_from_bag.assert_not_called()
                else:
                    self.assertIs(runtime.confirmed_preset, preset)
                    context.tasker.post_stop.assert_not_called()
                    runtime.notes.read_bag_setup.assert_called_once_with(context)
                    tracker.assert_called_once()

    def test_empty_preset_message_is_exact(self):
        messages = json.loads((ROOT / "agent/agent_config/i18n/zh_cn.json").read_text(encoding="utf-8"))
        self.assertEqual(messages["agent.custom.action.climb_tower_neural.preset_empty"],
                         "预设未设置，或未在背包打开“显示推荐”。")

    def test_failed_open_requires_both_cached_preset_and_confirmed_offer(self):
        preset = dict(party=[dict(name_ja_jp="A", targets=[dict(target_level=1), dict(target_level=2)]),
                             dict(name_ja_jp="B", targets=[dict(target_level=0)])])
        for targets, page, stop_reason in (
            (preset, "potential", None),
            (preset, None, "bag_return_unconfirmed"),
            (None, "potential", "bag_not_opened"),
        ):
            with self.subTest(cached=targets is not None, page=page):
                runtime = maa_hook.ValueModelRuntime(Path(maa_hook.__file__).parent, ROOT)
                runtime.confirmed_preset = targets
                runtime.fresh_entry = True
                runtime.model = Mock()
                runtime.logger = Mock()
                runtime.notes = Mock(observing=True)
                runtime.collector = Mock()
                runtime.stop_model = Mock()
                context = Mock()
                context.tasker.stopping = False
                context.get_node_data.return_value = {"attach": {"handler": "value_model", "lang_type": "jp"}}
                context.run_task.return_value.status.succeeded = False
                returned, fresh_image = [object()], object()
                with (
                    patch.object(maa_hook, "classify", return_value=page),
                    patch.object(maa_hook, "wait_for", side_effect=lambda ctx, observe, **kw: observe(fresh_image)),
                    patch.object(maa_hook, "ObservationTracker") as tracker,
                ):
                    continued = runtime.ensure_tracker(context, "potential", returned_frame=returned)
                self.assertEqual(continued, stop_reason is None)
                context.run_task.assert_called_once()
                runtime.notes.read_bag_setup.assert_not_called()
                if targets is not None:
                    runtime.collector.emit.assert_called_once_with(
                        "note_bag_setup_observed", complete=False, reason="bag_not_opened")
                if stop_reason:
                    self.assertEqual(runtime.stop_model.call_args.args[1], stop_reason)
                    tracker.assert_not_called()
                else:
                    self.assertIs(returned[0], fresh_image)
                    self.assertIs(runtime.tracker, tracker.return_value)
                    runtime.logger.info.assert_called_once_with("$agent.custom.action.climb_tower_neural.preset_loaded",
                        {"party": "A / B", "total": 3, "potential_targets": "3 + 0"})
                    runtime.stop_model.assert_not_called()
