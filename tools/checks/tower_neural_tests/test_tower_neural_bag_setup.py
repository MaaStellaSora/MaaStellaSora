"""A later sweep can skip its optional bag visit while the offer is still visible."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural import maa_hook


class BagSetupTests(unittest.TestCase):
    def test_failed_open_requires_both_cached_preset_and_confirmed_offer(self):
        preset = dict(party=[dict(name_ja_jp="A")], role_ordinary_totals=[2])
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
                    runtime.stop_model.assert_not_called()
