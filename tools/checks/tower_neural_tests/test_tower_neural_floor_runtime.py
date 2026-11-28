"""Read route floors without collecting notes or reusing stale click boxes."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural import note_floor, note_ocr, note_runtime
from custom.climb_tower_neural.session import RunSession
from test_tower_neural_wallet_state import select_quiz, tracker_at_quiz


def floor_observation(value):
    return dict(floor=value, complete=value is not None, source="floor_hud")


class TowerNeuralFloorRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.session = RunSession()
        self.session.begin_run()
        self.collector = SimpleNamespace(enabled=False, emit=Mock())
        runtime = SimpleNamespace(session=self.session, collector=self.collector,
                                  tracker=None, logger=Mock())
        self.notes = note_runtime.NoteRuntime(runtime)
        self.context = SimpleNamespace(tasker=SimpleNamespace(stopping=False), run_recognition=Mock(),
            get_node_data=Mock(return_value=dict(attach=dict(active_melodies=note_ocr.TOWER_ELEMENTS["光土"]))))
        self.image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.bar = dict(bar_present=True, bar_reason="clear_ordered_inventory_row")

    def test_recording_toggle_keeps_floor_prior_and_ocr_consensus_identical(self):
        observations = []
        for recording in (False, True):
            with self.subTest(recording=recording):
                session = RunSession()
                session.begin_run()
                collector = SimpleNamespace(enabled=recording, emit=Mock(return_value=None))
                runtime = SimpleNamespace(session=session, collector=collector, logger=Mock(),
                    tracker=SimpleNamespace(state=SimpleNamespace(floor=1), pending_event=None))
                notes = note_runtime.NoteRuntime(runtime)
                if recording:
                    notes.ledger = Mock(pending=None, revision=0)
                    notes.ledger.begin_battle.return_value = []
                    notes.ledger.observe.side_effect = lambda *args, **kwargs: dict(event_type="note_inventory_observed")
                bar = dict(self.bar, complete=True)
                def readings(*texts):
                    return [[dict(text=text, score=.99, box=[0, 0, 20, 20])] for text in texts]
                with (
                    patch.object(note_runtime, "locate_inventory_bar", return_value=bar),
                    patch.object(note_runtime, "read_inventory", return_value=bar),
                    patch.object(note_runtime, "read_until"),
                    patch.object(note_floor, "read_text", side_effect=readings("01", "01", "01", "01")),
                ):
                    notes.capture(self.context, self.image)
                self.assertEqual(session.last_confirmed_floor["floor"], 1)
                self.assertEqual(notes.floor, 1 if recording else None)
                with (
                    patch.object(note_runtime, "locate_inventory_bar", return_value=bar),
                    patch.object(note_runtime, "read_inventory", return_value=bar),
                    patch.object(note_runtime, "read_until") as retry,
                    patch.object(note_floor, "read_text", side_effect=readings("03", "03", "01", "01")) as ocr,
                ):
                    notes.capture(self.context, self.image)
                observed = session.current_floor_observation
                observations.append(observed)
                self.assertEqual(observed["expected_floors"], [1, 3])
                self.assertEqual((observed["floor"], observed["complete"], observed["early_completed"]), (3, True, True))
                self.assertEqual(ocr.call_count, 2)
                retry.assert_not_called()
        self.assertEqual(observations[0], observations[1])

    def test_confirmed_event_prior_uses_model_routes_without_note_ledger(self):
        tracker = tracker_at_quiz()
        select_quiz(tracker)
        tracker.event_action_confirmed()
        observations = []
        for recording in (False, True):
            with self.subTest(recording=recording):
                runtime = SimpleNamespace(session=tracker.session, tracker=tracker, logger=Mock(),
                    collector=SimpleNamespace(enabled=recording, emit=Mock()))
                notes = note_runtime.NoteRuntime(runtime)
                if recording:
                    notes.floor = 1
                    notes.ledger = SimpleNamespace(revision=1, pending=dict(source_type="event",
                        start_revision=0, metadata=dict(event_floor_candidates=[1, 2])))
                rows = [[dict(text=text, score=.99, box=[0, 0, 20, 20])]
                        for text in ("01", "01", "02", "02")]
                with patch.object(note_floor, "read_text", side_effect=rows) as ocr:
                    observation = notes._floor_observation(self.context, self.image, "result")
                observations.append(observation)
                self.assertEqual(set(observation["expected_floors"]), {1, 2})
                self.assertEqual((observation["floor"], observation["complete"]), (1, True))
                self.assertEqual(ocr.call_count, 2)
        self.assertEqual(observations[0], observations[1])

    def test_recording_disabled_reads_floor_without_note_counts_or_ledger(self):
        icons = [dict(melody="melody_of_pummel", box=[10, 10, 20, 20], score=.99)]
        with (
            patch.object(note_ocr, "find_icons", return_value=icons),
            patch.object(note_ocr, "_locate_inventory_bar", return_value=(icons, self.bar["bar_reason"], {})),
            patch.object(note_runtime, "read_floor_observation", return_value=floor_observation(3)) as floor,
            patch.object(note_runtime, "read_inventory") as inventory,
            patch.object(note_ocr, "_read_count") as count,
            patch.object(note_runtime, "read_until") as wait,
        ):
            self.assertEqual(self.notes.capture(self.context, self.image), (None, True, False))
        self.assertEqual(self.session.last_confirmed_floor["floor"], 3)
        self.assertTrue(self.session.current_floor_observation["complete"])
        floor.assert_called_once()
        inventory.assert_not_called()
        count.assert_not_called()
        wait.assert_not_called()
        self.collector.emit.assert_not_called()
        self.assertIsNone(self.notes.ledger)

    def test_generic_close_page_skips_floor_and_wait_but_retains_last_anchor(self):
        self.session.observe_floor("result", floor_observation(3))
        with (
            patch.object(note_runtime, "locate_inventory_bar", return_value=dict(
                bar_present=False, bar_reason="no_ordered_inventory_row")),
            patch.object(note_runtime, "read_floor_observation") as floor,
            patch.object(note_runtime, "read_until") as wait,
        ):
            self.assertEqual(self.notes.capture(self.context, self.image), (None, True, False))
        floor.assert_not_called()
        wait.assert_not_called()
        self.assertEqual(self.session.last_confirmed_floor["floor"], 3)
        self.assertIsNone(self.session.current_floor_observation["floor"])
        self.assertFalse(self.session.current_floor_observation["complete"])

    def test_retry_waits_for_page_anchor_and_replaces_continue_box(self):
        original_box, new_box = [10, 10, 20, 20], [30, 40, 50, 60]
        continue_box = list(original_box)
        frame = self.image.copy()
        self.context.run_recognition.side_effect = [None, SimpleNamespace(
            hit=True, best_result=SimpleNamespace(box=new_box))]

        def retry(context, observe):
            self.assertIs(context, self.context)
            self.assertEqual(continue_box, [])
            self.assertIsNone(observe(frame))
            self.assertEqual(floor.call_count, 1)
            self.assertTrue(observe(frame))

        with (
            patch.object(note_runtime, "locate_inventory_bar", return_value=self.bar) as bar,
            patch.object(note_runtime, "read_floor_observation", side_effect=[
                floor_observation(None), floor_observation(4)]) as floor,
            patch.object(note_runtime, "read_until", side_effect=retry) as wait,
        ):
            self.assertEqual(self.notes.capture(self.context, self.image, continue_box=continue_box),
                             (None, True, False))
        wait.assert_called_once()
        self.assertEqual(bar.call_count, 2)
        self.assertEqual(continue_box, new_box)
        self.assertEqual(self.session.last_confirmed_floor["floor"], 4)
        self.assertIs(floor.call_args.args[1], frame)
        self.assertEqual(self.context.run_recognition.call_args.args[0], "星塔_点击空白处关闭_agent")

    def test_disappeared_anchor_discards_old_continue_box(self):
        continue_box = [10, 10, 20, 20]
        self.context.run_recognition.return_value = SimpleNamespace(hit=False)

        def retry(context, observe):
            self.assertIsNone(observe(self.image))

        with (
            patch.object(note_runtime, "locate_inventory_bar", return_value=self.bar) as bar,
            patch.object(note_runtime, "read_floor_observation", return_value=floor_observation(None)) as floor,
            patch.object(note_runtime, "read_until", side_effect=retry),
        ):
            self.assertEqual(self.notes.capture(self.context, self.image, continue_box=continue_box),
                             (None, False, False))
        self.assertEqual(continue_box, [])
        floor.assert_called_once()
        bar.assert_called_once()
        self.assertFalse(self.session.current_floor_observation["complete"])


if __name__ == "__main__":
    unittest.main()
