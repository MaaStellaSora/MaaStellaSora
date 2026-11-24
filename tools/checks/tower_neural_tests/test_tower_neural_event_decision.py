"""One event decision serves repeated recognition and its confirmed UI click."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from maa.custom_recognition import CustomRecognition
from custom.climb_tower_neural import agent_callbacks, events
from custom.climb_tower_neural.event_policy import EventMenu, EventPolicy
from custom.climb_tower_neural.value_model import sim
from test_tower_neural_wallet_state import tracker_at_quiz


def event_fixture(*, selected_index=0):
    tracker = tracker_at_quiz()
    tracker.model.versions = {"policy_mode": "potential_only"}

    def decide(state, actions, preset, **inputs):
        values = [dict(action=sim.action_key(action), action_record=sim.to_jsonable(action),
                       predicted_value=float(len(actions) - index),
                       expected_total=float(6 + len(actions) - index))
                  for index, action in enumerate(actions)]
        return dict(action_record=values[selected_index]["action_record"], values=values,
                    expected_total=values[selected_index]["expected_total"], ordinary_score=0)

    # Keep the real tracker, route transaction, policy and public decision path;
    # only the neural estimates and Maa frame/click boundary are supplied.
    tracker.model.decide = Mock(side_effect=decide)
    context = SimpleNamespace(tasker=SimpleNamespace(stopping=False),
        override_pipeline=Mock(), run_action=Mock(return_value=SimpleNamespace(success=True)))
    runtime = SimpleNamespace(tracker=tracker, require_tracker=lambda: tracker,
        event_policy=EventPolicy(), stop_model=Mock(),
        decision_context=lambda kind, payload: deepcopy(dict(payload, kind=kind)),
        collector=SimpleNamespace(context=lambda value: value, event_clicked=Mock()),
        notes=SimpleNamespace(prepare_event_inventory=Mock(return_value=(True, False)),
                              confirm_event_choice=Mock()))
    argv = SimpleNamespace(node_name="星塔_节点_进行对话选择_agent", image=object(), box=None)
    return tracker, runtime, context, argv


def menu_read(*, coins=480, shift=0, attempts=1, quiz=False, duplicate_boxes=False):
    choices = [SimpleNamespace(text=f"option-{index}", consequence=f"effect-{index}",
        available=True, box=[10 + shift, 100 + index * 60, 200, 40], observations={})
        for index in range(3)]
    if duplicate_boxes:
        choices[0].box = list(choices[1].box)
    options = [dict(ui_index=index, model_option_id=10101 + index, available=True,
                    evaluation_status="not_evaluated", reason="model_value_not_recorded")
               for index in range(3)]
    menu = EventMenu(None if quiz else 101, {} if quiz else {i: 10101 + i for i in range(3)}, options)
    answer = CustomRecognition.AnalyzeResult(box=choices[1].box, detail={"official_rule": True}) if quiz else None
    return events.EventMenuRead(question="same visible question", choices=choices,
        coins=coins, attempts=attempts, missing=["coin"] if coins is None else [],
        menu=menu, quiz_id=116 if quiz else None, answer=answer)


class TowerNeuralEventDecisionTests(unittest.TestCase):
    def test_repeated_menu_reuses_decision_and_current_box_but_changed_wallet_revalues(self):
        for duplicate_boxes in (False, True):
            with self.subTest(duplicate_boxes=duplicate_boxes):
                tracker, runtime, context, argv = event_fixture(selected_index=1)
                before = tracker._event_snapshot()
                first = events.analyze_model_event(None, context, argv, runtime,
                    reading=menu_read(duplicate_boxes=duplicate_boxes))
                self.assertEqual(tracker.event_page_cache[1], 1)
                repeated = events.analyze_model_event(None, context, argv, runtime,
                    reading=menu_read(shift=25, attempts=2, duplicate_boxes=duplicate_boxes))

                tracker.model.decide.assert_called_once()
                self.assertEqual(first.detail["option"], repeated.detail["option"])
                self.assertEqual(repeated.detail["option"], 10102)
                self.assertEqual(repeated.box, [35, 160, 200, 40])
                self.assertEqual(repeated.detail["selected_box"], list(repeated.box))
                self.assertEqual(repeated.detail["ocr_attempts"], 2)
                self.assertEqual(tracker.event_page_cache[1], 1)
                self.assertEqual(tracker.event_page_cache[2], repeated.detail)
                self.assertEqual(tracker._event_snapshot(), before)

                changed = events.analyze_model_event(None, context, argv, runtime,
                    reading=menu_read(coins=500, attempts=3, duplicate_boxes=duplicate_boxes))

                self.assertEqual(tracker.model.decide.call_count, 2)
                self.assertEqual(changed.detail["observed_coins"], 500)
                self.assertEqual(tracker.event_page_cache[2], changed.detail)
                self.assertEqual(tracker._event_snapshot(), before)
                runtime.stop_model.assert_not_called()

    def test_click_commits_latest_choice_only_after_confirmed_transition(self):
        tracker, runtime, context, argv = event_fixture()
        before = tracker._event_snapshot()
        events.analyze_model_event(None, context, argv, runtime, reading=menu_read())
        result = events.analyze_model_event(None, context, argv, runtime, reading=menu_read(coins=500))
        argv.box = result.box
        click = agent_callbacks.NoteEventClick(runtime)

        with patch.object(agent_callbacks, "wait_for", return_value=None):
            self.assertFalse(click.run(context, argv))
        self.assertEqual(tracker._event_snapshot(), before)
        self.assertIsNotNone(tracker.event_page_cache)
        runtime.collector.event_clicked.assert_not_called()

        with patch.object(agent_callbacks, "wait_for", return_value="field"):
            self.assertTrue(click.run(context, argv))
        self.assertEqual(tracker.pending_event, (101, 10101))
        self.assertEqual(tracker.state.coins, 500)
        self.assertIsNone(tracker.event_page_cache)
        runtime.notes.confirm_event_choice.assert_called_once()
        runtime.collector.event_clicked.assert_called_once()
        self.assertFalse(click.run(context, argv))
        self.assertEqual(context.run_action.call_count, 2)

        # A separately confirmed event with the same menu/wallet is a new
        # decision, even though its visible text matches the previous event.
        events.analyze_model_event(None, context, argv, runtime, reading=menu_read(coins=500))
        self.assertEqual(tracker.model.decide.call_count, 3)
        runtime.stop_model.assert_not_called()

    def test_quiz_without_wallet_keeps_matched_official_answer_and_pending_transaction(self):
        tracker, runtime, context, argv = event_fixture()
        before = tracker._event_snapshot()
        reading = menu_read(coins=None, quiz=True)

        result = events.analyze_model_event(None, context, argv, runtime, reading=reading)

        self.assertEqual(result.box, reading.answer.box)
        self.assertEqual((result.detail["event"], result.detail["option"]), (116, 11603))
        self.assertEqual(result.detail["decision_method"], "official_answer_rule")
        self.assertEqual(result.detail["ocr_missing"], ["coin"])
        self.assertIsNone(result.detail["observed_coins"])
        self.assertEqual(tracker.event_page_cache[2], result.detail)
        tracker.model.decide.assert_not_called()
        self.assertEqual(tracker._event_snapshot(), before)
        tracker.event_action_confirmed()
        self.assertEqual(tracker.pending_event, (116, 11603))
        self.assertTrue(tracker.coins_uncertain)
        self.assertIsNone(tracker.event_page_cache)
        runtime.stop_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
