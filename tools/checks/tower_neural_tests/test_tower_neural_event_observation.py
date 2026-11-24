"""Existing event recording passes references to note collection explicitly."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tower_temp_directory import TowerTempDirectory


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural.note_ledger import NoteLedger
from custom.climb_tower_neural.note_runtime import NoteRuntime
from custom.climb_tower_neural.observation import PotentialObservationIncomplete
from custom.climb_tower_neural.sampling_data import SamplingCollector
from custom.climb_tower_neural.session import RunSession
from custom.climb_tower_neural.tower_sampling_collector import ObservedPotentialHandler


def potential_handler(source="default", *, core=False, level_up=False):
    return SimpleNamespace(
        data=SimpleNamespace(params=SimpleNamespace(potential_source=source),
                             core_potential=core, level_upped=level_up),
        screen=Mock(), initialize_potentials=Mock(return_value="initialized"),
    )


class TowerNeuralEventObservationTests(unittest.TestCase):
    def setUp(self):
        directory = TowerTempDirectory()
        self.addCleanup(directory.cleanup)
        self.collector = SamplingCollector(Path(directory.name), Mock())
        self.collector.begin_task("cn", "test")
        self.collector.start_intent("start")
        self.collector.confirm_start("entered")
        self.choice = dict(text="领取潜能", consequence="获得潜能", box=[1, 2, 3, 4])
        self.attach = dict(last_question="question", last_choice=self.choice["text"],
                           last_consequence=self.choice["consequence"])
        self.result = SimpleNamespace(box=self.choice["box"], detail={})

    def choose_event(self, *, click=False):
        self.collector.event_menu("question", [self.choice])
        reference = self.collector.event_choice(self.result, self.attach)
        if click:
            self.collector.event_clicked("stable_event_page_transition")
        return reference

    def rows(self):
        path = self.collector.directory / "runs" / f"{self.collector.run['run_id']}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_repeated_menu_and_choice_return_the_same_independent_reference(self):
        first = self.choose_event()
        second = self.choose_event()
        self.assertEqual(first, second)
        self.assertEqual(first["event_id"], self.collector.event["event_id"])
        choices = [row for row in self.rows() if row["event_type"] == "event_choice"]
        self.assertEqual(len(choices), 1)
        self.assertEqual(first["event_choice_record_id"], choices[0]["record_id"])
        second["event_id"] = "caller-local"
        self.assertEqual(self.collector.event["event_id"], first["event_id"])
        self.collector.event_menu("another question", [self.choice])
        self.assertNotEqual(self.collector.event["event_id"], first["event_id"])

    def test_choice_reference_keeps_event_identity_when_record_has_no_id(self):
        self.assertEqual(self.collector.event_choice(self.result, self.attach), {})
        self.collector.event_menu("question", [self.choice])
        with patch.object(self.collector, "emit", return_value=None) as emit:
            reference = self.collector.event_choice(self.result, self.attach)
            self.assertEqual(self.collector.event_choice(self.result, self.attach), reference)
        emit.assert_called_once()
        self.assertEqual(reference, dict(event_id=self.collector.event["event_id"], event_choice_record_id=None))
        self.assertEqual(self.collector.event_choice(SimpleNamespace(box=None), self.attach), {})

    def test_first_default_reward_uses_confirmed_event_and_refresh_keeps_its_source(self):
        reference = self.choose_event(click=True)
        handler = potential_handler()
        first = self.collector.prepare_offer(handler.data)
        self.assertEqual(self.collector.reward["source_fields"]["source_event_id"], reference["event_id"])
        self.assertFalse(self.collector.event["potential_reward_pending"])
        self.assertIs(self.collector.prepare_offer(handler.data), first)
        refreshed = self.collector.refresh_succeeded(first["offer_id"], "refresh")
        self.assertEqual(self.collector.prepare_offer(handler.data)["offer_id"], refreshed)
        self.assertEqual(self.collector.reward["source_fields"]["source"], "event")
        self.collector.reward["closed"] = True
        self.collector.prepare_offer(handler.data)
        self.assertEqual(self.collector.reward["source_fields"]["source"], "unknown")

    def test_explicit_sources_consume_event_candidate_without_borrowing_attribution(self):
        for source, core, level_up, expected in (
                ("default", True, False, "core"), ("default", False, True, "level_up"),
                ("drink", False, False, "drink"), ("specified_drink", False, False, "specified_drink"),
                ("enhance", False, False, "enhance")):
            with self.subTest(source=expected):
                self.choose_event(click=True)
                self.collector.prepare_offer(potential_handler(source, core=core, level_up=level_up).data)
                self.assertEqual(self.collector.reward["source_fields"]["source"], expected)
                self.assertFalse(self.collector.event["potential_reward_pending"])
                self.collector.reward["closed"] = True
                self.collector.prepare_offer(potential_handler().data)
                self.assertEqual(self.collector.reward["source_fields"]["source"], "unknown")

    def test_recording_disabled_or_unconfirmed_delegates_without_note_navigation(self):
        for enabled, confirmed in ((False, True), (True, False)):
            with self.subTest(enabled=enabled, confirmed=confirmed):
                self.collector.enabled, self.collector.run_confirmed = enabled, confirmed
                delegate, notes = potential_handler(), Mock()
                handler = ObservedPotentialHandler(delegate, self.collector, notes)
                with patch.object(self.collector, "prepare_offer") as prepare:
                    self.assertEqual(handler.initialize_potentials(), "initialized")
                prepare.assert_not_called()
                delegate.initialize_potentials.assert_called_once_with()
                notes.complete_potential_reward_inventory.assert_not_called()
                self.assertEqual(delegate.screen.mock_calls, [])

    def test_adapter_passes_event_source_and_stops_on_failed_page_restoration(self):
        reference = self.choose_event(click=True)
        delegate = potential_handler()
        notes = Mock(spec=["complete_potential_reward_inventory"])
        notes.complete_potential_reward_inventory.return_value = (False, True)
        handler = ObservedPotentialHandler(delegate, self.collector, notes)
        with self.assertRaises(PotentialObservationIncomplete):
            handler.initialize_potentials()
        notes.complete_potential_reward_inventory.assert_called_once_with(
            delegate.screen, source_event_id=reference["event_id"])
        delegate.initialize_potentials.assert_not_called()

        self.collector.reward["closed"] = True
        delegate = potential_handler("drink")
        notes.complete_potential_reward_inventory.reset_mock()
        notes.complete_potential_reward_inventory.return_value = (True, False)
        self.assertEqual(ObservedPotentialHandler(delegate, self.collector, notes).initialize_potentials(),
                         "initialized")
        notes.complete_potential_reward_inventory.assert_called_once_with(delegate.screen, source_event_id=None)


class TowerNeuralEventNoteBoundaryTests(unittest.TestCase):
    def setUp(self):
        session = RunSession()
        session.begin_run()
        # No collector.event exists here: note collection receives references.
        collector = SimpleNamespace(enabled=True, emit=Mock(), last_observed_floor_context=None)
        runtime = SimpleNamespace(collector=collector, session=session, tracker=None, logger=Mock())
        self.notes = NoteRuntime(runtime)
        self.context = SimpleNamespace()

    def test_before_event_uses_explicit_reference_and_repeated_menu_updates_metadata(self):
        self.notes.ledger = NoteLedger()
        self.notes.ledger.begin_run([], "光土", run_id="run")
        self.notes.before_event(self.context, None, event_id="event", event_choice_record_id="choice-record",
                                note_effect=dict(operation="gain"))
        pending = self.notes.ledger.pending
        self.assertEqual(pending["operation_id"], "event")
        self.assertEqual(pending["metadata"]["event_choice_record_id"], "choice-record")
        self.notes.before_event(self.context, None, event_id="event", event_choice_record_id=None,
                                note_effect=dict(operation="gain"))
        self.assertEqual(self.notes.ledger.pending["operation_id"], "event")
        self.assertIsNone(self.notes.ledger.pending["metadata"]["event_choice_record_id"])

    def test_only_matching_single_event_pending_can_request_inventory(self):
        screen = Mock()
        self.notes.complete_event_inventory = Mock(return_value=(True, False))
        for pending, source in (
                (None, "event"),
                (dict(source_type="event", operation_id="event"), None),
                (dict(source_type="event", operation_id="other-event"), "event"),
                (dict(source_type="combined_interval", operation_id="event"), "event"),
                (dict(source_type="battle", operation_id="event"), "event")):
            with self.subTest(pending=pending, source=source):
                self.notes.ledger = SimpleNamespace(pending=pending)
                self.assertEqual(self.notes.complete_potential_reward_inventory(screen, source_event_id=source),
                                 (True, False))
        self.notes.complete_event_inventory.assert_not_called()
        self.assertEqual(screen.mock_calls, [])
        self.notes.ledger = SimpleNamespace(pending=dict(source_type="event", operation_id="event"))
        self.assertEqual(self.notes.complete_potential_reward_inventory(screen, source_event_id="event"),
                         (True, False))
        self.notes.complete_event_inventory.assert_called_once_with(screen.context, page="stable_event_potential")
        screen.screenshot.assert_not_called()
        screen.context.run_recognition.assert_not_called()

    def test_reward_page_is_relocated_only_after_successful_bag_navigation(self):
        self.notes.ledger = SimpleNamespace(pending=dict(source_type="event", operation_id="event"))
        for restored, navigated, hit, expected in (
                (True, False, False, (True, False)), (False, True, True, (False, True)),
                (True, True, True, (True, True)), (True, True, False, (False, True))):
            with self.subTest(restored=restored, navigated=navigated, hit=hit):
                screen = Mock()
                screen.context.run_recognition.return_value = SimpleNamespace(hit=hit)
                self.notes.complete_event_inventory = Mock(return_value=(restored, navigated))
                self.assertEqual(self.notes.complete_potential_reward_inventory(screen, source_event_id="event"),
                                 expected)
                if restored and navigated:
                    screen.screenshot.assert_called_once_with()
                    screen.context.run_recognition.assert_called_once_with("星塔_节点_进入潜能选择_agent", screen.image)
                else:
                    screen.screenshot.assert_not_called()
                    screen.context.run_recognition.assert_not_called()


if __name__ == "__main__":
    unittest.main()
