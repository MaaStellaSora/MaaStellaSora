"""One refresh retry after a fresh unchanged wallet; debit and ready cards confirm it."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from tower_temp_directory import TowerTempDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural import potential
from custom.climb_tower_neural.interactor import Interactor
from custom.climb_tower_neural.agent_callbacks import PotentialRefresh
from custom.climb_tower_neural.critic.environment import sim
from custom.climb_tower_neural import research_data, sampling_data
from custom.climb_tower_neural.tower_sampling_collector import ObservedPotentialHandler


class TowerNeuralPotentialRefreshTests(unittest.TestCase):
    def setUp(self):
        self.states = {}
        self.in_observer = False
        self.refresh_frames = ["first", "second"]
        self.context = SimpleNamespace(tasker=SimpleNamespace(stopping=False))
        self.data = SimpleNamespace(current_coin=1230, refresh_cost=40, refresh_count=2,
            core_potential=False, potential_count=1, potential_rarities=["rare"])
        self.screen = Mock(context=self.context, image="before")
        self.handler = potential.ModelPotentialHandler(self.screen, self.data, SimpleNamespace())
        self.screen.refresh.side_effect = self.refresh
        self.screen.get_current_coin.side_effect = lambda: self.state().get("coin", 1230)
        self.screen.get_potentials_rarity.side_effect = lambda core: self.state().get("rarities", ["rare"])
        self.screen.screenshot.side_effect = self.capture

    def state(self):
        return self.states.get(self.screen.image, {})

    def refresh(self):
        self.assertFalse(self.in_observer, "refresh input must be outside recognition callbacks")
        self.screen.image = self.refresh_frames.pop(0)
        return True

    def capture(self):
        self.assertFalse(self.in_observer)
        self.screen.image = "fresh"
        if self.state().get("stop"):
            self.context.tasker.stopping = True
        return self.screen.image

    def run_frames(self, *batches):
        remaining = list(batches)
        def observe_frames(context, observe, **kwargs):
            for frame in remaining.pop(0):
                self.in_observer = True
                try:
                    result = observe(frame)
                finally:
                    self.in_observer = False
                if result is not None:
                    return result
            return None
        with patch.object(Interactor, "wait_for", side_effect=observe_frames):
            return self.handler.refresh()

    def assert_uncommitted(self):
        self.assertEqual((self.data.current_coin, self.data.refresh_count), (1230, 2))

    def test_unchanged_fresh_wallet_retries_same_refresh_and_reads_its_updated_image(self):
        self.states["second"] = dict(coin=1190)
        self.assertTrue(self.run_frames(["old_poll"]))
        self.screen.screenshot.assert_called_once_with()
        self.assertEqual(self.screen.refresh.call_count, 2)
        self.assertEqual((self.data.current_coin, self.data.refresh_count), (1190, 3))

    def test_normal_success_uses_first_refresh_image_without_extra_capture(self):
        self.states["first"] = dict(coin=1190)
        self.assertTrue(self.run_frames())
        self.screen.get_current_coin.assert_called_once_with()
        self.screen.refresh.assert_called_once_with()
        self.screen.screenshot.assert_not_called()
        self.assertEqual(self.data.refresh_count, 3)

    def test_observed_debit_waits_for_cards_without_retry(self):
        self.states["first"] = dict(coin=1190, rarities=[])
        self.states["ready"] = dict(coin=1190)
        self.assertTrue(self.run_frames(["loading", "ready"]))
        self.screen.refresh.assert_called_once_with()
        self.screen.screenshot.assert_not_called()
        self.assertEqual(self.data.refresh_count, 3)

    def test_debit_without_ready_cards_keeps_wallet_without_committing_page_or_retrying(self):
        self.states["first"] = dict(coin=1190, rarities=[])
        self.assertFalse(self.run_frames(["old_poll"]))
        self.screen.refresh.assert_called_once_with()
        self.screen.screenshot.assert_not_called()
        self.assertEqual((self.data.current_coin, self.data.refresh_count), (1190, 2))

    def test_fresh_late_debit_waits_for_cards_without_retry(self):
        self.states["fresh"] = dict(coin=1190, rarities=[])
        self.states["ready"] = dict(coin=1190)
        self.assertTrue(self.run_frames(["old_poll"], ["ready"]))
        self.screen.screenshot.assert_called_once_with()
        self.screen.refresh.assert_called_once_with()
        self.assertEqual(self.data.refresh_count, 3)

    def test_failed_page_confirmation_keeps_paid_wallet_in_model_before_stopping(self):
        tracker = SimpleNamespace(state=sim.initial_843())
        runtime = SimpleNamespace(pending_potential_refresh=(self.handler, tracker, False),
            collector=SimpleNamespace(context=lambda context: context), stop_model=Mock())
        self.states["first"] = dict(coin=1190, rarities=[])
        with patch.object(Interactor, "wait_for", return_value=None):
            self.assertFalse(PotentialRefresh(runtime).run(self.context, None))
        self.assertEqual(tracker.state.coins, 1190)
        self.assertEqual(self.data.refresh_count, 2)
        runtime.stop_model.assert_called_once()

    def test_paid_refresh_with_missing_cards_preserves_failed_result_through_sqlite_export(self):
        self.states["first"] = dict(coin=1190, rarities=[])
        self.data.params = SimpleNamespace(potential_source="battle")
        with TowerTempDirectory() as directory:
            collector = sampling_data.SamplingCollector(directory, Mock())
            collector.start_intent("start")
            collector.confirm_start("entered")
            handler = ObservedPotentialHandler(self.handler, collector)
            with patch.object(Interactor, "wait_for", return_value=None):
                self.assertFalse(handler.refresh())
            collector.emit("exploration_end", completion_observed=False)
            run_id = collector.run["run_id"]
            path = collector.directory / "runs" / f"{run_id}.jsonl"
            journal = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            recorded = next(row for row in journal if row["event_type"] == "refresh_result")
            database = Path(directory) / "research.sqlite3"
            research_data.import_payload(database, research_data.build_payload(path))
            payload = research_data.export_payload(database, run_id)
            exported = next(json.loads(row["payload_json"]) for row in payload["tables"]["records"]
                            if row["event_type"] == "refresh_result")
            for row in (recorded, exported):
                self.assertEqual((row["coin_before"], row["coin_after"]), (1230, 1190))
                self.assertIs(row["result"], False)

    def test_successful_callback_syncs_wallet_once_and_reentry_does_not_refresh_again(self):
        tracker = SimpleNamespace(state=sim.initial_843())
        runtime = SimpleNamespace(pending_potential_refresh=(self.handler, tracker, False),
            collector=SimpleNamespace(context=lambda context: context), stop_model=Mock())
        self.states["first"] = dict(coin=1190)
        callback = PotentialRefresh(runtime)
        self.assertTrue(callback.run(self.context, None))
        self.assertTrue(callback.run(self.context, None))
        self.assertEqual((tracker.state.coins, self.data.refresh_count), (1190, 3))
        self.screen.refresh.assert_called_once_with()
        runtime.stop_model.assert_not_called()

    def test_unreadable_or_changed_fresh_wallet_never_retries(self):
        for coin in (-1, 1180):
            with self.subTest(coin=coin):
                self.setUp()
                self.states["fresh"] = dict(coin=coin)
                self.assertFalse(self.run_frames(["old_poll"], []))
                self.screen.screenshot.assert_called_once_with()
                self.screen.refresh.assert_called_once_with()
                self.assert_uncommitted()

    def test_second_ignored_refresh_is_not_repeated(self):
        self.assertFalse(self.run_frames(["old_poll"], ["still_old"]))
        self.assertEqual(self.screen.refresh.call_count, 2)
        self.screen.screenshot.assert_called_once_with()
        self.assert_uncommitted()

    def test_failed_retry_does_not_wait_or_commit(self):
        self.screen.refresh.side_effect = [True, False]
        self.assertFalse(self.run_frames(["old_poll"]))
        self.assertEqual(self.screen.refresh.call_count, 2)
        self.assert_uncommitted()

    def test_stopping_or_initial_framework_failure_does_not_retry(self):
        self.context.tasker.stopping = True
        self.assertFalse(self.handler.refresh())
        self.screen.refresh.assert_not_called()
        self.context.tasker.stopping = False
        self.screen.refresh.side_effect = None
        self.screen.refresh.return_value = False
        self.assertFalse(self.handler.refresh())
        self.screen.screenshot.assert_not_called()
        self.assert_uncommitted()


if __name__ == "__main__":
    unittest.main()
