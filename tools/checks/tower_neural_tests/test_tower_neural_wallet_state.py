"""Event reward wallets and live floor facts restore the next draw's route."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tower_temp_directory import TowerTempDirectory


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural.observation import ObservationTracker, PotentialObservationIncomplete
from custom.climb_tower_neural.note_runtime import NoteRuntime
from custom.climb_tower_neural.pages import BY_NAME
from custom.climb_tower_neural.potential import PendingPotentialTake, commit_potential_take
from custom.climb_tower_neural import sampling_data
from custom.climb_tower_neural.session import RunSession
from custom.climb_tower_neural.shop_ui.context import Item
from custom.climb_tower_neural.value_model import ValueModel, replace, sim


def tracker_at_quiz(*, collector=None, session=None):
    party = []
    for role in range(1, 4):
        targets = [dict(potential_id=role * 100 + i, name_ja_jp=f"card_{role}_{i}",
                        rarity_class=quality, target_level=level)
                   for i, (quality, level) in enumerate(
                       [("rare", n) for n in (6, 6, 0)]
                       + [("common", n) for n in (6, 6, 6, 0, 0, 0, 0, 0, 0)])]
        party.append(dict(role=role, targets=targets))
    # Construction and all state transitions use the real adapter and simulator;
    # this regression needs no network inference or checkpoint load.
    model = SimpleNamespace(preset=ValueModel.preset, initial=ValueModel.initial, model_id="state-test")
    if session is None:
        session = RunSession()
        session.begin_run()
    tracker = ObservationTracker(ROOT, dict(name="wallet-state", party=party), Mock(), model,
                                 collector, session=session)
    tracker.observe_route("CORE", 480)
    tracker.state = replace(tracker.state, phase="ADVANCE")
    return tracker


def page(coins, quality):
    index = 0 if quality == "rare" else 3
    card = SimpleNamespace(name=f"card_1_{index}", rarity=quality, old_level=0,
                           new_level=1, trekker=SimpleNamespace(index=0))
    return SimpleNamespace(params=SimpleNamespace(potential_source="default"),
        current_coin=coins, potentials=[card], core_potential=False,
        level_upped=False, refresh_count=0)


def select_quiz(tracker):
    with tracker.event_evaluation(None, event_id=116):
        tracker.event_selected(116, 11603)
        tracker.stage_event_result()


def finish_quiz_reward(tracker):
    select_quiz(tracker)
    tracker.event_action_confirmed()
    reward = page(480, "rare")
    tracker.open_page(reward)
    tracker.selected(reward.potentials[0])


class TowerNeuralWalletStateTests(unittest.TestCase):
    def test_event_reward_confirms_wallet_after_transaction_commit(self):
        tracker = tracker_at_quiz()
        before = tracker._event_snapshot()
        select_quiz(tracker)
        self.assertEqual(tracker._event_snapshot(), before)
        tracker.event_action_confirmed()
        self.assertTrue(tracker.coins_uncertain)
        tracker.open_page(page(480, "rare"))
        self.assertFalse(tracker.coins_uncertain)
        self.assertTrue(tracker.event_reward_page)
        self.assertTrue(tracker.route_estimated)
        self.assertEqual({row[0] for row in tracker.route_candidates}, {4, 5})

    def test_next_draw_uses_reward_wallet_with_or_without_floor_hud(self):
        for hud in (False, True):
            with self.subTest(hud=hud):
                tracker = tracker_at_quiz()
                finish_quiz_reward(tracker)
                floor = dict(complete=True, source="floor_hud", floor=3) if hud else None
                if hud:
                    tracker.session.observe_floor("result", floor)
                tracker.open_page(page(560, "common"), observed_floor=floor)
                self.assertEqual((tracker.state.floor, tracker.state.node, tracker.state.coins), (3, 7, 560))
                self.assertEqual(tracker.state.cBel, ((0, 2, 0, sim.ONE),))
                self.assertFalse(tracker.route_estimated)
                self.assertTrue(tracker.decision_route_ready)
                self.assertEqual(len(tracker.route_candidates), 1)

    def test_hidden_floor_after_reward_uses_live_anchor_without_requiring_a_record(self):
        states = []
        for recording in ("enabled", "disabled", "write_failed"):
            with self.subTest(recording=recording), TowerTempDirectory() as directory:
                collector = sampling_data.SamplingCollector(Path(directory), Mock())
                collector.enabled = recording != "disabled"
                collector.begin_task("cn", "state-test")
                collector.start_intent("start")
                collector.confirm_start("entered")
                tracker = tracker_at_quiz(collector=collector)
                finish_quiz_reward(tracker)
                notes = NoteRuntime(SimpleNamespace(session=tracker.session, collector=collector,
                                                    tracker=tracker, logger=Mock()))
                failure = OSError("journal unavailable") if recording == "write_failed" else None
                with patch.object(sampling_data, "append_record", wraps=sampling_data.append_record,
                                  side_effect=failure) as write:
                    notes._read_floor(None, None, "result", observation=dict(
                        complete=True, source="floor_hud", floor=3))
                    hidden = notes._read_floor(None, None, "potential", observation=dict(
                        complete=False, source="floor_hud", floor=None, reason="floor_hud_obscured"))
                if recording == "disabled":
                    write.assert_not_called()
                else:
                    self.assertTrue(write.called)
                if recording == "write_failed":
                    collector.logger.error.assert_called()
                    self.assertIsNone(collector.last_observed_floor_context)
                # An unexplained wallet difference leaves the route estimated;
                # the clear result HUD still anchors the next obscured draw.
                tracker.open_page(page(570, "common"), observed_floor=hidden)
                self.assertIsNone(tracker.session.current_floor_observation["floor"])
                self.assertFalse(tracker.session.current_floor_observation["complete"])
                self.assertEqual((tracker.state.floor, tracker.state.node, tracker.state.coins), (3, 7, 570))
                self.assertTrue(tracker.route_estimated)
                self.assertTrue(tracker.decision_route_ready)
                states.append((tracker.state, tracker.route_candidates))
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[0], states[2])

    def test_new_run_does_not_bound_early_draw_by_previous_runs_final_floor(self):
        session = RunSession()
        session.begin_run()
        session.observe_floor("result", dict(complete=True, source="floor_hud", floor=20))
        session.begin_run()
        tracker = tracker_at_quiz(session=session)
        finish_quiz_reward(tracker)
        tracker.open_page(page(560, "common"))
        self.assertEqual((tracker.state.floor, tracker.state.node, tracker.state.coins), (3, 7, 560))
        self.assertFalse(tracker.route_estimated)
        self.assertTrue(tracker.decision_route_ready)

    def test_drink_reward_commits_stock_once_and_holding_only_after_take(self):
        tracker = tracker_at_quiz()
        tracker.state = replace(tracker.state, phase="SHOP", coins=410,
                                stock=((2, 0, 0), (0, 0, 0)))
        item = Item(grid_num=1, internal_name="potential_drink", price=100)
        purchase = dict(operation_id="buy", source_type="drink", item=item, committed=False)
        tracker.pending_shop_purchase = purchase
        reward = page(310, "rare")
        reward.params.potential_source = "drink"
        tracker.open_page(reward)
        self.assertEqual(tracker.state.stock, ((1, 0, 0), (0, 0, 0)))
        self.assertTrue(item.bought)
        self.assertTrue(purchase["committed"])
        self.assertFalse(tracker.identity_levels)
        self.assertFalse(tracker.acquired)

        # A changed wallet bypasses page-key deduplication and exercises the
        # purchase guard when a later reward refresh is read completely.
        refreshed = page(290, "rare")
        refreshed.params.potential_source = "drink"
        refreshed.refresh_count = 1
        tracker.open_page(refreshed)
        self.assertEqual(tracker.state.coins, 290)
        self.assertEqual(tracker.state.stock, ((1, 0, 0), (0, 0, 0)))
        self.assertEqual(tracker.source_observation["purchase_operation_id"], "buy")
        self.assertFalse(tracker.identity_levels)

        return_page = BY_NAME["shop_main"].node
        runtime = SimpleNamespace(pending_take=PendingPotentialTake(
            potential=refreshed.potentials[0], selected_offset=0, layouts=None,
            tracker=tracker, page_signature=(), evidence=dict(take_confirmation=return_page)))
        self.assertTrue(commit_potential_take(runtime))
        self.assertEqual(list(tracker.identity_levels.values()), [1])
        self.assertEqual(len(tracker.acquired), 1)
        self.assertEqual(tracker.state.stock, ((1, 0, 0), (0, 0, 0)))
        self.assertEqual(purchase["return_page"], return_page)
        self.assertIsNone(runtime.pending_take)

    def test_drink_source_mismatch_or_empty_stock_does_not_change_commit_flags(self):
        for source, committed, quantity, error in (
                ("specified_drink", False, 2, PotentialObservationIncomplete),
                ("specified_drink", True, 2, PotentialObservationIncomplete),
                ("drink", False, 0, ValueError)):
            with self.subTest(source=source, committed=committed, quantity=quantity):
                tracker = tracker_at_quiz()
                stock = ((quantity, 0, 0), (0, 0, 0))
                tracker.state = replace(tracker.state, phase="SHOP", coins=410, stock=stock)
                item = Item(grid_num=1, internal_name="potential_drink", price=100,
                            trekker_specified=source == "specified_drink", bought=committed)
                purchase = dict(operation_id="buy", source_type=source, item=item, committed=committed)
                tracker.pending_shop_purchase = purchase
                reward = page(310, "rare")
                reward.params.potential_source = "drink"
                with self.assertRaises(error):
                    tracker.open_page(reward)
                self.assertEqual(tracker.state.stock, stock)
                self.assertEqual((item.bought, purchase["committed"]), (committed, committed))
                # The visible wallet was read before purchase validation.
                self.assertEqual(tracker.state.coins, 310)
                self.assertFalse(tracker.identity_levels)


if __name__ == "__main__":
    unittest.main()
