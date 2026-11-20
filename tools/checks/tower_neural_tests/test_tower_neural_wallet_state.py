"""A readable event reward wallet restores exact route observations."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "agent/custom/action"))
from climb_tower_neural.observation import ObservationTracker
from climb_tower_neural.value_model import ValueModel, replace, sim


def tracker_at_quiz():
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
    tracker = ObservationTracker(ROOT, dict(name="wallet-state", party=party), Mock(), model)
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
                select_quiz(tracker)
                tracker.event_action_confirmed()
                reward = page(480, "rare")
                tracker.open_page(reward)
                tracker.selected(reward.potentials[0])
                floor = dict(complete=True, source="floor_hud", floor=3) if hud else None
                if hud:
                    tracker.collector = SimpleNamespace(enabled=False, last_observed_floor_context=floor)
                tracker.open_page(page(560, "common"), observed_floor=floor)
                self.assertEqual((tracker.state.floor, tracker.state.node, tracker.state.coins), (3, 7, 560))
                self.assertEqual(tracker.state.cBel, ((0, 2, 0, sim.ONE),))
                self.assertFalse(tracker.route_estimated)
                self.assertTrue(tracker.decision_route_ready)
                self.assertEqual(len(tracker.route_candidates), 1)


if __name__ == "__main__":
    unittest.main()
