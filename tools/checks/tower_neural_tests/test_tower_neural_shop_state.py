"""Compare observed shops with the same seeded simulator route and shelf."""
import sys
import unittest
from pathlib import Path
from random import Random
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "agent/custom/action"))
from climb_tower_neural.critic.environment import heuristic_action, legal_actions, sim
from climb_tower_neural.observation import ObservationTracker
from climb_tower_neural.potential_adapter import prepare_actions


def replay_shop(seed, ordinal):
    rng = Random(seed)
    state = sim.initial_843()
    latent = sim.sample_latent(state, rng)
    while state.phase != "END":
        action = sim.FORCED_ACTIONS.get(state.phase) or heuristic_action(state)
        step = sim.step(state, action, rng, latent, collapse_forced=False)
        if step.state.phase == "SHOP_UNSEEN" and step.state.shopOrdinal == ordinal:
            shelf = sim.advance_to_decision(step.state, rng, step.latent).state
            return state, step.state, shelf
        state, latent = step.state, step.latent
    raise AssertionError(f"Did not reach shop {ordinal}")


def tracker_from(state):
    tracker = ObservationTracker.__new__(ObservationTracker)
    tracker.state, tracker.collector, tracker.logger = state, None, Mock()
    tracker.coins_uncertain, tracker.route_estimated = False, False
    tracker.pending_event = None
    tracker.route_candidates = ((state.node, state.k, state.A, state.cBel, sim.ONE),)
    return tracker


class TowerNeuralShopStateTests(unittest.TestCase):
    def assert_observed_shop(self, seed, ordinal):
        previous, unseen, expected = replay_shop(seed, ordinal)
        tracker = tracker_from(previous)
        tracker.observe_route("SHOP_UNSEEN", unseen.coins)
        self.assertEqual(tracker.state, unseen)
        items, candidates = [], []
        for drink_type, row in enumerate(expected.stock, 1):
            for price, count in zip(sim.DRINK_PRICES, row):
                for _ in range(count):
                    slot = len(items) + 1
                    items.append(dict(slot=slot,
                        internal_name="potential_drink", quantity=1, price=price,
                        bought=False, trekker_specified=drink_type == 2))
                    candidates.append(sim.Action("BUY", index=slot, price=price))
        candidates.extend(action for action in legal_actions(expected) if action.kind != "BUY")
        context = dict(kind="shop", stock_read_complete=True, visible_items=items,
            enhancement_cost=0 if expected.freeEnhance else sim.PRICES[expected.paidIndex])
        actions = prepare_actions(tracker.state, candidates, context)
        self.assertEqual(context["model_state"], expected)
        self.assertEqual(actions, legal_actions(expected))

    def test_first_shop_matches_simulator_after_opening_shelf(self):
        self.assert_observed_shop(10, 1)

    def test_final_shop_matches_simulator_route_and_shelf(self):
        self.assert_observed_shop(16, 4)


if __name__ == "__main__":
    unittest.main()
