"""Compare observed shops with the same seeded simulator route and shelf."""
import sys
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from random import Random
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural.critic.environment import heuristic_action, legal_actions, sim
from custom.climb_tower_neural.critic.rules import MAX_POTENTIAL_LEVEL, MAX_SHOP_RESTOCKS
from custom.climb_tower_neural.observation import ObservationTracker
from custom.climb_tower_neural.potential_adapter import prepare_actions
from custom.climb_tower_neural.session import RunSession


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
    tracker.session = RunSession()
    tracker.coins_uncertain, tracker.route_estimated = False, False
    tracker.pending_event = None
    tracker.route_candidates = ((state.node, state.k, state.A, state.cBel, sim.ONE),)
    return tracker


class TowerNeuralShopStateTests(unittest.TestCase):
    def test_live_limits_match_the_frozen_simulator(self):
        pools = [[MAX_POTENTIAL_LEVEL] * count for count in sim.INITIAL_POOL_LENGTHS]
        self.assertEqual(sim.initial_from_pools(pools).restocks, MAX_SHOP_RESTOCKS)
        pools[0][0] += 1
        with self.assertRaises(sim.InvalidState):
            sim.initial_from_pools(pools)

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
        prepared = prepare_actions(tracker.state, candidates, context)
        self.assertEqual(prepared.state, expected)
        self.assertEqual(prepared.actions, legal_actions(expected))

    def test_first_shop_matches_simulator_after_opening_shelf(self):
        self.assert_observed_shop(10, 1)

    def test_final_shop_matches_simulator_route_and_shelf(self):
        self.assert_observed_shop(16, 4)

    def test_duplicate_drinks_keep_first_available_slot_without_changing_observations(self):
        state = replace(sim.initial_843(), phase="SHOP_UNSEEN", coins=500, restocks=0)
        items = [dict(slot=slot, internal_name=kind, price=100,
                      bought=False, trekker_specified=specified)
                 for slot, kind, specified in (
                     (4, "potential_drink", False), (7, "potential_drink", False),
                     (9, "potential_drink", True), (10, "note", False))]
        candidates = [sim.Action("BUY", index=slot, price=100) for slot in (7, 9, 4, 10)]
        candidates.append(sim.Action("LEAVE"))
        context = dict(kind="shop", stock_read_complete=True, visible_items=items, enhancement_cost=0)
        original = deepcopy(context)

        prepared = prepare_actions(state, candidates, context)

        ordinary = sim.Action("BUY", index=1, price=100, type=1)
        specified = sim.Action("BUY", index=1, price=100, type=2)
        self.assertEqual(prepared.actions, (ordinary, specified, sim.Action("LEAVE")))
        self.assertEqual(prepared.state.stock, ((2, 0, 0), (1, 0, 0)))
        self.assertEqual(prepared.action_slots,
                         {sim.action_key(ordinary): 7, sim.action_key(specified): 9})
        self.assertEqual(context, original)


if __name__ == "__main__":
    unittest.main()
