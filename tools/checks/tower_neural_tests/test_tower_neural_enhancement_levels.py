"""Shop enhancement uses each displayed level for inference and confirmed inventory."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from test_tower_neural_wallet_state import tracker_at_quiz
from custom.climb_tower.climb_tower_potential.context import Parameters
from custom.climb_tower_neural.potential import ModelPotentialHandler, commit_potential_take
from custom.climb_tower_neural.potential_reading import NeuralPotentialContext
from custom.climb_tower_neural.value_model import ValueModel, sim


def enhancement_handler(model, levels):
    tracker = tracker_at_quiz()
    tracker.model = model
    names = ("card_1_0", "card_2_2", "card_3_2")
    held = []
    for role, (name, (old, _)) in enumerate(zip(names, levels), 1):
        item = tracker.identify(name, role=role, rarity="rare")
        tracker.identity_levels[role, item["identity"]] = old
        tracker.acquired[item["identity"]] = item, old
        tracker.identity_pools.remove(2 * (role - 1), item["identity"])
        held.append((role, item["quality"], item["target_level"], old, 1))
    tracker.state = replace(tracker.state, phase="SHOP", H=tuple(held), pools=tracker.identity_pools.projection())

    params = Parameters(potential_source="enhance", max_refresh_count=0, reserved_coin=0,
        priority_list=[], handler="value_model", environment="tower_8", threshold_coef_str="0",
        threshold_decay_str="0", last_consequence="", lang="cn")
    data = NeuralPotentialContext(params, current_coin=10, potential_rarities=["rare"] * len(names))
    screen = Mock(context=SimpleNamespace(tasker=SimpleNamespace(stopping=False)))
    screen.get_potentials_rarity.return_value = data.potential_rarities
    screen.get_current_coin.return_value = data.current_coin
    screen.get_selected_potential_index.return_value = 0
    screen.get_recommended_potential.return_value = []
    screen.get_potential_name.side_effect = names
    screen.get_potential_level.side_effect = levels
    runtime = SimpleNamespace(model=model, require_tracker=lambda: tracker, pending_take=None,
                              decision_context=lambda kind, payload: dict(payload, kind=kind))
    return ModelPotentialHandler(screen, data, runtime), tracker


class TowerNeuralEnhancementLevelTests(unittest.TestCase):
    def test_observed_levels_reach_model_and_confirmed_take(self):
        model = ValueModel()
        cases = (
            (((1, 2), (1, 3), (2, 4)), None),
            (((1, 3), (1, 2), (2, 3)), None),
            (((1, 2), (1, 2), (2, 3)), False),
            (((1, 3), (1, 3), (2, 4)), True),
            (((5, 6), (1, 3), (2, 4)), True),
            (((5, 6), (5, 6), (5, 6)), None),
        )
        for levels, bonus in cases:
            with self.subTest(levels=levels):
                handler, tracker = enhancement_handler(model, levels)
                prior = tracker.state.eBel
                before = tracker.identity_levels.copy()
                held_before = tracker.state.H
                handler.read_potentials_info()
                selected = handler.choose_potential()

                self.assertEqual(tuple(card[4:6] for card in tracker.state.offer), levels)
                self.assertEqual(tracker.state.eBel, sim._enhance_posterior(prior, bonus))
                actions = tuple(sim.Action("SELECT", index=i) for i in (1, 2, 3))
                values = model.values(tracker.state, actions, tracker.preset)
                self.assertEqual([row["predicted_value"] for row in tracker.last_decision["values"]], list(values))
                self.assertEqual(selected.index, int(values.argmax()))

                handler.cache_potential_data(selected)
                self.assertEqual(tracker.identity_levels, before)
                self.assertTrue(commit_potential_take(handler.runtime))
                item = tracker.identify(selected.name, role=selected.trekker.index + 1, rarity=selected.rarity)
                expected = before.copy()
                expected[item["role"], item["identity"]] = levels[selected.index][1]
                self.assertEqual(tracker.identity_levels, expected)
                self.assertEqual(tracker.acquired[item["identity"]][1], levels[selected.index][1])
                self.assertEqual(tracker.state.H, tuple(
                    (role, quality, target, selected.new_level if role == item["role"] else level, count)
                    for role, quality, target, level, count in held_before))
