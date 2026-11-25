"""Evidence that corrects provisional bag identities."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "agent"))
from custom.climb_tower_neural import bag_preset_reader as bag, potential
from custom.climb_tower_neural.provisional_cards import ProvisionalCardMatcher


def slot(index, name, target=1):
    return dict(identity=f"local:{index}", local_identity=f"local:{index}", role=1,
                rarity_class="common", target_level=target, bag_display_index=index,
                name=name, provisional=True)


def tracker(items, known=None):
    known = known or {}
    return SimpleNamespace(provisional_items=items, provisional_matcher=ProvisionalCardMatcher(),
        local_names=known, name_candidates=lambda name, **kwargs: known.get(name, []).copy(),
        identity_levels={}, trekkers={}, identity_pools=SimpleNamespace(
            slots=[[], [item["identity"] for item in items]],
            inferred=[[], [True] * len(items)]))


def card(name, target, old=0):
    return SimpleNamespace(name=name, rarity="common", core=False, old_level=old,
                           recommended=True, recommended_level=target, selected=False,
                           layout=SimpleNamespace(trekker_roi=[0, 0, 40, 40]))


class ProvisionalIdentityTests(unittest.TestCase):
    def test_final_crop_vote_rechecks_same_frame_and_confirms_saved_titles(self):
        reader = bag._Reader(SimpleNamespace(tasker=SimpleNamespace(stopping=False)), "cn", None)
        reader.catalog = dict(characters=[], potentials=[dict(potential_id=i + 1,
            character_id=100, item_stype=42 if i == 0 else 40, item_rarity=1,
            roles=["assist"], name_zh_cn=name) for i, name in enumerate(("Core", "Rare"))])
        cards = [bag._Card(kind, i) for i, kind in enumerate(("core", "rare", "common"))]
        cards[0].name_evidence = dict(text="Core")
        member = dict(name="Wrong header", cards=cards)
        reader.members = [dict(name="Other", cards=[]), dict(name="Other2", cards=[]), member]
        reader.frame = 7
        for item, name in zip(cards[1:], ("Rare", "Saved title")):
            item.crop = np.full((39, 112, 3), 240, dtype=np.uint8)
            item.crop_box, item.crop_frame = [361, 245, 112, 39], 7
            if item is cards[2]:
                reader._add_name(item, [dict(text=name, score=.99, box=[365, 248, 80, 20])],
                                 item.crop_box, "page", 6)
        self.assertFalse(reader._known_member(member))
        # Crop confirmation supplies the second independent role vote, without a new frame.
        with patch.object(bag, "read_text", return_value=[dict(text="Rare", score=.99, box=[4, 3, 80, 20])]):
            reader._confirm_name(cards[1])
        self.assertTrue(reader._known_member(member))
        self.assertFalse(cards[2].name()[1])
        reader.finish_names()
        self.assertTrue(cards[2].name()[1])
        self.assertEqual(cards[2].name_evidence["original_regions"][0]["frame"], 6)

    def test_fresh_targets_correct_guessed_pool_positions_but_rereads_stay_inferred(self):
        items = [slot(0, "Same", 1), slot(1, "Same", 2)]
        state = tracker(items)
        state.identity_pools.slots[1].reverse()
        guesses = [card("Unreadable", -1), card("Unreadable", -1)]
        first = potential._resolve_potential_items(Mock(), state, guesses, 0, single_role=True)
        again = potential._resolve_potential_items(Mock(), state, guesses, 0, single_role=True)
        self.assertEqual([item["identity"] for item in first], ["local:1", "local:0"])
        self.assertEqual([item["identity"] for item in first], [item["identity"] for item in again])
        self.assertTrue(all(item["identity_runtime_status"] == "inferred" for item in again))
        corrected = potential._resolve_potential_items(
            Mock(), state, [card("Unreadable", 1), card("Unreadable", 2)], 0, single_role=True)
        self.assertEqual([item["identity"] for item in corrected], ["local:0", "local:1"])
        self.assertTrue(all(item["identity_runtime_status"] == "confirmed" for item in corrected))
        self.assertEqual(state.provisional_matcher.aliases, {})

    def test_corrected_alias_does_not_reserve_stale_teammate_name_in_mixed_enhancement(self):
        local, spare = slot(0, "Teammate"), slot(1, "Spare")
        teammate = dict(identity=200, role=2, rarity_class="common", target_level=1)
        third = dict(identity=300, role=3, rarity_class="common", target_level=1)
        for extra in ([], [spare]):
            with self.subTest(spare_same_level_slot=bool(extra)):
                state = tracker([local, *extra], {"Teammate": [teammate], "Third": [third]})
                state.provisional_matcher.remember("Correct local", local)
                state.identity_levels = {(1, "local:0"): 1, (1, "local:1"): 1,
                                         (2, 200): 1, (3, 300): 1}
                readings = [card(name, 1, 1) for name in ("Teammate", "Correct local", "Third")]
                resolved = potential._resolve_potential_items(Mock(), state, readings, 0)
                self.assertEqual([(item["role"], item["identity"]) for item in resolved],
                                 [(2, 200), (1, "local:0"), (3, 300)])
                self.assertEqual(state.provisional_matcher.name_candidates(
                    state.provisional_items, "Teammate", "common"), [])

    def test_known_portrait_prefix_cannot_become_a_local_alias(self):
        local = slot(0, "Local", 0)
        known = dict(identity=200, role=2, rarity_class="common", target_level=0)
        state = tracker([local], {"WeaknessMark": [known]})
        state.identity_levels = {(1, "local:0"): 1, (2, 200): 1}
        state.trekkers = {2: SimpleNamespace(image=np.ones((2, 2)))}
        screen = Mock()
        screen.match_trekker.return_value = True
        resolved = potential._resolve_potential_items(screen, state, [card("Weakness", 0, 1)], 0)
        self.assertEqual((resolved[0]["role"], resolved[0]["identity"]), (2, 200))
        self.assertEqual(state.provisional_matcher.aliases, {})

    def test_clear_upgrade_transfers_an_inferred_holding_and_restores_its_value(self):
        from test_tower_neural_wallet_state import tracker_at_quiz
        from custom.climb_tower_neural.observation import ObservationTracker
        from custom.climb_tower_neural.value_model import replace, sim

        base = tracker_at_quiz()
        member = base.preset_data["party"][0]
        member["local_potential_catalogue"] = True
        for index, item in enumerate(member["targets"]):
            item.update(local_identity=f"local:{index}", provisional=True, bag_display_index=index,
                        name_ja_jp="Unreadable", name_confirmed=False)
            item.pop("potential_id")
        member["targets"][3]["target_level"] = 2
        member["targets"][5]["target_level"] = 0
        state = ObservationTracker(base.root, dict(name="holding-correction", party=base.preset_data["party"]),
                                   Mock(), base.model)
        state.observe_route("CORE", 480)
        state.state = replace(state.state, phase="ADVANCE")

        def open_cards(readings, source):
            for index, reading in enumerate(readings):
                reading.index, reading.new_level = index, 2 if reading.old_level == 0 else 3
            resolved = potential._resolve_potential_items(Mock(), state, readings, 0, single_role=True)
            state.open_page(SimpleNamespace(params=SimpleNamespace(potential_source=source),
                current_coin=480, potentials=readings, resolved_items=resolved,
                core_potential=False, level_upped=False, refresh_count=0))
            return readings[0]

        first = open_cards([card("Unreadable", -1), card("Unreadable", -1), card("Unreadable", 0)], "default")
        state.selected(first)
        state.close_potential_page()
        state.state = replace(state.state, phase="SHOP")
        upgrade = open_cards([card("Unreadable", 6, 2)], "enhance")
        self.assertEqual(state.identity_levels, {(1, "local:4"): 2})
        self.assertEqual(sim.score(sim._select(state.state, 1)) - sim.score(state.state), 1)
        state.selected(upgrade)
        self.assertEqual(state.state.H, ((1, 2, 6, 3, 1),))
        self.assertIn("local:3", state.identity_pools.slots[1])
