"""Game language aliases preserve catalogue identities and task eligibility."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tower_temp_directory import TowerTempDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural import events, language, research_data
from custom.climb_tower_neural.agent_callbacks import Preparation
from custom.climb_tower_neural.language import catalog_name_field
from custom.climb_tower_neural.maa_hook import ValueModelRuntime
from custom.climb_tower_neural.observation import ObservationTracker
from custom.climb_tower_neural.paths import data_path
from custom.climb_tower_neural.pool_identity import pool_contract
from custom.climb_tower_neural.preset_tool import compile_bag_preset
from custom.climb_tower_neural.sampling_data import SamplingCollector
from custom.climb_tower_neural.value_model import ValueModel
from custom.climb_tower.climb_tower_preparation import AscensionPreparation
from tools.catalogs import tower_catalog


def known_party(catalog, characters, name_field):
    party = []
    for role, character in enumerate((162, 115, 107), 1):
        role_type = "master" if role == 1 else "assist"
        rows = [row for row in catalog["potentials"]
                if row["character_id"] == character and role_type in row["roles"]]
        party.append(dict(role=role, name=characters[character][name_field], obtained_count=0,
            cards=[dict(name=row[name_field], name_confirmed=True, level=0, target_level=0,
                rarity_class="core" if row["item_stype"] == 42 else
                             "rare" if row["item_rarity"] == 1 else "common") for row in rows]))
    return party


def preparation(language):
    context = SimpleNamespace(get_node_data=lambda name: dict(attach=(
        dict(lang_type=language) if name == "星塔_节点_进行对话选择_agent" else dict(native_windows=True))))
    runtime = SimpleNamespace(requested=Mock(return_value=True), settings={},
        language=ValueModelRuntime.language, logger=Mock(), begin_task=Mock(),
        ui_logs=SimpleNamespace(scope=nullcontext))
    return Preparation(runtime), runtime, context


class TowerNeuralLanguageTests(unittest.TestCase):
    def configured_catalog(self):
        catalog = json.loads(data_path("potential_name_catalog.json").read_text(encoding="utf-8"))
        characters = json.loads(data_path("character_name_catalog.json").read_text(encoding="utf-8"))["characters"]
        for row in catalog["potentials"]:
            row["name_ko_kr"] = f"시험잠재{row['potential_id']}"
        for row in characters:
            row["name_ko_kr"] = f"시험캐릭터{row['character_id']}"
        catalog["characters"] = characters
        return catalog

    def test_configured_name_survives_known_targets_and_both_runtime_indexes(self):
        catalog = self.configured_catalog()
        characters = {row["character_id"]: row for row in catalog["characters"]}
        party = known_party(catalog, characters, "name_ko_kr")
        card = next(card for card in party[0]["cards"] if card["rarity_class"] == "rare")
        card.update(level=1, target_level=6)
        party[0]["obtained_count"] = 1
        with patch.dict(language._LOCALES, {"kr": dict(aliases=[], catalog_name_field="name_ko_kr")}):
            compiled = compile_bag_preset(party, "kr", catalog)
            targets = [target for member in compiled["party"] for target in member["targets"]]
            self.assertEqual(len(targets), 36)
            self.assertTrue(all(target.get("name_ko_kr") for target in targets))
            self.assertEqual(len(compiled["recognition_catalog"]), 12)
            self.assertTrue(all(target.get("name_ko_kr") for target in compiled["recognition_catalog"]))
            self.assertTrue(all(slot.get("name_ko_kr") for member in compiled["party"] for slot in member["bag_slots"]))
            self.assertEqual(compiled["initial_inventory"][0]["name_ko_kr"], card["name"])
            model = SimpleNamespace(preset=ValueModel.preset, initial=ValueModel.initial, model_id="language-test")
            tracker = ObservationTracker(ROOT, compiled, Mock(), model)
            with TowerTempDirectory() as directory:
                collector = SamplingCollector(directory, Mock())
                collector.run_confirmed, collector.language = True, "kr"
                collector.emit = Mock()
                collector.bind_preset(compiled, tracker)
                for target in targets:
                    name = target["name_ko_kr"]
                    identified = tracker.identify(name, role=target["role"], rarity=target["rarity_class"])
                    self.assertEqual(identified["potential_id"], target["potential_id"])
                    self.assertEqual(collector.local_names[name][0]["potential_id"], target["potential_id"])

    def test_catalog_roundtrip_uses_configured_names_and_retains_existing_aliases(self):
        catalog = self.configured_catalog()
        catalog["potentials"][0]["name_en_us"] = "Legacy catalogue alias"
        with patch.dict(tower_catalog.language._LOCALES, {"kr": dict(aliases=[], catalog_name_field="name_ko_kr")}):
            expanded = tower_catalog.expand(catalog)
            for row in catalog["potentials"]:
                aliases = expanded["by_normalized_name"][tower_catalog.name_key(row["name_ko_kr"])]
                self.assertIn(row["potential_id"], [item["potential_id"] for item in aliases])
            self.assertIn("Legacycataloguealias", expanded["by_normalized_name"])
            self.assertEqual(tower_catalog.compact(expanded), catalog)
            self.assertEqual(language.catalog_name_fields()[:4],
                ("name_ja_jp", "name_zh_cn", "name_en_us", "name_zh_tw"))
        self.assertNotIn("catalog_name_alias_fields", language.SUPPORTED_LANGUAGES)
        self.assertNotIn("kr", language.SUPPORTED_LANGUAGES)

    def test_compact_preset_retains_configured_local_names_and_compacts_known_ids(self):
        unknown = dict(role=1, character_id=None, name_ko_kr="신규캐릭터", local_character_identity="local-role",
            targets=[dict(potential_id=None, local_identity="local-target", name_ko_kr="신규잠재", target_level=3)])
        known = deepcopy(unknown)
        known.update(character_id=162)
        known["targets"][0]["potential_id"] = 101
        with patch.dict(language._LOCALES, {"kr": dict(aliases=[], catalog_name_field="name_ko_kr")}):
            result = research_data.compact_preset(dict(party=[unknown, known]))
        local, canonical = result["party"]
        self.assertEqual(local["name_ko_kr"], unknown["name_ko_kr"])
        self.assertEqual(local["targets"][0]["name_ko_kr"], unknown["targets"][0]["name_ko_kr"])
        self.assertEqual(local["targets"][0]["target_level"], 3)
        self.assertNotIn("name_ko_kr", canonical)
        self.assertNotIn("name_ko_kr", canonical["targets"][0])
        self.assertEqual(canonical["targets"][0]["target_level"], 3)

    def test_language_aliases_compile_the_same_known_party_and_confirmed_pools(self):
        catalog = json.loads(data_path("potential_name_catalog.json").read_text(encoding="utf-8"))
        characters = {row["character_id"]: row for row in
                      json.loads(data_path("character_name_catalog.json").read_text(encoding="utf-8"))["characters"]}
        for canonical, alias, field in (("cn", "zh_cn", "name_zh_cn"), ("jp", "ja_jp", "name_ja_jp")):
            with self.subTest(language=alias):
                party = known_party(catalog, characters, field)
                expected = compile_bag_preset(party, canonical, catalog)
                actual = compile_bag_preset(party, alias, catalog)
                self.assertEqual(actual, expected)
                self.assertEqual([member["character_id"] for member in actual["party"]], [162, 115, 107])
                self.assertTrue(all(row["status"] == "confirmed" for row in actual["pool_order_evidence"]))
                self.assertEqual(pool_contract(actual["party"], alias), pool_contract(actual["party"], canonical))

    def test_existing_languages_and_aliases_reach_normal_preparation(self):
        for language in ("cn", "zh_cn", "jp", "ja_jp"):
            with self.subTest(language=language):
                callback, runtime, context = preparation(language)
                with patch.object(AscensionPreparation, "run", return_value=True) as original:
                    self.assertTrue(callback.run(context, SimpleNamespace(node_name="preparation")))
                original.assert_called_once()
                runtime.begin_task.assert_called_once_with(context)
                runtime.logger.error.assert_not_called()

    def test_unknown_language_is_rejected_despite_legacy_catalogue_fallback(self):
        for language in ("unsupported", None):
            with self.subTest(language=language):
                self.assertEqual(catalog_name_field(language), "name_ja_jp")
                callback, runtime, context = preparation(language)
                with patch.object(AscensionPreparation, "run", return_value=True) as original:
                    self.assertFalse(callback.run(context, SimpleNamespace(node_name="preparation")))
                original.assert_not_called()
                runtime.begin_task.assert_not_called()
                runtime.logger.error.assert_called_once_with(
                    "$agent.custom.action.climb_tower_neural.preset_language_unsupported")

    def test_event_menu_normalizes_its_own_nodes_game_language(self):
        context = SimpleNamespace(tasker=SimpleNamespace(stopping=False), get_node_data=Mock(
            return_value=dict(attach=dict(lang_type="ja_jp"))))
        choices = [SimpleNamespace(text="leave", consequence="no reward", observations={})]
        recognizer = SimpleNamespace(_get_question_text=Mock(return_value="question"),
                                     _get_choice_texts=Mock(return_value=choices))
        runtime = SimpleNamespace(collector=SimpleNamespace(capture=lambda *args: nullcontext()),
            event_policy=SimpleNamespace(menu=Mock(return_value=SimpleNamespace(event_id=101))),
            observe_wallet=Mock(),
            language=Mock(return_value="cn"))
        image = object()
        with patch.object(events, "_read_event_coins", return_value=500):
            result = events._read_event_menu(recognizer, context,
                SimpleNamespace(node_name="current_event_node", image=image), runtime)
        self.assertEqual(result.required_missing, [])
        context.get_node_data.assert_called_once_with("current_event_node")
        recognizer._get_question_text.assert_called_once_with(context, image, "jp")
        recognizer._get_choice_texts.assert_called_once_with(context, image, "jp")
        runtime.event_policy.menu.assert_called_once_with("question", choices, "jp")
        runtime.language.assert_not_called()


if __name__ == "__main__":
    unittest.main()
