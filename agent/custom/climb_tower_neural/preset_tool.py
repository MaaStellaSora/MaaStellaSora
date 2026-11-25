"""Compile ID-based deployment targets using this package's local catalog."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

from .critic.rules import MAX_POTENTIAL_LEVEL
from .name_matching import (
    BAG_HEADER_MIN_SCORE, clean_name, identity_key, is_anonymous_non_target, match_name,
)
from .pool_identity import pool_contract
from .paths import data_path
from .language import catalog_name_field, catalog_name_fields, normalize_language


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def catalog_rows(catalog):
    if "potentials" in catalog:
        return catalog["potentials"]
    rows = {}
    for aliases in catalog["by_normalized_name"].values():
        for row in aliases:
            key = (row["potential_id"], tuple(row["roles"]))
            rows.setdefault(key, {}).update({k: v for k, v in row.items() if v is not None})
    return list(rows.values())


def bag_member_character_id(member, language, catalog):
    """Recover a weak header from two distinct core/rare catalogue matches.

    Raw text votes only on a unique exact name; confirmed text keeps fuzzy matching.
    A character vote never confirms a known card.
    The reader uses the same vote before deciding which names need strict OCR.
    """
    name_field = catalog_name_field(language)
    characters = catalog.get("characters")
    if characters is None:
        characters = read(data_path("character_name_catalog.json"))["characters"]
    character_id = next((row["character_id"] for row in characters
        if row.get(name_field) and clean_name(row[name_field]) == clean_name(member.get("name"))), None)
    if character_id is not None:
        return character_id
    role_type = "master" if member["role"] == 1 else "assist"
    aliases = defaultdict(list)
    for row in catalog_rows(catalog):
        kind = "core" if row["item_stype"] == 42 else "rare" if row["item_rarity"] == 1 else "common"
        if role_type in row["roles"] and kind in {"core", "rare"} and row.get(name_field):
            aliases[clean_name(row[name_field])].append(dict(row, role=member["role"], rarity_class=kind))
    votes = defaultdict(set)
    for card in member["cards"]:
        if card.get("rarity_class") not in {"core", "rare", ""}:
            continue
        if card.get("name_confirmed"):
            matches = match_name(card.get("name"), aliases, role=member["role"],
                                 rarity=card.get("rarity_class") or None, party_bound=True).items
        else:
            matches = [row for row in aliases.get(clean_name(card.get("name")), ())
                       if not card.get("rarity_class") or row["rarity_class"] == card["rarity_class"]]
        if len(matches) == 1:
            row = matches[0]
            votes[row["character_id"]].add(row["potential_id"])
    ranked = sorted(votes, key=lambda key: len(votes[key]), reverse=True)
    if ranked and len(votes[ranked[0]]) >= 2 and (len(ranked) == 1 or len(votes[ranked[0]]) > len(votes[ranked[1]])):
        return ranked[0]
    return None


def compile_bag_preset(party, language, catalog):
    """Compile the three visible 16-card bags; unknown IDs stay explicitly null.

    Known target names retain independent confirmation. Uncatalogued members
    use display-slot identities; their OCR names are observations, not identities.
    """
    language = normalize_language(language)
    name_field = catalog_name_field(language)
    catalog_fields = (*catalog_name_fields(), "catalog_id_namespace",
                      "region_validation_status", "cn_server_id_equivalence_verified")
    rows = catalog_rows(catalog)

    def rarity(row):
        return "core" if row["item_stype"] == 42 else "rare" if row["item_rarity"] == 1 else "common"

    def resolve(name, available, kind):
        aliases = defaultdict(list)
        for row in available:
            if rarity(row) == kind and row.get(name_field):
                aliases[clean_name(row[name_field])].append(dict(row, role=1, rarity_class=kind))
        result = match_name(name, aliases, rarity=kind, party_bound=True)
        return result.items[0] if len(result.items) == 1 else None

    def raw_names(value):
        if isinstance(value, list):
            return [row for child in value for row in raw_names(child)]
        if not isinstance(value, dict):
            return []
        return ([value] if "text" in value and "score" in value else
                [row for child in value.values() for row in raw_names(child)])

    members, recognition, inventory, original = [], [], [], []
    if sorted(member["role"] for member in party) != [1, 2, 3]:
        raise ValueError("背包需要按主控、辅助、辅助确认三个角色。")
    for member in sorted(party, key=lambda row: row["role"]):
        role, cards = member["role"], member["cards"]
        role_type = "master" if role == 1 else "assist"
        if Counter(card["rarity_class"] for card in cards) != {"core": 4, "rare": 3, "common": 9}:
            raise ValueError(f"角色{role}背包未确认完整4/3/9潜能布局。")
        for card in cards:
            cap = 1 if card["rarity_class"] == "core" else MAX_POTENTIAL_LEVEL
            if any(type(card[key]) is not int or not 0 <= card[key] <= cap for key in ("level", "target_level")):
                raise ValueError(f"角色{role}潜能等级未完整读取。")
        if member["obtained_count"] != sum(card["level"] > 0 for card in cards):
            raise ValueError(f"角色{role}已获得数量与可见等级行不一致。")
        available = [row for row in rows if role_type in row["roles"]]
        character_id = bag_member_character_id(member, language, catalog)
        known = [row for row in available if row["character_id"] == character_id]
        catalogue_complete = Counter(rarity(row) for row in known) == {"core": 4, "rare": 3, "common": 9}
        if not catalogue_complete:
            headers = raw_names(member.get("raw", {}).get("header_readings", member.get("raw", {}).get("name_readings", [])))
            header_confirmed = any(clean_name(row["text"]) == clean_name(member["name"])
                                   and row["score"] >= BAG_HEADER_MIN_SCORE for row in headers)
            if not header_confirmed:
                raise ValueError(f"Role {role} uncatalogued character header lacks trusted matching evidence")
        for card in cards:
            if catalogue_complete and card["target_level"] and not (card.get("name_confirmed") and clean_name(card.get("name"))):
                raise ValueError(f"角色{role}目标潜能名称缺少独立确认。")
            if not catalogue_complete and not clean_name(card.get("name")):
                raise ValueError(f"角色{role}库外潜能缺少可见名称文本。")
        targets, named, card_items = [], {}, []
        for index, card in enumerate(cards):
            kind, name = card["rarity_class"], card.get("name", "")
            match = resolve(name, known, kind) if catalogue_complete and card.get("name_confirmed") else None
            if catalogue_complete and card["target_level"] and match is None:
                raise ValueError(f"角色{role}目标潜能不属于已确认角色或职能：{name}")
            item = dict(potential_id=match["potential_id"] if match else None,
                local_identity=None, character_id=character_id, role=role,
                rarity_class=kind, target_level=card["target_level"], **{name_field: name or None})
            if match:
                item.update({key: match[key] for key in catalog_fields if key in match})
            elif not catalogue_complete:
                display_index = card.get("raw", {}).get("bag_display_index", index)
                item.update(local_identity=f"bag-slot:{role}:{kind}:{display_index}",
                    local_identity_namespace="bag_role_rarity_display_index",
                    identity_basis="bag_role_rarity_display_index", provisional=True,
                    bag_display_index=display_index, bag_name=name, name_confirmed=bool(card.get("name_confirmed")))
            key = identity_key(item)
            if key is not None:
                if key in named:
                    raise ValueError(f"角色{role}背包重复潜能行：{name}")
                named[key] = item
            card_items.append(item)
        if catalogue_complete:
            for row in sorted(known, key=lambda row: row["potential_id"]):
                observed = named.get(row["potential_id"])
                item = dict(potential_id=row["potential_id"], character_id=character_id, role=role,
                    rarity_class=rarity(row), target_level=observed["target_level"] if observed else 0,
                    **{key: row[key] for key in catalog_fields if key in row})
                (targets if item["rarity_class"] != "core" or item["target_level"] else recognition).append(item)
        else:
            for item in card_items:
                (targets if item["rarity_class"] != "core" or item["target_level"] else recognition).append(item)
        for card, item in zip(cards, card_items):
            if card["level"]:
                inventory.append(dict(item, name=card.get("name") or None, level=card["level"],
                    anonymous=identity_key(item) is None or is_anonymous_non_target(item), raw=deepcopy(card.get("raw")),
                    level_basis="bag_level_OCR", identity_confirmed=identity_key(item) is not None and not item.get("provisional", False)))
        members.append(dict(role=role, character_id=character_id, **{name_field: member["name"]},
            character_id_namespace="JP_artifact" if character_id is not None else None,
            local_character_identity=f"bag:{role}:{clean_name(member['name'])}",
            local_potential_catalogue=not catalogue_complete, targets=targets,
            catalogue_order=member.get("catalogue_order", {}),
            bag_slots=[dict(item, bag_display_index=card.get("raw", {}).get("bag_display_index", index)
                if isinstance(card.get("raw"), dict) else index)
                for index, (card, item) in enumerate(zip(cards, card_items))]))
        original.append(deepcopy({key: member[key] for key in ("role", "name", "obtained_count", "cards", "raw") if key in member}))
    pools, identities, order_evidence = pool_contract(members, language)
    totals = [sum(pools[i])+sum(pools[i+1]) for i in (0, 2, 4)]
    return dict(name="游戏当前背包目标", deployment_only=True, language=language,
        pools=pools, pool_identity_order=identities, pool_order_evidence=order_evidence,
        party=members, recognition_catalog=recognition, initial_inventory=inventory,
        bag_party=original, ordinary_total=sum(totals), role_ordinary_totals=totals,
        source=dict(target_authority="independently_confirmed_bag_targets",
                    inventory_authority="visible_bag_levels_and_obtained_counts"))
