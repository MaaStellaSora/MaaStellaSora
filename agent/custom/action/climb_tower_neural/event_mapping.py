"""Map Maa's visible event OCR through the shared event_rules.json contract.

Quiz answer text stays in Maa's ascension_event_<language>.json tables. This
module interprets their rewards and the complete menu without assuming UI order.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import permutations
import json
from pathlib import Path
import re
import unicodedata


RULES = json.loads(Path(__file__).with_name("event_rules.json").read_text(encoding="utf-8"))
TEXT_RULES = RULES["text"]


@dataclass(frozen=True)
class Effect:
    kind: str
    amount: int = 0
    quality: str = ""
    scope: str = ""
    chance: int = 0


MENUS = {int(eid): {int(oid): Effect(**effect) for oid, effect in menu["options"].items()}
         for eid, menu in RULES["menus"].items()}
QUIZZES = {row["event_id"]: row for row in RULES["quiz_rules"]}


def _text(value):
    return unicodedata.normalize("NFKC", str(value or "")).lower()


def _has(text, pattern):
    return re.search(pattern, text, re.I) is not None


def choice_complete(choice):
    field = choice.get if isinstance(choice, dict) else lambda key, default=None: getattr(choice, key, default)
    observations = field("observations", {})
    return all(field(key) and (key not in observations or observations[key].get("complete"))
               for key in ("text", "consequence"))


def _matches(context, conditions):
    return all((bool(context.get(key)) == expected if isinstance(expected, bool)
                else context.get(key) == expected) for key, expected in conditions.items())


def _context(text):
    context = {name: _has(text, pattern) for name, pattern in TEXT_RULES["patterns"].items()}
    context["numbers"] = re.findall(TEXT_RULES["number_pattern"], text)
    for name, rows in TEXT_RULES["classifiers"].items():
        context[name] = next((row["value"] for row in rows if _has(text, row["pattern"])),
                             TEXT_RULES["classifier_defaults"][name])
    for name, rows in TEXT_RULES["captures"].items():
        context[name] = 0
        for row in rows:
            if not _matches(context, row.get("when", {})):
                continue
            match = re.search(row["pattern"], text, re.I)
            if match:
                context[name] = int(match.group(1))
                break
    return context


def _note_kind(text):
    return next((row["melody"] for row in TEXT_RULES["note_kinds"]
                 if _has(text, row["pattern"])), None)


def is_quiz_consequence(text):
    return _has(_text(text), TEXT_RULES["patterns"]["quiz"])


def has_potential_reward(text):
    """Use the same potential reward semantics for live and collected events."""
    return _has(_text(text), TEXT_RULES["patterns"]["potential"])


def quiz_event_id(choices):
    """Identify a shared quiz reward, leaving answer selection to Maa's table."""
    consequences = [_text(choice.get("consequence", "") if isinstance(choice, dict) else choice.consequence)
                    for choice in choices]
    if not consequences or len(set(consequences)) != 1:
        return None
    context = _context(consequences[0])
    return next((row["event_id"] for row in RULES["quiz_rules"] if _matches(context, row["when"])), None)


def quiz_correct_option(event_id):
    return QUIZZES[event_id]["correct_option_id"]


def quiz_potential_source(event_id):
    return deepcopy(QUIZZES[event_id]["potential_source"])


def _effect(text: str) -> Effect | None:
    text = _text(text)
    if not text or is_quiz_consequence(text):
        return None
    context = _context(text)
    for row in RULES["effect_rules"]:
        if _matches(context, row["when"]):
            fields = {key: context[value["field"]] if isinstance(value, dict) else value
                      for key, value in row["effect"].items()}
            fields["amount"] = int(fields.get("amount", 0))
            return Effect(**fields)
    return None


def _score(observed, expected):
    if (observed.kind, observed.amount, observed.chance) != (expected.kind, expected.amount, expected.chance):
        return None
    if observed.scope and observed.scope != expected.scope:
        return None
    if observed.quality and observed.quality != expected.quality:
        return None
    return 2 + int(observed.scope == expected.scope) + int(observed.quality == expected.quality)


def _match_note_menu(choices, effects):
    for menu in RULES["note_menus"]:
        if len(choices) != menu["visible_count"]:
            continue
        indices, note_kinds = {}, []
        for index, (choice, effect) in enumerate(zip(choices, effects)):
            other = next((int(oid) for oid, expected in menu.get("other_options", {}).items()
                          if effect == Effect(**expected)), None)
            if other is not None:
                indices[index] = other
                continue
            field = choice.get if isinstance(choice, dict) else lambda key, default="": getattr(choice, key, default)
            kind = _note_kind(_text(field("consequence", "") + " " + field("text", "")))
            option = menu["options"].get(kind)
            if option is None or (effect.kind, effect.amount) != (option["effect"]["kind"], option["effect"]["amount"]):
                break
            indices[index] = option["option_id"]
            note_kinds.append(kind)
        if (len(indices) == len(choices) and len(note_kinds) == menu["note_choice_count"]
                and len(set(note_kinds)) == len(note_kinds)
                and set(menu["required_kinds"]).issubset(note_kinds)):
            return menu["event_id"], indices
    return None


def note_descriptor(event_id, option_id):
    """Retain actual note effects even when the current objective ignores them."""
    return deepcopy(RULES["note_effects"].get(str(option_id)))


def match_event(question: str, choices: list, lang: str = "jp"):
    """Return (event ID, zero-based visible choice -> option ID), or None."""
    if not all(choice_complete(choice) for choice in choices):
        return None
    effects = []
    for choice in choices:
        consequence = choice.get("consequence", "") if isinstance(choice, dict) else choice.consequence
        effect = _effect(consequence)
        if effect is None:
            return None
        effects.append(effect)
    if len(effects) < 2:
        return None
    note_menu = _match_note_menu(choices, effects)
    if note_menu is not None:
        return note_menu
    matches = []
    for eid, options in MENUS.items():
        if len(effects) != len(options):
            continue
        for ids in permutations(options):
            scores = [_score(actual, options[oid]) for actual, oid in zip(effects, ids)]
            if None not in scores:
                matches.append((sum(scores), eid, dict(enumerate(ids))))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        return None
    return matches[0][1], matches[0][2]
