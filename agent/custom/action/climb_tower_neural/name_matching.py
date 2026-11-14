"""Shared OCR name resolution for neural decisions and recorded identities."""
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import json
from pathlib import Path
import unicodedata


GLYPHS = str.maketrans(json.loads(Path(__file__).with_name("bag_preset_rules.json")
                                  .read_text(encoding="utf-8"))["name_variant_glyphs"])


def clean_name(value, *, keep_long_vowel=False):
    text = unicodedata.normalize("NFKC", "" if value is None else str(value)).translate(GLYPHS)
    text = re.sub(r"\W", "", text)
    return text if keep_long_vowel else text.replace("ー", "")


@dataclass(frozen=True)
class NameMatch:
    items: tuple
    method: str
    similarity: float
    matched_alias: str = ""
    runner_up_similarity: float = 0.
    similarity_gap: float = 0.
    minimum_similarity: float = 0.
    minimum_gap: float = 0.


def _identity(item):
    return identity_key(item), item.get("role")


def identity_key(item):
    """Game catalogue IDs and observed local identities keep separate namespaces."""
    return item.get("potential_id") if item.get("potential_id") is not None else item.get("local_identity", item.get("id"))


def local_potential_identity(role, character, rarity, name):
    return f"bag:{role}:{clean_name(character)}:{rarity}:{clean_name(name)}"


def is_anonymous_non_target(item):
    return item.get("potential_id") is None and item.get("target_level") == 0


def match_name(name, aliases, *, role=None, rarity=None, preferred=None, party_bound=False):
    """Prefer normalized aliases; bound approximate matches by lexical evidence.

    Similarity is a string comparison score, not an identity probability.
    Shared names keep their multiple roles and remain unresolved.
    """
    key = clean_name(name)
    if not key:
        return NameMatch((), "unresolved", 0.)

    def in_role(items):
        return tuple(item for item in items if role is None or item["role"] == role)

    def by_rarity(items):
        if not rarity:
            return items
        return tuple(item for item in items if item["rarity_class"] == rarity) or items

    exact = in_role((preferred or {}).get(key, ())) or in_role(aliases.get(key, ()))
    if exact:
        return NameMatch(by_rarity(exact), "normalized_name", 1.)

    candidates = [(alias, in_role(items)) for alias, items in aliases.items()]
    candidates = [(alias, items) for alias, items in candidates if items]
    if rarity:
        same_rarity = [(alias, tuple(item for item in items if item["rarity_class"] == rarity))
                       for alias, items in candidates]
        candidates = [(alias, items) for alias, items in same_rarity if items] or candidates
    ranked = []
    for alias, items in candidates:
        ratio = SequenceMatcher(None, key, alias).ratio()
        short_name = 4 <= min(len(key), len(alias)) <= 5 and abs(len(key) - len(alias)) <= 1
        threshold = .75 if party_bound and short_name else (.82 if len(key) > 4 else .91)
        ranked.append((ratio, alias, items, threshold))
    if not ranked:
        return NameMatch((), "unresolved", 0.)
    ratio, alias, items, threshold = max(ranked, key=lambda entry: entry[0])
    identities = {_identity(item) for item in items}
    runner_up = max((score for score, _, others, _ in ranked
                     if any(_identity(item) not in identities for item in others)), default=0.)
    margin = .1 if ratio < .82 else .035
    evidence = dict(matched_alias=alias, runner_up_similarity=runner_up, similarity_gap=ratio-runner_up,
                    minimum_similarity=threshold, minimum_gap=margin)
    if ratio < threshold or ratio - runner_up < margin:
        return NameMatch((), "unresolved", ratio, **evidence)
    return NameMatch(items, "closest_name", ratio, **evidence)
