"""Bind out-of-catalogue offers to stable bag slots, without inventing IDs.

Known candidates are supplied by the tracker and left for its existing name /
portrait disambiguation. Only the one local character's slots enter this matcher.
Guessed display names break ties but never become trusted aliases.
"""
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import product

from .language import catalog_name_fields
from .name_matching import clean_name, identity_key


def matches_target(observed, target):
    """None ignores the ribbon; -1 is a visible badge with unread digits."""
    if observed is None:
        return True
    return target > 0 if observed < 0 else observed == target


@dataclass(frozen=True)
class CardReading:
    name: str
    rarity_class: str
    target_level: int | None = None
    old_level: int | None = None


@dataclass(frozen=True)
class CardBinding:
    items: tuple
    estimated: bool = False
    method: str = "known_catalogue"

    @property
    def item(self):
        return self.items[0] if len(self.items) == 1 else None


def _key(item):
    return item["role"], item.get("identity", identity_key(item))


def _names(item):
    return {clean_name(item.get(field)) for field in
            (*catalog_name_fields(), "name", "display_name", "bag_name")} - {""}


class ProvisionalCardMatcher:
    """Pure page parsing with explicitly remembered per-run aliases.

    ``known_candidates`` is one candidate sequence per reading; its nonempty
    entries always win. ``identity_levels`` uses (role, identity) keys.
    Only ``remember`` changes state, after the caller has accepted a decision
    page. Refresh / Take probes can freely resolve without recording evidence.
    ``full_pool_slots`` maps page indices to identities at already known pool
    positions; position evidence binds a slot without confirming its OCR name.
    ``preferred_pool_slots`` preserves guessed positions while names remain
    noisy; reliable names and recommendation compatibility take precedence.
    Items themselves remain untouched.
    """

    def __init__(self):
        self.aliases = {}
        self.display_names = {}
        self.inferred_held_groups = set()
        self.inferred_acquisitions = {}

    def name_candidates(self, items, name, rarity):
        key = clean_name(name)
        return [item for item in items if item["rarity_class"] == rarity
                and key in (self.aliases.get(_key(item)) or _names(item))]

    def correct_held_slots(self, readings, candidates, items, levels):
        """Propose transfers from tentative holdings; a page probe must stay read-only."""
        corrections = {}
        if not self.inferred_acquisitions:
            return corrections
        reserved = {_key(row) for choices in candidates for row in choices}
        for index, (reading, known) in enumerate(zip(readings, candidates)):
            if known or (reading.old_level or 0) <= 0 or (reading.target_level or 0) <= 0:
                continue
            # Identify the visible card without letting a previous guess veto its name or target.
            unheld_reading = CardReading(reading.name, reading.rarity_class, reading.target_level)
            binding = self.resolve_page([unheld_reading], [[]], items, {})[0]
            if (binding.estimated or binding.item is None or _key(binding.item) in reserved
                    or levels.get(_key(binding.item), 0)):
                continue
            donors = [_key(item) for item in items if item["rarity_class"] == reading.rarity_class
                      and _key(item) in self.inferred_acquisitions and _key(item) not in reserved
                      and levels.get(_key(item)) == reading.old_level]
            matching = [key for key in donors if self.inferred_acquisitions[key] == clean_name(reading.name)]
            donors = matching or donors
            if len(donors) == 1:
                corrections[index] = donors[0], _key(binding.item)
        # One held card cannot explain two separate upgrades on the same screen.
        keys = [key for transfer in corrections.values() for key in transfer]
        return {i: pair for i, pair in corrections.items() if all(keys.count(key) == 1 for key in pair)}

    def resolve_page(self, readings, known_candidates, provisional_items,
                     identity_levels, *, pending_level=None, full_pool_slots=None,
                     preferred_pool_slots=None, ribbon_hints=None):
        def held_levels(item):
            key = _key(item)
            levels = {identity_levels.get(key, 0)}
            if pending_level is not None and key == pending_level[0]:
                levels.add(pending_level[1])
            return levels

        slots = sorted(provisional_items, key=lambda item:
                       (item["bag_display_index"], item["role"], str(_key(item)[1])))
        provisional_role = slots[0]["role"] if slots else None
        result = [CardBinding(tuple(items)) if items else None for items in known_candidates]
        # Alternatives await role / portrait disambiguation; only an actual
        # singleton occupies a slot. Reserving every alternative can force a
        # different card into a wrong slot and manufacture name confirmation.
        used = {_key(binding.item) for binding in result if binding and binding.item}
        choices = {}
        evidence_choices = {}
        strong_names = {}
        for index, (reading, binding) in enumerate(zip(readings, result)):
            if binding is not None:
                continue
            metadata = [item for item in slots
                          if item["rarity_class"] == reading.rarity_class
                          and matches_target(reading.target_level, item["target_level"])]
            compatible = [item for item in metadata
                          if (reading.old_level is None or reading.old_level < 0 or
                               reading.old_level in held_levels(item))]
            # A guessed acquisition changes the model's levels, but neither
            # that level nor the other slots' inferred absence proves identity.
            if (provisional_role, reading.rarity_class) in self.inferred_held_groups:
                evidence_choices[index] = metadata
            name = clean_name(reading.name)
            ranked = []
            for item in compatible:
                aliases = self.aliases.get(_key(item)) or _names(item)
                score = max((SequenceMatcher(None, name, alias).ratio()
                             for alias in aliases), default=0.) if name else 0.
                confirmed = name in self.aliases.get(_key(item), set())
                tentative = bool(name and name == clean_name(self.display_names.get(_key(item))))
                ranked.append((score, item, confirmed, tentative))
            choices[index] = ranked
            if preferred_pool_slots and ranked:
                by_score = sorted(ranked, key=lambda row: row[0], reverse=True)
                score, item, _, _ = by_score[0]
                runner_up = by_score[1][0] if len(by_score) > 1 else 0.
                if item.get("name_confirmed") and score >= .82 and score - runner_up >= .1:
                    strong_names[index] = _key(item)

        # Solve the whole small page, rather than letting an early guess steal
        # the only exact lexical match of a later card.
        best_rank, best_assignment, evidence_rank = None, None, None
        feasible, optimal = {}, {}
        for rows in product(*choices.values()):
            assigned = dict(zip(choices, rows))
            keys = [_key(item) for _, item, _, _ in rows]
            if len(set(keys)) != len(keys) or used.intersection(keys):
                continue
            if any(index in (full_pool_slots or {}) and _key(item)[1] != full_pool_slots[index]
                   for index, (_, item, _, _) in assigned.items()):
                continue
            for index, (_, item, _, _) in assigned.items():
                feasible.setdefault(index, set()).add(_key(item))
            base = (sum(confirmed for _, _, confirmed, _ in rows),
                    sum(score == 1. for score, _, _, _ in rows),
                    sum(score for score, _, _, _ in rows))
            # Keep certainty independent of the tentative display-name tie break.
            if evidence_rank is None or base > evidence_rank:
                evidence_rank = base
                optimal = {index: {_key(item)} for index, (_, item, _, _) in assigned.items()}
            elif base == evidence_rank:
                for index, (_, item, _, _) in assigned.items():
                    optimal[index].add(_key(item))
            rank = base
            if preferred_pool_slots:
                reliable_names = sum(strong_names.get(index) == _key(row[1]) for index, row in assigned.items())
                preferred = sum(preferred_pool_slots.get(index) == _key(row[1])[1] for index, row in assigned.items())
                rank = (*base[:2], reliable_names, preferred, base[2])
            if ribbon_hints is not None:
                agreements = sum(matches_target(ribbon_hints[i], row[1]["target_level"])
                                 for i, row in assigned.items())
                rank = (rank[0], agreements, *rank[1:])
            rank = (*rank, sum(tentative for _, _, _, tentative in rows))
            if best_rank is None or rank > best_rank:
                best_rank, best_assignment = rank, assigned
        if best_assignment is None:
            return [binding or CardBinding((), True, "unresolved") for binding in result]
        for index, (score, item, _, _) in best_assignment.items():
            # Fuzzy evidence must distinguish this slot from every other
            # compatible slot, not merely the slots left after greedy guesses.
            if index in evidence_choices:
                name = clean_name(readings[index].name)
                runner_up = max((SequenceMatcher(None, name, alias).ratio()
                    for candidate in evidence_choices[index] if _key(candidate) != _key(item)
                    for alias in (self.aliases.get(_key(candidate)) or _names(candidate))), default=0.)
            else:
                runner_up = max((other for other, candidate, _, _ in choices[index]
                                 if _key(candidate) != _key(item)), default=0.)
            # A known full-pool position constrains identity, but cannot turn
            # an unreadable name into confirmed OCR evidence.
            metadata_unique = len([candidate for _, candidate, _, _ in choices[index]
                                   if _key(candidate) not in used]) == 1
            if full_pool_slots is None:
                metadata_unique = len(feasible[index]) == 1
            if index in evidence_choices:
                metadata_unique = len([candidate for candidate in evidence_choices[index]
                                       if _key(candidate) not in used]) == 1
            confirmed = clean_name(readings[index].name) in self.aliases.get(_key(item), set())
            supported_name = score == 1. or (item.get("name_confirmed") and score >= .82)
            clear = len(optimal[index]) == 1 and _key(item) in optimal[index] and (confirmed or
                supported_name and score - runner_up >= .1)
            estimated = not (metadata_unique or clear)
            binding = CardBinding((item,), estimated,
                                  "provisional_name" if not estimated else "provisional_order")
            result[index] = binding
        return result

    def remember(self, name, item, *, estimated=False):
        """Remember one accepted slot; guessed names remain display-only hints."""
        key = _key(item)
        if not estimated:
            self.aliases.setdefault(key, set()).add(clean_name(name))
        self.display_names[key] = name
