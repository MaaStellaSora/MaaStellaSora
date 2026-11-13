"""Small recording-only types; no Maa calls, model state or export transforms."""
from dataclasses import dataclass, field
from collections import Counter
import ast
from functools import lru_cache
from pathlib import Path
import uuid

from .name_matching import identity_key, is_anonymous_non_target


def planned_combined_interval(value, *, started=False):
    """The explicit stable-page policy excludes inseparable source statistics."""
    sources = value.get("mixed_sources")
    return (value.get("source_type") == "combined_interval"
        and (value.get("reason") == "stable_page_combined_observation"
             or started and value.get("status") == "combined" and "reason" not in value)
        and value.get("interval_basis") in {"event_then_battle_without_intermediate_capture",
                                            "consecutive_events_without_intermediate_capture"}
        and value.get("independently_observed") is False
        and value.get("analysis_eligible") is False
        and value.get("probability_eligible") is False
        and value.get("analysis_exclusion_reason") == "combined_sources_not_separable"
        and isinstance(sources, list) and len(sources) >= 2
        and all(isinstance(source, dict) and source.get("operation_id")
                and source.get("source_type") in {"event", "battle"} for source in sources)
        and ({source["source_type"] for source in sources} == {"event", "battle"}
             if value.get("interval_basis") == "event_then_battle_without_intermediate_capture" else
             all(source["source_type"] == "event" for source in sources)))


def reused_interval_components(records):
    """Locate sources claimed again after their earlier interval was closed."""
    closed, reused = {}, []
    for row in records:
        if row.get("event_type") != "note_delta":
            continue
        sources = row["mixed_sources"] if planned_combined_interval(row) else [row]
        duplicates = []
        for source in sources:
            key = source.get("source_type"), str(source.get("operation_id"))
            if planned_combined_interval(row) and key in closed:
                duplicates.append(dict(source_type=key[0], operation_id=source["operation_id"],
                    previous_record_id=closed[key]))
            if source.get("operation_id"):
                closed[key] = row["record_id"]
        if duplicates:
            reused.append((row, duplicates))
    return reused


@lru_cache(maxsize=1)
def _route_battle_floors():
    source = Path(__file__).with_name("critic") / "vendor/stella_sim.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    route = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "ROUTE" for target in node.targets))
    return {floor for kind, floor, _ in route if kind == "BATTLE"}


def battle_floor_outside_route(value):
    floor = value.get("floor")
    return (value.get("source_type") == "battle" and value.get("floor_complete") is True
            and isinstance(floor, int) and not isinstance(floor, bool) and floor not in _route_battle_floors())


@dataclass
class ReadEvidence:
    kind: str
    links: dict = field(default_factory=dict)
    read_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    recognitions: list = field(default_factory=list)

    def recognition_by_id(self, reco_id):
        if reco_id is None:
            return None
        return next((r for r in reversed(self.recognitions) if r.get("native_reco_id") == reco_id), None)

    def matching(self, node, roi=None):
        nodes = (node,) if isinstance(node, str) else node
        return [r for r in self.recognitions if r["node_name"] in nodes
                and (roi is None or r["roi"] == list(roi))]

    def matching_within(self, node, roi):
        x, y, width, height = roi
        return [r for r in self.matching(node) if r.get("roi")
                and x <= r["roi"][0] and y <= r["roi"][1]
                and r["roi"][0] + r["roi"][2] <= x + width
                and r["roi"][1] + r["roi"][3] <= y + height]

    @property
    def record_ids(self):
        return [r["record_id"] for r in self.recognitions]


def recognition_roi(entry, args, kwargs):
    override = kwargs.get("pipeline_override", args[1] if len(args) > 1 else {})
    node = override.get(entry, {}) if isinstance(override, dict) else {}
    return node.get("recognition", {}).get("param", {}).get("roi")


def raw_ocr(records):
    return [item for r in records for item in (r["raw_result"] or {}).get("all_results", [])]


class ObservedInventory:
    """Inventory reconstructed only from a fresh start and confirmed take actions.

    Maa's eager cache remains separate. Completeness describes this observation
    history, never an independently read bag or a server response.
    """
    def __init__(self, full_start=False):
        self.items = {}
        self.anonymous = Counter()
        self.bag_verified = False
        self.bag_initialized = False
        self.revision = 0
        self.complete = full_start
        self.issues = [] if full_start else [dict(reason="run_start_not_observed")]

    def initialize_bag_inventory(self, rows):
        self.items, self.anonymous = {}, Counter()
        for row in rows:
            identity = identity_key(row)
            kind = row["rarity_class"]
            if identity is None or is_anonymous_non_target(row):
                self.anonymous[row["role"], kind, row["level"]] += row.get("count", 1)
            else:
                self.items[identity] = dict(row, canonical_rarity=kind, core=kind == "core",
                    level_after=row["level"], inventory_basis="visible_bag_level_OCR")
        self.complete, self.bag_verified, self.issues = True, True, []
        self.bag_initialized = True
        self.revision += 1

    def uncertain(self, reason, **details):
        self.complete = False
        issue = dict(reason=reason, **details)
        if issue not in self.issues:
            self.issues.append(issue)

    def audit(self, candidates, offer_id):
        conflicts = []
        for p in candidates:
            identity, before = identity_key(p), p.get("level_before", -1)
            if p.get("recognition_status") == "unresolved" or identity is None or before is None or before < 0:
                continue
            known = self.items.get(identity)
            anonymous = self.anonymous[p.get("role"), p.get("canonical_rarity"), before]
            expected = known["level"] if known else before if anonymous else 0 if self.complete else None
            if expected is not None and before != expected:
                conflict = dict(potential_id=identity, history_level=expected, card_level_before=before)
                conflicts.append(conflict)
                self.uncertain("card_level_disagrees_with_history", offer_id=offer_id, **conflict)
        return conflicts

    def confirm(self, selected, record_id):
        identity = identity_key(selected)
        level = selected.get("level_after", selected.get("new_level", -1))
        self.revision += 1
        if (selected.get("recognition_status") == "unresolved"
                or identity is None or level is None or level < 1 or selected.get("role") not in (1, 2, 3)
                or selected.get("canonical_rarity") not in ("core", "rare", "common") or selected.get("core") is None):
            self.uncertain("selected_potential_unresolved", confirmation_record_id=record_id)
            return False
        row = dict(selected, level=level, confirmation_record_id=record_id,
                   level_basis=selected.get("level_basis", "card_level_OCR"))
        bucket = selected["role"], selected["canonical_rarity"], selected.get("level_before")
        if is_anonymous_non_target(selected):
            if selected.get("level_before", 0) > 0:
                self.anonymous[bucket] -= 1
            self.anonymous[selected["role"], selected["canonical_rarity"], level] += 1
            self.bag_verified = False
            return True
        if identity not in self.items and self.anonymous[bucket]:
            self.anonymous[bucket] -= 1
        self.items[identity] = row
        self.bag_verified = False
        return True

    def fields(self):
        owned = [dict(p) for _, p in sorted(self.items.items(), key=lambda row: str(row[0]))]
        anonymous = [dict(role=role, rarity_class=kind, canonical_rarity=kind, level=level,
                         count=count, core=kind == "core", potential_id=None, local_identity=None,
                         anonymous=True, target_level=0)
                     for (role, kind, level), count in sorted(self.anonymous.items()) if count]
        counts = []
        for role in (1, 2, 3):
            members = [p for p in owned+anonymous if p["role"] == role]
            noncore = [p for p in members if not p["core"]]
            counts.append(dict(role=role, core_count=sum(p.get("count", 1) for p in members if p["core"]),
                               noncore_count=sum(p.get("count", 1) for p in noncore),
                               rare_count=sum(p.get("count", 1) for p in noncore if p["canonical_rarity"] == "rare"),
                               common_count=sum(p.get("count", 1) for p in noncore if p["canonical_rarity"] == "common"),
                               below_max_count=sum(p.get("count", 1) for p in noncore if p["level"] < 6),
                               noncore_level_sum=sum(p["level"]*p.get("count", 1) for p in noncore)))
        return dict(owned=owned, owned_anonymous=anonymous, owned_complete=self.complete,
                    owned_basis="bag_OCR_and_UI_confirmed_selection_history" if self.bag_initialized else "fresh_start_and_UI_confirmed_selection_history",
                    owned_source="bag_inventory_and_confirmed_selection_history" if self.bag_initialized else "observed_selection_history", owned_revision=self.revision,
                    owned_issues=list(self.issues), owned_counts=counts,
                    owned_independently_verified=self.bag_verified)
