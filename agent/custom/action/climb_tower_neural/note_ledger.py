"""Run-scoped note accounting from inventory snapshots, without Maa or OCR calls.

Rows returned here can be passed directly to ``SamplingCollector.emit(**row)``.
Completeness follows the page's display rules. In-run inventory bars contain
the selected tower's nine notes; supply and settlement have their own scope.
"""
from copy import deepcopy
from datetime import datetime


MELODIES = tuple("melody_of_" + name for name in (
    "pummel", "luck", "burst", "stamina", "focus", "skill", "ultimate",
    "ignis", "umbra", "aqua", "ventus", "lux", "terra",
))
COMMON_MELODIES = MELODIES[:7]
TOWER_ELEMENTS = {
    "光土": ("melody_of_lux", "melody_of_terra"),
    "水风": ("melody_of_aqua", "melody_of_ventus"),
    "火暗": ("melody_of_ignis", "melody_of_umbra"),
}
TOWER_INVENTORY_PAGES = frozenset({
    "bag", "event", "event_menu", "field", "result", "associated_skill", "level_up",
})


def _unknown_counts():
    return dict.fromkeys(MELODIES)


def _timestamp():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _difference(before, after):
    return {melody: after[melody] - before[melody]
            if before[melody] is not None and after[melody] is not None else None
            for melody in MELODIES}


class NoteLedger:
    """Keep one initial baseline, fresh snapshots and one pending source.

    ``tower`` accepts a Chinese tower name or Maa's ``active_melodies`` pair.
    ``observe`` accepts a canonical count mapping or ``read_inventory``'s row.
    A partial observation replaces the whole current snapshot, including its
    unknown components. Source IDs, rather than page identities, deduplicate
    battle and operation results.
    """

    def __init__(self):
        self.tower = None
        self.active_melodies = ()
        self.visible_melodies = ()
        self.tracked_melodies = ()
        self.run_id = None
        self.supply = None
        self.initial = _unknown_counts()
        self.initial_sources = dict.fromkeys(MELODIES, "unknown")
        self.current = _unknown_counts()
        self.last_displayed_counts = _unknown_counts()
        self.revision = 0
        self._started = False
        self._pending = None
        self._closed = set()
        self._gap_before = None
        self._gap_sources = []
        self._current_snapshot = self._unknown_snapshot()
        self._spend_extras = {}
        self._possible_spend_extras = set()

    @property
    def pending(self):
        return deepcopy(self._pending)

    @property
    def current_observation(self):
        return deepcopy(self._current_snapshot)

    def observed_in_scope(self, scope, *, after_revision=None, page=None):
        """Return a complete reading for this boundary, never one before an unobserved change."""
        current = self._current_snapshot
        if (self._gap_sources or (after_revision is not None and self.revision <= after_revision)
                or (page is not None and current["page"] != page) or not self.complete_in_scope(current, scope)):
            return None
        return self.current_observation

    def inventory_scope(self, displayed=()):
        """The tower's nine types, held extras and additional displayed types."""
        return tuple(dict.fromkeys((*self.visible_melodies,
            *(key for key, value in self.last_displayed_counts.items() if value is not None and value > 0),
            *displayed)))

    def cache_spend_extras(self, observation, *, page="bag_detail", observed_at=None):
        """Remember actual off-tower reads without advancing an inventory frame.

        Ordinary gains cannot add off-tower notes. These component readings
        stay usable until a real random-spend click invalidates them. An
        absent component is untouched; an explicitly failed read is unknown.
        """
        if observation is None:
            return
        wrapped = "counts" in observation
        raw = observation if wrapped else {}
        counts = raw["counts"] if wrapped else observation
        readings = [row for row in raw.get("readings", ()) if row.get("kind") != "skipped"]
        unresolved = set(raw.get("missing", ())) | set(raw.get("incomplete", ())) | set(raw.get("conflicts", ()))
        attempted = {key for key, value in counts.items() if value is not None}
        attempted.update(row.get("melody") for row in readings)
        attempted.update(unresolved)
        if not wrapped:
            attempted.update(counts)
        extras = [key for key in MELODIES if key not in self.visible_melodies and key in attempted]
        self._track(extras)
        for melody in extras:
            count = None if melody in unresolved else counts.get(melody)
            self._spend_extras[melody] = dict(count=count, valid=count is not None,
                page=page or raw.get("page") or raw.get("layout") or "inventory",
                observed_at=observed_at or raw.get("observed_at") or raw.get("timestamp") or _timestamp(),
                revision=self.revision, source=raw.get("sources", {}).get(melody, "inventory"),
                readings=deepcopy([row for row in readings if row.get("melody") == melody]))
            if count == 0:
                self._possible_spend_extras.discard(melody)
            else:
                self._possible_spend_extras.add(melody)

    def known_spend_extras(self):
        """Return only explicitly read off-tower counts that remain valid."""
        return {key: row["count"] for key, row in self._spend_extras.items() if row["valid"]}

    def unresolved_spend_extras(self):
        """Off-tower stock that a spend needs but no trustworthy read supplies."""
        known = self.known_spend_extras()
        return tuple(key for key in MELODIES if key in self._possible_spend_extras and key not in known)

    def spend_scope(self):
        """The nine tower types and extras that may participate in this spend."""
        extras = set(self._possible_spend_extras)
        pending = self._pending
        if pending:
            extras.update(key for key, count in pending["before"]["counts"].items()
                          if key not in self.visible_melodies and count is not None and count > 0)
            if (pending.get("spend_baseline_locked") or pending["metadata"].get("event_components")
                    or pending["source_type"] == "combined_interval"):
                extras.update(pending.get("required_scope", ()))
        return self.visible_melodies + tuple(key for key in MELODIES
                                            if key not in self.visible_melodies and key in extras)

    def prepare_spend_baseline(self):
        """Enrich a pre-click baseline with still valid, evidenced extra stock.

        The inventory observation stays unchanged. Once a spend has clicked,
        or sources have merged, later reads cannot replace its first baseline.
        """
        pending = self._pending
        if pending is None:
            return None
        before = pending["before"]
        previous = (dict(before["counts"]), list(before["scope_melodies"]),
                    before["complete"], pending.get("required_scope"))
        scope = self.spend_scope()
        pending["required_scope"] = list(scope)
        frozen = (pending.get("spend_baseline_locked") or pending["metadata"].get("event_components")
                  or pending["source_type"] == "combined_interval")
        if not frozen:
            original_complete = pending.setdefault("spend_original_complete",
                self.complete_in_scope(before, self.visible_melodies))
            blocked = set(before["conflicts"] + before["incomplete"])
            for melody in scope:
                cached = self._spend_extras.get(melody)
                if (melody in self.visible_melodies or melody in blocked or before["counts"][melody] is not None
                        or not cached or not cached["valid"] or cached["count"] <= 0):
                    continue
                before["counts"][melody] = cached["count"]
                before["sources"][melody] = "carried_from_" + cached["page"]
                before.setdefault("carried_sources", {})[melody] = deepcopy(cached)
            before["scope_melodies"] = list(dict.fromkeys((*before["scope_melodies"], *scope)))
            before["missing"] = [key for key in before["scope_melodies"] if before["counts"][key] is None]
            before["complete"] = bool(original_complete and not (
                before["missing"] or before["incomplete"] or before["conflicts"]))
        if previous == (before["counts"], before["scope_melodies"], before["complete"], pending["required_scope"]):
            return None
        return self._event("note_source_baseline_updated", source_type=pending["source_type"],
            operation_id=pending["operation_id"], before=deepcopy(before["counts"]),
            before_sources=deepcopy(before["sources"]),
            before_carried_sources=deepcopy(before.get("carried_sources", {})),
            before_revision=before["revision"], before_observed_at=before["observed_at"],
            scope_melodies=list(scope), complete=self.complete_in_scope(before, scope))

    def invalidate_spend_extras(self):
        """Expire earlier extra readings only after a confirmed spend click."""
        for melody in self._possible_spend_extras:
            if melody in self._spend_extras:
                self._spend_extras[melody]["valid"] = False
        if self._pending is not None:
            self._pending["spend_baseline_locked"] = True

    def _set_tower(self, tower):
        if isinstance(tower, str):
            self.tower = tower
            self.active_melodies = TOWER_ELEMENTS[tower]
        else:
            self.tower = next((name for name, pair in TOWER_ELEMENTS.items()
                               if set(pair) == set(tower)), None)
            self.active_melodies = TOWER_ELEMENTS[self.tower]
        self.visible_melodies = COMMON_MELODIES + self.active_melodies
        self._track(self.visible_melodies)

    def _track(self, melodies):
        self.tracked_melodies += tuple(key for key in MELODIES
                                      if key in melodies and key not in self.tracked_melodies)

    def _event(self, event_type, **payload):
        row = dict(event_type=event_type, tower=self.tower,
                   active_melodies=list(self.active_melodies),
                   tracked_melodies=list(self.tracked_melodies))
        if self.run_id is not None:
            row["run_id"] = self.run_id
        row.update(payload)
        return row

    def _unknown_snapshot(self):
        return dict(counts=_unknown_counts(), sources=dict.fromkeys(MELODIES, "unknown"),
                    scope_melodies=list(self.visible_melodies),
                    complete=False, missing=list(self.tracked_melodies), incomplete=[],
                    conflicts=[], page=None, observed_at=None, revision=None)

    def _snapshot(self, observation, page, observed_at):
        wrapped = isinstance(observation, dict) and "counts" in observation
        raw = observation if wrapped else {}
        if not self.visible_melodies and raw.get("tower"):
            self._set_tower(raw["tower"])
        elif not self.visible_melodies and raw.get("active_melodies"):
            self._set_tower(raw["active_melodies"])
        counts = _unknown_counts()
        counts.update({key: value for key, value in
                       (raw["counts"] if wrapped else observation).items()
                       if key in counts})
        observed = {key for key, value in counts.items() if value is not None}
        if not wrapped:
            observed.update(observation)
        observed.update(item["melody"] for item in raw.get("readings", ())
                        if item.get("kind") != "skipped" and item.get("melody") in MELODIES)
        observed.update(raw.get("conflicts", ()))
        observed.update(raw.get("incomplete", ()))
        self._track(observed)
        page = page or raw.get("page") or raw.get("layout")
        # In-run bars always show the tower's nine types, regardless of the
        # party or extra elements supplied by discs. OCR does not choose this
        # scope. Settlement appends held extras, but omits unheld extras even
        # if a requirement previously displayed their explicit zero count.
        scope = (self.visible_melodies if page in TOWER_INVENTORY_PAGES else
                 self.inventory_scope(raw.get("scope_melodies", ())) if page == "settlement" else
                 tuple(raw.get("scope_melodies", self.tracked_melodies)))
        conflicts = list(raw.get("conflicts", ()))
        for melody in conflicts:
            if melody in counts:
                counts[melody] = None
        missing = [melody for melody in scope if counts[melody] is None]
        incomplete = list(raw.get("incomplete", ()))
        complete = bool(self.visible_melodies) and not (missing or conflicts or incomplete)
        complete = complete and raw.get("complete", True)
        for melody in scope:
            if counts[melody] is not None:
                self.last_displayed_counts[melody] = counts[melody]
        return dict(counts=counts,
                    sources={key: "inventory" if value is not None else "unknown"
                             for key, value in counts.items()},
                    scope_melodies=list(scope),
                    complete=bool(complete), missing=missing, incomplete=incomplete,
                    conflicts=conflicts, page=page,
                    observed_at=observed_at or raw.get("observed_at") or raw.get("timestamp") or _timestamp(),
                    revision=self.revision, observation=deepcopy(observation))

    def begin_run(self, supply, tower, *, run_id=None, **metadata):
        """Initialize a new run from the right-hand supply total exactly once.

        Supply may be a sparse count mapping, a ``counts/readings`` row, or a
        list of ``melody/current`` readings. An actually observed empty list
        means all nine selected-tower counts start at rule-derived zero.
        Explicitly unread listed entries stay unknown. An incomplete reading
        with no supply values leaves the entire initial baseline unknown.
        """
        if self._started and (run_id is None or run_id == self.run_id):
            return self._event("note_run_start_repeated", reason="run_already_started",
                               counts=deepcopy(self.current), inventory_revision=self.revision)
        self.__init__()
        self._set_tower(tower)
        self.run_id = run_id
        self.supply = deepcopy(supply)
        wrapped = isinstance(supply, dict) and "counts" in supply
        entries = []
        if isinstance(supply, list):
            entries = supply
            values = {}
            for item in entries:
                melody = item["melody"]
                value = item.get("current", item.get("count"))
                values[melody] = value if melody not in values or values[melody] == value else None
        elif wrapped:
            values = supply["counts"]
            if "readings" in supply:
                entries = [item for item in supply["readings"] if item.get("kind") != "skipped"]
                values = {item["melody"]: values.get(item["melody"])
                          for item in entries if item.get("melody") in MELODIES}
        else:
            values = supply
        conflicts = list(supply.get("conflicts", ())) if wrapped else []
        if conflicts:
            values = dict(values)
        for melody in conflicts:
            if melody in MELODIES:
                values[melody] = None
        self._track(values)
        supply_missing = wrapped and not supply.get("complete", True) and not values
        for melody in self.tracked_melodies:
            if melody in values:
                self.initial[melody] = values[melody]
                self.initial_sources[melody] = ("supply_total_ocr" if values[melody] is not None
                                                else "unknown")
            elif not supply_missing:
                self.initial[melody] = 0
                self.initial_sources[melody] = "initial_rule"
        missing = [melody for melody in self.tracked_melodies if self.initial[melody] is None]
        complete = not missing and (not wrapped or supply.get("complete", True))
        self.current = deepcopy(self.initial)
        self.last_displayed_counts = deepcopy(self.initial)
        self._current_snapshot = dict(counts=deepcopy(self.initial), sources=deepcopy(self.initial_sources),
                                      scope_melodies=list(self.tracked_melodies),
                                      complete=bool(complete), missing=missing, incomplete=[], conflicts=conflicts,
                                      page="initial_supply", observed_at=_timestamp(), revision=0)
        if not complete:
            self._possible_spend_extras.update(key for key in MELODIES if key not in self.visible_melodies)
        self.cache_spend_extras(dict(counts=values, sources=self.initial_sources,
            readings=entries,
            missing=[key for key, value in values.items() if value is None], conflicts=conflicts),
            page="initial_supply", observed_at=self._current_snapshot["observed_at"])
        self._started = True
        return self._event("note_run_started", **metadata, supply=deepcopy(self.supply),
                           initial=deepcopy(self.initial), initial_sources=deepcopy(self.initial_sources),
                           scope_melodies=list(self.tracked_melodies),
                           complete=bool(complete), missing=missing, conflicts=conflicts, inventory_revision=0)

    def begin_battle(self, battle_id, **metadata):
        return self._begin_source("battle", battle_id, metadata)

    def combine_event_battle(self, battle_id, **metadata):
        """Keep the event baseline for one explicitly inseparable interval."""
        return self._combine_source(dict(source_type="battle", operation_id=battle_id, **metadata))

    def combine_events(self, event_id, **metadata):
        """Keep each event's selection with its own ID when no boundary was read."""
        return self._combine_source(dict(source_type="event", operation_id=event_id,
            source_basis="confirmed_event_choice", **metadata))

    def _combine_source(self, source):
        pending = self._pending
        if not pending or pending["source_type"] not in {"event", "combined_interval"}:
            return []
        sources = (deepcopy(pending["mixed_sources"]) if pending["source_type"] == "combined_interval" else
            deepcopy(pending["metadata"].get("event_components")) or [dict(
                source_type="event", operation_id=pending["operation_id"],
                source_basis="confirmed_event_choice", **deepcopy(pending["metadata"]))])
        if any((row["source_type"], str(row["operation_id"])) ==
               (source["source_type"], str(source["operation_id"])) for row in sources):
            return []
        sources.append(deepcopy(source))
        pending["source_type"] = "combined_interval"
        pending["operation_id"] = "combined_" + "_".join(str(row["operation_id"]) for row in sources)
        pending["key"] = (pending["source_type"], pending["operation_id"])
        pending["mixed_sources"] = sources
        events = [row for row in sources if row["source_type"] == "event"]
        interval_basis = ("event_then_battle_without_intermediate_capture"
            if any(row["source_type"] == "battle" for row in sources) else
            "consecutive_events_without_intermediate_capture")
        for key in ("choice", "consequence", "note_effect", "spends_notes", "model_event_id", "option_id",
                    "event_choice_record_id", "floor_observation",
                    "stable_event_result_confirmed"):
            pending["metadata"].pop(key, None)
        pending["metadata"].update(floor=None, floor_complete=False,
            floor_source="combined_interval_spans_sources", interval_basis=interval_basis,
            event_components=deepcopy(events), event_floor_candidates=sorted({
                floor for event in events for floor in event.get("event_floor_candidates", ())}),
            independently_observed=False, analysis_eligible=False, probability_eligible=False,
            analysis_exclusion_reason="combined_sources_not_separable")
        return [self._event("note_source_started", **pending["metadata"],
            source_type=pending["source_type"], operation_id=pending["operation_id"],
            mixed_sources=deepcopy(pending["mixed_sources"]), before=deepcopy(pending["before"]["counts"]),
            before_revision=pending["before"]["revision"],
            scope_melodies=list(pending["before"]["scope_melodies"]),
            complete=pending["before"]["complete"], status="combined")]

    def update_event_metadata(self, event_id, **metadata):
        """Re-reading one menu updates that component without changing other events."""
        pending = self._pending
        if pending is None:
            return
        if pending["source_type"] == "event" and str(pending["operation_id"]) == str(event_id):
            pending["metadata"].update(deepcopy(metadata))
        elif pending["source_type"] == "combined_interval":
            for sources in (pending["mixed_sources"], pending["metadata"]["event_components"]):
                for source in sources:
                    if source["source_type"] == "event" and str(source["operation_id"]) == str(event_id):
                        source.update(deepcopy(metadata))

    def separate_event_boundary(self):
        """A complete pre-click read can still separate the latest, unclicked event."""
        pending = self._pending
        if not pending or pending["source_type"] != "combined_interval":
            return []
        latest = pending["mixed_sources"][-1]
        if latest["source_type"] != "event" or "event_click_inventory_revision" in latest:
            return []
        source_fields = {"source_type", "operation_id", "source_basis"}
        next_metadata = {key: deepcopy(value) for key, value in latest.items() if key not in source_fields}
        sources = pending["mixed_sources"][:-1]
        if len(sources) == 1:
            source = sources[0]
            pending.update(source_type=source["source_type"], operation_id=source["operation_id"],
                metadata={key: deepcopy(value) for key, value in source.items() if key not in source_fields},
                mixed_sources=[])
            reason = "next_event_menu_observation"
        else:
            pending["operation_id"] = "combined_" + "_".join(str(source["operation_id"]) for source in sources)
            pending["mixed_sources"] = deepcopy(sources)
            pending["metadata"]["event_components"] = [deepcopy(source) for source in sources
                                                       if source["source_type"] == "event"]
            pending["metadata"]["interval_basis"] = ("event_then_battle_without_intermediate_capture"
                if any(source["source_type"] == "battle" for source in sources) else
                "consecutive_events_without_intermediate_capture")
            reason = "stable_page_combined_observation"
        pending["key"] = (pending["source_type"], str(pending["operation_id"]))
        row = self.finish_pending(reason=reason)
        return [row, *self.begin_operation("event", latest["operation_id"], **next_metadata)]

    def begin_operation(self, source_type, operation_id, **metadata):
        return self._begin_source(source_type, operation_id, metadata)

    def update_pending_metadata(self, **metadata):
        """Attach a newly observed floor without changing the frozen baseline."""
        if self._pending is not None:
            self._pending["metadata"].update(metadata)

    def require_pending_scope(self, melodies):
        if self._pending is not None:
            self._pending["required_scope"] = list(melodies)

    def refresh_pending_baseline(self):
        """Replace the event baseline after a detail read, before its click."""
        if (self._pending is None or self._pending.get("spend_baseline_locked")
                or self._pending["metadata"].get("event_components")
                or self._pending["source_type"] == "combined_interval"):
            return None
        self._pending["before"] = deepcopy(self._current_snapshot)
        self._pending["start_revision"] = self.revision
        self._pending.pop("spend_original_complete", None)
        return self._event("note_source_baseline_updated",
            source_type=self._pending["source_type"], operation_id=self._pending["operation_id"],
            before=deepcopy(self.current), before_revision=self.revision,
            scope_melodies=list(self._current_snapshot["scope_melodies"]),
            complete=self._current_snapshot["complete"])

    def _begin_source(self, source_type, operation_id, metadata):
        key = (source_type, str(operation_id))
        if key in self._closed or (self._pending and self._pending["key"] == key):
            return []
        rows = []
        previous = self.finish_pending(reason="next_source")
        if previous is not None:
            rows.append(previous)
        before = deepcopy(self._current_snapshot)
        self._pending = dict(key=key, source_type=source_type, operation_id=operation_id,
                             before=before, start_revision=self.revision,
                             metadata=deepcopy(metadata), mixed=bool(self._gap_sources),
                             mixed_sources=deepcopy(self._gap_sources), issues=[])
        rows.append(self._event("note_source_started", **metadata, source_type=source_type,
                                operation_id=operation_id, before=deepcopy(before["counts"]),
                                before_revision=before["revision"], before_page=before["page"],
                                before_observed_at=before["observed_at"], complete=before["complete"],
                                scope_melodies=list(before["scope_melodies"]),
                                missing=list(before["missing"]),
                                status="mixed" if self._gap_sources else
                                       "complete" if before["complete"] else "incomplete"))
        return rows

    def observe(self, observation, *, page=None, observed_at=None, source_id=None):
        """Replace current counts; prompt pages never create their own delta."""
        self.revision += 1
        snapshot = self._snapshot(observation, page, observed_at)
        self.cache_spend_extras(observation, page=snapshot["page"], observed_at=snapshot["observed_at"])
        self.current = deepcopy(snapshot["counts"])
        self._current_snapshot = snapshot
        links = {}
        if self._pending:
            links = {key: self._pending[key] for key in ("source_type", "operation_id")}
            if self._pending["metadata"].get("independently_observed") is False:
                links.update({key: self._pending["metadata"][key] for key in (
                    "independently_observed", "analysis_eligible", "probability_eligible", "analysis_exclusion_reason")})
            if source_id is not None and str(source_id) != str(self._pending["operation_id"]):
                self._pending["mixed"] = True
                self._pending["issues"].append("observation_source_mismatch")
        row = self._event("note_inventory_observed", **links, counts=deepcopy(self.current),
                          sources=deepcopy(snapshot["sources"]), complete=snapshot["complete"],
                          scope_melodies=list(snapshot["scope_melodies"]),
                          missing=list(snapshot["missing"]), incomplete=list(snapshot["incomplete"]),
                          conflicts=list(snapshot["conflicts"]), page=snapshot["page"],
                          observed_at=snapshot["observed_at"], inventory_revision=self.revision,
                          observation=deepcopy(observation))
        if source_id is not None:
            row["observed_source_id"] = source_id
        if self._gap_before is not None:
            row["unattributed_interval"] = dict(
                status="mixed", complete=False, before=deepcopy(self._gap_before["counts"]),
                after=deepcopy(self.current), delta=_difference(self._gap_before["counts"], self.current),
                total=None, sources=deepcopy(self._gap_sources))
            self._gap_before = None
            self._gap_sources = []
        return row

    def finish_pending(self, *, reason="boundary", mixed=False):
        """Compare the shared page scope; never sum unshown types as zero."""
        pending = self._pending
        if pending is None:
            return None
        before = pending["before"]
        has_after = self.revision > pending["start_revision"]
        after = self._current_snapshot if has_after else self._unknown_snapshot()
        is_mixed = bool(mixed or pending["mixed"])
        delta = _difference(before["counts"], after["counts"])
        scope = pending.get("required_scope") or [melody for melody in before["scope_melodies"]
                 if not has_after or melody in after["scope_melodies"]]
        before_missing = [melody for melody in scope if before["counts"][melody] is None]
        after_missing = [melody for melody in scope if after["counts"][melody] is None]
        missing = [melody for melody in scope if delta[melody] is None]
        before_complete = self.complete_in_scope(before, scope)
        after_complete = self.complete_in_scope(after, scope)
        complete = bool(scope and before_complete and after_complete and not is_mixed and not missing)
        issues = list(pending["issues"])
        if not before_complete or before_missing:
            issues.append("incomplete_before")
        if not has_after:
            issues.append("missing_after")
        elif not after_complete or after_missing:
            issues.append("incomplete_after")
        if is_mixed:
            issues.append("mixed_source_interval")
        incomplete = list(dict.fromkeys(melody for melody in
            before["incomplete"] + after["incomplete"] + missing if melody in scope))
        row = self._event("note_delta", **pending["metadata"], source_type=pending["source_type"],
                          operation_id=pending["operation_id"], before=deepcopy(before["counts"]),
                          after=deepcopy(after["counts"]), delta=delta,
                          scope_melodies=scope,
                          total=sum(delta[key] for key in scope) if complete else None,
                          complete=complete, status="mixed" if is_mixed else
                          "complete" if complete else "incomplete", missing=missing, incomplete=incomplete,
                          before_missing=before_missing, after_missing=after_missing,
                          before_revision=before["revision"], after_revision=after["revision"],
                          before_observed_at=before["observed_at"], after_observed_at=after["observed_at"],
                          before_observation_complete=before["complete"], after_observation_complete=after["complete"],
                          before_scope_complete=before_complete, after_scope_complete=after_complete,
                          issues=issues, mixed_sources=deepcopy(pending["mixed_sources"]), reason=reason)
        if before.get("carried_sources"):
            row.update(before_sources=deepcopy(before["sources"]),
                       before_carried_sources=deepcopy(before["carried_sources"]))
        if not has_after:
            if self._gap_before is None:
                self._gap_before = deepcopy(before)
            self._gap_sources.append({key: pending[key] for key in ("source_type", "operation_id")})
        self._closed.add(pending["key"])
        self._pending = None
        return row

    @staticmethod
    def complete_in_scope(snapshot, scope):
        """An extra note's unknown stock does not invalidate another page scope."""
        unresolved = set(snapshot["missing"] + snapshot["incomplete"] + snapshot["conflicts"])
        if any(snapshot["counts"][melody] is None for melody in scope) or unresolved.intersection(scope):
            return False
        if snapshot["complete"]:
            return True
        observation = snapshot.get("observation") or {}
        return bool(unresolved.difference(scope)) and not (
            observation.get("stopped") or observation.get("unresolved_regions"))
