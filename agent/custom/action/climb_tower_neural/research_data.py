"""State-space observations as database rows, shared by local import and upload.

This format stores observations and their evidence, not a copy of the collector
log.  Game readings, confirmed-history reconstruction and model inference keep
separate provenance throughout the format.
"""
from collections import defaultdict
from contextlib import closing
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

from .purchase_evidence import purchase_wallet_evidence
from .name_matching import identity_key
from .settlement_counts import check_potential_counts
from .sampling_evidence import planned_combined_interval, reused_interval_components, battle_floor_outside_route
from .sampling_journal import DAMAGE_SUFFIX


SCHEMA_VERSION = 1
FORMAT = "maa-state-space-database"
TABLES = ("presets", "runs", "records", "objects", "record_objects", "readings",
          "record_readings", "quality_issues")
BASE_FIELDS = {"schema_version", "collector_version", "event_type", "record_type", "timestamp",
               "record_id", "participant_id", "session_id", "task_id", "run_id", "exploration_id",
               "sequence", "run_sequence", "lang_type", "observer", "server_confirmed",
               "preset_id", "target_vector_complete", "target_vector", "_original_record_id"}
TRANSPORT_FIELDS = {"arguments", "keyword_arguments", "raw_detail", "reco_id", "maa_task_id",
                    "maa_node_id", "image_path", "source_image_path"}
SNAPSHOTS = {"owned": ("owned", "confirmed_selection_history"),
             "final_owned": ("owned", "confirmed_selection_history"),
             "owned_before": ("owned", "confirmed_selection_history"),
             "observed_state_after": ("maa_owned", "maa_cache_unverified"),
             "maa_owned_cache": ("maa_owned", "maa_cache_unverified"),
             "state": ("maa_state", "maa_cache_unverified"),
             "previous_state": ("maa_state", "maa_cache_unverified"),
             "tracker_estimate": ("tracker", "model_inference"),
             "counters": ("tracker", "model_inference"),
             "public_state": ("model_state", "model_inference"),
             "observed_state": ("decision_observations", "game_UI_observation"),
             "pools": ("model_remaining_targets", "model_inference"),
             "tracker_public_observation": ("tracker", "model_inference")}
POLICY_FIELDS = {"score", "probability", "rank", "sub_rank", "in_preset", "in_owned",
                 "recommended", "recommended_level"}
QUALITY_RANK = {"valid": 0, "partial": 1, "suspect": 2, "invalid": 3}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _references(value):
    """Original recognition IDs, including IDs held by a selected candidate."""
    result = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"owned", "final_owned", "owned_before", "maa_owned_cache", "observed_state_after"}:
                continue
            if ("recognition_record_id" in key or key in {"recognition_record_ids", "source_locator_record_id"}) and item:
                result.update(item if isinstance(item, list) else [item])
            result.update(_references(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_references(item))
    return result


def _normalize_wallet_attempts(rows):
    """One acquisition has one wallet observation and retains every OCR attempt."""
    groups, active = [], {}
    for row in rows:
        if row["event_type"] == "wallet_observed" and row.get("read_id"):
            key = row.get("run_id"), row["read_id"]
            # A refresh-result read can poll multiple fresh screenshots. Only
            # the retry nodes belong to its preceding original-image attempt.
            if key not in active or row.get("node_name") in {
                    "星塔_神经网络_识别当前金币", "星塔_通用_识别当前金币_agent"}:
                active[key] = []
                groups.append(active[key])
            active[key].append(row)
    replacements, wallets = {}, {}
    for attempts in groups:
        if len(attempts) < 2:
            continue
        trusted = [row for row in attempts if row.get("coin_complete") and row.get("coin") is not None]
        final = (trusted or attempts)[-1]
        conflicted = [row for row in attempts if row.get("recognition_status") == "conflict"]
        values = sorted({row["coin"] for row in trusted} | {
            value for row in conflicted for value in row.get("conflicts", row.get("parsed_values", [])) if value is not None})
        final["attempt_record_ids"] = [row["record_id"] for row in attempts if row is not final]
        final["recognition_record_ids"] = [row["recognition_record_id"] for row in attempts
                                           if row.get("recognition_record_id")]
        if conflicted or len(values) > 1:
            final.update(coin=None, coin_complete=False, recognition_status="conflict", conflicts=values)
        elif not trusted:
            status = "invalid" if any(row.get("recognition_status") == "invalid" for row in attempts) else "missing"
            final.update(coin=None, coin_complete=False, recognition_status=status)
        wallets[final["record_id"]] = final
        for row in attempts:
            replacements[row["record_id"]] = final["record_id"]
            if row is not final:
                row["event_type"] = row["record_type"] = "wallet_recognition_attempt"
                row["wallet_observation_id"] = final["record_id"]
    for row in rows:
        for key, value in tuple(row.items()):
            if key.endswith("wallet_observation_id") or key == "wallet_before_observation_id":
                row[key] = replacements.get(value, value)
        if row["event_type"] == "field_wallet_observed":
            wallet = wallets.get(row.get("wallet_observation_id"))
            if wallet:
                row.update(coin=wallet["coin"], coin_complete=wallet["coin_complete"],
                           complete=wallet["coin_complete"] and row.get("floor_complete", False))


def _raw_result(value):
    """Keep all OCR/template candidates once, plus their all/filtered/best roles."""
    value = value or {}
    result = {key: deepcopy(item) for key, item in value.items()
              if key not in {"reco_id", "raw_detail", "all_results", "filtered_results", "best_result"}}
    candidates, indexes = [], {}

    def index(item):
        token = _json(item)
        if token not in indexes:
            indexes[token] = len(candidates)
            candidates.append(item)
        return indexes[token]

    result["all_indexes"] = [index(item) for item in value.get("all_results", [])]
    result["filtered_indexes"] = [index(item) for item in value.get("filtered_results", [])]
    result["best_index"] = index(value["best_result"]) if value.get("best_result") is not None else None
    result["results"] = candidates
    return result


def _normalize_event_choices(rows):
    """A repeated recognition can replace a choice before the actual click."""
    choices, pending = {}, {}
    for row in rows:
        event = row.get("event_id")
        if row["event_type"] == "event_choice":
            previous = pending.get(event)
            if previous and previous["selection_status"] == "awaiting_click":
                previous.update(selection_status="superseded", superseded_by_record_id=row["record_id"])
            row["selection_status"] = "awaiting_click"
            choices[row["record_id"]] = row
            pending[event] = row
        elif row["event_type"] == "event_choice_click_observed":
            # Older journals stored only the event ID. Their click belongs to
            # the last choice observed before it, never every choice in a menu.
            choice_id = row.get("event_choice_record_id")
            selected = choices.get(choice_id) if choice_id else pending.get(event)
            if selected and selected.get("event_id") == event:
                row["event_choice_record_id"] = selected["record_id"]
                selected.update(selection_status="clicked", click_record_id=row["record_id"])


def _snapshot(value):
    """Current owned facts; the confirmation record contains change evidence."""
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: value[key] for key in ("role", "potential_id", "local_identity", "level", "confirmation_record_id",
              "anonymous", "count", "rarity_class", "canonical_rarity", "core", "target_level",
              "local_identity_namespace") if key in value}
    if "level" not in result:
        result["level"] = value.get("level_after", value.get("new_level"))
    if identity_key(value) is None or value.get("role") is None or value.get("local_identity"):
        identity_fields = {"name", "character_id", "trekker", "trekker_index", "rarity", "canonical_rarity",
            "core", "identity_basis", "identity_name", "identity_similarity", "identity_inferred", "id_candidates",
            "recognition_status", "recognition_errors", "level_basis"}
        result.update({key: item for key, item in value.items() if key in identity_fields})
    return result


def _normalize_floor(value):
    if isinstance(value, list):
        return [_normalize_floor(item) for item in value]
    if isinstance(value, dict):
        value = {key: _normalize_floor(item) for key, item in value.items()}
        if value.get("floor", object()) is None and value.get("sequence_consistent") is False:
            value["sequence_consistent"] = None
    return value


def _session_rows(run_path, run, session_path, tail_damage):
    """Read the shared session without changing an unfinished final record."""
    first = run[0]
    if session_path is None:
        session_path = Path(run_path).parent.parent / "sessions" / f"{first.get('session_id')}.jsonl"
    if not Path(session_path).exists():
        return []
    task = first.get("task_id")
    data = Path(session_path).read_bytes()
    lines = data.splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line.decode("utf-8-sig" if index == 0 else "utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith((b"\n", b"\r")):
                raise
            tail_damage.update(scope="session", session_id=first.get("session_id"),
                byte_offset=len(data) - len(line), byte_count=len(line),
                sha256=hashlib.sha256(line).hexdigest())
        else:
            if row.get("task_id") == task:
                rows.append(row)
    return rows


def _session_record_ids(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_session_record_ids(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _session_record_ids(item, mapping) for key, item in value.items()}
    return value


class _Builder:
    def __init__(self, run_id):
        self.run_id = run_id
        self.tables = {table: [] for table in TABLES}
        self.objects = {}
        self.readings = {}
        self.reading_links = set()
        self.object_links = set()
        self.issues = {}
        self.recognitions = {}
        self.evidence_index = defaultdict(set)
        self.read_groups = defaultdict(set)
        self.record_quality = defaultdict(lambda: "valid")

    def issue(self, record_id, path, code, severity, message, evidence=None):
        identity = [self.run_id, record_id, path, code]
        key = _hash(identity)
        self.issues[key] = dict(issue_id=key, run_id=self.run_id, record_id=record_id,
            field_path=path, code=code, severity=severity, message=message,
            evidence_json=_json(evidence or {}))
        current = self.record_quality[record_id]
        if QUALITY_RANK[severity] > QUALITY_RANK[current]:
            self.record_quality[record_id] = severity

    def link_reading(self, record_id, reading_id, path):
        self.reading_links.add((record_id, reading_id, path))

    def add_recognition(self, row, input_crop=None):
        rid = row["record_id"]
        raw = row.get("raw_result", {}) or {}
        content = _raw_result(raw)
        content.update({key: row[key] for key in ("purpose", "native_reco_id", "source_node", "source_roi",
            "preprocessing", "source_locator_reco_id", "source_locator_record_id", "detected_text_roi", "padding") if key in row})
        coordinate_space = row.get("coordinate_space", "maa_client_1280x720")
        if input_crop is not None:
            coordinate_space = "preset_level_input_3x"
            content.update(source_coordinate_space="maa_client_1280x720",
                           source_box=input_crop["source_box"], input_scale=3,
                           source_frame_id=input_crop.get("source_frame_id"))
        elif coordinate_space != "maa_client_1280x720":
            content.update(source_coordinate_space=row.get("source_coordinate_space", "maa_client_1280x720"),
                           source_box=row.get("source_roi"), input_scale=row.get("input_scale"))
        self.readings[rid] = dict(reading_id=rid, run_id=self.run_id,
            original_record_id=row.get("_original_record_id", rid),
            timestamp=row.get("timestamp"), sequence=row.get("sequence"), node_name=row.get("node_name"),
            method=raw.get("algorithm", row.get("method")), read_id=row.get("read_id"),
            coordinate_space=coordinate_space, roi_json=_json(row.get("roi")),
            status="hit" if raw.get("hit") else "missing", content_json=_json(content))
        self.recognitions[rid] = row
        self.evidence_index[_json(_raw_result(raw))].add(rid)
        for key in ("read_id", "offer_id", "operation_id", "stock_scan_id", "event_id"):
            if row.get(key):
                self.read_groups[(key, row[key])].add(rid)

    def inline_reading(self, value, record, path):
        matching = self.evidence_index.get(_json(value), set())
        primary = record.get("recognition_record_id")
        own_readings = matching & _references(record)
        if record.get("read_id"):
            own_readings.update(matching & self.read_groups[("read_id", record["read_id"])])
        existing = (primary if primary in matching else
                    next(iter(own_readings)) if len(own_readings) == 1 else None)
        if existing is not None:
            self.link_reading(record["record_id"], existing, path)
            return existing
        rid = "inline:" + _hash([self.run_id, value])
        self.readings.setdefault(rid, dict(reading_id=rid, run_id=self.run_id, original_record_id=None,
            timestamp=None, sequence=None, node_name=None, method="embedded_UI_evidence", read_id=None,
            coordinate_space="maa_client_1280x720", roi_json="null", status="captured",
            content_json=_json(value)))
        self.link_reading(record["record_id"], rid, path)
        return rid

    def clean(self, value, record, path="", owned=False):
        if isinstance(value, list):
            return [self.clean(item, record, f"{path}[{index}]", owned=owned)
                    for index, item in enumerate(value)]
        if not isinstance(value, dict):
            return value
        result = {}
        policy = {}
        for key, item in value.items():
            child = f"{path}.{key}" if path else key
            if key in TRANSPORT_FIELDS:
                continue
            if key in ("original_target_vector", "original_target_pools", "original_preset"):
                continue
            if key in SNAPSHOTS:
                kind, basis = SNAPSHOTS[key]
                if kind == "owned" and value.get("owned_basis", "").startswith("bag_OCR"):
                    basis = "bag_OCR_and_confirmed_selection_history"
                content = _snapshot(item) if kind in ("owned", "maa_owned") else self.clean(item, record, child)
                oid = _hash([kind, basis, content])
                self.objects.setdefault(oid, dict(object_id=oid, kind=kind, basis=basis, content_json=_json(content)))
                self.object_links.add((record["record_id"], child, oid))
                result[key + "_object_id"] = oid
                continue
            if key in {"raw_ocr", "raw_result", "raw", "score_raw"}:
                evidence = _raw_result(item) if key == "raw_result" else item
                result[key + "_reading_id"] = self.inline_reading(evidence, record, child)
                continue
            if key in POLICY_FIELDS and ("potential_id" in value or "old_level" in value):
                policy[key] = item
                continue
            if key == "derived_concertos":
                result[key] = [dict(self.clean(entry, record, f"{child}[{index}]"), basis=entry.get("basis", "local_rules_derivation"))
                               for index, entry in enumerate(item)]
                continue
            result[key] = self.clean(item, record, child)
        if policy:
            # These are Maa policy fields; probability=0 is not a game probability.
            result["maa_policy"] = dict(basis="maa_policy_metadata_not_game_probability", **policy)
        return result

    def inspect_quality(self, row, value, path=""):
        if row["event_type"] == "wallet_recognition_attempt":
            return  # Its original status is evidence; the grouped wallet carries quality.
        if (row["event_type"] == "exploration_start" and not path
                and value.get("start_status") == "awaiting_entry" and value.get("start_confirmation_record_id")):
            return  # Entry confirmation resolves this provisional empty history.
        if not isinstance(value, (dict, list)):
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                self.inspect_quality(row, item, f"{path}[{index}]")
            return
        # Only quality attached to observations; model flags describe inference.
        if path and any(part in path for part in ("state", "decision", "tracker", "maa_owned_cache",
                                                   "owned[", "owned_before[", "final_owned[")):
            return
        if value.get("reason") in {"floor_hud_not_displayed", "floor_hud_obscured", "inventory_layout_not_present"}:
            return
        rid = row["record_id"]
        if not path and row["event_type"] in {"note_source_started", "note_delta"} and battle_floor_outside_route(value):
            self.issue(rid, "floor", "battle_floor_outside_route", "suspect", "已确认战斗楼层不属于正式路线的战斗节点。")
        for key, item in value.items():
            child = f"{path}.{key}" if path else key
            if key == "source_inferred" and item is True and value.get("source_attribution") == "confirmed_event_choice_and_following_reward_page":
                continue
            if key in {"identity_inferred", "source_inferred"} and item is True:
                self.issue(rid, child, key, "suspect", "身份或来源经过推断，未由独立实读确认。")
            elif key == "recognition_status" and item not in (None, "ok", "complete", "confirmed"):
                severity = "invalid" if item == "invalid" else "suspect" if item == "conflict" else "partial"
                self.issue(rid, child, "recognition_" + str(item), severity, "识别状态未达到完整确认。",
                           {"recognition_record_ids": sorted(_references(value))})
            elif key == "owned_issues" and item:
                incomplete = all(issue.get("reason") in {"run_start_not_observed", "selected_potential_unresolved",
                                                         "take_not_confirmed", "journal_record_missing"} for issue in item)
                self.issue(rid, child, "incomplete_owned_history" if incomplete else "observation_conflict",
                           "partial" if incomplete else "suspect", "持有历史不完整或存在冲突。", {"values": item})
            elif key == "recognition_errors" and item:
                missing_only = all(error in {"unresolved_identity", "name_input_incomplete", "level_not_read"}
                                   for error in item)
                self.issue(rid, child, "incomplete_observation" if missing_only else "observation_conflict",
                           "partial" if missing_only else "suspect", "潜能识别存在缺读或冲突。", {"values": item})
            elif key in {"conflicts", "inventory_conflicts",
                         "potential_counts_conflicts"} and item:
                self.issue(rid, child, "observation_conflict", "suspect", "保留了相互冲突或不完整的观测。", {"values": item})
            elif key in {"candidate_read_complete", "candidates_complete", "recognition_consistent",
                         "owned_complete", "inventory_complete", "supply_complete",
                         "potential_counts_complete", "shelf_generation_complete", "record_level_complete",
                         "selection_observation_complete", "ocr_complete"} and item is False:
                self.issue(rid, child, "incomplete_observation", "partial", "该观测的完整性未确认。")
            elif key == "complete" and item is False and not path and row["event_type"].startswith("note_"):
                self.issue(rid, child, "incomplete_note_observation", "partial", "音符观测或区间未完整。")
            elif key == "complete" and item is False and "observations" in path:
                self.issue(rid, child, "incomplete_observation", "partial", "该区域的识别完整性未确认。")
            elif key in {"missing", "unknown_slots", "incomplete", "ocr_missing"} and item:
                self.issue(rid, child, "missing_observation", "partial", "声明可见或跟踪范围内存在缺失。", {"values": item})
            self.inspect_quality(row, item, child)
        if row["event_type"] == "offer_observed" and not path:
            if value.get("source") in (None, "", "unknown"):
                self.issue(rid, "source", "unknown_offer_source", "partial", "独立候选页的语义来源未确认。")
        if row["event_type"] == "runtime_recovery" and not path:
            reason = value.get("reason")
            if reason == "wallet_delta_unattributed":
                self.issue(rid, "observed_coins", reason, "partial", "已按实读金币继续，先前金币变化的来源未确认。")
            elif reason == "wallet_not_observed":
                self.issue(rid, "observed_coins", reason, "partial", "当前金币未实读，路线按当前页面继续；余额保持未确认。")
            elif reason == "shop_observation_incomplete":
                self.issue(rid, "missing", reason, "partial", "商店关键字段未读完整，已按确认的商店导航离店继续。",
                           {key: value.get(key) for key in ("missing", "recovery_action", "continuation_basis")})
            elif reason == "route_observation_gap":
                self.issue(rid, "recovered_node", reason, "suspect", "路线状态曾与当前页面不符，已重新定位；恢复位置属于推断。",
                           {key: value.get(key) for key in ("previous_node", "recovered_node", "skipped_observation_count")})
            elif value.get("model_decision_recorded") is False and value.get("recovery_action") != "use_native_for_page":
                self.issue(rid, "recovery_action", reason or "native_fallback", "partial",
                           "识别信息未完整，已按记录的界面依据继续；本次没有模型决策。",
                           {key: value.get(key) for key in ("missing", "recovery_action", "continuation_basis")})
        if row["event_type"] == "note_delta" and not path:
            combined = planned_combined_interval(value)
            if len(value.get("event_components") or []) > 1 and (
                    value.get("source_type") != "combined_interval"
                    or any(value.get(key) is True for key in (
                        "independently_observed", "analysis_eligible", "probability_eligible"))):
                self.issue(rid, "event_components", "multi_event_interval_attribution", "suspect",
                           "连续事件的净变化属于综合区间，不能关联单个事件或进入独立统计。")
            if value.get("source_type") == "combined_interval" and not combined:
                self.issue(rid, "mixed_sources", "combined_interval_contract", "suspect",
                           "综合区间必须保留各个来源并明确排除独立收益分析。")
            if not combined and (value.get("floor") is None or value.get("floor_complete") is False):
                self.issue(rid, "floor", "unknown_interval_floor", "partial", "该音符增量的楼层未确认。")
            if not value.get("source_type") or value.get("source_type") == "unknown":
                self.issue(rid, "source_type", "unknown_interval_source", "partial", "该音符增量的独立来源未确认。")


def build_payload(run_path, session_path=None):
    """Build JSON-serializable rows for one explicitly selected run.

    A matching session is found by session_id and filtered by task_id. No other
    run, historical data directory, network service or trainer is accessed.
    """
    run_path = Path(run_path)
    run = _rows(run_path)
    if not run:
        raise ValueError("The selected run has no records")
    damage_path = run_path.with_suffix(DAMAGE_SUFFIX)
    journal_damage = json.loads(damage_path.read_text(encoding="utf-8")) if damage_path.exists() else None
    run_id = next((row.get("run_id") for row in run if row.get("run_id")), run_path.stem)
    builder = _Builder(run_id)
    session_tail = {}
    session = _session_rows(run_path, run, session_path, session_tail)
    session_ids = {row["record_id"]: f"session:{run_id}:{row['record_id']}" for row in session}
    for row in session:
        row["_original_record_id"] = row["record_id"]
    session = _session_record_ids(session, session_ids)
    # Keep the original ID separately; only database identities and references
    # are scoped to the run when the same session serves consecutive runs.
    for row in session:
        row["_original_record_id"] = row["record_id"].split(":", 2)[2]
    run = _session_record_ids(run, session_ids)
    all_rows = sorted(session + run, key=lambda row: row.get("sequence") or 0)
    _normalize_wallet_attempts(all_rows)
    _normalize_event_choices(run)
    start = next((row for row in run if row["event_type"] == "exploration_start"), run[0])
    start_confirmation = next((row for row in run if row["event_type"] == "exploration_start_confirmed"
                               and row.get("full_run_start_observed") is True), None)
    full_start = bool(start.get("full_run_start_observed") or start_confirmation)
    end = next((row for row in reversed(run) if row["event_type"] == "exploration_end"), None)
    binding = next((row for row in reversed(run) if row["event_type"] == "preset_observed"
                    and row.get("original_preset")), None)
    original = binding["original_preset"] if binding else next((row.get("original_preset") for row in run + session if row.get("original_preset")), None)
    vector = binding.get("target_vector") if binding and binding.get("target_vector_complete") else next(
        (row.get("target_vector") for row in run + session if row.get("target_vector_complete")), None)
    preset_content = deepcopy(original) if original is not None else {}
    preset_content.pop("pools", None)
    static_source_keys = {"target_authority", "name_and_id_catalog", "catalog_provenance", "catalog_language_support", "method"}
    original_source = preset_content.pop("source", {})
    preset_content["source"] = {key: value for key, value in original_source.items() if key in static_source_keys}
    preset_evidence = {key: value for key, value in original_source.items() if key not in static_source_keys}
    for key in ("game_display_role_totals", "game_display_total", "initial_inventory", "bag_party"):
        if key in preset_content:
            preset_evidence[key] = preset_content.pop(key)
    preset_key = _hash([preset_content, vector])
    preset_evidence_record = binding["record_id"] if binding else next((row["record_id"] for row in session if row["event_type"] == "preset_observed"), start["record_id"])
    builder.tables["presets"].append(dict(preset_key=preset_key,
        source_preset_id=binding.get("preset_id") if binding else start.get("preset_id"), target_vector_json=_json(vector),
        target_vector_complete=vector is not None, content_json=_json(preset_content)))

    task_end = next((row for row in reversed(session) if row["event_type"] == "task_end"), None)
    completed = bool(end and end.get("completion_observed"))
    lifecycle = "completed" if completed else "interrupted" if end or task_end else "open"
    termination = end.get("reason") if end else ("task_" + str(task_end.get("status")) if task_end else None)
    configs = {}
    for row in run:
        params = (row.get("state") or {}).get("params") if isinstance(row.get("state"), dict) else None
        if params:
            configs[_hash(params)] = params
    builder.tables["runs"].append(dict(run_id=run_id, participant_id=start.get("participant_id"),
        session_id=start.get("session_id"), task_id=start.get("task_id"), preset_key=preset_key,
        language=start.get("lang_type"), started_at=start.get("timestamp"),
        ended_at=(end or task_end or {}).get("timestamp"), lifecycle_status=lifecycle,
        termination_reason=termination,
        termination_basis=("local_log_recovery" if end and end.get("boundary_recovered") else
                           "exploration_end" if end else "task_end" if task_end else "no_end_observed"),
        full_run_start_observed=full_start,
        completion_observed=completed, quality_status="valid",
        collector_version=start.get("collector_version"), source_schema_version=start.get("schema_version"),
        runtime_versions_json=_json(start.get("runtime_versions", {})),
        model_versions_json=_json((binding or {}).get("model_versions") or start.get("model_versions", {})),
        config_json=_json(configs), observation_kind="local_game_UI_not_teacher_labels"))

    semantic = []
    for row in all_rows:
        event = row["event_type"]
        if event in ("context_run_recognition", "context_run_task", "collector_loaded", "preset_screenshot"):
            continue
        if event == "lifecycle_observed" and row.get("lifecycle_type") == "pipeline_node_completed":
            continue
        semantic.append(row)
    referenced = set().union(*(_references(row) for row in semantic)) if semantic else set()
    last_preset_crop = None
    for row in all_rows:
        if row["event_type"] == "preset_level_crop":
            last_preset_crop = row
        if row["event_type"] == "context_run_recognition":
            raw = row.get("raw_result") or {}
            if (raw.get("algorithm") == "OCR" or row["record_id"] in referenced
                    or (row in session and raw.get("algorithm") == "ColorMatch")):
                args = row.get("arguments") or []
                override = row.get("keyword_arguments", {}).get("pipeline_override", args[1] if len(args) > 1 else {})
                params = (override.get(row.get("node_name"), {}).get("recognition", {}).get("param", {})
                          if isinstance(override, dict) else {})
                input_crop = last_preset_crop if row.get("node_name", "").endswith("预设OCR") and params.get("only_rec") else None
                builder.add_recognition(row, input_crop)
                if input_crop:
                    builder.link_reading(input_crop["record_id"], row["record_id"], "preset_level_OCR")

    wallets = [row for row in run if row["event_type"] == "wallet_observed"]
    attempts = {row.get("operation_id"): row for row in run if row["event_type"] == "purchase_attempt"}
    action_attempts = {}
    results = defaultdict(list)
    for row in run:
        if row["event_type"] == "wallet_recognition_attempt":
            continue
        if row["event_type"].endswith("_attempt") and row.get("operation_id"):
            action_attempts[row["operation_id"]] = row
        elif row.get("operation_id"):
            results[row["operation_id"]].append(row)

    normalized = {}
    for source in semantic:
        row = _normalize_floor(source)
        event, rid = row["event_type"], row["record_id"]
        if event == "purchase_result":
            attempt = attempts.get(row.get("operation_id"))
            before_seq = (attempt or row).get("sequence", 0)
            end_seq = row.get("sequence", 0)
            before = next((wallet for wallet in reversed(wallets) if wallet.get("sequence", 0) < before_seq
                           and wallet.get("coin_complete") and wallet.get("record_id")), None)
            observed = [wallet for wallet in wallets if before_seq < wallet.get("sequence", 0) < end_seq]
            pipeline = observed[-1] if observed else None
            evidence = purchase_wallet_evidence(row.get("operation_id"), (row.get("item") or {}).get("price"),
                observed, before_wallet=before, pipeline_wallet=pipeline)
            row["pipeline_reported_success"] = row.get("pipeline_succeeded", row.get("success"))
            row.update(evidence)
            row["success"] = row["debit_confirmed"]
            if row["confirmation_status"] != "confirmed":
                builder.issue(rid, "coins_after", "purchase_" + row["confirmation_status"],
                    "partial" if row["confirmation_status"] == "wallet_missing" else "suspect",
                    "购买首后钱包差额未确认；未使用价格反推余额。",
                    {key: value for key, value in evidence.items() if key.endswith("record_id") or key.endswith("observation_id")})
        if event in ("event_menu_observed", "event_menu_reobserved", "event_choice", "event_choice_click_observed"):
            row["event_identity_basis"] = "local_menu_correlation"
            row["game_event_id"] = None
            row["game_option_id"] = None
            if "model_mapping" in row:
                row["model_mapping_basis"] = "model_inference_not_game_server_identity"
        if event == "settlement_observed":
            previous_complete = row.get("potential_counts_complete")
            conflicts = list(row.get("potential_counts_conflicts") or [])
            checked = check_potential_counts(row.get("potential_counts") or [], row)
            for conflict in checked["potential_counts_conflicts"]:
                if conflict not in conflicts:
                    conflicts.append(conflict)
            row.update(checked)
            row["potential_counts_conflicts"] = conflicts
            row["potential_counts_complete"] &= previous_complete is not False and not conflicts
            row["potential_counts_order"] = "ui_position_order"
            row["potential_count_roles"] = None
        if event.startswith("note_"):
            row["completeness_scope"] = ("page_rule_scope" if "scope_melodies" in row else
                                         "tracked_visible_melodies_not_all_elements")
        if row.get("source_attribution") == "confirmed_event_choice_and_following_reward_page":
            row["source_basis"] = "confirmed_history_reconstruction"
        basis = ("local_log_recovery" if row.get("observer") == "local_log_recovery" else
                 "maa_policy_decision" if row.get("policy_mode") == "native_fallback" and event in ("policy_decision", "fallback_policy_decision") else
                 "model_inference" if event in ("model_decision", "policy_decision") else
                 "maa_cache_unverified" if event == "selection_cache" else
                 "inventory_difference_derivation" if event == "note_delta" else "game_UI_observation")
        for key in ("original_preset", "model_versions", "runtime_versions"):
            row.pop(key, None)
        payload = {key: item for key, item in row.items() if key not in BASE_FIELDS}
        if event == "note_setup_observed":
            payload.pop("raw", None)  # repeated main_discs[].raw and readings[].raw
        if event == "note_inventory_observed" and isinstance(payload.get("observation"), dict):
            if payload["observation"].get("counts") == payload.get("counts"):
                payload["observation"].pop("counts", None)
        if event == "shop_observed":
            shop_payload = payload.pop("payload", {})
            payload["slots"] = shop_payload.get("slots", [])
            payload.pop("stock_snapshot", None)  # same shelf items already live in slots
        if event in ("exploration_start", "preset_observed"):
            payload["preset_key"] = preset_key
        if event == "exploration_start" and start_confirmation:
            payload["start_confirmation_record_id"] = start_confirmation["record_id"]
        if rid == preset_evidence_record and preset_evidence:
            payload["preset_read_evidence"] = preset_evidence
        if isinstance(payload.get("decision"), dict):
            payload["decision"]["basis"] = "model_inference"
        # Parameters are run configuration, not per-page observed state.
        for field in ("state", "previous_state"):
            if isinstance(payload.get(field), dict) and "params" in payload[field]:
                params = payload[field].pop("params")
                payload[field]["run_config_id"] = _hash(params)
        builder.inspect_quality(row, payload)
        if row.get("journal_gap"):
            builder.issue(rid, "journal_gap", "journal_record_missing", "partial",
                "本地写入失败后仍未补写的记录形成缺口；此前独立确认的前缀保留。",
                dict(row["journal_gap"], affected_through_sequence=run[-1].get("sequence")))
        normalized[rid] = row
        clean = builder.clean(payload, row)
        builder.tables["records"].append(dict(record_id=rid, original_record_id=row.get("_original_record_id", rid), run_id=run_id,
            origin="run" if source.get("run_id") == run_id else "session",
            sequence=source.get("sequence"), run_sequence=source.get("run_sequence"),
            timestamp=source.get("timestamp"), event_type=event, basis=basis,
            quality_status="valid", payload_json=_json(clean)))
        linked = _references(row)
        # A read_id identifies one acquisition. Event/operation IDs can cover
        # many later pages and must not attach their entire OCR history here.
        for key in ("read_id",):
            if row.get(key):
                linked.update(builder.read_groups.get((key, row[key]), set()))
        if event in ("event_menu_observed", "event_menu_reobserved"):
            # Event OCR precedes menu emission and is linked by the menu ID.
            linked.update(builder.read_groups.get(("event_id", row.get("event_id")), set()))
        for reading_id in linked & builder.readings.keys():
            builder.link_reading(rid, reading_id, "recognition_evidence")
        if event == "preset_observed":
            for reading_id, reading in builder.recognitions.items():
                if reading.get("task_id") == source.get("task_id") and reading in session:
                    builder.link_reading(rid, reading_id, "preset_OCR_and_card_boundaries")

    record_ids = {row["record_id"] for row in builder.tables["records"]}
    offer_ids = {row.get("offer_id") for row in run if row["event_type"] == "offer_observed"}
    for row, reused in reused_interval_components(normalized.values()):
        builder.issue(row["record_id"], "mixed_sources", "combined_source_already_attributed", "suspect",
            "联合区间重复使用已归入更早区间的来源。", {"reused_sources": reused})

    for operation, attempt in action_attempts.items():
        outcomes = results.get(operation, [])
        result_event = ("purchase_result" if attempt["event_type"] == "purchase_attempt" else
                        attempt["event_type"].removesuffix("_attempt"))
        actual = [row for row in outcomes if row["event_type"] == result_event]
        if not actual:
            builder.issue(attempt["record_id"], "operation_id", "unconfirmed_action_tail", "partial",
                "已记录动作尝试，尚未记录对应结果；此前已确认观测保留。",
                {"operation_id": operation, "affected_from_sequence": attempt.get("sequence"),
                 "affected_through_sequence": run[-1].get("sequence")})
        else:
            result = normalized.get(actual[-1]["record_id"], actual[-1])
            if result_event == "selection_result":
                confirmed = any(row["event_type"] == "selection_ui_transition_observed"
                                and row.get("operation_confirmed") is True for row in outcomes)
            elif "operation_confirmed" in result:
                confirmed = result["operation_confirmed"] is True
            elif result_event == "refresh_result":
                confirmed = result.get("next_offer_id") in offer_ids
            elif result_event == "enhancement_result":
                confirmed = any(row["event_type"] == "offer_observed" and row.get("origin_operation_id") == operation
                                and row.get("source") == "enhance" for row in run)
            else:
                confirmed = False
            if not confirmed:
                builder.issue(result["record_id"], "operation_confirmed", "action_not_confirmed", "partial",
                    "动作管线已有返回，尚无关联的游戏界面或钱包确认。",
                    {"operation_id": operation, "affected_from_sequence": attempt.get("sequence"),
                     "affected_through_sequence": run[-1].get("sequence")})
    for row in run:
        if row["event_type"] == "event_choice" and row["selection_status"] == "awaiting_click":
            builder.issue(row["record_id"], "operation_confirmed", "event_click_not_observed", "partial",
                "已记录事件选择，尚无对应的实际点击确认。",
                {"event_id": row.get("event_id"), "affected_from_sequence": row.get("sequence"),
                 "affected_through_sequence": run[-1].get("sequence")})
    previous = None
    known_gaps = [row["journal_gap"] for row in run if row.get("journal_gap")]
    known_gaps.extend((journal_damage or {}).get("fragments", []))
    for row in run:
        seq = row.get("run_sequence")
        if (seq is not None and previous is not None and seq != previous + 1
                and not any(previous + 1 >= gap["affected_from_run_sequence"]
                            and seq - 1 <= gap.get("affected_through_missing_run_sequence",
                                                   (run[-1].get("run_sequence") or seq) - 1)
                            for gap in known_gaps)):
            following = next((item for item in semantic if (item.get("sequence") or 0) >= (row.get("sequence") or 0)), None)
            builder.issue(following["record_id"] if following else None, "run_sequence", "sequence_gap", "suspect",
                          "对局记录序号不连续，缺失后的持有历史与增量归属可能受影响。",
                          {"previous": previous, "current": seq, "affected_from_sequence": row.get("sequence"),
                           "affected_through_sequence": run[-1].get("sequence")})
        if seq is not None:
            previous = seq
    if not full_start:
        builder.issue(None, "full_run_start_observed", "start_not_observed", "partial", "未观察到完整开局边界。")
    if lifecycle != "completed":
        builder.issue(None, "lifecycle_status", "run_" + lifecycle, "partial",
                      "对局未观察到完整结算；已确认前缀仍可按用途查询。", {"termination_reason": termination})
    if vector is None:
        builder.issue(None, "preset.target_vector", "original_targets_missing", "partial", "完整原始目标向量缺失。")
    if journal_damage:
        builder.issue(None, "records.journal", "journal_damage", "partial",
            "恢复时已隔离无法解码的物理行；未猜测坏行内的内容，后续独立记录保留。",
            dict(journal_damage,
                 affected_from_sequence=min(part["affected_from_sequence"] for part in journal_damage["fragments"]),
                 affected_through_sequence=run[-1].get("sequence")))
    if session_tail:
        builder.issue(None, "session.journal", "truncated_session_tail", "partial",
            "共享会话末条记录未写完；已保留可读记录，未推断缺失内容或所属任务。", session_tail)
    interrupted_tail = run_path.with_suffix(".interrupted-tail")
    if interrupted_tail.exists():
        builder.issue(end["record_id"] if end else None, "records.tail", "truncated_record_tail", "partial",
                      "本地恢复已移出未写完的末条记录；其字节不进入结构化观测。",
                      {"tail_byte_count": interrupted_tail.stat().st_size,
                       "affected_from_sequence": (end or run[-1]).get("sequence"),
                       "affected_through_sequence": run[-1].get("sequence")})
    for record in builder.tables["records"]:
        record["quality_status"] = builder.record_quality[record["record_id"]]
    builder.tables["runs"][0]["quality_status"] = max(builder.record_quality.values(), key=QUALITY_RANK.get, default="valid")
    builder.tables["objects"] = sorted(builder.objects.values(), key=lambda row: row["object_id"])
    builder.tables["record_objects"] = [dict(record_id=rid, field_path=path, object_id=oid)
        for rid, path, oid in sorted(builder.object_links)]
    builder.tables["readings"] = sorted(builder.readings.values(), key=lambda row: row["reading_id"])
    builder.tables["record_readings"] = [dict(record_id=rid, reading_id=reading, field_path=path)
        for rid, reading, path in sorted(builder.reading_links)]
    builder.tables["quality_issues"] = sorted(builder.issues.values(), key=lambda row: row["issue_id"])
    return dict(format=FORMAT, schema_version=SCHEMA_VERSION, tables=builder.tables)


def import_payload(database_path, payload):
    """Idempotently insert the same rows used by transport, returning table totals."""
    if payload.get("format") != FORMAT or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported state-space database payload")
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection, connection:
        # Refresh views and replace the run in one commit, including on old databases.
        connection.executescript("PRAGMA foreign_keys=ON;\nBEGIN;\n" +
            Path(__file__).with_name("research_schema.sql").read_text(encoding="utf-8"))
        for run in payload["tables"]["runs"]:
            connection.execute("DELETE FROM quality_issues WHERE run_id=?", (run["run_id"],))
            for table in ("record_readings", "record_objects"):
                connection.execute(f'DELETE FROM "{table}" WHERE record_id IN (SELECT record_id FROM records WHERE run_id=?)',
                                   (run["run_id"],))
            connection.execute("DELETE FROM readings WHERE run_id=?", (run["run_id"],))
            connection.execute("DELETE FROM records WHERE run_id=?", (run["run_id"],))
        for table in TABLES:
            rows = payload["tables"].get(table, [])
            if not rows:
                continue
            columns = tuple(rows[0])
            names = ",".join('"' + column + '"' for column in columns)
            values = ",".join("?" for _ in columns)
            primary = [column[1] for column in connection.execute(f'PRAGMA table_info("{table}")') if column[5]]
            target = ",".join('"' + column + '"' for column in primary)
            updates = ",".join(f'"{column}"=excluded."{column}"' for column in columns if column not in primary)
            action = "DO UPDATE SET " + updates if updates else "DO NOTHING"
            # Updating in place avoids deleting a parent referenced by existing
            # rows when the same run is imported again.
            connection.executemany(f'INSERT INTO "{table}" ({names}) VALUES ({values}) ON CONFLICT({target}) {action}',
                                   ([row[column] for column in columns] for row in rows))
        connection.execute("DELETE FROM objects WHERE NOT EXISTS (SELECT 1 FROM record_objects r WHERE r.object_id=objects.object_id)")
        connection.execute("DELETE FROM presets WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.preset_key=presets.preset_key)")
        return {table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in TABLES}


def export_payload(database_path, run_id):
    """Read one run and its referenced normalized rows, without collector logs."""
    database_path = Path(database_path).resolve()
    with closing(sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        if not connection.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone():
            raise ValueError(f"Unknown state-space run: {run_id}")
        queries = {
            "presets": "SELECT p.* FROM presets p JOIN runs r USING(preset_key) WHERE r.run_id=? ORDER BY p.preset_key",
            "runs": "SELECT * FROM runs WHERE run_id=?",
            "records": "SELECT * FROM records WHERE run_id=? ORDER BY sequence,record_id",
            "objects": "SELECT DISTINCT o.* FROM objects o JOIN record_objects l USING(object_id) JOIN records r USING(record_id) WHERE r.run_id=? ORDER BY o.object_id",
            "record_objects": "SELECT l.* FROM record_objects l JOIN records r USING(record_id) WHERE r.run_id=? ORDER BY l.record_id,l.field_path,l.object_id",
            "readings": "SELECT * FROM readings WHERE run_id=? ORDER BY reading_id",
            "record_readings": "SELECT l.* FROM record_readings l JOIN records r USING(record_id) WHERE r.run_id=? ORDER BY l.record_id,l.reading_id,l.field_path",
            "quality_issues": "SELECT * FROM quality_issues WHERE run_id=? ORDER BY issue_id",
        }
        tables = {table: [dict(row) for row in connection.execute(queries[table], (run_id,))]
                  for table in TABLES}
    return dict(format=FORMAT, schema_version=SCHEMA_VERSION, tables=tables)


def write_payload(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json(payload).encode("utf-8")
    path.write_bytes(gzip.compress(encoded, mtime=0) if path.suffix == ".gz" else encoded)
    return path


def read_payload(path):
    path = Path(path)
    encoded = path.read_bytes()
    return json.loads(gzip.decompress(encoded) if path.suffix == ".gz" else encoded)
