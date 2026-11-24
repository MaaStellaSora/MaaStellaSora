"""State-space observations as database rows, shared by local import and upload.

Game readings and confirmed actions retain their provenance. Runtime calls,
recognition geometry, policy predictions and reconstructed caches are omitted.
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
from .language import catalog_name_fields
from .settlement_counts import check_potential_counts
from .sampling_evidence import planned_combined_interval, reused_interval_components, battle_floor_outside_route
from .sampling_journal import DAMAGE_SUFFIX


SCHEMA_VERSION = 2
FORMAT = "maa-state-space-database"
TABLES = ("presets", "runs", "records", "quality_issues")
BASE_FIELDS = {"schema_version", "collector_version", "event_type", "record_type", "timestamp",
               "record_id", "participant_id", "session_id", "task_id", "run_id", "exploration_id",
               "sequence", "run_sequence", "lang_type", "observer", "server_confirmed",
               "preset_id", "target_vector_complete", "target_vector", "_original_record_id"}
TRANSPORT_FIELDS = {"arguments", "keyword_arguments", "raw_detail", "reco_id", "maa_task_id",
                    "maa_node_id", "image_path", "source_image_path"}
QUALITY_RANK = {"valid": 0, "partial": 1, "suspect": 2, "invalid": 3}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


_DISCARD_EVENTS = {
    "context_run_recognition", "context_run_task", "collector_loaded", "preset_screenshot",
    "preset_level_crop", "model_decision", "policy_decision", "fallback_policy_decision",
    "selection_cache", "task_start", "task_end", "exploration_start_attempt",
}
_DISCARD_FIELDS = (BASE_FIELDS - {"record_id", "run_id", "sequence"}) | TRANSPORT_FIELDS | {
    "legacy_identity", "metadata", "_identity", "_legacy", "pc", "policy_mode", "maa_policy",
    "model_mapping", "model_mapping_basis", "model_decision", "model_decision_recorded",
    "model_versions", "runtime_versions", "versions", "config", "params", "decision",
    "tracker_estimate", "tracker_public_observation", "counters", "public_state", "pools",
    "state", "previous_state", "observed_state_after", "maa_owned_cache", "maa_owned_complete",
    "owned_counts", "owned_revision", "owned_independently_verified", "owned_source",
    "raw", "raw_ocr", "raw_result", "raw_detail", "raw_values", "score_raw", "raw_rarity",
    "raw_clear_text", "raw_difficulty_text", "source_raw", "raw_increment", "server_raw_increment",
    "recognition_record_id", "recognition_record_ids", "native_reco_id", "source_locator_record_id",
    "source_locator_reco_id", "source_frame_id", "source_image_path", "image_path", "source_document",
    "operation_document", "source_sample_id", "source_preset_id", "original_preset_source",
    "source_documents", "original_record", "payload_json", "content_json", "path", "checkpoint", "checkpoint_sha256",
    "machine_id", "client_id", "account_id", "user_id", "username", "computer_name", "hostname", "device_id", "pid",
    "source_database", "database", "dataset_id", "source_dataset_id", "source_session_id",
    "source_task_id", "source_participant_id", "teacher_label_available", "training_eligible",
    "training_exclusion_reason", "training_exclusion_reasons", "training_scope_match",
    "training_scope_basis", "training_scope_version", "coordinate_space", "source_coordinate_space",
    "frame", "preprocessing", "padding", "input_scale", "arguments", "keyword_arguments",
    "node_name", "source_node", "capture_node", "verification_node", "off_verification_node",
    "on_verification_node", "resume_action", "message", "icon_quality", "icon_score",
    "white_level", "origin_x", "row_y", "spacing", "detail", "score", "similarity",
    "identity_similarity", "identity_name", "ocr_text", "text_score", "name_score", "confidence",
    "derived_concertos", "game_event_id", "game_option_id", "read_id", "stock_read_id",
    "stock_recognition_record_ids", "pipeline_wallet_recognition_record_id",
    "wallet_before_recognition_record_id", "wallet_recognition_record_id", "attempt_record_ids",
    "original_target_vector", "original_target_pools", "original_preset", "original_record_id",
    "resumed_from_decision_index", "confirmed_at", "observed_at", "before_observed_at",
    "after_observed_at", "start_timestamp", "time", "timestamp", "reco_id", "maa_node_id", "maa_task_id",
}
_BOUNDARY_FIELDS = set("""full_run_start_observed completion_observed reason start_status
    start_confirmation_record_id start_attempt_record_id resumed boundary_basis boundary_evidence
    boundary_payload start_basis boundary_recovered entry_page continuation_basis confirmation_basis
    history_estimated pending_operation pending_selection owned final_owned owned_complete owned_basis
    owned_issues owned_anonymous last_wallet_observation last_coin_observation last_floor outcome
    record_level record_level_complete potential_counts potential_counts_complete potential_counts_conflicts
    potential_counts_order potential_count_roles potential_count observed_counts settlement end_reason
    physical_game_id segment_index segment_count""".split())
_OFFER_FIELDS = set("""offer_id reward_chain_id offer_index refresh_index same_page_read_index
    source source_type source_attribution source_inferred source_basis origin_operation_id
    refresh_operation_id refresh_parent_offer_id preceding_event_id preceding_event_relation
    floor floor_basis floor_observation_id floor_candidates last_observed_floor_context coin
    core_potential candidates candidates_complete candidate_read_complete recognition_status
    recognition_consistent recognition_errors inventory_conflicts owned_complete owned_basis owned_issues
    main_trekker_index trekker_index_basis level_basis level_upped page_generation_basis refresh_price
    pool_identity_observation visible_shared_bonus""".split())
_ACTION_FIELDS = set("""operation_id offer_id next_offer_id reward_chain_id stock_scan_id next_stock_scan_id
    shop_visit_id shelf_generation_id item item_instance_id slot source source_type source_attribution
    source_inferred source_basis phase selected selected_slot selected_ui_index planned_name
    selection_confirmed selection_basis selection_observation_complete selected_candidate_index operation_confirmed
    confirmation_basis confirmation_status debit_confirmed success pipeline_succeeded reward_committed result coin_before coin_after
    coins coins_before coins_after observed_coins purchase_debit pipeline_coins_after
    pipeline_wallet_observation_id wallet_before_observation_id wallet_observation_id
    wallet_observation_complete refresh_price refresh_remaining_before refresh_remaining_after
    refresh_remaining_decreased shelf_changed stock_read_complete owned_complete owned_basis owned_issues
    owned_before_complete selection_status superseded_by_record_id click_record_id
    source_record_id confirmation_record_id affected_from_sequence affected_through_sequence""".split())
_WALLET_FIELDS = set("""coin coin_complete parsed_values values recognition_status conflicts read_kind
    wallet_observation_id wallet_before_observation_id before_wallet_observation_id after_wallet_observation_id
    previous_wallet_observation_id retry_of_wallet_observation_id resolved_by_wallet_record_id
    floor floor_complete floor_observation_id preceding_battle_id preceding_event_id offer_id operation_id
    reward_chain_id stock_scan_id shop_visit_id source_page source_type capture_timing complete probe_only
    probe_outcome coins_before coins_after delta source_type candidate_components component_amounts_basis
    event_choice_record_id event_click_record_id event_id interval_basis independently_observed
    analysis_eligible probability_eligible analysis_exclusion_reason""".split())
_SHOP_FIELDS = set("""coin wallet_complete slots refresh_price refresh_remaining phase stock_scan_id
    shop_visit_id shop_entry_id shelf_generation_id previous_shelf_generation_id stock_generation_id
    shelf_generation_basis shelf_generation_complete repeated_scan shop_event_type invalid_fields
    recognition_errors owned_complete owned_basis owned_issues operation_id missing
    refresh_remaining_observed refresh_remaining_basis""".split())
_EVENT_FIELDS = set("""event_id event_identity_basis choices question selected_text displayed_consequence
    selected_ui_index event_choice_record_id operation_confirmed selection_status superseded_by_record_id
    click_record_id floor floor_complete floor_confirmed floor_candidates full_visible_menu ocr_missing evidence
    owned_complete owned_basis owned_issues event_mapping""".split())
_OBSERVATION_FIELDS = {"observation_alternatives", "observed_state", "physical_source",
                       "physical_source_evidence", "source_record_id", "journal_gap", "unresolved_values",
                       "decision_observations"}
_OWNED_EVENTS = {"exploration_start", "exploration_start_confirmed", "exploration_end",
                 "run_start", "run_end", "bag_inventory_observed", "settlement_observed"}


def _reading_adds_information(reading, counts):
    return (reading.get("conflicting_values") or reading.get("conflicts") or reading.get("complete") is False
            or reading.get("recognition_status") not in {None, "ok", "confirmed", "complete"}
            or reading.get("current") is None or reading.get("current") != counts.get(reading.get("melody")))


def _compact_value(value, event_type="", *, potential=False):
    """Keep game meaning; IDs for OCR calls, geometry and policy data have no role here."""
    if isinstance(value, list):
        return [_compact_value(item, event_type, potential=potential) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if (key in _DISCARD_FIELDS or key.endswith(("_roi", "_box", "_boxes", "_reading_id", "_reading_ids", "_object_id"))
                or key in {"box", "roi", "regions", "level_regions", "crop", "source_box", "selected_box"}
                or key.startswith(("model_", "evaluation_", "training_"))):
            if key == "score" and event_type in {"settlement_observed", "note_settlement_observed"}:
                result[key] = item
            continue
        result[key] = _compact_value(item, event_type,
            potential=key in {"selected", "candidates", "owned", "owned_before", "final_owned", "targets", "recognition_catalog"})
    if "potential_id" in value or potential:
        for key in ("rank", "sub_rank", "probability", "recommended", "recommended_level", "in_preset",
                    "in_owned", "target_level", "initial_level", "is_new", "selected"):
            result.pop(key, None)
        if "canonical_rarity" in result:
            result["rarity"] = result.pop("canonical_rarity")
        for canonical, aliases in (("level_before", ("old_level",)),
                                    ("level_after", ("new_level", "offered_level_after"))):
            for alias in aliases:
                if canonical not in result and alias in result:
                    result[canonical] = result[alias]
                result.pop(alias, None)
        if "slot" in result:
            result.pop("index", None)
        elif "index" in result:
            result["slot"] = result.pop("index")
        if value.get("potential_id") is not None and value.get("role") is not None:
            if len(value.get("id_candidates") or []) <= 1 and not value.get("recognition_errors"):
                result.pop("name", None)
            result.pop("trekker", None)
        if value.get("potential_id") is not None:
            for key in catalog_name_fields():
                result.pop(key, None)
        if result.get("rarity") == "core":
            result.pop("core", None)
    # The inventory map already gives all successful reads. Keep only extra
    # quality/alternative information from the embedded acquisition object.
    if isinstance(result.get("observation"), dict):
        observation = result["observation"]
        if observation.get("counts") == result.get("counts"):
            observation.pop("counts", None)
        if observation.get("complete") == result.get("complete"):
            observation.pop("complete", None)
        if isinstance(result.get("counts"), dict):
            readings = observation.get("readings")
            if isinstance(readings, list):
                observation["readings"] = [reading for reading in readings
                    if _reading_adds_information(reading, result["counts"])]
        if not observation:
            result.pop("observation")
    counts = result.get("counts")
    if not isinstance(counts, dict):
        counts = result.get("starting_supply")
    readings = result.get("readings")
    if isinstance(counts, dict) and isinstance(readings, list):
        result["readings"] = [reading for reading in readings
            if _reading_adds_information(reading, counts)]
        if not result["readings"]:
            result.pop("readings")
    if result.get("final_owned") is not None and result.get("owned") == result["final_owned"]:
        result.pop("owned")
    if result.get("supply_complete") and isinstance(result.get("skipped_regions"), list):
        result["skipped_regions"] = [region for region in result["skipped_regions"]
            if region.get("reason") != "outside_supply_column_without_number"]
        if not result["skipped_regions"]:
            result.pop("skipped_regions")
    for disc in result.get("main_discs", []):
        if disc.get("id") is not None and disc.get("status") == "confirmed":
            for key in ("text", "name", "name_ja", "name_zh", *catalog_name_fields()):
                disc.pop(key, None)
    return result


def compact_record(event_type, payload):
    """Project one semantic observation/action; return None for collector/policy bookkeeping."""
    if event_type in _DISCARD_EVENTS:
        return None
    if event_type in {"note_bag_inventory_read", "note_detail_read"} and payload.get("complete") and payload.get("bag_closed"):
        return None
    payload = deepcopy(payload)
    if event_type == "lifecycle_observed":
        if payload.get("lifecycle_type") in {"pipeline_node_completed", "task_started", "task_ended"}:
            return None
    if event_type in {"shop_observed", "shop_reobserved"}:
        nested = payload.pop("payload", {})
        if isinstance(nested, dict):
            for key in ("slots", "coin", "refresh_price", "refresh_remaining", "invalid_fields", "recognition_errors"):
                if key not in payload and key in nested:
                    payload[key] = nested[key]
        payload.pop("stock_snapshot", None)
        fields = _SHOP_FIELDS
    elif event_type in {"offer_observed", "offer_reobserved"}:
        fields = _OFFER_FIELDS
    elif event_type in {"event_menu_observed", "event_menu_reobserved", "event_choice", "event_choice_click_observed"}:
        fields = _EVENT_FIELDS
        mapping = payload.get("model_mapping") or {}
        if "selected_ui_index" not in payload and mapping.get("selected_ui_index") is not None:
            payload["selected_ui_index"] = mapping["selected_ui_index"]
    elif event_type.startswith("wallet_") or event_type == "field_wallet_observed":
        fields = _WALLET_FIELDS
        if payload.get("parsed_values"):
            payload.pop("values", None)
    elif event_type in _OWNED_EVENTS:
        fields = _BOUNDARY_FIELDS
        if event_type == "bag_inventory_observed":
            fields |= {"original_bag_party", "source", "complete"}
    elif event_type.endswith(("_attempt", "_result")) or event_type in {"selection_ui_transition_observed", "selection_attempt"}:
        fields = _ACTION_FIELDS
    elif event_type == "preset_observed":
        fields = {"preset_read_evidence", "source"}
    else:
        fields = None  # Notes/manual evidence carry task-specific game observations.
    if fields is not None:
        payload = {key: item for key, item in payload.items() if key in fields | _OBSERVATION_FIELDS}
    if event_type not in _OWNED_EVENTS:
        for field in ("owned", "owned_before", "final_owned", "maa_owned_cache"):
            payload.pop(field, None)
    result = _compact_value(payload, event_type)
    if event_type == "preset_observed" and isinstance(result.get("preset_read_evidence"), dict):
        evidence = result["preset_read_evidence"]
        for key in ("name_matches", "frames_read", "displayed_role_total_sources"):
            evidence.pop(key, None)
        readings = evidence.pop("displayed_role_total_readings", {})
        alternatives = {role: sorted({item.get("value") for item in values if item.get("value") is not None})
                        for role, values in readings.items()}
        alternatives = {role: values for role, values in alternatives.items() if len(values) > 1}
        if alternatives:
            evidence["role_total_alternatives"] = alternatives
    if event_type == "note_inventory_observed":
        result.pop("bar_evidence", None)
    return result


def compact_preset(original):
    """Keep the party and each target's identity/level, without the OCR/catalog copies."""
    content = {}
    name_fields = catalog_name_fields()
    for key in ("party", "recognition_catalog"):
        if key in original:
            content[key] = _compact_value(original[key])
    for party, original_party in zip(content.get("party", []), original.get("party", [])):
        for key in tuple(party):
            if key not in {"role", "character_id", "character_id_namespace", "local_character_identity",
                           "local_potential_catalogue", "targets", "catalogue_order",
                           "name", *name_fields}:
                party.pop(key)
        for target, original_target in zip(party.get("targets", []), original_party.get("targets", [])):
            if "target_level" in original_target:
                target["target_level"] = original_target["target_level"]
        if party.get("character_id") is not None:
            for key in ("name", *name_fields, "local_character_identity"):
                party.pop(key, None)
    for target, original_target in zip(content.get("recognition_catalog", []), original.get("recognition_catalog", [])):
        if "target_level" in original_target:
            target["target_level"] = original_target["target_level"]
    return content


def compact_payload(payload):
    """Project the native format or its explicitly marked v3 transport envelope."""
    native = payload.get("schema_version") == SCHEMA_VERSION
    envelope = payload.get("schema_version") == 1 and payload.get("content_schema_version") == SCHEMA_VERSION
    if payload.get("format") != FORMAT or not (native or envelope):
        raise ValueError("Unsupported state-space database payload")
    source = payload
    tables = {table: [] for table in TABLES}
    preset_keys = {}
    for row in source["tables"].get("presets", []):
        original = json.loads(row["content_json"])
        content = compact_preset(original)
        vector = json.loads(row.get("target_vector_json", "null"))
        key = _hash([content, vector])
        preset_keys[row["preset_key"]] = key
        candidate = dict(preset_key=key, target_vector_json=_json(vector),
            target_vector_complete=bool(row.get("target_vector_complete")), content_json=_json(content))
        if candidate not in tables["presets"]:
            tables["presets"].append(candidate)
    for row in source["tables"].get("runs", []):
        tables["runs"].append(dict(run_id=row["run_id"],
            physical_game_id=row.get("physical_game_id") or row["run_id"],
            segment_index=row.get("segment_index", 1), segment_count=row.get("segment_count", 1),
            preset_key=preset_keys.get(row.get("preset_key")), language=row.get("language"),
            lifecycle_status=row["lifecycle_status"], termination_reason=row.get("termination_reason"),
            termination_basis=row.get("termination_basis", "observation"),
            full_run_start_observed=bool(row.get("full_run_start_observed")),
            completion_observed=bool(row.get("completion_observed")), quality_status=row.get("quality_status", "partial")))
    removed = {}
    for row in source["tables"].get("records", []):
        event = row["event_type"]
        data = json.loads(row["payload_json"])
        data = compact_record(event, data)
        if data is None:
            removed[row["record_id"]] = row
            continue
        tables["records"].append(dict(record_id=row["record_id"], run_id=row["run_id"],
            sequence=row.get("sequence"), event_type=event,
            basis=row.get("basis", "game_UI_observation"),
            quality_status=row.get("quality_status", "partial"), payload_json=_json(data)))
    seen = set()
    for row in source["tables"].get("quality_issues", []):
        evidence = _compact_value(json.loads(row.get("evidence_json", "{}")))
        rid = row.get("record_id")
        if rid in removed:
            original = removed[rid]
            if original["event_type"] in {"model_decision", "policy_decision", "fallback_policy_decision", "context_run_task"}:
                continue
            evidence.setdefault("affected_from_sequence", original.get("sequence"))
            evidence.setdefault("affected_through_sequence", original.get("sequence"))
            rid = None
        identity = [row["run_id"], rid, row["field_path"], row["code"], row["severity"], evidence]
        key = _hash(identity)
        if key in seen:
            continue
        seen.add(key)
        tables["quality_issues"].append(dict(issue_id=key, run_id=row["run_id"], record_id=rid,
            field_path=row["field_path"], code=row["code"], severity=row["severity"], evidence_json=_json(evidence)))
    return dict(format=FORMAT, schema_version=SCHEMA_VERSION, tables=tables)


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
        self.issues = {}
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
            elif (value.get("recovery_action") in {"stop_task", "use_native_fallback", "leave_shop"}
                  or value.get("missing")) and value.get("recovery_action") != "use_native_for_page":
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
    session = _session_record_ids(session, session_ids)
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
    if vector is None and original:
        vector = original.get("target_vector") if original.get("target_vector_complete") else None
    preset_content = compact_preset(original or {})
    preset_evidence = {}
    for key in ("game_display_role_totals", "game_display_total", "initial_inventory", "bag_party"):
        if key in (original or {}):
            preset_evidence[key] = original[key]
    preset_key = _hash([preset_content, vector])
    preset_evidence_record = binding["record_id"] if binding else next((row["record_id"] for row in session if row["event_type"] == "preset_observed"), start["record_id"])
    builder.tables["presets"].append(dict(preset_key=preset_key,
        target_vector_json=_json(vector),
        target_vector_complete=vector is not None, content_json=_json(preset_content)))

    task_end = next((row for row in reversed(session) if row["event_type"] == "task_end"), None)
    completed = bool(end and end.get("completion_observed"))
    lifecycle = "completed" if completed else "interrupted" if end or task_end else "open"
    termination = end.get("reason") if end else ("task_" + str(task_end.get("status")) if task_end else None)
    builder.tables["runs"].append(dict(run_id=run_id, physical_game_id=run_id,
        segment_index=1, segment_count=1, preset_key=preset_key, language=start.get("lang_type"),
        lifecycle_status=lifecycle, termination_reason=termination,
        termination_basis=("local_log_recovery" if end and end.get("boundary_recovered") else
                           "exploration_end" if end else "task_end" if task_end else "no_end_observed"),
        full_run_start_observed=full_start, completion_observed=completed, quality_status="valid"))

    semantic = []
    for row in all_rows:
        event = row["event_type"]
        if event in _DISCARD_EVENTS:
            continue
        if event == "lifecycle_observed" and row.get("lifecycle_type") == "pipeline_node_completed":
            continue
        semantic.append(row)
    wallets = [dict(row, observation_id=row["record_id"])
               for row in run if row["event_type"] == "wallet_observed"]
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
            if attempt and "wallet_before_observation_id" in attempt:
                before = next((wallet for wallet in wallets
                               if wallet["observation_id"] == attempt["wallet_before_observation_id"]), None)
            else:
                before = next((wallet for wallet in reversed(wallets) if wallet.get("sequence", 0) < before_seq
                               and wallet.get("coin_complete") and wallet.get("record_id")), None)
            observed = [wallet for wallet in wallets if before_seq < wallet.get("sequence", 0) < end_seq]
            pipeline = observed[-1] if observed else None
            evidence = purchase_wallet_evidence(row.get("operation_id"), (row.get("item") or {}).get("price"),
                observed, before_wallet=before, pipeline_wallet=pipeline)
            row.update(evidence)
            if row["confirmation_status"] != "confirmed":
                builder.issue(rid, "coins_after", "purchase_" + row["confirmation_status"],
                    "partial" if row["confirmation_status"] == "wallet_missing" else "suspect",
                    "购买首后钱包差额未确认；未使用价格反推余额。",
                    {key: value for key, value in evidence.items() if key.endswith("record_id") or key.endswith("observation_id")})
        if event in ("event_menu_observed", "event_menu_reobserved", "event_choice", "event_choice_click_observed"):
            row["event_identity_basis"] = "local_menu_correlation"
        if event == "settlement_observed":
            previous_complete = row.get("potential_counts_complete")
            conflicts = list(row.get("potential_counts_conflicts") or [])
            if row.get("owned_complete") and not row.get("owned_counts") and ("final_owned" in row or "owned" in row):
                owned = row.get("final_owned", row.get("owned", [])) + row.get("owned_anonymous", [])
                row["owned_counts"] = [dict(role=role,
                    core_count=sum(item.get("count", 1) for item in owned if item.get("role") == role
                                   and (item.get("core") or item.get("rarity", item.get("canonical_rarity")) == "core")),
                    noncore_level_sum=sum(item.get("level", item.get("level_after")) * item.get("count", 1) for item in owned
                        if item.get("role") == role and not (item.get("core")
                            or item.get("rarity", item.get("canonical_rarity")) == "core"))) for role in (1, 2, 3)]
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
                 "inventory_difference_derivation" if event == "note_delta" else
                 "confirmed_history_reconstruction" if event == "pool_observed" else "game_UI_observation")
        if event == "pool_observed" and any(any(flags) for flags in
                row.get("pool_identity_observation", {}).get("position_identity_inferred", [])):
            basis = "inferred_identity_history_reconstruction"
        row.pop("original_preset", None)
        payload = {key: item for key, item in row.items() if key not in BASE_FIELDS}
        if event == "note_setup_observed":
            payload.pop("raw", None)  # repeated main_discs[].raw and readings[].raw
        if event == "note_inventory_observed" and isinstance(payload.get("observation"), dict):
            if payload["observation"].get("counts") == payload.get("counts"):
                payload["observation"].pop("counts", None)
        if event == "shop_observed":
            shop_payload = payload.pop("payload", {})
            if "slots" not in payload:
                payload["slots"] = shop_payload.get("slots", [])
            payload.pop("stock_snapshot", None)  # same shelf items already live in slots
        if event in ("exploration_start", "preset_observed"):
            payload["preset_key"] = preset_key
        if event == "exploration_start" and start_confirmation:
            payload["start_confirmation_record_id"] = start_confirmation["record_id"]
        if rid == preset_evidence_record and preset_evidence:
            payload["preset_read_evidence"] = preset_evidence
        builder.inspect_quality(row, payload)
        if row.get("journal_gap"):
            builder.issue(rid, "journal_gap", "journal_record_missing", "partial",
                "本地写入失败后仍未补写的记录形成缺口；此前独立确认的前缀保留。",
                dict(row["journal_gap"], affected_through_sequence=run[-1].get("sequence")))
        normalized[rid] = row
        builder.tables["records"].append(dict(record_id=rid, run_id=run_id,
            sequence=source.get("sequence"), event_type=event, basis=basis,
            quality_status="valid", payload_json=_json(payload)))

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
    builder.tables["quality_issues"] = sorted(builder.issues.values(), key=lambda row: row["issue_id"])
    return compact_payload(dict(format=FORMAT, schema_version=SCHEMA_VERSION, tables=builder.tables))


def import_payload(database_path, payload):
    """Idempotently insert the same rows used by transport, returning table totals."""
    payload = compact_payload(payload)
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection, connection:
        # Replace a checkpoint and its quality together in one commit.
        connection.executescript("PRAGMA foreign_keys=ON;\nBEGIN;\n" +
            Path(__file__).with_name("research_schema.sql").read_text(encoding="utf-8"))
        for run in payload["tables"]["runs"]:
            connection.execute("DELETE FROM quality_issues WHERE run_id=?", (run["run_id"],))
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
