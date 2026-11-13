"""Run/session observations with linked OCR readings and original game evidence."""
from collections import defaultdict
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import hashlib
import json
import os
import re
import sqlite3
import threading
import sysconfig
import uuid

from .ocr_number import assess_amount, decimal_amount

from .event_mapping import has_potential_reward
from .sampling_evidence import ObservedInventory, ReadEvidence, raw_ocr, recognition_roi
from .name_matching import clean_name, identity_key, match_name
from .research_store import DATABASE_FILE, ResearchStore, valid_upload_consent
from .sampling_journal import JournalRollbackError, append_record, repair_journal, rollback


VERSION = "maa-run-observations-20261113.1"
SCHEMA_VERSION = 3
SKIP_FIELDS = {"image", "raw_image", "draw_images", "layout", "potential_layouts"}
LINK_FIELDS = {"operation_id", "origin_operation_id", "offer_id", "reward_chain_id",
               "stock_scan_id", "shop_visit_id", "event_id", "source_type"}


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def wallet_reading(raw, *, observed_extra=()):
    """Parse complete decimal amounts without changing OCR letters or digits."""
    raw = raw or {}
    values = [row.get("text", "") for row in raw.get("filtered_results", [])]
    raw_values = [row.get("text", "") for row in raw.get("all_results", [])]
    observed = [(row.get("text", ""), row.get("score", 0)) for row in raw.get("all_results", [])]
    observed.extend(observed_extra)
    parsed = [decimal_amount(text) for text in values]
    accepted = [row.get("text", "") for row in raw.get("filtered_results", []) if row.get("score", 0) >= .85]
    coin, status, conflicts = assess_amount(accepted if raw.get("hit") else (), observed)
    result = dict(values=values, raw_values=raw_values, parsed_values=parsed,
                  coin=coin, coin_complete=status == "ok", recognition_status=status)
    if conflicts:
        result["conflicts"] = list(conflicts)
    return result


def snapshot(value):
    if isinstance(value, Enum):
        return value.name
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.name
    if is_dataclass(value):
        return {f.name: snapshot(getattr(value, f.name)) for f in fields(value)
                if not f.name.startswith("_") and f.name not in SKIP_FIELDS}
    if isinstance(value, dict):
        return {str(k): snapshot(v) for k, v in value.items() if str(k) not in SKIP_FIELDS}
    if isinstance(value, (list, tuple)):
        return [snapshot(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: snapshot(v) for k, v in vars(value).items()
                if not k.startswith("_") and k not in SKIP_FIELDS}
    if hasattr(value, "__slots__"):
        return {k: snapshot(getattr(value, k)) for k in value.__slots__
                if not k.startswith("_") and k not in SKIP_FIELDS and hasattr(value, k)}
    return None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def process_alive(pid):
    if pid == os.getpid():
        return True
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return ctypes.get_last_error() == 5  # An elevated, running Maa may deny access.
    try:
        code = ctypes.c_uint32()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


class SamplingCollector:
    def __init__(self, app_dir, logger, module_root=None, owned_provider=None):
        self.app_dir, self.logger = Path(app_dir), logger
        self.module_root = Path(module_root) if module_root else None
        config = self.app_dir / "agent/agent_config/tower_sampling.json"
        try:
            self.config = json.loads(config.read_text(encoding="utf-8-sig")) if config.exists() else {}
            if not isinstance(self.config, dict):
                raise ValueError("Sampling config must be a JSON object")
        except (OSError, ValueError) as error:
            self.logger.error("$agent.custom.action.climb_tower_neural.sampling_config_load_failed", {"error": str(error)})
            self.config = {"enabled": False}
        self.enabled = self.config.get("enabled", True)
        self.directory = self.app_dir / "debug/tower_sampling"
        self.research_store = ResearchStore(self.directory)
        self.session_id = uuid.uuid4().hex
        self.task_id = None
        self.sequence = 0
        self.lock = threading.RLock()
        self.owned_provider = owned_provider or (lambda: ())
        self._read = ContextVar("tower_sampling_read", default=None)
        self._operations = []
        self._purchase_wallets = {}
        self.inventory = ObservedInventory()
        self.offer_index = 0
        self.last_wallet = None
        self.last_note_state_record_id = None
        self.last_observed_floor_context = None
        self._pending_potential_floor = None
        self._potential_floor_reads = {}
        self.last_preset_frame = None
        self.tracker = None
        self.preset = None
        self.preset_id = None
        self.target_vector = None
        self.policy_mode = "awaiting_preset"
        self.names = defaultdict(list)
        self.local_names = defaultdict(list)
        self.language = "unknown"
        self.run = None
        self.reward = None
        self.pending_selection = None
        self.start_pending = False
        self.run_confirmed = False
        self._journal_rollback = {}
        self._pending_journal_gap = None
        self._pending_archives = set()
        self.settlement_seen = False
        self.settlement_result = None
        self.event = None
        self.event_question = ""
        self.stock_scan_id = None
        self.upload = None
        self.runtime_versions = {}
        self.participant_id = None

    def context(self, context):
        from .tower_sampling_collector import SamplingContext
        return context if not self.enabled or isinstance(context, SamplingContext) else SamplingContext(context, self)

    def loaded(self):
        if not self.enabled:
            return
        try:
            self._load_collector()
        except (OSError, KeyError, TypeError, ValueError, sqlite3.Error) as error:
            self.enabled = False
            self.logger.error("$agent.custom.action.climb_tower_neural.sampling_initialize_failed", {"error": str(error)})

    def _load_collector(self):
        legacy_identity = self.directory / "collector_identity.json"
        identity = self.app_dir / "config/neural_network/collector_identity.json"
        if not identity.exists():
            value = (json.loads(legacy_identity.read_text(encoding="utf-8-sig")) if legacy_identity.exists()
                     else {"participant_id": uuid.uuid4().hex, "created_at": timestamp()})
            write_json(identity, value)
        self.participant_id = json.loads(identity.read_text(encoding="utf-8-sig"))["participant_id"]
        write_json(legacy_identity, json.loads(identity.read_text(encoding="utf-8-sig")))
        self.recover_interrupted()
        interface = self.app_dir / "interface.json"
        if interface.exists():
            # Only public software versions, never the app's instance/account settings.
            import jsonc
            data = jsonc.loads(interface.read_text(encoding="utf-8-sig"))
            self.runtime_versions = {key: data[key] for key in ("version", "interface_version") if key in data}
        self.runtime_versions["python_platform"] = sysconfig.get_platform()
        rules = (self.module_root or Path(__file__).resolve().parent) / "note_rules.json"
        self.runtime_versions["note_rules_sha256"] = hashlib.sha256(rules.read_bytes()).hexdigest()
        self.emit("collector_loaded", supported_languages=["cn", "jp"],
                  capture="all visible tower OCR and actions; local UI boundaries; no images or server IDs")
        self.logger.debug("Tower game recording enabled")

    def _safe(self, value):
        if isinstance(value, str):
            for path, replacement in ((self.app_dir, "$APP"), (Path.home(), "$USER")):
                value = value.replace(str(path), replacement).replace(path.as_posix(), replacement)
            return value
        if isinstance(value, dict):
            return {k: self._safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._safe(v) for v in value]
        return value

    def emit(self, event_type, **payload):
        if not self.enabled:
            return
        try:
            with self.lock:
                if not self.run_confirmed and event_type not in {
                        "collector_loaded", "task_start", "task_end", "exploration_start",
                        "exploration_start_attempt", "exploration_start_confirmed", "exploration_end"}:
                    return
                # A potential-bearing event can instead roll a non-potential
                # outcome. Once a real scene boundary is seen, its next reward
                # must not be borrowed by a later battle page.
                # Field inventory reads inside the dialogue action precede its
                # click confirmation and do not end that event's reward chain.
                if ((event_type == "note_source_started" and payload.get("source_type") == "battle"
                     and payload.get("floor_complete"))
                        or (event_type == "note_inventory_observed" and payload.get("page") == "field"
                            and payload.get("complete") and self.event and self.event["clicked"])
                        or (event_type == "lifecycle_observed" and payload.get("lifecycle_type") == "shop_entry")):
                    self._clear_event_reward()
                sequence = self.sequence + 1
                run_id = self.run["run_id"] if self.run else None
                run_sequence = self.run["record_count"] + 1 if self.run else None
                row = dict(schema_version=SCHEMA_VERSION, collector_version=VERSION, event_type=event_type,
                           record_type=event_type, timestamp=timestamp(), record_id=uuid.uuid4().hex,
                           participant_id=self.participant_id, session_id=self.session_id,
                           task_id=self.task_id, run_id=run_id, exploration_id=run_id,
                           sequence=sequence, run_sequence=run_sequence,
                           lang_type=self.language, observer="maa_local_UI_and_cached_OCR",
                           policy_mode=self.policy_mode,
                           server_confirmed=False, preset_id=self.preset_id,
                           target_vector_complete=self.target_vector is not None,
                           target_vector=self.target_vector)
                row.update(payload)
                row.update(participant_id=self.participant_id, session_id=self.session_id,
                           task_id=self.task_id, run_id=run_id, exploration_id=run_id,
                           sequence=sequence, run_sequence=run_sequence)
                folder = self.directory / ("runs" if run_id else "sessions")
                path = folder / f"{run_id or self.session_id}.jsonl"
                self.sequence = sequence
                if self.run:
                    self.run["record_count"] = run_sequence
                    self.run["last_sequence"] = sequence
                if self._pending_journal_gap:
                    row["journal_gap"] = self._pending_journal_gap
                try:
                    folder.mkdir(parents=True, exist_ok=True)
                    if path in self._journal_rollback:
                        rollback(path, self._journal_rollback[path])
                        self._journal_rollback.pop(path)
                    data = (json.dumps(self._safe(snapshot(row)), ensure_ascii=False) + "\n").encode("utf-8")
                    append_record(path, data)
                except OSError as error:
                    if isinstance(error, JournalRollbackError):
                        self._journal_rollback[path] = error.offset
                    if self.run:
                        gap = self.run.setdefault("journal_gap", dict(
                            affected_from_sequence=sequence, affected_from_run_sequence=run_sequence,
                            missing_record_count=0))
                        gap.update(affected_through_missing_sequence=sequence,
                                   affected_through_missing_run_sequence=run_sequence,
                                   missing_record_count=gap["missing_record_count"] + 1)
                        self.run["journal_integrity_complete"] = False
                        self.inventory.uncertain("journal_record_missing",
                                                 from_sequence=gap["affected_from_sequence"])
                        self._pending_journal_gap = dict(gap)
                        # Preserve the missing interval even if the process exits
                        # before a later successful record can carry it.
                        try:
                            write_json(self.directory / "queue" / f"{run_id}.open.json", self.run)
                        except OSError as marker_error:
                            self.logger.error("$agent.custom.action.climb_tower_neural.sampling_marker_open_failed",
                                              {"error": str(marker_error)})
                    raise
                self._pending_journal_gap = None
                if event_type in ("note_run_started", "note_inventory_observed"):
                    self.last_note_state_record_id = row["record_id"]
                if event_type == "note_floor_observed":
                    floor = (dict(floor=payload["floor"], record_id=row["record_id"],
                                  page=payload.get("page"), source="floor_hud", timestamp=row["timestamp"])
                             if payload.get("complete") and payload.get("source") == "floor_hud"
                             and payload.get("floor") is not None else None)
                    if floor:
                        self.last_observed_floor_context = floor
                    if payload.get("page") == "potential":
                        self._pending_potential_floor = floor
                try:
                    if self.run:
                        self.research_store.observe(path, event_type)
                    elif event_type == "task_end":
                        self.research_store.release_session(self.session_id)
                except (OSError, TypeError, ValueError, sqlite3.Error) as error:
                    self.logger.error("$agent.custom.action.climb_tower_neural.sampling_database_failed", {"error": str(error)})
                return row["record_id"]
        except (OSError, TypeError, ValueError) as error:
            self.logger.error("$agent.custom.action.climb_tower_neural.sampling_record_failed", {"error": str(error)})

    def new_read(self, kind, **links):
        parent = self._read.get()
        with self.lock:
            inherited = dict(self._operations[-1]) if self._operations else {}
        if parent:
            inherited.update(parent.links)
        inherited.update({key: value for key, value in links.items() if key in LINK_FIELDS})
        read = ReadEvidence(kind, inherited)
        if kind == "potential_callback":
            self._potential_floor_reads[read.read_id] = self._pending_potential_floor
        self._pending_potential_floor = None
        return read

    def capture_preset(self, event, image, **metadata):
        """Keep frame/crop coordinates; source pixels are an opt-in local aid."""
        if not self.enabled or not self.run_confirmed:
            return
        if event == "preset_screenshot":
            record_id = uuid.uuid4().hex
            self.last_preset_frame = dict(source_frame_id=record_id)
            if self.config.get("capture_preset_screenshots", False):
                import cv2
                folder = self.directory / "screenshots" / "presets" / self.session_id
                try:
                    folder.mkdir(parents=True, exist_ok=True)
                    target = folder / f"{record_id}.png"
                    cv2.imencode(".png", image)[1].tofile(target)
                except OSError as error:
                    self.logger.error("$agent.custom.action.climb_tower_neural.sampling_record_failed", {"error": str(error)})
                else:
                    self.last_preset_frame["source_image_path"] = target.relative_to(self.app_dir).as_posix()
            self.emit(event, record_id=record_id,
                      image_path=self.last_preset_frame.get("source_image_path"),
                      coordinate_space="maa_client_1280x720",
                      image_size=[image.shape[1], image.shape[0]], **metadata)
        elif event == "preset_level_crop":
            self.emit(event, **(self.last_preset_frame or {}),
                      coordinate_space="maa_client_1280x720", **metadata)

    @property
    def current_read(self):
        return self._read.get()

    @contextmanager
    def action_scope(self, **links):
        # Nested Maa run_task callbacks can execute on another Python thread.
        # Keep their causal action on this collector, not in official classes.
        links = {key: value for key, value in links.items() if key in LINK_FIELDS}
        with self.lock:
            self._operations.append(links)
        try:
            yield
        finally:
            with self.lock:
                self._operations.remove(links)

    @contextmanager
    def purchase_wallet_capture(self, operation_id):
        evidence = dict(before_wallet=self.last_wallet, wallets=[])
        with self.lock:
            self._purchase_wallets[operation_id] = evidence
        try:
            yield evidence
        finally:
            with self.lock:
                self._purchase_wallets.pop(operation_id)

    @contextmanager
    def capture(self, kind=None, *, read=None, **links):
        read = read if read is not None else self.new_read(kind, **links)
        token = self._read.set(read)
        try:
            yield read
        finally:
            self._read.reset(token)
            if read.kind == "potential_callback":
                self._potential_floor_reads.pop(read.read_id, None)

    def record_recognition(self, entry, args, kwargs, result, language):
        if not self.enabled or not self.run_confirmed:
            return
        read = self._read.get()
        links = dict(read_id=read.read_id, read_kind=read.kind, **read.links) if read else {}
        raw = snapshot(result)
        roi = recognition_roi(entry, args, kwargs)
        coin_node = next((name for name in ("星塔_神经网络_识别当前金币", "星塔_通用_识别当前金币_agent")
                          if entry in (name, name + "__数字重读")), None)
        override = kwargs.get("pipeline_override", args[1] if len(args) > 1 else {})
        origin = dict(override.get(entry, {}).get("attach", {})) if isinstance(override, dict) else {}
        native_reco_id = raw.get("reco_id") if isinstance(raw, dict) else None
        if native_reco_id is None:
            native_reco_id = getattr(result, "reco_id", None)
        locator_id = origin.get("source_locator_reco_id")
        locator = read.recognition_by_id(locator_id) if read else None
        if locator:
            origin["source_locator_record_id"] = locator["record_id"]
        recognition_links = dict(links)
        recognition_links.update({key: origin[key] for key in (
            "coordinate_space", "source_coordinate_space", "source_node", "source_roi", "input_scale",
            "purpose", "preprocessing", "source_locator_reco_id", "source_locator_record_id",
            "detected_text_roi", "padding") if key in origin})
        if native_reco_id is not None:
            recognition_links["native_reco_id"] = native_reco_id
        record_id = self.emit("context_run_recognition", phase="after", node_name=entry,
                              lang_type=language, method="run_recognition", roi=roi,
                              arguments=snapshot(args), keyword_arguments=snapshot(kwargs),
                              raw_result=raw, **recognition_links)
        if read:
            read.recognitions.append(dict(record_id=record_id, node_name=entry, roi=roi, raw_result=raw,
                native_reco_id=native_reco_id, purpose=origin.get("purpose"), source_locator_reco_id=locator_id))
        if coin_node is not None:
            # The color box supplies the crop; only its OCR parent can contradict the amount.
            text_locator = (read.recognition_by_id(locator["source_locator_reco_id"])
                            if locator and locator["purpose"] == "wallet_digit_locator" else locator)
            locator_rows = (text_locator.get("raw_result") or {}).get("all_results", []) if text_locator else []
            wallet = wallet_reading(raw, observed_extra=[(row.get("text", ""), row.get("score", 0))
                for row in locator_rows])
            if locator:
                wallet["locator_raw_values"] = [row.get("text", "") for row in locator_rows]
            previous = self.last_wallet
            if (read and entry != coin_node and previous
                    and previous.get("read_id") == read.read_id):
                values = set(previous.get("conflicts", [])) | set(wallet.get("conflicts", []))
                values.update(row["coin"] for row in (previous, wallet) if row.get("coin_complete"))
                if len(values) > 1 or previous["recognition_status"] == "conflict":
                    wallet.update(coin=None, coin_complete=False, recognition_status="conflict", conflicts=sorted(values))
            wallet_id = self.emit("wallet_observed", node_name=entry, **wallet,
                preceding_event_id=self.event["event_id"] if self.event else None,
                recognition_record_id=record_id, raw_result=raw,
                source_node=origin.get("source_node", entry), source_roi=origin.get("source_roi", roi),
                preprocessing=origin.get("preprocessing", "original"),
                **{key: origin[key] for key in ("purpose", "source_locator_reco_id", "source_locator_record_id",
                    "detected_text_roi", "padding") if key in origin}, **links)
            self.last_wallet = dict(**wallet, record_id=wallet_id, recognition_record_id=record_id,
                **{key: origin[key] for key in ("purpose", "source_locator_reco_id", "source_locator_record_id") if key in origin}, **links)
            with self.lock:
                purchase = self._purchase_wallets.get(links.get("operation_id"))
                if purchase is not None:
                    purchase["wallets"].append(self.last_wallet)

    def owned_fields(self):
        return dict(self.inventory.fields(), maa_owned_cache=self.owned(), maa_owned_complete=False)

    def begin_task(self, language, strategy):
        self.finish_at_boundary("task_restarted")
        self.run_confirmed = self.start_pending = False
        if not self.enabled:
            return
        if self.upload:
            self.upload.set_consent(False)
        try:
            self.research_store.release_session(self.session_id)
        except (OSError, TypeError, ValueError) as error:
            self.logger.error("$agent.custom.action.climb_tower_neural.sampling_journal_cleanup_failed", {"error": str(error)})
        self.task_id = uuid.uuid4().hex
        self.language = language
        self.tracker = self.preset = self.preset_id = self.target_vector = None
        self.policy_mode = "awaiting_preset"
        self.last_note_state_record_id = None
        self.names.clear()
        self.local_names.clear()
        self.start_pending = False
        self.last_preset_frame = None
        self.last_observed_floor_context = None
        self._pending_potential_floor = None
        self._potential_floor_reads.clear()
        self.reward = self.pending_selection = self.event = None
        self.event_question = ""
        self.emit("task_start", strategy=strategy, observation_policy="stable_pages_only_v1")

    def bind_preset(self, preset, tracker):
        if not self.enabled or not self.run_confirmed:
            return
        if self.run:
            tracker.run_id = self.run["run_id"]
        self.preset, self.tracker = snapshot(preset), tracker
        self.policy_mode = "model"
        self.target_vector = [level for pool in preset["pools"] for level in pool]
        identities = []
        self.names.clear()
        self.local_names.clear()
        for member in preset["party"]:
            for target in member["targets"]:
                identities.append(dict(target, role=member["role"], character_id=member["character_id"]))
        identities.extend(preset.get("recognition_catalog", []))
        for item in identities:
            for field in ("name_ja_jp", "name_zh_cn", "name_en_us", "name_zh_tw"):
                if item.get(field):
                    key = clean_name(item[field])
                    if item not in self.names[key]:
                        self.names[key].append(item)
                    if field == ("name_zh_cn" if self.language == "cn" else "name_ja_jp"):
                        self.local_names[key].append(item)
        key = [{"role": m["role"], "character_id": m["character_id"], "targets": m["targets"]}
               for m in preset["party"]]
        self.preset_id = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
        self.emit("preset_observed", original_preset=self.preset,
                  source="in_game_bag_targets_OCR" if "bag_party" in preset else "in_game_current_preset_OCR",
                  model_versions=snapshot(tracker.model.versions))

    def initialize_bag_inventory(self, preset):
        """Import the complete visible stock without requiring unnamed non-target IDs."""
        if not self.ensure_run("bag_inventory_observed"):
            return
        self.inventory.initialize_bag_inventory(preset["initial_inventory"])
        self.emit("bag_inventory_observed", original_bag_party=preset.get("bag_party"),
            source="visible_bag_inventory", **self.owned_fields())

    def start_intent(self, node):
        with self.lock:
            if self.start_pending:
                return self.run["run_id"] if self.run else None
            self.finish_at_boundary("new_game_started")
            self.start_pending = True
            self.run_confirmed = False
            if not self.enabled:
                return
            self._start_run(node)
            self.run["start_attempt_record_id"] = self.emit("exploration_start_attempt", node_name=node,
                start_status="awaiting_entry", full_run_start_observed=False)
            return self.run["run_id"]

    def confirm_start(self, evidence):
        """Confirm only the pending new run at its first successful entry anchor."""
        with self.lock:
            if not self.start_pending or (self.enabled and (not self.run or not self.run.get("awaiting_start"))):
                return False
            self.start_pending = False
            self.run_confirmed = True
            if not self.enabled:
                return True
            self.run.update(full_run_start_observed=True, awaiting_start=False)
            self.inventory = ObservedInventory(full_start=True)
            record_id = self.emit("exploration_start_confirmed", full_run_start_observed=True,
                boundary_evidence=evidence, start_attempt_record_id=self.run.get("start_attempt_record_id"),
                **self.owned_fields())
            self.run["start_confirmation_record_id"] = record_id
            try:
                write_json(self.directory / "queue" / f"{self.run['run_id']}.open.json", self.run)
            except OSError as error:
                self.logger.error("$agent.custom.action.climb_tower_neural.sampling_marker_open_failed", {"error": str(error)})
            return True

    def ensure_run(self, evidence):
        """Only an explicit new-game click creates a run; callbacks cannot attach."""
        return self.run["run_id"] if self.enabled and self.run_confirmed and self.run else None

    def _start_run(self, evidence):
        self._pending_journal_gap = None
        run_id = uuid.uuid4().hex
        if self.tracker:
            self.tracker.run_id = run_id
        self.run = dict(run_id=run_id, session_id=self.session_id, participant_id=self.participant_id,
                        pid=os.getpid(), started_at=timestamp(), record_count=0,
                        full_run_start_observed=False, awaiting_start=self.start_pending, language=self.language,
                        unresolved_candidate_count=0)
        if self.upload:
            self.run["research_upload"] = self.upload.consent_for_run(run_id)
        self.inventory = ObservedInventory()
        self.offer_index = 0
        self.last_wallet = None
        self.last_note_state_record_id = None
        self.last_observed_floor_context = None
        self._pending_potential_floor = None
        self._potential_floor_reads.clear()
        self.reward = self.pending_selection = self.event = None
        self.event_question = ""
        self.stock_scan_id = None
        self.settlement_seen = False
        self.settlement_result = None
        try:
            write_json(self.directory / "queue" / f"{run_id}.open.json", self.run)
        except OSError as error:
            self.logger.error("$agent.custom.action.climb_tower_neural.sampling_marker_open_failed", {"error": str(error)})
        self.emit("exploration_start", full_run_start_observed=self.run["full_run_start_observed"],
                  observation_policy="stable_pages_only_v1",
                  resumed=False, boundary_evidence=evidence, start_status="awaiting_entry",
                  original_preset=self.preset, runtime_versions=self.runtime_versions,
                  model_versions=snapshot(self.tracker.model.versions) if self.tracker else None,
                  **self.owned_fields())

    def owned(self):
        return [self.potential(p) for p in self.owned_provider()]

    def potential(self, potential, *, name_readings=()):
        row = snapshot(potential)
        role = getattr(getattr(potential, "trekker", None), "index", -1) + 1
        resolution = match_name(row.get("name", ""), self.names, role=role or None,
                                rarity=row.get("rarity"), preferred=self.local_names, party_bound=True)
        matches = resolution.items
        item = matches[0] if len(matches) == 1 else None
        if item is None and self.tracker is not None and role in getattr(self.tracker, "local_roles", {}):
            item = self.tracker.identify(row.get("name", ""), role=role, rarity=row.get("rarity"))
            matches = (item,)
        resolved_role = item["role"] if item else role or None
        core = row.get("rarity") == "core"
        before, after = row.get("old_level"), row.get("new_level")
        before = before if before is not None and before >= 0 else None
        after = after if after is not None and after >= 1 else None
        # Fragments belong to one OCR result; repeated reads are independent evidence.
        name = clean_name(row.get("name"))
        sources = [reading.get("raw_result") or {} for reading in name_readings]
        name_complete = bool(name and sources and len(name) >= len(resolution.matched_alias) and all(
            source.get("hit") and name == clean_name(" ".join(r.get("text", "") for r in source.get("filtered_results", [])))
            for source in sources))
        name_confidence = min((r.get("score", 0.) for source in sources
                               for r in source.get("filtered_results", [])), default=None)
        validated = bool(item and role in (1, 2, 3) and resolution.method == "closest_name"
                         and resolution.similarity >= resolution.minimum_similarity
                         and resolution.similarity_gap >= resolution.minimum_gap
                         and name_complete and name_confidence >= .85)
        row.update(potential_id=item.get("potential_id") if item else None,
                   local_identity=item.get("local_identity") if item else None,
                   local_identity_namespace=item.get("local_identity_namespace") if item else None,
                   catalog_id_namespace=item.get("catalog_id_namespace") if item else None,
                   character_id=item.get("character_id") if item else None,
                   role=resolved_role,
                   target_level=item["target_level"] if item else None,
                   identity_basis=(item.get("identity_basis") or resolution.method + ("_and_role" if role else "_unique_in_party")) if item else "unresolved",
                   identity_name=(item.get("name_zh_cn" if self.language == "cn" else "name_ja_jp") if item else None),
                   identity_similarity=resolution.similarity,
                   identity_inferred=bool(item and resolution.method == "closest_name" and not validated),
                   identity_match_validated=validated,
                   identity_matched_alias=resolution.matched_alias or None,
                   identity_runner_up_similarity=resolution.runner_up_similarity,
                   identity_similarity_gap=resolution.similarity_gap,
                   identity_minimum_similarity=resolution.minimum_similarity,
                   identity_minimum_gap=resolution.minimum_gap,
                   identity_name_input_complete=name_complete,
                   identity_name_min_confidence=name_confidence,
                   id_candidates=[m["potential_id"] for m in matches if m.get("potential_id") is not None],
                   local_identity_candidates=[m["local_identity"] for m in matches if m.get("local_identity")],
                   canonical_rarity=item["rarity_class"] if item else None,
                   trekker_index=resolved_role-1 if resolved_role else None, core=core,
                   level_before=before, offered_level_after=after,
                   level_after=min(after, 1 if core else 6) if after is not None and after >= 0 else None,
                   level_delta=after-before if after is not None and before is not None else None,
                   is_new=before == 0 if before is not None else None,
                   level_basis="Maa_core_0_to_1_convention" if core else "card_level_OCR")
        return row

    def inferred_state(self):
        if not self.tracker:
            return None
        return dict(floor=self.tracker.state.floor, node=self.tracker.state.node,
                    phase=self.tracker.state.phase, source="neural_network_route_inference",
                    observation_not_game_truth=True)

    def finish_at_boundary(self, reason):
        with self.lock:
            for marker in tuple(self._pending_archives):
                self._try_recover_marker(marker)
        self.finish_run("settlement_saved" if self.settlement_seen else reason,
                        completion_observed=self.settlement_seen, settlement=self.settlement_result)

    def finish_run(self, reason, *, completion_observed, settlement=None):
        if not self.enabled or not self.run:
            self.run_confirmed = self.start_pending = False
            return
        ready = None
        with self.lock:
            if not self.run:
                return
            if self.pending_selection:
                self.inventory.uncertain("take_not_confirmed", offer_id=self.pending_selection["offer_id"])
            end_record = self.emit("exploration_end", completion_observed=completion_observed, reason=reason,
                      final_owned=self.inventory.fields()["owned"], **self.owned_fields(),
                      settlement=settlement, tracker_estimate=self.inferred_state(),
                      last_wallet_observation=self.last_wallet,
                      pending_selection=snapshot(self.pending_selection),
                      full_run_start_observed=self.run["full_run_start_observed"])
            manifest = dict(self.run, ended_at=timestamp(), reason=reason,
                            completion_observed=completion_observed,
                            complete=(self.run["full_run_start_observed"] and completion_observed
                                      and self.run.get("journal_integrity_complete", True)),
                            owned_history_complete=self.inventory.complete,
                            target_vector_complete=self.target_vector is not None, preset_id=self.preset_id,
                            research_database=DATABASE_FILE, collector_version=VERSION)
            marker = self.directory / "queue" / f"{self.run['run_id']}.open.json"
            if not end_record:
                try:
                    write_json(marker, manifest)
                except OSError as error:
                    self.logger.error("$agent.custom.action.climb_tower_neural.sampling_marker_open_failed", {"error": str(error)})
            if end_record:
                try:
                    path = self.directory / "runs" / f"{self.run['run_id']}.jsonl"
                    self.research_store.checkpoint(path)
                    ready = self._release_run(path, manifest)
                except (OSError, TypeError, ValueError, sqlite3.Error) as error:
                    self._pending_archives.add(marker)
                    self.logger.error("$agent.custom.action.climb_tower_neural.sampling_finalize_failed", {"error": str(error)})
            self.run = None
            self.run_confirmed = False
            self._pending_journal_gap = None
            self.start_pending = False
            self.last_note_state_record_id = None
            self.reward = self.pending_selection = self.event = None
            self.last_observed_floor_context = None
            self._pending_potential_floor = None
            self._potential_floor_reads.clear()
        if ready and self.upload:
            self.upload.finish_run(ready)

    def _release_run(self, path, manifest):
        ready = self.directory / "queue" / f"{manifest['run_id']}.ready.json"
        authorized = valid_upload_consent(manifest)
        if authorized and not ready.exists():
            # Publish once, only after the database commit. The worker may pick it up immediately.
            write_json(ready, {key: manifest[key] for key in ("run_id", "session_id", "pid",
                                                              "research_database", "research_upload")})
        self.research_store.release_run(path)
        return ready if authorized else None

    def recover_interrupted(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        # A separate database serializes recovery through journal cleanup without
        # holding the research database's writer lock. Closing releases it on exit.
        with closing(sqlite3.connect(self.directory / ".recovery.lock", timeout=0)) as lock:
            try:
                lock.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as error:
                if error.sqlite_errorcode != sqlite3.SQLITE_BUSY:
                    raise
                return
            with self.lock:
                self._recover_interrupted()

    def _recover_interrupted(self):
        sessions = set()
        for marker in (self.directory / "queue").glob("*.open.json"):
            session = self._try_recover_marker(marker)
            if session and session != self.session_id:
                sessions.add(session)
        # A process can exit after committing its last run but before task_end.
        for folder, pattern in (("queue", "*.ready.json"), ("sent", "*.json")):
            for marker in (self.directory / folder).glob(pattern):
                try:
                    manifest = json.loads(marker.read_text(encoding="utf-8"))
                    session = manifest.get("session_id")
                    if (manifest.get("research_database") and session
                            and (self.directory / "sessions" / f"{session}.jsonl").exists()
                            and not process_alive(manifest["pid"])):
                        sessions.add(session)
                except FileNotFoundError:
                    continue
                except (OSError, KeyError, TypeError, ValueError) as error:
                    self.logger.error("$agent.custom.action.climb_tower_neural.sampling_recovery_failed", {"marker": marker.name, "error": str(error)})
        for session in sessions:
            try:
                self.research_store.release_session(session)
            except (OSError, TypeError, ValueError) as error:
                self.logger.error("$agent.custom.action.climb_tower_neural.sampling_journal_cleanup_failed", {"error": str(error)})

    def _try_recover_marker(self, marker):
        try:
            session = self._recover_marker(marker)
        except (OSError, KeyError, TypeError, ValueError, sqlite3.Error) as error:
            self.logger.error("$agent.custom.action.climb_tower_neural.sampling_recovery_failed", {"marker": marker.name, "error": str(error)})
        else:
            self._pending_archives.discard(marker)
            return session

    def _recover_marker(self, marker):
        try:
            manifest = json.loads(marker.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        run_id = manifest["run_id"]
        if uuid.UUID(run_id).hex != run_id or marker.name != f"{run_id}.open.json":
            raise ValueError("Invalid sampling run ID")
        # Only this collector can retry its own closed runs while its PID is alive.
        if marker not in self._pending_archives and process_alive(manifest["pid"]):
            return
        path = self.directory / "runs" / f"{manifest['run_id']}.jsonl"
        status = self.research_store.run_status(run_id)
        if status and status != "open":
            # A previous recovery may have committed before cleanup was interrupted.
            manifest["research_database"] = DATABASE_FILE
            self._release_run(path, manifest)
            return manifest["session_id"]
        if not path.exists():
            if status:
                raise ValueError("Missing journal for a checkpointed sampling run")
            # The open marker is written before the first observation.
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        count, last, settlement_save, start_confirmation = 0, {}, None, None
        damage = repair_journal(path)
        if damage:
            manifest["journal_integrity_complete"] = False
        with path.open("rb+") as stream:
            for line in stream:
                last = json.loads(line)
                count += 1
                if (last.get("event_type") == "lifecycle_observed"
                        and last.get("lifecycle_type") == "settlement_save"):
                    settlement_save = last
                if (last.get("event_type") == "exploration_start_confirmed"
                        and last.get("full_run_start_observed") is True):
                    start_confirmation = last
            has_end = last.get("event_type") == "exploration_end"
            full_start = bool(manifest["full_run_start_observed"] or start_confirmation)
            completion = last.get("completion_observed", False) if has_end else settlement_save is not None
            reason = (last.get("reason") if has_end else "settlement_saved" if completion
                      else "process_ended_without_settlement")
            if not has_end:
                row = {key: last.get(key) for key in ("task_id", "lang_type", "preset_id",
                       "target_vector_complete", "target_vector")}
                row.update(schema_version=SCHEMA_VERSION, collector_version=VERSION,
                           event_type="exploration_end", record_type="exploration_end", record_id=uuid.uuid4().hex,
                           run_id=manifest["run_id"], exploration_id=manifest["run_id"],
                           session_id=manifest["session_id"], participant_id=manifest["participant_id"],
                           sequence=max(last.get("sequence", 0), manifest.get("last_sequence", 0)) + 1,
                           run_sequence=max(last.get("run_sequence") or count, manifest["record_count"]) + 1,
                           timestamp=timestamp(), reason=reason, observer="local_log_recovery",
                           completion_observed=completion,
                           full_run_start_observed=full_start,
                           completion_evidence_record_id=(settlement_save or {}).get("record_id"),
                           boundary_recovered=True, server_confirmed=False)
                if manifest.get("journal_gap"):
                    row["journal_gap"] = manifest["journal_gap"]
        if not has_end:
            append_record(path, (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        manifest.pop("data_file", None)
        manifest.update(research_database=DATABASE_FILE,
                        full_run_start_observed=full_start, awaiting_start=False,
                        start_confirmation_record_id=(start_confirmation or {}).get("record_id"),
                        complete=full_start and completion and manifest.get("journal_integrity_complete", True),
                        completion_observed=completion,
                        record_count=(last if has_end else row).get("run_sequence", count + (not has_end)),
                        target_vector_complete=last.get("target_vector_complete", False),
                        preset_id=last.get("preset_id"), collector_version=VERSION,
                        reason=reason, recovered_at=timestamp(),
                        ended_at=last.get("timestamp", manifest["started_at"]))
        self.research_store.checkpoint(path)
        self._release_run(path, manifest)
        return manifest["session_id"]

    def reward_chain(self, source):
        if not self.ensure_run("potential_page"):
            return
        if not self.reward or self.reward["closed"]:
            read = self._read.get()
            self.reward = dict(reward_chain_id=uuid.uuid4().hex, source=source, pages=[],
                               page=None, refresh_operation_id=None, closed=False,
                               origin_operation_id=read.links.get("operation_id") if read else None)
        return self.reward

    def prepare_offer(self, handler):
        chain = self.reward_chain(handler.data.params.potential_source)
        if chain is None:
            return
        if chain["page"] is None:
            data = handler.data
            source_raw = data.params.potential_source
            source = ("core" if data.core_potential else "level_up" if data.level_upped
                      else source_raw if source_raw != "default" else "unknown")
            chain["source_fields"] = dict(source=source, source_attribution="Maa_attach_and_banner")
            event = self.event
            pending = event and event.get("potential_reward_pending")
            if event:
                # Only the first reward after this click can consume the event;
                # explicit sources and core/level banners consume it as well.
                event["potential_reward_pending"] = False
            if source == "unknown" and pending:
                selection = event["selection"]
                chain["source_fields"].update(source="event", source_inferred=True,
                    source_attribution="confirmed_event_choice_and_following_reward_page",
                    event_id=event["event_id"], source_event_id=event["event_id"],
                    source_evidence=dict(event_choice_record_id=selection["record_id"],
                        event_click_record_id=event["click_record_id"],
                        selected_box=selection["selected_box"], selected_text=selection["selected_text"],
                        displayed_consequence=selection["displayed_consequence"],
                        selection_basis="selected_box_in_observed_event_menu",
                        reward_page_basis="default_without_core_or_level_up_banner"))
            self._new_offer_page(chain)
        return chain["page"]

    def _clear_event_reward(self):
        if self.event:
            self.event["potential_reward_pending"] = False
            self.event["reward_boundary_seen"] = True
        if self.reward and self.reward.get("source_fields", {}).get("source") == "event":
            self.reward["closed"] = True

    def _new_offer_page(self, chain):
        self.offer_index += 1
        previous = chain["pages"][-1] if chain["pages"] else None
        page = dict(offer_id=uuid.uuid4().hex, offer_index=self.offer_index,
                    refresh_index=len(chain["pages"]),
                    refresh_parent_offer_id=previous["offer_id"] if previous else None,
                    refresh_operation_id=chain["refresh_operation_id"],
                    observation_count=0, fingerprint=None)
        chain["pages"].append(page)
        chain["page"] = page
        return page

    def observe_offer(self, handler, read=None):
        if not self.ensure_run("potential_page"):
            return
        data = handler.data
        chain = self.reward_chain(data.params.potential_source)
        page = self.prepare_offer(handler)
        repeated = page["observation_count"] > 0
        offer_id = page["offer_id"]
        read = read if read is not None else self.new_read("potential_page", offer_id=offer_id)
        name_node = "星塔_神经网络_潜能_识别潜能名称"
        names = read.matching(name_node)
        level_nodes = ("星塔_神经网络_潜能_识别潜能等级", "星塔_神经网络_潜能_识别潜能等级后值")
        levels = read.matching(level_nodes)
        candidates = []
        for slot, potential in enumerate(data.potentials):
            name_roi = getattr(potential, "core_potential_name_roi" if potential.rarity == "core" else "general_potential_name_roi", None)
            level_roi = getattr(potential, "general_potential_level_roi", None)
            def adjusted(roi):
                if roi is None:
                    return None
                x, y, width, height = roi
                return [x, max(0, y-data.params.selected_potential_offset) if potential.selected else y, width, height]
            name_roi, level_roi = adjusted(name_roi), adjusted(level_roi)
            name = read.matching_within(name_node, name_roi) if name_roi else names[slot:slot+1]
            level = read.matching_within(level_nodes, level_roi) if level_roi else levels[slot:slot+1]
            p = self.potential(potential, name_readings=name)
            candidates.append(p)
            errors = []
            if identity_key(p) is None:
                errors.append("unresolved_identity")
            elif not p["identity_name_input_complete"]:
                errors.append("name_input_incomplete")
            if p["level_before"] is None or p["offered_level_after"] is None or p["level_before"] < 0 or p["offered_level_after"] < 1:
                errors.append("level_not_read")
            if p["canonical_rarity"] is not None and p["rarity"] != p["canonical_rarity"]:
                errors.append("rarity_catalog_conflict")
            p.update(slot=slot, initial_level=p["offered_level_after"] if p["is_new"] else None,
                     server_raw_increment=None, raw_ocr=dict(name=raw_ocr(name), level=raw_ocr(level)),
                     name_roi=name_roi or (name[0]["roi"] if name else None),
                     level_roi=level_roi or (level[0]["roi"] if level else None),
                     level_regions=[dict(node_name=r["node_name"], roi=r["roi"], recognition_record_id=r["record_id"], raw=raw_ocr([r])) for r in level],
                     recognition_record_ids=[r["record_id"] for r in name+level],
                     recognition_status="ok" if not errors else "unresolved",
                     recognition_errors=errors)
        unresolved = sum(p["recognition_status"] != "ok" for p in candidates)
        read_complete = bool(candidates) and unresolved == 0
        fingerprint = [(identity_key(p) or p["name"], p["role"], p["rarity"],
                        p["level_before"], p["offered_level_after"]) for p in candidates]
        consistent = not repeated or page["fingerprint"] == fingerprint
        if not repeated:
            page["fingerprint"] = fingerprint
        page["observation_count"] += 1
        page["candidates"] = candidates
        conflicts = self.inventory.audit(candidates, offer_id)
        if not repeated:
            self.run["unresolved_candidate_count"] += unresolved
        source_raw = data.params.potential_source
        source_fields = chain["source_fields"]
        callback = self._read.get()
        callback_floor = self._potential_floor_reads.pop(callback.read_id, None) if callback else None
        if source_raw == "default" and callback_floor and not chain.get("floor_observation"):
            chain["floor_observation"] = callback_floor
        floor_observation = chain.get("floor_observation")
        floor_basis = ("same_potential_callback_floor_hud" if floor_observation and not page["refresh_index"]
                       else "same_reward_chain_floor_hud" if floor_observation
                       else "not_observed_on_this_reward_chain")
        self.emit("offer_reobserved" if repeated else "offer_observed", offer_id=offer_id,
                  offer_index=page["offer_index"], reward_chain_id=chain["reward_chain_id"],
                  refresh_index=page["refresh_index"], refresh_parent_offer_id=page["refresh_parent_offer_id"],
                  refresh_operation_id=page["refresh_operation_id"], maa_refresh_count=data.refresh_count,
                  page_generation_basis="successful_refresh" if page["refresh_index"] else "new_reward_after_confirmed_take_or_run_start",
                  same_page_read_index=page["observation_count"], recognition_consistent=consistent,
                  read_id=read.read_id, recognition_record_ids=read.record_ids,
                  origin_operation_id=chain["origin_operation_id"],
                  source_raw=source_raw, source_type=source_raw, **source_fields,
                  level_upped=data.level_upped, core_potential=data.core_potential,
                  candidates=candidates, candidate_read_complete=read_complete,
                  candidates_complete=read_complete,
                  recognition_status="ok" if read_complete and consistent else "unresolved",
                  raw_rarity=list(data.potential_rarities),
                  level_basis="Maa_core_0_to_1_convention" if data.core_potential else "card_level_OCR",
                  inventory_conflicts=conflicts, **self.owned_fields(),
                  main_trekker_index=0, trekker_index_basis="game_preset_party_roles",
                  coin=data.current_coin, refresh_price=data.refresh_cost,
                  floor=floor_observation["floor"] if floor_observation else None,
                  floor_basis=floor_basis,
                  floor_observation_id=floor_observation["record_id"] if floor_observation else None,
                  last_observed_floor_context=self.last_observed_floor_context,
                  state=snapshot(data), tracker_estimate=self.inferred_state(),
                  preceding_event_id=self.event["event_id"] if self.event else None,
                  preceding_event_relation=("confirmed_event_choice_reward_chain" if source_fields["source"] == "event"
                                            else "last_observed_event_not_reward_attribution"))
        return offer_id

    def observed_offer_candidate(self, offer_id, candidate_index):
        """Return evidence only from this exact open offer and card slot."""
        chain = self.reward
        page = chain.get("page") if chain and not chain["closed"] else None
        if page is None or page["offer_id"] != offer_id:
            return None
        return next((snapshot(candidate) for candidate in page["candidates"]
                     if candidate["slot"] == candidate_index), None)

    def cache_selection(self, potential, offer_id, operation_id):
        if not self.enabled or not self.run_confirmed:
            return
        page = self.reward["page"]
        observed = next((p for p in page["candidates"] if p["slot"] == potential.index), {})
        selected = dict(observed, **self.potential(potential))
        self._reuse_identity_evidence(selected, observed)
        self.pending_selection = dict(offer_id=offer_id, operation_id=operation_id,
            reward_chain_id=self.reward["reward_chain_id"], selected=selected,
            source_type=self.reward["source"], source_raw=self.reward["source"],
            **self.reward["source_fields"])
        self.emit("selection_cache", **self.pending_selection, **self.owned_fields(),
                  operation_confirmed=False, cache_timing="before_final_take_button_confirmation")

    @staticmethod
    def _reuse_identity_evidence(selected, observed):
        if (selected.get("name"), identity_key(selected), selected.get("role")) == (
                observed.get("name"), identity_key(observed), observed.get("role")):
            selected.update({key: value for key, value in observed.items() if key.startswith("identity_")})

    def selection_confirmed(self, evidence):
        if not self.enabled or not self.run_confirmed:
            return
        if self.event:
            self.event["potential_reward_pending"] = False
        if not self.pending_selection:
            return
        record_id = uuid.uuid4().hex
        previous = self.inventory.fields()
        self.pending_selection["selection_observation_complete"] = self.inventory.confirm(
            self.pending_selection["selected"], record_id)
        self.emit("selection_ui_transition_observed", **self.pending_selection,
                  record_id=record_id, owned_before=previous["owned"],
                  owned_before_complete=previous["owned_complete"],
                  operation_confirmed=True, confirmation_basis=evidence,
                  observed_state_after=self.owned(), **self.owned_fields())
        self.pending_selection = None
        if self.reward:
            self.reward["closed"] = True

    def refresh_succeeded(self, offer_id, operation_id):
        if not self.enabled or not self.run_confirmed:
            return
        chain = self.reward
        if chain and chain["page"] and chain["page"]["offer_id"] == offer_id:
            chain["refresh_operation_id"] = operation_id
            return self._new_offer_page(chain)["offer_id"]

    def model_decision(self, record):
        if self.policy_mode != "model" or not self.ensure_run("model_decision"):
            return
        # The preset and software versions are already in this run's header.
        reduced = {k: v for k, v in record.items()
                   if k not in {"original_preset", "versions", "run_id"}}
        chain = self.reward if self.reward and not self.reward["closed"] else None
        phase = record.get("phase")
        self.emit("model_decision", reward_chain_id=chain["reward_chain_id"] if chain else None,
                  offer_id=chain["page"]["offer_id"] if chain and chain["page"] else None,
                  event_id=self.event["event_id"] if phase == "EVENT" and self.event else None,
                  stock_scan_id=self.stock_scan_id if phase == "SHOP" else None, decision=reduced,
                  observation_context={
                      "basis": "latest_captured_observations_not_model_input",
                      "note_state_record_id": self.last_note_state_record_id,
                      "wallet_record_id": (self.last_wallet or {}).get("record_id"),
                      "floor_record_id": (self.last_observed_floor_context or {}).get("record_id"),
                  })

    def event_menu(self, choices):
        if not self.ensure_run("event_menu"):
            return
        menu = dict(question=self.event_question, choices=snapshot(choices))
        signature = json.dumps(menu, sort_keys=True, ensure_ascii=False)
        repeated = (self.event and self.event["signature"] == signature
                    and not self.event["clicked"] and not self.event.get("reward_boundary_seen"))
        if not repeated:
            if self.reward:
                self.reward["closed"] = True
            self.event = dict(event_id=uuid.uuid4().hex, signature=signature, clicked=False, **menu)
        self.emit("event_menu_reobserved" if repeated else "event_menu_observed",
                  event_id=self.event["event_id"], **menu, **self.owned_fields(),
                  tracker_estimate=self.inferred_state())

    def event_choice(self, result, attach):
        if not self.run_confirmed or self.event is None or result.box is None:
            return
        self.event["wallet_before"] = (snapshot(self.last_wallet)
                                       if result.detail.get("observed_coins") is not None else None)
        selected_box = snapshot(result.box)
        selected_ui_index = next((index for index, choice in enumerate(self.event["choices"])
                                  if choice.get("box") == selected_box), None)
        record_id = self.emit("event_choice", event_id=self.event["event_id"],
                  question=attach.get("last_question"), selected_text=attach.get("last_choice"),
                  displayed_consequence=attach.get("last_consequence"),
                  selected_box=selected_box, selected_ui_index=selected_ui_index,
                  model_mapping=snapshot(result.detail),
                  **{key: result.detail[key] for key in ("floor", "floor_confirmed", "floor_candidates")
                     if key in result.detail},
                  operation_confirmed=False)
        self.event["choice_record_id"] = record_id
        matches = [choice for choice in self.event["choices"] if choice.get("box") == selected_box]
        self.event["selection"] = (dict(record_id=record_id, selected_box=selected_box,
            selected_text=matches[0]["text"], displayed_consequence=matches[0]["consequence"])
            if len(matches) == 1 else None)

    def event_clicked(self, evidence):
        if not self.run_confirmed or self.event is None or self.event["clicked"]:
            return
        self.event["clicked"] = True
        self.event["click_record_id"] = self.emit("event_choice_click_observed", event_id=self.event["event_id"],
                  event_choice_record_id=self.event.get("choice_record_id"),
                  evidence=evidence, server_confirmed=False)
        selection = self.event.get("selection")
        self.event["potential_reward_pending"] = bool(selection and not self.event.get("reward_boundary_seen")
            and has_potential_reward(selection["displayed_consequence"]))

    def operation(self, event, handler, function, *args, before=False, offer_id=None, operation_id=None):
        if not self.ensure_run(event):
            return function(*args)
        data = handler.data
        initial = snapshot(data)
        common = dict(operation_id=operation_id or uuid.uuid4().hex, offer_id=offer_id,
                      reward_chain_id=self.reward["reward_chain_id"] if self.reward and not self.reward["closed"] else None,
                      stock_scan_id=self.stock_scan_id, arguments=snapshot(args),
                      source_type=getattr(data.params, "potential_source", None))
        if offer_id and self.reward and not self.reward["closed"]:
            common.update(self.reward["source_fields"], source_raw=data.params.potential_source)
        if before:
            self.emit(event + "_attempt", phase="before", state=initial, **self.owned_fields(), **common)
        with self.action_scope(**common), self.capture(event, **common) as read:
            result = function(*args)
        next_offer_id = (self.refresh_succeeded(offer_id, common["operation_id"])
                         if event == "refresh_result" and result else None)
        outcome = dict(previous_state=initial, result=snapshot(result),
                       Maa_reported_success=result if isinstance(result, bool) else None,
                       next_offer_id=next_offer_id, coin_before=initial.get("current_coin"),
                       read_id=read.read_id, recognition_record_ids=read.record_ids, **common)
        if event == "shop_refresh_result" and result is True and not handler.interactor.context.tasker.stopping:
            # The adapter publishes the final confirmed frame before completing this operation.
            outcome["refresh_remaining_after_action"] = data.refresh_remaining
            handler.pending_refresh = outcome
        else:
            self.emit(event, phase="after", state=snapshot(data),
                      coin_after=None if event == "shop_refresh_result" else getattr(data, "current_coin", None),
                      **self.owned_fields(), **outcome)
        return result

    def complete_shop_refresh(self, handler, read):
        if not self.enabled or not self.run_confirmed:
            return
        pending = handler.pending_refresh
        if pending is None:
            return
        handler.pending_refresh = None
        before, data = pending["previous_state"], handler.data
        wallet = self.last_wallet
        coin_after = (wallet["coin"] if wallet and wallet["coin_complete"]
                      and wallet["recognition_record_id"] in read.record_ids else None)
        coin_before, price = before.get("current_coin"), before.get("refresh_cost")
        debit_confirmed = (coin_before is not None and coin_before >= 0 and price is not None
                           and price > 0 and coin_after is not None and coin_after >= 0
                           and coin_before - coin_after == price)
        remaining_before, remaining_after = before.get("refresh_remaining"), data.refresh_remaining
        action_remaining = pending.get("refresh_remaining_after_action")
        remaining_decreased = (remaining_before is not None and remaining_before > 0
                               and (remaining_after == remaining_before - 1
                                    or action_remaining == remaining_before - 1))
        items = snapshot(data.items)
        stock_complete = (len(items) == len(data.params.item_rois)
                          and {item["grid_num"] for item in items} == set(range(1, len(items) + 1)))
        shelf_changed = items != before.get("items")
        success = True if debit_confirmed and stock_complete and (remaining_decreased or shelf_changed) else None
        pending.update(result=success, Maa_return_value=pending["result"])
        self.emit("shop_refresh_result", phase="after", state=snapshot(data), coin_after=coin_after,
                  success=success, operation_confirmed=success is True,
                  confirmation_basis="next_shelf_wallet_and_refresh_count_or_stock_change",
                  confirmation_status="confirmed" if success else "incomplete",
                  debit_confirmed=debit_confirmed, refresh_remaining_before=remaining_before,
                  refresh_remaining_after=remaining_after, refresh_remaining_decreased=remaining_decreased,
                  stock_read_complete=stock_complete, shelf_changed=shelf_changed,
                  next_stock_scan_id=self.stock_scan_id, stock_read_id=read.read_id,
                  stock_recognition_record_ids=read.record_ids,
                  wallet_observation_id=wallet["record_id"] if coin_after is not None else None,
                  **self.owned_fields(), **pending)

    def observe_shop(self, handler, read, previous_state):
        if not self.ensure_run("shop_page"):
            return
        data = handler.data
        remaining = data.refresh_remaining
        previous_generation = handler.stock_generation_id
        previous_remaining = next((count for count, generation in handler.stock_generations.items()
                                   if generation == previous_generation), None)
        pending = handler.pending_refresh
        refreshed = bool(pending and pending.get("result") is True)
        advanced = remaining >= 0 and previous_remaining is not None and remaining < previous_remaining
        generation = (uuid.uuid4().hex if refreshed or advanced or
                      (previous_generation is None and remaining >= 0) else previous_generation)
        generation_remaining = pending.get("refresh_remaining_after_action", remaining) if refreshed else remaining
        if generation is not None and generation_remaining >= 0 and (refreshed or advanced or previous_generation is None):
            handler.stock_generations[generation_remaining] = generation
        repeated = generation is not None and generation == previous_generation
        handler.stock_generation_id = generation
        if repeated:
            previous_items = {item["grid_num"]: item for item in previous_state.get("items", [])}
            identity = ("internal_name", "quantity", "price", "trekker_specified")
            for item in data.items:
                previous = previous_items.get(item.grid_num)
                if previous and all(previous.get(key) == getattr(item, key) for key in identity):
                    item.bought = item.bought or previous.get("bought", False)
                    item.checked = item.checked or previous.get("checked", False)
        items = snapshot(data.items)
        by_slot = {p["grid_num"]-1: p for p in items}
        slots = []
        for slot, rois in enumerate(data.params.item_rois):
            name = read.matching("星塔_节点_商店_购物_识别物品内容_agent", rois["name_roi"])
            price = read.matching(("星塔_神经网络_商店_识别物品价格",
                                   "星塔_节点_商店_购物_识别物品价格_agent"), rois["price_roi"])
            item = by_slot.get(slot)
            observed_item = dict(item) if item else None
            if observed_item:
                missing = observed_item.get("missing", [])
                for field, column in (("name", "internal_name"), ("quantity", "quantity"), ("price", "price")):
                    if field in missing:
                        observed_item[column] = None
                if "name" in missing:
                    observed_item["trekker_specified"] = None
            effective_price = observed_item.get("price") if observed_item else None
            slots.append(dict(slot=slot, **snapshot(rois),
                item_instance_id=f"{generation}:{slot}" if generation else None, item=observed_item,
                effective_price=effective_price if effective_price is not None and effective_price >= 0 else None,
                raw_ocr=dict(name=raw_ocr(name), price=raw_ocr(price)),
                recognition_record_ids=[r["record_id"] for r in name+price]))
        self.emit("shop_observed", phase="after", shop_event_type="stock_snapshot",
                  stock_scan_id=self.stock_scan_id, shop_visit_id=handler.shop_visit_id,
                  shop_entry_id=handler.shop_visit_id, stock_generation_id=generation,
                  shelf_generation_id=generation, previous_shelf_generation_id=previous_generation,
                  shelf_generation_basis=("confirmed_refresh_operation" if refreshed else
                      "confirmed_global_refresh_counter" if generation and not repeated
                      and getattr(data, "refresh_remaining_basis", "ocr") == "global_confirmed_refreshes_exhausted" else
                      "visible_remaining_refresh_count" if generation and not repeated else
                      "same_visit_without_confirmed_refresh" if repeated else "refresh_count_not_observed"),
                  shelf_generation_complete=generation is not None,
                  repeated_scan=repeated, refresh_remaining=remaining,
                  refresh_remaining_observed=getattr(data, "refresh_remaining_observed", remaining),
                  refresh_remaining_basis=getattr(data, "refresh_remaining_basis", "ocr"),
                  refresh_controls_expected_visible=(getattr(data, "refresh_remaining_basis", "ocr")
                                                     != "global_confirmed_refreshes_exhausted"),
                  stock_snapshot=items,
                  payload=dict(slots=slots, maa_items=items), state=snapshot(data),
                  previous_state=previous_state, coin=data.current_coin,
                  refresh_price=data.refresh_cost, read_id=read.read_id,
                  recognition_record_ids=read.record_ids, **self.owned_fields())
