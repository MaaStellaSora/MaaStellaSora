"""Silently upload normalized runs after explicit task checkbox consent."""
from pathlib import Path
import hashlib
import http.client
import io
import json
import random
import sqlite3
import time
import threading
import urllib.error
import urllib.request
import zipfile

from .sampling_data import timestamp, write_json
from .research_data import FORMAT, SCHEMA_VERSION, export_payload
from .research_store import CONSENT_SOURCE, DATABASE_FILE, recycle_file, valid_upload_consent


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


class SampleUpload:
    def __init__(self, app_dir, module_root, directory, logger):
        self.app_dir, self.module_root, self.directory = Path(app_dir), Path(module_root), Path(directory)
        self.logger = logger
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.thread = None
        self.attempt_finished = threading.Condition()
        self.opener = urllib.request.build_opener(_RejectRedirects())
        self.settings = json.loads((self.module_root / "upload_settings.json").read_text(encoding="utf-8-sig"))
        if not isinstance(self.settings, dict):
            raise ValueError("upload settings must be an object")
        local_settings = self.app_dir / "config/neural_network/research_upload.json"
        if local_settings.exists():
            local = json.loads(local_settings.read_text(encoding="utf-8-sig"))
            if not isinstance(local, dict):
                raise ValueError("upload settings must be an object")
            self.settings.update({key: local[key] for key in ("endpoint", "access_token") if key in local})
        if not all(isinstance(self.settings.get(key), str)
                for key in ("endpoint", "access_token", "policy_version", "recipient", "purpose")):
            raise ValueError("invalid upload settings")
        self.consent_enabled = False
        self.consent_granted_at = None

    def set_consent(self, enabled):
        """Apply this task's explicit option; a later task must set it again."""
        self.consent_enabled = enabled is True
        self.consent_granted_at = timestamp() if self.consent_enabled else None
        self.wake.set()

    def consent_for_run(self, run_id):
        """Freeze the task's decision in the new run's persistent manifest."""
        return dict(enabled=self.consent_enabled, source=CONSENT_SOURCE,
                    policy_version=self.settings["policy_version"],
                    granted_at=self.consent_granted_at, run_id=run_id)

    def upload_allowed(self, manifest=None):
        if not self.consent_enabled:
            return False
        return manifest is None or valid_upload_consent(manifest, self.settings["policy_version"])

    def start(self):
        if not self.settings["endpoint"] or self.thread is not None:
            return
        self.thread = threading.Thread(target=self._worker, name="tower-sample-upload", daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stop.is_set():
            self.wake.clear()
            try:
                delay = self.flush()
            except (OSError, ValueError):
                delay = 30
            self.wake.wait(delay)

    def finish_run(self, marker):
        """Give the queued run one attempt before Maa can close the Agent."""
        if self.thread is None or not self.upload_allowed():
            return
        with self.attempt_finished:
            self.wake.set()
            self.attempt_finished.wait_for(
                lambda: not marker.exists() or self._retry_path(marker).exists(), timeout=5)

    def _retry_path(self, marker):
        return self.directory / "retries" / marker.name

    def _defer(self, marker, state, permanent=False):
        attempts = state.get("attempts", 0) + 1
        delay = 1800 if permanent else min(120, 3 * 2 ** min(attempts - 1, 6))
        write_json(self._retry_path(marker), {
            "attempts": attempts, "next_attempt": time.time() + delay + random.uniform(0, delay * 0.25)
        })

    def flush(self):
        delay = 30
        if not self.settings["endpoint"] or not self.upload_allowed():
            return delay
        # Serve the just-finished run ahead of older retries. Retry metadata stays
        # outside research data and does not change the ready marker's age.
        for marker in sorted((self.directory / "queue").glob("*.ready.json"),
                             key=lambda path: path.stat().st_mtime_ns, reverse=True):
            if not self.upload_allowed():
                return delay
            retry = self._retry_path(marker)
            state = json.loads(retry.read_text(encoding="utf-8")) if retry.exists() else {}
            remaining = state.get("next_attempt", 0) - time.time()
            if remaining > 0:
                delay = min(delay, remaining)
                continue
            try:
                manifest = json.loads(marker.read_text(encoding="utf-8"))
                if not self.upload_allowed(manifest):
                    continue
                self.send(marker, manifest)
            except urllib.error.HTTPError as error:
                code = error.code
                error.close()
                self.logger.debug("Tower upload %s: HTTP %s; queued for retry", marker.stem, code)
                self._defer(marker, state, permanent=400 <= code < 500 and code not in (408, 425, 429))
            except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, http.client.HTTPException) as error:
                self.logger.debug("Tower upload %s: %s; queued for retry", marker.stem, type(error).__name__)
                self._defer(marker, state)
            finally:
                with self.attempt_finished:
                    self.attempt_finished.notify_all()
            if retry.exists():
                next_attempt = json.loads(retry.read_text(encoding="utf-8"))["next_attempt"]
                delay = min(delay, max(0, next_attempt - time.time()))
            if self.wake.is_set():
                return 0  # A newly finished run preempts older retries.
        return delay

    def send(self, marker, manifest):
        if not self.upload_allowed(manifest):
            return
        run_id = manifest["run_id"]
        payload = export_payload(self.directory / DATABASE_FILE, run_id)
        package = io.BytesIO()
        with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.writestr("research.json", json.dumps(payload, ensure_ascii=False,
                separators=(",", ":"), allow_nan=False))
            archive.writestr("manifest.json", json.dumps(dict(run_id=run_id,
                format=FORMAT, schema_version=SCHEMA_VERSION,
                consent_source=CONSENT_SOURCE,
                consent_policy=self.settings["policy_version"],
                research_upload=manifest["research_upload"],
                recipient=self.settings["recipient"], purpose=self.settings["purpose"],
                images_included=False), ensure_ascii=False))
        raw = package.getvalue()
        headers = {"Content-Type": "application/zip", "X-Upload-ID": run_id,
                   "X-Content-SHA256": hashlib.sha256(raw).hexdigest()}
        if self.settings.get("access_token"):
            headers["Authorization"] = "Bearer " + self.settings["access_token"]
        request = urllib.request.Request(self.settings["endpoint"], data=raw, headers=headers, method="POST")
        if not self.upload_allowed(manifest):
            return
        # Acknowledgement is any HTTP 2xx; the receiver deduplicates X-Upload-ID.
        with self.opener.open(request, timeout=max(60, min(180, 20 + len(raw) * 8 / 3_400_000 * 3))) as response:
            if not 200 <= response.status < 300:
                raise OSError(f"接收端返回 HTTP {response.status}")
            self.logger.debug("Tower upload %s: HTTP %s", run_id, response.status)
        recycle_file(self._retry_path(marker))
        recycle_file(marker)
