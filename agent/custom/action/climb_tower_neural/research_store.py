"""Live normalized storage; JSONL exists only as the active task/run journal."""
import ctypes
from pathlib import Path
import time

from .research_data import build_payload, import_payload
from .sampling_journal import BAD_BYTES_SUFFIX, DAMAGE_SUFFIX


DATABASE_FILE = "research.sqlite3"
CHECKPOINT_RECORDS = 256
CHECKPOINT_SECONDS = 30
CONSENT_SOURCE = "task.attach.research_upload"


def valid_upload_consent(manifest, policy_version=None):
    consent = manifest.get("research_upload", {})
    return (isinstance(consent, dict) and consent.get("enabled") is True
            and consent.get("source") == CONSENT_SOURCE
            and bool(consent.get("policy_version"))
            and (policy_version is None or consent["policy_version"] == policy_version)
            and bool(consent.get("granted_at"))
            and consent.get("run_id") == manifest.get("run_id"))


def recycle_file(path):
    """Move a committed journal to Windows' Recycle Bin without showing UI."""
    path = Path(path)
    if not path.exists():
        return

    class FileOperation(ctypes.Structure):
        _fields_ = [("hwnd", ctypes.c_void_p), ("operation", ctypes.c_uint),
                    ("source", ctypes.c_wchar_p), ("destination", ctypes.c_wchar_p),
                    ("flags", ctypes.c_ushort), ("aborted", ctypes.c_int),
                    ("mappings", ctypes.c_void_p), ("progress", ctypes.c_wchar_p)]

    operation = FileOperation(operation=3, source=str(path.resolve()) + "\0\0",
                              flags=0x0040 | 0x0010 | 0x0004 | 0x0400)
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    if result or operation.aborted:
        raise OSError(result, f"Could not recycle committed journal: {path.name}")


class ResearchStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.database_path = self.directory / DATABASE_FILE
        self._checkpoints = {}

    def observe(self, run_path, event_type):
        """Checkpoint batches, while the append-only journal persists every row."""
        path = Path(run_path)
        count, previous = self._checkpoints.get(path.stem, (0, time.monotonic()))
        count += 1
        now = time.monotonic()
        self._checkpoints[path.stem] = (count, previous)
        if event_type != "exploration_end" and (event_type in ("exploration_start", "exploration_start_confirmed")
                or count >= CHECKPOINT_RECORDS or now - previous >= CHECKPOINT_SECONDS):
            self.checkpoint(path)

    def checkpoint(self, run_path):
        """Commit all evidence and quality together, using the shared converter."""
        path = Path(run_path)
        import_payload(self.database_path, build_payload(path))
        self._checkpoints[path.stem] = (0, time.monotonic())

    def release_run(self, run_path):
        """Recycle an open marker and its journal once the run is committed."""
        path = Path(run_path)
        recycle_file(self.directory / "queue" / f"{path.stem}.open.json")
        recycle_file(path)
        recycle_file(path.with_suffix(".interrupted-tail"))
        recycle_file(path.with_suffix(BAD_BYTES_SUFFIX))
        recycle_file(path.with_suffix(DAMAGE_SUFFIX))
        self._checkpoints.pop(path.stem, None)

    def release_session(self, session_id):
        """Keep one task's preset evidence available to all its consecutive runs."""
        import json
        for marker in (self.directory / "queue").glob("*.open.json"):
            manifest = json.loads(marker.read_text(encoding="utf-8"))
            if manifest.get("session_id") == session_id:
                return
        recycle_file(self.directory / "sessions" / f"{session_id}.jsonl")
