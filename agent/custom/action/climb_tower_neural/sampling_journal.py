"""Whole-record appends and explicit quarantine of undecodable journal lines."""
import hashlib
import json
from pathlib import Path


DAMAGE_SUFFIX = ".journal-damage.json"
BAD_BYTES_SUFFIX = ".damaged-records"


class JournalRollbackError(OSError):
    def __init__(self, offset, error):
        super().__init__(f"Journal rollback to {offset} failed: {error}")
        self.offset = offset


def rollback(path, offset):
    with Path(path).open("r+b", buffering=0) as stream:
        stream.truncate(offset)


def append_record(path, data):
    """Retry the same bytes once, after undoing a write/flush/close failure."""
    for attempt in range(2):
        offset = None
        try:
            with Path(path).open("a+b", buffering=0) as stream:
                offset = stream.tell()
                if stream.write(data) != len(data):
                    raise OSError("Journal record was only partly written")
                stream.flush()
            return
        except OSError:
            if offset is not None:
                try:
                    rollback(path, offset)
                except OSError as error:
                    raise JournalRollbackError(offset, error) from error
            if attempt:
                raise


def _replace_bytes(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.write_bytes(data) != len(data):
        raise OSError("Journal repair was only partly written")
    temporary.replace(path)


def repair_journal(path):
    """Keep complete independent JSON objects; never guess within a bad line.

    Bad bytes stay local until the committed journal is recycled. Only byte
    counts, hashes and the affected sequence boundary enter structured data.
    """
    path = Path(path)
    metadata_path = path.with_suffix(DAMAGE_SUFFIX)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    valid, damaged, offset, previous = [], [], 0, {}
    changed = False
    for line in path.read_bytes().splitlines(keepends=True):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            damaged.append((line, dict(byte_offset=offset, byte_count=len(line),
                sha256=hashlib.sha256(line).hexdigest(),
                affected_from_sequence=(previous.get("sequence") or 0) + 1,
                affected_from_run_sequence=(previous.get("run_sequence") or 0) + 1)))
            changed = True
        else:
            if not isinstance(row, dict):
                raise ValueError("Journal records must be JSON objects")
            for _, fragment in damaged:
                if "affected_through_missing_run_sequence" not in fragment:
                    fragment["affected_through_missing_run_sequence"] = (row.get("run_sequence") or 1) - 1
            valid.append(line if line.endswith(b"\n") else line + b"\n")
            changed |= not line.endswith(b"\n")
            previous = row
        offset += len(line)
    if damaged:
        raw_path = path.with_suffix(BAD_BYTES_SUFFIX)
        old_bytes = raw_path.read_bytes() if metadata and raw_path.exists() else b""
        metadata = metadata or dict(fragments=[])
        for raw, fragment in damaged:
            fragment["quarantine_offset"] = len(old_bytes)
            old_bytes += raw
            metadata["fragments"].append(fragment)
        _replace_bytes(raw_path, old_bytes)
        _replace_bytes(metadata_path, (json.dumps(metadata, ensure_ascii=False) + "\n").encode("utf-8"))
    if changed:
        _replace_bytes(path, b"".join(valid))
    return metadata
