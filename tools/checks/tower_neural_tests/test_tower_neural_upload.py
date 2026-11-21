"""Exercise the compact research ZIP using an entirely offline transport."""
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch
import zipfile

from tower_temp_directory import TowerTempDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural import research_data, sample_upload
from custom.climb_tower_neural.research_wire import WIRE_TABLES


class TowerNeuralUploadTests(unittest.TestCase):
    def test_compact_zip_round_trips_through_the_public_v3_envelope(self):
        with TowerTempDirectory() as temporary:
            app = Path(temporary)
            directory = app / "debug/tower_sampling"
            run_id = "a" * 32
            events = [
                dict(event_type="exploration_start", full_run_start_observed=True,
                     original_preset={"party": []}, owned=[], owned_complete=True),
                dict(event_type="note_delta", operation_id="battle-1", source_type="battle",
                     floor=1, before={"alpha": 0}, after={"alpha": 2}, delta={"alpha": 2},
                     scope_melodies=["alpha"], independently_observed=True,
                     complete=True, analysis_eligible=True, probability_eligible=True),
                dict(event_type="exploration_end", reason="saved", completion_observed=True),
            ]
            rows = [dict(event, record_id=f"record-{index}", run_id=run_id,
                         sequence=index, run_sequence=index, lang_type="cn",
                         participant_id="private-participant", session_id="private-session",
                         task_id="private-task", target_vector=[3], target_vector_complete=True)
                    for index, event in enumerate(events, 1)]
            journal = directory / "runs" / f"{run_id}.jsonl"
            journal.parent.mkdir(parents=True)
            journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            research_data.import_payload(directory / "research.sqlite3", research_data.build_payload(journal))
            original = research_data.export_payload(directory / "research.sqlite3", run_id)
            (app / "upload_settings.json").write_text(json.dumps(dict(
                endpoint="https://offline.invalid/v3/runs", access_token="",
                policy_version="offline-fixture", recipient="fixture", purpose="fixture")), encoding="utf-8")
            uploader = sample_upload.SampleUpload(app, app, directory, Mock())
            uploader.set_consent(True)
            manifest = dict(run_id=run_id, research_upload=uploader.consent_for_run(run_id))
            marker = directory / "queue" / f"{run_id}.ready.json"
            marker.parent.mkdir()
            marker.write_text(json.dumps(manifest), encoding="utf-8")
            requests = []

            @contextmanager
            def receive(request, timeout):
                requests.append(request)
                yield SimpleNamespace(status=201)

            uploader.opener = Mock()
            uploader.opener.open.side_effect = receive
            with patch.object(uploader, "_recycle") as recycle:
                uploader.send(marker, manifest)
            recycle.assert_has_calls([call(marker), call(uploader._retry_path(marker))])
            request, = requests
            self.assertEqual(request.get_header("X-upload-id"), run_id)
            self.assertEqual(request.get_header("X-content-sha256"), hashlib.sha256(request.data).hexdigest())
            self.assertIsNone(request.get_header("Authorization"))
            with zipfile.ZipFile(io.BytesIO(request.data)) as archive:
                self.assertEqual(set(archive.namelist()), {"manifest.json", "research.json"})
                received = json.loads(archive.read("research.json"))
                sent_manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual((received["schema_version"], received["content_schema_version"]), (1, 2))
            self.assertEqual(set(received["tables"]), set(WIRE_TABLES))
            for table in ("objects", "record_objects", "readings", "record_readings"):
                self.assertEqual(received["tables"][table], [])
            self.assertEqual(sent_manifest["schema_version"], 1)
            self.assertEqual(sent_manifest["content_schema_version"], 2)
            self.assertEqual(sent_manifest["research_upload"], manifest["research_upload"])
            self.assertFalse(sent_manifest["images_included"])
            for private in ("private-participant", "private-session", "private-task", "participant_id"):
                self.assertNotIn(private, json.dumps(received))
            restored = app / "received.sqlite3"
            research_data.import_payload(restored, received)
            self.assertEqual(research_data.export_payload(restored, run_id), original)
            uploader.set_consent(False)
            uploader.send(marker, manifest)
            self.assertEqual(len(requests), 1)


if __name__ == "__main__":
    unittest.main()
