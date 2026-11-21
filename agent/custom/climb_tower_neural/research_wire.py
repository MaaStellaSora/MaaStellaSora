"""Keep compact research content inside the deployed public-v3 envelope.

The receiver accepts schema 1 and its eight table names.  The content version
separately identifies the compact rows; the four unused legacy tables stay
empty.  No legacy identity or runtime fields are reconstructed for transport.
"""


WIRE_SCHEMA_VERSION = 1
WIRE_TABLES = ("presets", "runs", "records", "objects", "record_objects",
               "readings", "record_readings", "quality_issues")


def upload_payload(payload):
    tables = payload["tables"]
    return dict(payload, schema_version=WIRE_SCHEMA_VERSION,
                content_schema_version=payload["schema_version"],
                tables={name: tables[name] if name in tables else [] for name in WIRE_TABLES})
