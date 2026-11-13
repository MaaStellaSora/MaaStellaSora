"""Compact tower catalogues for distribution, or restore the maintenance format."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[2] / "agent/custom/action/climb_tower_neural"
NAME_FIELDS = ("name_ja_jp", "name_zh_cn")
NAME_PROOF_FIELDS = {"source_language", "exact_name", "item_title_key", "localization_key"}


def name_key(value):
    return re.sub(r"\W", "", unicodedata.normalize("NFKC", value or ""))


def unique_potentials(catalog):
    """Match the maintenance tool's merge, retaining both languages' evidence."""
    rows = {}
    for aliases in catalog["by_normalized_name"].values():
        for row in aliases:
            merged = rows.setdefault(row["potential_id"], {})
            proofs = list(merged.get("name_provenance", ()))
            for proof in row.get("name_provenance", ()):
                if proof not in proofs:
                    proofs.append(proof)
            merged.update({key: value for key, value in row.items() if value is not None})
            if proofs:
                merged["name_provenance"] = proofs
    return sorted(rows.values(), key=lambda row: row["potential_id"])


def expand(catalog):
    if "potentials" not in catalog:
        return deepcopy(catalog)
    result = deepcopy(catalog)
    sources = result.pop("name_provenance_sources")
    rows = result.pop("potentials")
    for row in rows:
        row["name_provenance"] = [dict(sources[proof["source_index"]], **{
            key: value for key, value in proof.items() if key != "source_index"})
            for proof in row.get("name_provenance", ())]
    aliases = {}
    for row in rows:
        for key in sorted({name_key(row.get(field)) for field in NAME_FIELDS} - {""}):
            aliases.setdefault(key, []).append(row)
    result["schema_version"] = result.pop("source_schema_version")
    result["by_normalized_name"] = dict(sorted(aliases.items()))
    return result


def compact(catalog):
    result = expand(catalog)
    rows = unique_potentials(result)
    sources = []
    for row in rows:
        proofs = []
        for proof in row.get("name_provenance", ()):
            source = {key: value for key, value in proof.items() if key not in NAME_PROOF_FIELDS}
            if source not in sources:
                sources.append(source)
            proofs.append(dict(source_index=sources.index(source), **{
                key: value for key, value in proof.items() if key in NAME_PROOF_FIELDS}))
        if "name_provenance" in row:
            row["name_provenance"] = proofs
    result["source_schema_version"] = result["schema_version"]
    result["schema_version"] = 3
    result.pop("by_normalized_name")
    result["name_provenance_sources"] = sources
    result["potentials"] = rows
    return result


def write_catalog(path, value, template):
    original = template.decode("utf-8")
    indentation = re.search(r"\n([ \t]+)\S", original)
    text = json.dumps(value, ensure_ascii=False, indent=indentation[1] if indentation else None)
    if original.endswith("\n"):
        text += "\n"
    newline = "\r\n" if "\r\n" in original else "\n"
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compact", "expand"))
    parser.add_argument("source", type=Path, help="Directory containing the three tower catalogues")
    parser.add_argument("destination", type=Path, nargs="?", default=ROOT)
    args = parser.parse_args()
    original = (args.source / "potential_name_catalog.json").read_bytes()
    catalog = json.loads(original.decode("utf-8-sig"))
    converted = compact(catalog) if args.command == "compact" else expand(catalog)
    args.destination.mkdir(parents=True, exist_ok=True)
    target = args.destination / "potential_name_catalog.json"
    template = target.read_bytes() if target.exists() else original
    write_catalog(target, converted, template)
    # These small catalogues already store each identity once. Copy them intact
    # so expanded output is ready for the existing three-catalogue updater.
    if args.source.resolve() != args.destination.resolve():
        for name in ("character_name_catalog.json", "note_rules.json"):
            (args.destination / name).write_bytes((args.source / name).read_bytes())
    print(f"{args.command}: {len(unique_potentials(expand(converted)))} potentials -> {target}")


if __name__ == "__main__":
    main()
