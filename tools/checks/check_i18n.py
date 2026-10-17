import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


CHINESE_TEXT = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
I18N_FIELDS = {"contact", "description", "label", "license", "pattern_msg", "title"}
I18N_REFERENCE_FIELDS = I18N_FIELDS | {"icon", "welcome"}
DISPLAY_ARRAYS = {
    "cases",
    "controller",
    "group",
    "hotkey",
    "inputs",
    "preset",
    "pretask",
    "resource",
    "setting",
    "task",
}
SERVER_SUFFIX = re.compile(r"_(?:cn|tw|jp|en|kr)\.json$", re.IGNORECASE)


@dataclass
class Finding:
    message: str
    location: str


@dataclass
class ScanResult:
    references: dict[str, list[str]] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    def add_reference(self, key: str, location: str) -> None:
        self.references.setdefault(key, []).append(location)


class CheckScope(Protocol):
    name: str

    def scan(self, root: Path, interface: dict[str, Any]) -> ScanResult: ...


class DuplicateKeyError(ValueError):
    pass


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strip_json_extensions(text: str) -> str:
    """Remove JSON comments and trailing commas without changing quoted strings."""
    output = []
    index = 0
    in_string = False
    escaped = False

    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""

        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            output.append(char)
            index += 1
        elif char == "/" and following == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
        elif char == "/" and following == "*":
            index += 2
            while index + 1 < len(text) and text[index : index + 2] != "*/":
                if text[index] in "\r\n":
                    output.append(text[index])
                index += 1
            index = min(index + 2, len(text))
        elif char == ",":
            lookahead = index + 1
            while lookahead < len(text):
                if text[lookahead].isspace():
                    lookahead += 1
                elif text.startswith("//", lookahead):
                    newline = text.find("\n", lookahead + 2)
                    lookahead = len(text) if newline == -1 else newline + 1
                elif text.startswith("/*", lookahead):
                    end_comment = text.find("*/", lookahead + 2)
                    lookahead = len(text) if end_comment == -1 else end_comment + 2
                else:
                    break
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
            else:
                output.append(char)
                index += 1
        else:
            output.append(char)
            index += 1

    return "".join(output)


def load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(strip_json_extensions(text), object_pairs_hook=reject_duplicate_keys)


def collect_strings(
    value: Any,
    location: str,
    result: ScanResult,
    parent_key: str = "",
    in_preset_task: bool = False,
) -> None:
    if isinstance(value, dict):
        name = value.get("name")
        if (
            parent_key in DISPLAY_ARRAYS
            and not in_preset_task
            and "label" not in value
            and isinstance(name, str)
            and CHINESE_TEXT.search(name)
        ):
            result.findings.append(
                Finding("Chinese name is displayed as a fallback; add an i18n label", location)
            )

        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key in I18N_REFERENCE_FIELDS and isinstance(child, str) and child.startswith("$"):
                i18n_key = child[1:]
                if not i18n_key:
                    result.findings.append(Finding("empty i18n key", child_location))
                else:
                    result.add_reference(i18n_key, child_location)

            if (
                key in I18N_FIELDS
                and isinstance(child, str)
                and not child.startswith("$")
                and CHINESE_TEXT.search(child)
            ):
                result.findings.append(
                    Finding("hardcoded Chinese in an i18n-capable display field", child_location)
                )

            if key == "pipeline_override":
                continue
            collect_strings(
                child,
                child_location,
                result,
                key,
                in_preset_task or (parent_key == "preset" and key == "task"),
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            collect_strings(
                child,
                f"{location}[{index}]",
                result,
                parent_key,
                in_preset_task,
            )


def imported_task_paths(root: Path, interface: dict[str, Any], result: ScanResult) -> list[Path]:
    paths = []
    imports = interface.get("import", [])
    if not isinstance(imports, list):
        result.findings.append(
            Finding("interface.import must be an array", "assets/interface.json")
        )
        return paths

    assets = (root / "assets").resolve()
    for index, imported in enumerate(imports):
        location = f"assets/interface.json.import[{index}]"
        if not isinstance(imported, str):
            result.findings.append(Finding("import path must be a string", location))
            continue
        if SERVER_SUFFIX.search(Path(imported).name):
            continue

        path = (assets / imported).resolve()
        if not path.is_relative_to(assets):
            result.findings.append(Finding("import path escapes the assets directory", location))
            continue
        paths.append(path)
    return paths


class InterfaceScope:
    name = "interface"

    def scan(self, root: Path, interface: dict[str, Any]) -> ScanResult:
        result = ScanResult()
        sources = [root / "assets" / "interface.json"]
        sources.extend(imported_task_paths(root, interface, result))

        for path in sources:
            try:
                source = load_json(path)
            except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError) as error:
                result.findings.append(Finding(f"cannot read JSON: {error}", str(path)))
                continue
            collect_strings(source, path.relative_to(root).as_posix(), result)
        return result


# Add future checks (for example, Pipeline sources) by implementing CheckScope here.
CHECK_SCOPES: tuple[CheckScope, ...] = (InterfaceScope(),)


def load_translations(
    root: Path, interface: dict[str, Any]
) -> tuple[dict[str, dict[str, str]], list[Finding]]:
    translations = {}
    findings = []
    languages = interface.get("languages")
    if not isinstance(languages, dict) or not languages:
        return {}, [
            Finding(
                "interface.languages must map locales to translation files",
                "assets/interface.json.languages",
            )
        ]

    interface_dir = (root / "assets").resolve()
    for locale, relative_path in languages.items():
        location = f"assets/interface.json.languages.{locale}"
        if not isinstance(relative_path, str):
            findings.append(Finding("translation path must be a string", location))
            continue
        path = (interface_dir / relative_path).resolve()
        if not path.is_relative_to(interface_dir):
            findings.append(Finding("translation path escapes the assets directory", location))
            continue
        try:
            data = load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError) as error:
            findings.append(Finding(f"cannot read translation file: {error}", location))
            continue
        if not isinstance(data, dict):
            findings.append(Finding("translation file must contain a flat JSON object", str(path)))
            continue

        invalid = [key for key, value in data.items() if not isinstance(value, str)]
        if invalid:
            findings.append(
                Finding(f"translation values must be strings: {', '.join(invalid)}", str(path))
            )
            continue
        translations[locale] = data
    return translations, findings


def check(root: Path) -> list[Finding]:
    interface_path = root / "assets" / "interface.json"
    try:
        interface = load_json(interface_path)
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError) as error:
        return [Finding(f"cannot read interface JSON: {error}", str(interface_path))]
    if not isinstance(interface, dict):
        return [Finding("interface.json must contain a JSON object", str(interface_path))]

    result = ScanResult()
    for scope in CHECK_SCOPES:
        scoped = scope.scan(root, interface)
        for key, locations in scoped.references.items():
            result.references.setdefault(key, []).extend(locations)
        result.findings.extend(scoped.findings)

    translations, translation_findings = load_translations(root, interface)
    result.findings.extend(translation_findings)
    for locale, entries in translations.items():
        missing = result.references.keys() - entries.keys()
        unused = entries.keys() - result.references.keys()
        for key in sorted(missing):
            result.findings.append(
                Finding(
                    f"missing translation for locale {locale}: {key}",
                    ", ".join(result.references[key]),
                )
            )
        for key in sorted(unused):
            result.findings.append(
                Finding(f"unused translation for locale {locale}: {key}", f"languages.{locale}")
            )
    return result.findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Check i18n strings and translation-key coverage.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="project root (defaults to the repository containing this script)",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    findings = check(root)
    if findings:
        for finding in findings:
            print(f"ERROR: {finding.location}: {finding.message}")
        print(f"i18n check failed with {len(findings)} error(s).")
        return 1

    print("i18n check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
