import argparse
import ast
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
LANGUAGE_SUFFIX = re.compile(r"_(?:cn|tw|jp|en|kr)\.json$", re.IGNORECASE)


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


def is_language_specific(path: Path) -> bool:
    return LANGUAGE_SUFFIX.search(path.name) is not None


def collect_focus_template(value: Any, location: str, result: ScanResult) -> None:
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            collect_focus_template(content, f"{location}.content", result)
    elif isinstance(value, str):
        if value.startswith("$"):
            i18n_key = value[1:]
            if not i18n_key:
                result.findings.append(Finding("empty i18n key", location))
            else:
                result.add_reference(i18n_key, location)
        elif CHINESE_TEXT.search(value):
            result.findings.append(
                Finding("hardcoded Chinese in a Pipeline focus message", location)
            )


def collect_pipeline_override_focus(pipeline: Any, location: str, result: ScanResult) -> None:
    if not isinstance(pipeline, dict):
        return

    for node_name, node in pipeline.items():
        if not isinstance(node, dict):
            continue
        focus = node.get("focus")
        if not isinstance(focus, dict):
            continue
        for message, template in focus.items():
            collect_focus_template(
                template,
                f"{location}.{node_name}.focus.{message}",
                result,
            )


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
        if is_language_specific(Path(imported)):
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


class PipelineOverrideScope:
    name = "pipeline_override"

    def scan(self, root: Path, interface: dict[str, Any]) -> ScanResult:
        result = ScanResult()
        for path in imported_task_paths(root, interface, result):
            try:
                source = load_json(path)
            except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError) as error:
                result.findings.append(Finding(f"cannot read JSON: {error}", str(path)))
                continue
            self._scan_value(source, path.relative_to(root).as_posix(), result)
        return result

    def _scan_value(self, value: Any, location: str, result: ScanResult) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_location = f"{location}.{key}"
                if key == "pipeline_override":
                    collect_pipeline_override_focus(child, child_location, result)
                    continue

                self._scan_value(child, child_location, result)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self._scan_value(child, f"{location}[{index}]", result)


class PipelineScope:
    name = "pipeline"

    def scan(self, root: Path, interface: dict[str, Any]) -> ScanResult:
        result = ScanResult()
        pipeline_root = root / "assets" / "resource"
        for path in sorted(pipeline_root.rglob("*.json")):
            relative_path = path.relative_to(root)
            if is_language_specific(path) or any(
                part.startswith(".") for part in relative_path.parts
            ):
                continue
            try:
                source = load_json(path)
            except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError) as error:
                result.findings.append(Finding(f"cannot read JSON: {error}", str(path)))
                continue
            if not isinstance(source, dict):
                continue

            location = relative_path.as_posix()
            for node_name, node in source.items():
                if not isinstance(node, dict) or not isinstance(node.get("focus"), dict):
                    continue
                for message, template in node["focus"].items():
                    collect_focus_template(
                        template,
                        f"{location}.{node_name}.focus.{message}",
                        result,
                    )
        return result


class AgentScope:
    name = "agent"

    def scan(self, root: Path, interface: dict[str, Any]) -> ScanResult:
        result = ScanResult()
        agent_root = root / "agent"
        for path in sorted(agent_root.rglob("*.py")):
            if path == agent_root / "utils" / "logger.py":
                continue
            location = path.relative_to(root).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=location)
            except (OSError, UnicodeError, SyntaxError) as error:
                result.findings.append(Finding(f"cannot read Python source: {error}", location))
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self._collect_key(node.value, location, node, result)
                if not isinstance(node, ast.Call):
                    continue
                if self._is_logger_call(node):
                    self._scan_log_call(node, location, result)
        return result

    @staticmethod
    def _is_logger_call(node: ast.Call) -> bool:
        if not isinstance(node.func, ast.Attribute) or node.func.attr not in {
            "info",
            "warning",
            "error",
        }:
            return False
        receiver = node.func.value
        return (isinstance(receiver, ast.Name) and receiver.id == "logger") or (
            isinstance(receiver, ast.Attribute) and receiver.attr == "logger"
        )

    def _scan_log_call(self, node: ast.Call, location: str, result: ScanResult) -> None:
        if not node.args:
            return
        message = node.args[0]
        if isinstance(message, ast.Constant) and isinstance(message.value, str):
            if message.value.startswith("$agent."):
                return
            text = message.value
        elif isinstance(message, ast.JoinedStr):
            parts = []
            for item in message.values:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    parts.append(item.value)
            text = "".join(parts)
        else:
            return

        if CHINESE_TEXT.search(text):
            result.findings.append(
                Finding(
                    "hardcoded Chinese in an Agent info/warning/error log",
                    self._node_location(location, message),
                )
            )

    @staticmethod
    def _collect_key(value: str, location: str, node: ast.AST, result: ScanResult) -> bool:
        if not value.startswith("$agent."):
            return False
        key = value[1:]
        if key == "agent.":
            result.findings.append(
                Finding("empty i18n key", AgentScope._node_location(location, node))
            )
        else:
            result.add_reference(key, AgentScope._node_location(location, node))
        return True

    @staticmethod
    def _node_location(location: str, node: ast.AST) -> str:
        line = getattr(node, "lineno", None)
        return f"{location}:{line}" if isinstance(line, int) else location


CHECK_SCOPES: tuple[CheckScope, ...] = (
    InterfaceScope(),
    PipelineOverrideScope(),
    PipelineScope(),
    AgentScope(),
)


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
        data, finding = load_translation_file(path, location)
        if finding:
            findings.append(finding)
        else:
            translations[locale] = data
    return translations, findings


def load_translation_file(path: Path, location: str) -> tuple[dict[str, str], Finding | None]:
    try:
        data = load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError) as error:
        return {}, Finding(f"cannot read translation file: {error}", location)
    if not isinstance(data, dict):
        return {}, Finding("translation file must contain a flat JSON object", location)

    invalid = [key for key, value in data.items() if not isinstance(value, str)]
    if invalid:
        return {}, Finding(f"translation values must be strings: {', '.join(invalid)}", location)
    return data, None


def load_agent_translations(root: Path) -> tuple[dict[str, dict[str, str]], list[Finding]]:
    translations = {}
    findings = []
    i18n_root = root / "agent" / "agent_config" / "i18n"
    if not i18n_root.is_dir():
        return {}, [
            Finding("Agent i18n directory must contain zh_cn.json", str(i18n_root))
        ]

    for path in sorted(i18n_root.glob("*.json")):
        locale = path.stem
        data, finding = load_translation_file(path, str(path))
        if finding:
            findings.append(finding)
        else:
            translations[locale] = data
    if "zh_cn" not in translations:
        findings.append(
            Finding("Agent i18n directory must contain zh_cn.json", str(i18n_root))
        )
    return translations, findings


def check_translation_coverage(
    references: dict[str, list[str]],
    translations: dict[str, dict[str, str]],
    location_root: str = "languages",
) -> list[Finding]:
    findings = []
    for locale, entries in translations.items():
        missing = references.keys() - entries.keys()
        unused = entries.keys() - references.keys()
        for key in sorted(missing):
            findings.append(
                Finding(
                    f"missing translation for locale {locale}: {key}",
                    ", ".join(references[key]),
                )
            )
        for key in sorted(unused):
            findings.append(
                Finding(
                    f"unused translation for locale {locale}: {key}",
                    f"{location_root}.{locale}",
                )
            )
    return findings


def check(root: Path) -> list[Finding]:
    root = root.resolve()
    interface_path = root / "assets" / "interface.json"
    try:
        interface = load_json(interface_path)
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError) as error:
        return [Finding(f"cannot read interface JSON: {error}", str(interface_path))]
    if not isinstance(interface, dict):
        return [Finding("interface.json must contain a JSON object", str(interface_path))]

    result = ScanResult()
    agent_result = ScanResult()
    for scope in CHECK_SCOPES:
        scoped = scope.scan(root, interface)
        target = agent_result if scope.name == "agent" else result
        for key, locations in scoped.references.items():
            target.references.setdefault(key, []).extend(locations)
        result.findings.extend(scoped.findings)

    translations, translation_findings = load_translations(root, interface)
    result.findings.extend(translation_findings)
    result.findings.extend(check_translation_coverage(result.references, translations))

    agent_translations, agent_translation_findings = load_agent_translations(root)
    result.findings.extend(agent_translation_findings)
    result.findings.extend(
        check_translation_coverage(
            agent_result.references,
            agent_translations,
            "agent/agent_config/i18n",
        )
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
