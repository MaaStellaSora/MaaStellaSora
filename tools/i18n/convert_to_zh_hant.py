#!/usr/bin/env python3
"""Convert Simplified Chinese i18n files to Traditional Chinese with Taiwan-style wording."""

import json
from pathlib import Path
from typing import Any

from opencc import OpenCC


ROOT = Path(__file__).resolve().parents[2]
I18N_FILES = (
    (ROOT / "assets/i18n/zh_cn.json", ROOT / "assets/i18n/zh_hant.json"),
    (
        ROOT / "agent/agent_config/i18n/zh_cn.json",
        ROOT / "agent/agent_config/i18n/zh_hant.json",
    ),
)


def convert_values(value: Any, converter: OpenCC) -> Any:
    if isinstance(value, str):
        return converter.convert(value)
    if isinstance(value, dict):
        return {key: convert_values(item, converter) for key, item in value.items()}
    if isinstance(value, list):
        return [convert_values(item, converter) for item in value]
    return value


def main() -> None:
    converter = OpenCC("s2twp")
    outputs = []
    for source, target in I18N_FILES:
        with source.open(encoding="utf-8") as file:
            translations = json.load(file)
        if not isinstance(translations, dict):
            raise ValueError(f"Expected a JSON object in {source}")
        converted = convert_values(translations, converter)
        outputs.append((target, json.dumps(converted, ensure_ascii=False, indent=4) + "\n"))

    for target, content in outputs:
        target.write_text(content, encoding="utf-8")
        print(f"Converted {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
