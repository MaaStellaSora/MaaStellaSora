"""Resolve the active game's language and its catalogue name field."""
import json

if __package__:
    from .paths import data_path
else:
    from paths import data_path


_LOCALES = json.loads(data_path("locales.json").read_text(encoding="utf-8"))
_CATALOG_NAME_ALIASES = tuple(_LOCALES.pop("catalog_name_alias_fields"))
SUPPORTED_LANGUAGES = tuple(_LOCALES)
_ALIASES = {alias: language for language, entry in _LOCALES.items() for alias in entry["aliases"]}


def normalize_language(language):
    """Leave unknown values unchanged so the task entry can reject them."""
    return _ALIASES.get(language, language)


def catalog_name_field(language):
    """Preserve the existing Japanese fallback for incomplete preset metadata."""
    return _LOCALES.get(normalize_language(language), _LOCALES["jp"])["catalog_name_field"]


def catalog_name_fields():
    """Keep configured names and existing aliases without enabling extra servers."""
    return tuple(dict.fromkeys((*_CATALOG_NAME_ALIASES,
        *(entry["catalog_name_field"] for entry in _LOCALES.values()))))
