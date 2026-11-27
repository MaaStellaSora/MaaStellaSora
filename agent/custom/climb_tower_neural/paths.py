"""Locate shipped data independently of the importing module's directory."""
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent


def data_path(name, *, module_root=MODULE_ROOT):
    return Path(module_root) / "data" / name
