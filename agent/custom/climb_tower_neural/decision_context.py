"""Snapshot the visible candidates used by the frozen potential-only adapter."""
from copy import deepcopy


def decision_context(runtime, kind, payload):
    return deepcopy(dict(payload, kind=kind))
