"""Ordered 36-slot presets and deterministic event seeds; live bags use DeploymentPreset."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

POOL_LENGTHS = (3, 9, 3, 9, 3, 9)
BUCKETS = ((72, 95), (96, 119), (120, 144))
BUCKET_NAMES = ("72–95", "96–119", "120–144")
ROLE_TARGET_RANGE = (18, 54)


def validate_role_totals(totals) -> tuple:
    totals = tuple(totals)
    if len(totals) != 3:
        raise ValueError("预设须包含三名角色的原始普通目标总级数。")
    for role, total in enumerate(totals, 1):
        if not ROLE_TARGET_RANGE[0] <= total <= ROLE_TARGET_RANGE[1]:
            raise ValueError(f"角色{role}的原始普通目标总级数 {total} 不在18–54；不包括核心。")
    return totals


def canonical_seed(*parts) -> int:
    """Domain-separated seed; never incorporates model IDs or checkpoints."""
    payload = json.dumps(parts, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def validate_pools(pools: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    ordered = tuple(tuple(row) for row in pools)
    if tuple(map(len, ordered)) != POOL_LENGTHS:
        raise ValueError("预设必须包含完整 36 个普通槽位，六池长度须为 3/9/3/9/3/9；保留零目标。")
    if any(type(v) is not int or not 0 <= v <= 6 for row in ordered for v in row):
        raise ValueError("每个普通目标等级必须是 0–6 的整数。")
    total = sum(map(sum, ordered))
    if not 72 <= total <= 144:
        raise ValueError(f"普通目标总级数 {total} 不在 72–144；不包括核心。")
    validate_role_totals(sum(ordered[2*r]) + sum(ordered[2*r+1]) for r in range(3))
    return ordered


@dataclass(frozen=True, slots=True)
class Preset:
    name: str
    pools: tuple
    group: str = "custom"

    def __post_init__(self):
        object.__setattr__(self, "pools", validate_pools(self.pools))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Preset name must be nonempty")

    @property
    def total(self) -> int:
        return sum(map(sum, self.pools))

    @property
    def cap(self) -> int:
        return self.total

    @property
    def role_totals(self) -> tuple[int, int, int]:
        return tuple(sum(self.pools[2*r]) + sum(self.pools[2*r+1]) for r in range(3))

    @property
    def flat(self) -> tuple[int, ...]:
        return tuple(v for row in self.pools for v in row)

    @property
    def bucket(self) -> int:
        return next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= self.total <= hi)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.pools, separators=(",", ":")).encode()).hexdigest()

    def to_dict(self) -> dict:
        return {"name": self.name, "pools": [list(p) for p in self.pools],
                "group": self.group, "total": self.total,
                "role_totals": list(self.role_totals), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: dict) -> "Preset":
        preset = cls(str(value["name"]), tuple(tuple(p) for p in value["pools"]), value.get("group", "custom"))
        if "total" in value and value["total"] != preset.total:
            raise ValueError("Preset total disagrees with the full target vector")
        if "role_totals" in value and tuple(value["role_totals"]) != preset.role_totals:
            raise ValueError("Preset role totals disagree with the full target vector")
        if "fingerprint" in value and value["fingerprint"] != preset.fingerprint:
            raise ValueError("Preset fingerprint mismatch")
        return preset
