"""Identity-preserving pool swaps, with unweighted unknown-order compatibility."""
import json
from pathlib import Path

if __package__:
    from .name_matching import identity_key
else:
    from name_matching import identity_key


ORDERS = json.loads(Path(__file__).with_name("pool_orders.json").read_text(encoding="utf-8"))


def pool_contract(party, language):
    identities, labels, evidence = [], [], []
    for member in party:
        role_type = "master" if member["role"] == 1 else "assist"
        for quality in ("rare", "common"):
            items = [item for item in member["targets"] if item["rarity_class"] == quality]
            by_identity = {identity_key(item): item for item in items}
            supplied = member.get("catalogue_order", {}).get(quality)
            order = (dict(ids=supplied, basis="provided_catalogue_order") if supplied is not None else
                next((row for row in ORDERS["orders"] if row["character_id"] == member.get("character_id")
                      and row["role_type"] == role_type and row["quality"] == quality
                      and language in row["regions"]), None))
            if order is not None:
                if len(order["ids"]) != len(items) or set(order["ids"]) != set(by_identity):
                    raise ValueError("已确认池序与当前角色、品质的完整潜能目录不一致")
                items = [by_identity[key] for key in order["ids"]]
            identities.append([identity_key(item) for item in items])
            labels.append([item["target_level"] for item in items])
            evidence.append(dict(role=member["role"], character_id=member.get("character_id"),
                quality=quality, region=language, status="confirmed" if order is not None else "unknown",
                basis=order["basis"] if order is not None else "initial_order_not_observed",
                source=order))
    return labels, identities, evidence


class IdentityPools:
    """Fixed positions plus an unordered set for every remaining unknown slot.

    No probabilities are assigned. Partial Fisher-Yates observations preserve
    this form exactly; full-pool draws reveal the current order without swaps.
    The target projection is exact when the remaining unknowns share a label.
    """
    def __init__(self, data):
        _labels, derived_rows, derived_evidence = pool_contract(data["party"], data.get("language", "jp"))
        targets = [[] for _ in range(6)]
        for member in data["party"]:
            for item in member["targets"]:
                if item["rarity_class"] != "core":
                    targets[2 * (member["role"] - 1) + (item["rarity_class"] == "common")].append(item)
        rows = data.get("pool_identity_order", derived_rows)
        self.targets = [{identity_key(item): item["target_level"] for item in row} for row in targets]
        self.available = tuple(map(len, rows)) == tuple(map(len, data.get("pools", ()))) and all(
            None not in row and len(set(row)) == len(row) for row in rows)
        self.reference = [list(row) for row in rows]
        evidence = data.get("pool_order_evidence", derived_evidence)
        confirmed = [row["status"] == "confirmed" for row in evidence]
        self.slots = [list(row) if known else [None] * len(row) for row, known in zip(rows, confirmed)]
        self.unknown = [set() if known else set(row) for row, known in zip(rows, confirmed)]

    @property
    def ready(self):
        return self.available and all(len({target[key] for key in unknown}) <= 1
                                     for target, unknown in zip(self.targets, self.unknown))

    def projection(self):
        """Keep observed positions; fill unknowns in preset catalogue order only for inference."""
        result = []
        for slots, unknown, target, reference in zip(self.slots, self.unknown, self.targets, self.reference):
            remaining = iter(key for key in reference if key in unknown)
            result.append(tuple(target[key if key is not None else next(remaining)] for key in slots))
        return tuple(result)

    def copy(self):
        obj = object.__new__(type(self))
        obj.__dict__ = dict(self.__dict__, slots=[list(row) for row in self.slots],
                            unknown=[set(row) for row in self.unknown])
        return obj

    def observe(self, groups):
        slots = [list(row) for row in self.slots]
        unknown = [set(row) for row in self.unknown]
        for pi, observed in groups.items():
            row, missing = slots[pi], unknown[pi]
            if len(observed) == len(row):
                if set(observed) != {*missing, *(key for key in row if key is not None)} or any(
                        known is not None and known != seen for known, seen in zip(row, observed)):
                    raise ValueError("整池展示与已确认潜能身份或顺序不一致")
                slots[pi], unknown[pi] = list(observed), set()
                continue
            for offset, key in enumerate(observed):
                if key in row[offset:]:
                    found = row.index(key, offset)
                    row[offset], row[found] = row[found], row[offset]
                elif key in missing:
                    missing.remove(key)
                    if row[offset] is not None:
                        missing.add(row[offset])
                    row[offset] = key
                else:
                    raise ValueError("候选新潜能身份不在已确认剩余池中")
        self.slots, self.unknown = slots, unknown

    def remove(self, pi, key):
        row, unknown = self.slots[pi], self.unknown[pi]
        if key in row:
            index = row.index(key)
            row[index] = row[-1]
        elif key in unknown:
            unknown.remove(key)
            if row[-1] is not None:
                unknown.add(row[-1])
        else:
            raise ValueError("已取得潜能身份不在剩余池中")
        row.pop()

    def forget_order(self):
        for index, row in enumerate(self.slots):
            self.unknown[index].update(key for key in row if key is not None)
            self.slots[index] = [None] * len(row)

    def snapshot(self):
        return dict(positions=[list(row) for row in self.slots], unknown_identities=[
            [key for key in reference if key in unknown] for reference, unknown in zip(self.reference, self.unknown)],
            projection_unique=self.ready, compatibility="unknown_slots_without_prior")
