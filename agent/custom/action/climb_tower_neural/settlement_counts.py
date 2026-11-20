"""Check settlement UI totals without assigning UI positions to party roles."""
from collections import Counter


def check_potential_counts(parts, owned_fields):
    """Check actual level sums including cores; target-capped score is separate."""
    counts = [value if type(value) is int and value >= 0 else None for value in parts[:3]]
    counts += [None] * (3 - len(counts))
    conflicts = []
    history = owned_fields.get("owned_counts") or []
    if (owned_fields.get("owned_complete") is True and len(history) == 3
            and {row.get("role") for row in history} == {1, 2, 3}):
        expected = [row["noncore_level_sum"] + row["core_count"] for row in history]
        expected_counts = Counter(expected)
        observed_counts = Counter(value for value in counts if value is not None)
        for position, observed in enumerate(counts, 1):
            if observed is None:
                continue
            if observed_counts[observed] > expected_counts[observed]:
                # UI positions do not identify roles. If a repeated total is
                # overrepresented, no occurrence can be chosen as the true one.
                conflicts.append(dict(ui_position=position, observed=observed,
                                      expected_totals=list(expected)))
                counts[position - 1] = None
    return dict(potential_counts=counts,
                potential_counts_semantics="actual_level_sum_including_core",
                potential_counts_complete=len(parts) == 3 and all(value is not None for value in counts),
                potential_counts_conflicts=conflicts)
