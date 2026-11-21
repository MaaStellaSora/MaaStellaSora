"""Human-readable terminal forecasts and changes from the previous forecast."""
from __future__ import annotations

from utils.logger import translate


def action_name(action):
    kind = action.kind
    if kind == "SELECT":
        return translate("$agent.custom.action.climb_tower_neural.action_select", index=action.index)
    if kind == "REROLL":
        return translate("$agent.custom.action.climb_tower_neural.action_reroll", price=action.price)
    if kind == "BUY":
        if action.type == 2:
            return translate("$agent.custom.action.climb_tower_neural.action_buy_specified_drink", price=action.price)
        return translate("$agent.custom.action.climb_tower_neural.action_buy_ordinary_drink", price=action.price)
    if kind in ("ENHANCE", "FREE_ENHANCE"):
        return translate("$agent.custom.action.climb_tower_neural.action_enhance", price=action.price)
    if kind == "RESTOCK":
        return translate("$agent.custom.action.climb_tower_neural.action_restock", price=action.price)
    if kind == "LEAVE":
        return translate("$agent.custom.action.climb_tower_neural.action_leave")
    if kind == "EVENT_OPTION":
        return translate("$agent.custom.action.climb_tower_neural.action_event_option", index=action.index % 100)
    if kind == "SKIP_EVENT":
        return translate("$agent.custom.action.climb_tower_neural.action_skip_event")
    if kind == "ADVANCE":
        return translate("$agent.custom.action.climb_tower_neural.action_advance")
    return kind


def forecast_rows(decision, actions, ordinary_score, previous_total, labels=None):
    labels = labels or {}
    rows = []
    for action, item in zip(actions, decision["values"]):
        total = item.get("expected_total")
        if total is None:
            current = decision.get("objective_current_value", 6 + ordinary_score)
            value = item.get("predicted_value")
            if value is None:
                value = item["expected_remaining_ordinary_return"]
            total = current + value
        rows.append({"action": item["action"], "label": labels[action] if action in labels else action_name(action),
                     "expected_total": float(total), "change": float(total - previous_total),
                     "selected": item["action_record"] == decision["action_record"]})
    return rows


def log_forecast(logger, selected_total, rows):
    options = []
    for row in rows:
        change = row["change"] if abs(row["change"]) >= .005 else 0.0
        label = " ".join(row["label"].split())
        # MFA's Markdown color span keeps the same arrow advance while hiding
        # it; whitespace alone is trimmed and has a different glyph width.
        marker = "→" if row["selected"] else "%{color:transparent}→%"
        options.append(f"{marker} {change:+.2f} {label}")
    # MFA decodes Unicode escapes after framing stdout records. Keep one
    # physical record, then render Markdown hard breaks inside its log item.
    logger.info("$agent.custom.action.climb_tower_neural.decision_forecast", {
        "expected_total": selected_total,
        "options": "".join(r"  \u000a" + option for option in options),
    })
