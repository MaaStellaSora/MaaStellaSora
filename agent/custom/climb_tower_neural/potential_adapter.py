"""Translate visible actions to the frozen potential-only action catalogue."""
from dataclasses import dataclass, replace

from .critic.vendor import stella_sim as sim


@dataclass(frozen=True)
class PreparedActions:
    state: sim.PublicState
    actions: tuple[sim.Action, ...]
    action_slots: dict[str, int]


def consume_drink_stock(state, *, price, trekker_specified):
    """Project one confirmed drink purchase onto the model's stock."""
    drink_type = 2 if trekker_specified else 1
    price_index = sim.DRINK_PRICES.index(price)
    stock = [list(row) for row in state.stock]
    if stock[drink_type - 1][price_index] < 1:
        raise ValueError("已确认饮品奖励没有对应的在售库存")
    stock[drink_type - 1][price_index] -= 1
    return replace(state, stock=tuple(map(tuple, stock)))


def prepare_core(state, candidates, decision_context):
    cards = {row["ui_index"]: row for row in decision_context["visible_cards"]}
    return max(candidates, key=lambda action: cards[action.index]["target_level"] > 0)


def prepare_actions(state, candidates, decision_context):
    """Return the model projection and UI slot mapping without changing observations."""
    candidates = tuple(candidates)
    if decision_context.get("kind") == "shop":
        if decision_context.get("stock_read_complete") is False:
            raise ValueError("商店货架尚有未确认的商品字段")
        stock = [list(row) for row in sim.EMPTY_STOCK]
        available = {}
        slots = {}
        by_slot = {row["slot"]: row for row in decision_context["visible_items"]}
        for row in by_slot.values():
            if row["bought"] or row["internal_name"] != "potential_drink":
                continue
            if row["price"] not in sim.DRINK_PRICES:
                continue
            drink_type = 2 if row["trekker_specified"] else 1
            index = sim.DRINK_PRICES.index(row["price"]) + 1
            stock[drink_type - 1][index - 1] += 1
        for candidate in candidates:
            if candidate.kind != "BUY":
                available[candidate] = candidate
                continue
            row = by_slot[candidate.index]
            if row["internal_name"] != "potential_drink" or row["price"] not in sim.DRINK_PRICES:
                continue
            action = sim.Action("BUY", sim.DRINK_PRICES.index(row["price"]) + 1,
                                row["price"], 2 if row["trekker_specified"] else 1)
            if action not in available:
                available[action] = candidate
                slots[sim.action_key(action)] = row["slot"]
        cost = decision_context["enhancement_cost"]
        state = replace(state, phase="SHOP", stock=tuple(map(tuple, stock)), firstInitial=False,
                        paidIndex=sim.PRICES.index(cost) if cost else state.paidIndex)
        actions = tuple(action for action in sim.legal_actions(state) if action in available)
        return PreparedActions(state, actions, slots)
    available = {(action.kind, action.index, action.price) for action in candidates}
    actions = tuple(action for action in sim.legal_actions(state)
                    if (action.kind, action.index, action.price) in available)
    return PreparedActions(state, actions, {})
