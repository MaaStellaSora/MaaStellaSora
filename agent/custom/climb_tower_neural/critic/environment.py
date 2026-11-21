"""Public actions and the frozen heuristic feature used by potential-only inference.

The event evaluator uses vendor.stella_sim to advance forced outcomes to the
next decision. Training episodes, branching controllers and full-game runners
are kept in the research repository, outside the release runtime.
"""
from __future__ import annotations

from functools import lru_cache

from .vendor import stella_sim as sim

PublicState = sim.PublicState
Action = sim.Action
ModelGap = sim.ModelGap
score = sim.score
action_key = sim.action_key


@lru_cache(maxsize=2048)
def original_role_totals(inventory, pools):
    """Recover original targets from acquired items AND remaining pools."""
    totals = [sum(pools[2*r]) + sum(pools[2*r+1]) for r in range(3)]
    for role, _quality, target, _level, count in inventory:
        totals[role-1] += target*count
    return tuple(totals)


def legal_actions(state: PublicState) -> tuple[Action, ...]:
    """Modeled legal actions; no value threshold or baseline-action lock.

    The reference simulator deliberately leaves negative-wallet event outcomes
    undefined. These are the sole additional exclusions: all positive-mass
    outcomes must retain a nonnegative wallet. This is a rule-domain check,
    not an action-value heuristic or teacher protection mechanism.
    """
    actions = sim.legal_actions(state)
    if state.phase == "EVENT":
        actions = tuple(a for a in actions if a.kind == "SKIP_EVENT" or all(
            n == 0 or state.coins + delta >= 0
            for n, _d, delta, _kind, _scope in sim.EVENT_DATA[state.eventId][2][a.index][1]))
    if not actions and state.phase != "END":
        raise ModelGap("NO_COVERED_ACTION", state.eventId)
    return actions


def heuristic_action(state: PublicState, actions: tuple[Action, ...] | None = None) -> Action:
    """Frozen preferred-action feature; it never selects the live action.

    The potential-only model was trained with this feature, so its exact tie-breaking and utility
    remain part of the feature schema even after training helpers are removed.
    """
    actions = legal_actions(state) if actions is None else actions
    if not actions:
        raise ValueError("No action in a terminal state")
    if state.phase in ("PAGE", "ENHANCE_PAGE"):
        choices = [a for a in actions if a.kind == "SELECT"]
        def utility(action):
            new, _role, _quality, target, old, level, _index = state.offer[action.index-1]
            gain = min(target, level) - min(target, old)
            # A zero-valued potential can remain a future offer distractor.
            # Small tie-breaking terms do not block learned-policy decisions.
            return gain, int(target > 0), max(target-level, 0) * int(new), -action.index
        return max(choices, key=utility)
    if state.phase == "SHOP":
        enhances = [a for a in actions if a.kind == "ENHANCE"]
        if enhances and enhances[0].price == 0:
            return enhances[0]
        buys = [a for a in actions if a.kind == "BUY"]
        if buys:
            return min(buys, key=lambda a: (a.price, a.type))
        if enhances:
            return enhances[0]
        return next(a for a in actions if a.kind == "LEAVE")
    if state.phase == "EVENT":
        def utility(action):
            if action.kind == "SKIP_EVENT":
                return 0.0
            outcomes = sim.EVENT_DATA[state.eventId][2][action.index][1]
            return sum(n/d * (delta/100 + int(kind != "none"))
                       for n, d, delta, kind, _scope in outcomes)
        return max(actions, key=utility)
    if len(actions) == 1:
        return actions[0]
    raise ValueError(f"Unexpected decision phase: {state.phase}")
