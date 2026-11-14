"""Compact, complete public-state/action features for the independent Q critic.

Nothing here accepts an Episode, Transition, latent counter, environment RNG,
future outcome, model identifier or rollout label. Ordered pools and offers
remain ordered. Features are fixed by the shipped public rule table.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from functools import lru_cache
import math
import numpy as np

from .environment import Action, PublicState, heuristic_action, score, sim, original_role_totals
from .presets import POOL_LENGTHS, Preset, validate_role_totals

FEATURE_VERSION = "public_state_action_compact_v1"
TARGET_SCALE = 20.0
PHASES = ("ADVANCE", "CORE", "DRAW", "EVENT_DRAW", "SHOP_UNSEEN", "PAGE",
          "ENHANCE_PAGE", "EVENT", "SHOP", "END")
RETURNS = ("ADVANCE", "SHOP", "END")
RARITIES = ("regular", "common", "rare")
MODES = ("any", "main", "bound")
EVENTS = tuple(sorted(sim.EVENT_DATA))
OPTIONS = tuple(sorted({oid for row in sim.EVENT_DATA.values() for oid in row[2]}))
KINDS = ("SELECT", "REROLL", "ENHANCE", "BUY", "RESTOCK", "LEAVE",
         "EVENT_OPTION", "SKIP_EVENT", "CONTINUE", "ADVANCE", "CONFIRM_CORE",
         "GENERATE_PAGE", "GENERATE_EVENT", "OPEN_SHOP")
THRESHOLDS = np.array((40, 60, 100, 120, 160, 180, 200, 260, 340, 540, 740,
                       360, 620, 960, 1500, 2240), dtype=np.float32)
_blocks: list[dict] = []
_position = 0


def _block(name, shape, meaning):
    global _position
    size = math.prod(shape)
    _blocks.append({"name": name, "shape": list(shape), "start": _position,
                    "stop": _position + size, "meaning": meaning})
    _position += size


_block("inventory", (6, 7, 6), "counts/9: role-quality, target, actual level")
_block("ordered_pools", (36, 2), "target/6, present; zero targets retain present=1")
_block("original_histogram", (6, 7), "H plus retained pools, counts/9")
_block("event_posterior", (6, 4), "public exact posterior converted to float")
_block("coin_posterior", (6, 4, 2), "public posterior over gC/fC/effective G1; event callback folds G1 to zero")
_block("phase", (len(PHASES),), "one-hot")
_block("position", (5,), "node/46, floor/20, shopOrdinal/4, nodes left/46, floors left/20")
_block("return_phase", (3,), "one-hot")
_block("wallet", (13,), "coins/2000, log wallet/10, enhancement/stock/restock state")
_block("new_history", (3,), "P,Q,h/10")
_block("offers", (3, 31), "ordered visible cards, fixed one-hot/numeric fields")
_block("source", (15,), "source existence, rarity, role mode, bound role, return phase, reroll")
_block("seen_events", (len(EVENTS),), "public event membership")
_block("event_id", (len(EVENTS) + 1,), "one-hot including no event")
_block("event_options", (len(OPTIONS),), "visible option membership")
_block("event_history", (2,), "k/4,A")
_block("objective", (3,), "ordinary cap/144, score/144, remaining/144")
_block("inventory_summary", (6, 8), "counts, level sum, effective sum, deficit, improvable, pool sum/count, mean target")
_block("affordability", (len(THRESHOLDS), 2), "wallet threshold indicator and signed distance/500")
STATE_DIM = _position
_block("action_kind", (len(KINDS),), "one-hot")
_block("action_numbers", (6,), "price/740, wallet after/2000, slot/9, drink type/2, free, immediate gain/6")
_block("selected_offer", (31,), "the exact publicly chosen card")
_block("action_option", (len(OPTIONS),), "one-hot event option including skip")
_block("event_expectation", (10,), "expected coin, absolute coin, variance, reward type/scope public probabilities")
_block("heuristic_q_prior", (1,), "(common public remaining-value prior + .25 preferred-action marker)/20")
FEATURE_DIM = DIMENSION = _position
HEURISTIC_Q_INDEX = FEATURE_DIM - 1
_slices = {block["name"]: slice(block["start"], block["stop"]) for block in _blocks}
_SUMMARY_H_SCALE = np.array((9, 54, 54, 54, 9))
_SUMMARY_POOL_SCALE = np.array((54, 9, 6))
_TARGET_VALUES = np.arange(7)
# Bounded per-worker caches. They contain public, model-independent features
# only. Arrays are read-only, and encode_state always returns a fresh vector.
_BELIEF_CACHE = OrderedDict()
_BELIEF_CACHE_MAX = 256
_BELIEF_CACHE_COUNTS = {"hits": 0, "misses": 0}


def _readonly(array):
    array.flags.writeable = False
    return array


@lru_cache(maxsize=2048)
def _inventory_features(H):
    inventory = np.zeros((6, 7, 6), dtype=np.float32)
    histogram = np.zeros((6, 7), dtype=np.int16)
    # All accumulated quantities are small exact integers. Accumulating them
    # in Python then rounding once to float32 equals the former float32 sums,
    # while avoiding a NumPy array allocation/ufunc for every learned category.
    summary = [[0] * 5 for _ in range(6)]
    for role, quality, target, level, count in H:
        i = 2*(role-1)+quality-1
        inventory[i, target, level-1] += count/9
        histogram[i, target] += count
        row = summary[i]
        row[0] += count
        row[1] += count*level
        row[2] += count*min(target, level)
        row[3] += count*max(target-level, 0)
        row[4] += count*(target > level)
    normalized = (np.asarray(summary, dtype=np.float32)/_SUMMARY_H_SCALE).astype(np.float32)
    return (_readonly(inventory.reshape(-1)), _readonly(histogram),
            _readonly(normalized), tuple(int(x) for x in histogram.sum(1)),
            int((histogram*_TARGET_VALUES).sum()))


@lru_cache(maxsize=1024)
def _pool_features(state_pools):
    pools = np.zeros((36, 2), dtype=np.float32)
    histogram = np.zeros((6, 7), dtype=np.int16)
    summary = np.zeros((6, 3), dtype=np.float32)
    offset = 0
    for i, (pool, length) in enumerate(zip(state_pools, POOL_LENGTHS)):
        if len(pool) > length:
            raise ValueError("Public pool exceeds its catalogue")
        for j, target in enumerate(pool):
            if not 0 <= target <= 6:
                raise ValueError("Invalid target")
            pools[offset+j] = target/6, 1
            histogram[i, target] += 1
        offset += length
        total = sum(pool)
        summary[i] = total, len(pool), total/max(1, len(pool))
    normalized = (summary/_SUMMARY_POOL_SCALE).astype(np.float32)
    return (_readonly(pools.reshape(-1)), _readonly(histogram),
            _readonly(normalized), tuple(int(x) for x in histogram.sum(1)),
            int((histogram*_TARGET_VALUES).sum()))


def _belief_features(rows, shape):
    # These immutable tuple objects are shared across many public decisions.
    # An identity cache avoids expensive Fraction hashing. Keeping the tuple
    # itself alive in each entry prevents Python id reuse from giving a hit.
    key = (id(rows), shape)
    hit = _BELIEF_CACHE.get(key)
    if hit is not None and hit[0] is rows:
        _BELIEF_CACHE.move_to_end(key)
        _BELIEF_CACHE_COUNTS["hits"] += 1
        return hit[1]
    _BELIEF_CACHE_COUNTS["misses"] += 1
    result = np.zeros(shape, dtype=np.float32)
    for row in rows:
        result[tuple(row[:-1])] = float(row[-1])
    result = _readonly(result.reshape(-1))
    _BELIEF_CACHE[key] = (rows, result)
    if len(_BELIEF_CACHE) > _BELIEF_CACHE_MAX:
        _BELIEF_CACHE.popitem(last=False)
    return result


def clear_feature_caches():
    """Benchmark/debug helper; clearing changes speed only, never outputs."""
    _inventory_features.cache_clear()
    _pool_features.cache_clear()
    _inventory_context.cache_clear()
    _card.cache_clear()
    _BELIEF_CACHE.clear()
    _BELIEF_CACHE_COUNTS.update(hits=0, misses=0)


def feature_cache_info():
    return {"inventory": _inventory_features.cache_info()._asdict(),
            "pools": _pool_features.cache_info()._asdict(),
            "cards": _card.cache_info()._asdict(),
            "beliefs": {**_BELIEF_CACHE_COUNTS, "currsize": len(_BELIEF_CACHE),
                        "maxsize": _BELIEF_CACHE_MAX}}


def schema() -> dict:
    return {"feature_version": FEATURE_VERSION, "dimension": FEATURE_DIM,
            "state_dimension": STATE_DIM, "target_scale": TARGET_SCALE,
            "blocks": [dict(block) for block in _blocks],
            "forbidden": ["latent", "environment_rng", "future_draw", "model_id", "rollout_return"],
            "heuristic_prior_is_trainable_initialization_only": True}


def _hot(out, name, value, values):
    try:
        index = values.index(value)
    except ValueError as exc:
        raise ValueError(f"Unknown {name}: {value!r}") from exc
    out[_slices[name].start + index] = 1.0


@lru_cache(maxsize=8192)
def _card(card) -> np.ndarray:
    out = np.zeros(31, dtype=np.float32)
    if card is None:
        return _readonly(out)
    is_new, role, quality, target, old, level, index = card
    out[:2] = 1, is_new
    offset = 2
    for value, minimum, maximum in ((role, 1, 3), (quality, 1, 2), (target, 0, 6),
                                     (old, 0, 6), (level, 0, 6)):
        if not minimum <= value <= maximum:
            raise ValueError("Unknown visible card category")
        out[offset + value - minimum] = 1
        offset += maximum - minimum + 1
    out[offset:] = (index / 9, (min(target, level) - min(target, old)) / 6,
                    max(target - level, 0) / 6)
    return _readonly(out)


@lru_cache(maxsize=2048)
def _inventory_context(H, state_pools, cap, deployment_only=False):
    """Reuse public features that stay fixed between inventory/pool changes."""
    if not deployment_only:
        validate_role_totals(original_role_totals(H, state_pools))
    out = np.zeros(STATE_DIM, dtype=np.float32)
    inventory, h_histogram, h_summary, h_counts, h_total = _inventory_features(H)
    pools, p_histogram, p_summary, p_counts, p_total = _pool_features(state_pools)
    histogram = h_histogram+p_histogram
    out[_slices["inventory"]] = inventory
    out[_slices["ordered_pools"]] = pools
    if tuple(a+b for a, b in zip(h_counts, p_counts)) != POOL_LENGTHS:
        raise ValueError("Public state does not retain all 36 original ordinary targets")
    if h_total+p_total != cap:
        raise ValueError("Public full target inventory disagrees with cap")
    out[_slices["original_histogram"]] = histogram.reshape(-1) / 9
    summary = out[_slices["inventory_summary"]].reshape(6, 8)
    summary[:, :5] = h_summary
    summary[:, 5:] = p_summary
    return _readonly(out), _readonly(histogram)


def encode_state(state: PublicState, preset: Preset | None = None) -> np.ndarray:
    """Encode one immutable PUBLIC state; an optional preset verifies identity."""
    if not isinstance(state, PublicState):
        raise TypeError("Only PublicState may enter the value network")
    deployment_only = bool(getattr(preset, "deployment_only", False))
    if state.model not in sim.MODEL_ALIASES or (not deployment_only and not 72 <= state.cap <= 144):
        raise ValueError("Unsupported public rule version or target scope")
    context, histogram = _inventory_context(state.H, state.pools, state.cap, deployment_only)
    out = context.copy()
    if preset is not None:
        if not isinstance(preset, Preset) or preset.total != state.cap:
            raise ValueError("A complete original preset is required")
        if any(Counter(preset.pools[i]) != Counter({t: int(n) for t, n in enumerate(histogram[i]) if n})
               for i in range(6)):
            raise ValueError("Public state differs from its original preset")
    out[_slices["event_posterior"]] = _belief_features(state.eBel, (6, 4))
    coin = _belief_features(state.cBel, (6, 4, 2))
    if state.npc3EventReward:
        coin = coin.reshape(6, 4, 2).copy()
        coin[:, :, 0] += coin[:, :, 1]
        coin[:, :, 1] = 0
        coin = coin.reshape(-1)
    out[_slices["coin_posterior"]] = coin
    _hot(out, "phase", state.phase, PHASES)
    out[_slices["position"]] = (state.node / 46, state.floor / 20, state.shopOrdinal / 4,
                                (46 - state.node) / 46, (20 - state.floor) / 20)
    _hot(out, "return_phase", state.shopReturnPhase, RETURNS)
    out[_slices["wallet"]] = (state.coins / 2000, math.log1p(state.coins) / 10,
                              state.freeEnhance, state.paidIndex / 6,
                              *(n / 8 for row in state.stock for n in row),
                              state.restocks / 2, state.firstInitial,
                              sim.PRICES[min(state.paidIndex, 6)] / 740)
    out[_slices["new_history"]] = state.P, state.Q, state.h / 10
    offers = out[_slices["offers"]].reshape(3, 31)
    if len(state.offer) > 3:
        raise ValueError("At most three public offer rows are supported")
    for i, card in enumerate(state.offer):
        offers[i] = _card(card)
    if state.source is not None:
        src = state.source
        values = out[_slices["source"]]
        values[0] = 1
        values[1 + RARITIES.index(src.rarity)] = 1
        values[4 + MODES.index(src.roleMode)] = 1
        values[7 + src.boundRole] = 1
        values[11 + RETURNS.index(src.returnPhase)] = 1
        values[14] = src.rerollAllowed
    for event in state.seen:
        out[_slices["seen_events"].start + EVENTS.index(event)] = 1
    _hot(out, "event_id", state.eventId, (0, *EVENTS))
    for option in state.eventOptions:
        out[_slices["event_options"].start + OPTIONS.index(option)] = 1
    out[_slices["event_history"]] = state.k / 4, state.A
    current_score = score(state)
    out[_slices["objective"]] = state.cap / 144, current_score / 144, (state.cap-current_score) / 144
    affordability = out[_slices["affordability"]].reshape(-1, 2)
    affordability[:, 0] = state.coins >= THRESHOLDS
    affordability[:, 1] = np.clip((state.coins-THRESHOLDS) / 500, -3, 3)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite public state feature")
    return out


def encode_actions(state: PublicState, actions, preset: Preset | None = None, *, out=None) -> np.ndarray:
    """Encode every supplied legal root action once, with shared public context."""
    actions = tuple(actions)
    if not actions:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    base = encode_state(state, preset)
    result = np.zeros((len(actions), FEATURE_DIM), dtype=np.float32) if out is None else out
    result[:, :STATE_DIM] = base
    preferred = heuristic_action(state, actions)
    remaining = max(0, state.cap-score(state))
    # Deliberately coarse cold-start estimate, never a claim of an optimal value.
    common_prior = min(max(0, remaining - .5), 3.1 * (20-state.floor) + state.coins / 180)
    for i, action in enumerate(actions):
        if not isinstance(action, Action):
            raise TypeError("Only a public Action may enter the Q network")
        out = result[i]
        _hot(out, "action_kind", action.kind, KINDS)
        immediate = 0.0
        if action.kind == "SELECT":
            card = state.offer[action.index-1]
            out[_slices["selected_offer"]] = _card(card)
            immediate = min(card[3], card[5]) - min(card[3], card[4])
        out[_slices["action_numbers"]] = (action.price / 740, (state.coins-action.price) / 2000,
                (action.index / 9 if action.kind in ("SELECT", "BUY") else 0),
                (action.type / 2 if action.kind == "BUY" else 0), action.price == 0, immediate / 6)
        if action.kind in ("EVENT_OPTION", "SKIP_EVENT"):
            oid = action.index if action.kind == "EVENT_OPTION" else 0
            _hot(out, "action_option", oid, OPTIONS)
            desc = out[_slices["event_expectation"]]
            outcomes = sim.EVENT_DATA[state.eventId][2][oid][1]
            for n, d, delta, rarity, scope in outcomes:
                probability = n / d
                desc[0] += probability * delta / 500
                desc[1] += probability * abs(delta) / 500
                desc[2] += probability * (delta / 500) ** 2
                desc[3 + ("none", "regular", "common", "rare").index(rarity)] += probability
                if scope != "none":
                    desc[7 + ("party", "main", "support_random").index(scope)] += probability
        out[HEURISTIC_Q_INDEX] = (common_prior + .25 * (action == preferred)) / TARGET_SCALE
    return result


def encode(state: PublicState, action: Action, preset: Preset | None = None) -> np.ndarray:
    """Convenience entry point. Full action sets are preferred for inference."""
    from .environment import legal_actions
    actions = legal_actions(state)
    try:
        index = actions.index(action)
    except ValueError as exc:
        raise ValueError("Action does not belong to this public root") from exc
    return encode_actions(state, actions, preset)[index]
