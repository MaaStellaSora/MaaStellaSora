"""Fast generative sampler of the StellaSora game kernel.

The current rule revision applies E6 shelves, level .49 and NPC event callbacks. Random
choices use integer randrange draws.  Public beliefs use small exact
Fractions; the hidden counters in Latent must never be given to a policy.
This sampler reports ModelGap rather than clamping, deleting or renormalizing
undefined branches.  It is a simulation kernel, not an optimality certificate.
"""
from __future__ import annotations

from ast import literal_eval
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields, is_dataclass, replace
from fractions import Fraction
from functools import lru_cache
from math import comb, gcd, isfinite
from numbers import Real
from random import Random
from typing import Any, Iterable

F = Fraction
ONE = F(1)
ZERO = F(0)
MODEL_VERSION = "state_space_e6_l49_npc_event_v3"
MODEL_ALIASES = frozenset((MODEL_VERSION,))
INITIAL_POOL_LENGTHS = (3, 9, 3, 9, 3, 9)
PRICES = (60, 120, 180, 260, 340, 540, 740)
DRINK_PRICES = (100, 160, 200)
EMPTY_STOCK = ((0, 0, 0), (0, 0, 0))
FORCED_PHASES = frozenset(("ADVANCE", "CORE", "DRAW", "EVENT_DRAW", "SHOP_UNSEEN"))


class ModelGap(RuntimeError):
    def __init__(self, reason: str, details: Any = None):
        self.reason = reason
        self.details = details
        super().__init__(reason if details is None else f"{reason}: {details}")


class InvalidState(ValueError):
    def __init__(self, reason: str, details: Any = None):
        self.reason = reason
        self.details = details
        super().__init__(reason if details is None else f"{reason}: {details}")


@dataclass(frozen=True, slots=True)
class Source:
    rarity: str = "regular"
    roleMode: str = "any"
    boundRole: int = 0
    returnPhase: str = "ADVANCE"
    rerollAllowed: bool = True


@dataclass(frozen=True, slots=True)
class Action:
    kind: str
    index: int = 0
    price: int = 0
    type: int = 0
    key: str = ""


FORCED_ACTIONS = {
    "ADVANCE": Action("ADVANCE"), "CORE": Action("CONFIRM_CORE"),
    "DRAW": Action("GENERATE_PAGE"), "EVENT_DRAW": Action("GENERATE_EVENT"),
    "SHOP_UNSEEN": Action("OPEN_SHOP"),
}


@dataclass(frozen=True, slots=True)
class PublicState:
    model: str = MODEL_VERSION
    phase: str = "ADVANCE"
    node: int = 1                         # next route instruction, ONE based
    floor: int = 0
    H: tuple = ()                         # role,quality,target,level,count
    # These two fields are intentionally REQUIRED. PublicState() must not
    # silently choose a particular preset or invent the missing catalogue.
    pools: tuple = field(kw_only=True)
    cap: int = field(kw_only=True)
    P: bool = False
    Q: bool = False
    h: int = 0
    eBel: tuple = ((0, 0, ONE),)
    cBel: tuple = ((0, 0, 0, ONE),)
    coins: int = 400
    freeEnhance: bool = True
    paidIndex: int = 0
    stock: tuple = EMPTY_STOCK
    restocks: int = 2
    firstInitial: bool = False
    shopReturnPhase: str = "ADVANCE"
    shopOrdinal: int = 0
    offer: tuple = ()                     # new,role,q,target,old,new,index(1)
    source: Source | None = None
    npc3EventReward: bool = False
    A: bool = False
    k: int = 0
    seen: tuple = ()
    eventId: int = 0
    eventOptions: tuple = ()


@dataclass(frozen=True, slots=True)
class Latent:
    gE: int = 0
    fE: int = 0
    gC: int = 0
    fC: int = 0
    G1: int = 0


@dataclass(frozen=True, slots=True)
class Transition:
    state: PublicState
    observation: PublicState
    latent: Latent
    reward: int
    terminal: bool


ROUTE = (
    ("BATTLE",1,80),("CORE",1,0),("EXTRA",1,0),("EVENT",2,0),
    ("BATTLE",3,80),("DRAW",3,0),("EXTRA",3,0),
    ("BATTLE",4,320),("DRAW",4,0),("SHOP",5,1),
    ("BATTLE",6,700),("CORE",6,0),("DRAW",6,0),
    ("BATTLE",7,80),("BATTLE",8,320),("CORE",8,0),("DRAW",8,0),
    ("EVENT",9,0),("BATTLE",10,80),("CORE",10,0),("EXTRA",10,0),
    ("BATTLE",11,320),("DRAW",11,0),("DRAW",11,0),("SHOP",12,2),
    ("BATTLE",13,700),("CORE",13,0),("DRAW",13,0),
    ("BATTLE",14,80),("EXTRA",14,0),
    ("BATTLE",15,320),("CORE",15,0),("DRAW",15,0),("EVENT",16,0),
    ("BATTLE",17,80),("DRAW",17,0),("EXTRA",17,0),
    ("BATTLE",18,320),("DRAW",18,0),("DRAW",18,0),("SHOP",19,3),
    ("BATTLE",20,700),("DRAW",20,0),("DRAW",20,0),("SHOP",20,4),
)

# EVENT_DATA is embedded below from the frozen R12 option configuration.
# id -> (ordinary weight, predictive IDs, {option:(cost,outcomes)})
# outcome = (numerator,denominator,net_coin_delta,rarity,role_scope)
EVENT_DATA: dict = {101: (5,
       (10101, 10102, 10103, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        10101: (100, ((1, 1, -100, 'common', 'party'),)),
        10102: (120, ((1, 1, -120, 'regular', 'party'),)),
        10103: (0, ((1, 1, 30, 'none', 'none'),))}),
 102: (3,
       (10201, 10202, 10203, 10204, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        10201: (120, ((1, 1, -120, 'regular', 'support_random'),)),
        10202: (160, ((1, 1, -160, 'regular', 'main'),)),
        10203: (200, ((1, 1, -200, 'rare', 'party'),)),
        10204: (0, ((1, 1, 30, 'none', 'none'),))}),
 103: (30,
       (10301, 10302, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        10301: (0, ((1, 1, 150, 'none', 'none'),)),
        10302: (0, ((1, 1, 30, 'none', 'none'),))}),
 104: (2,
       (10401, 10402, 10403, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        10401: (0, ((1, 1, 0, 'rare', 'party'),)),
        10402: (200, ((1, 1, -200, 'rare', 'party'),)),
        10403: (0, ((1, 1, 30, 'none', 'none'),))}),
 105: (0,
       (10501, 10502, 10503, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        10501: (0, ((1, 2, 200, 'none', 'none'), (1, 2, -100, 'none', 'none'))),
        10502: (0, ((50, 99, 650, 'none', 'none'), (49, 99, -200, 'none', 'none'))),
        10503: (0, ((1, 1, 30, 'none', 'none'),))}),
 106: (2,
       (10601, 10602, 10603, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        10601: (0, ((1, 2, 0, 'rare', 'party'), (1, 2, 0, 'none', 'none'))),
        10602: (0, ((1, 1, 0, 'common', 'party'),)),
        10603: (0, ((1, 1, 30, 'none', 'none'),))}),
 107: (20,
       (0,),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        10701: (0, ((1, 1, 0, 'none', 'none'),)),
        10702: (0, ((1, 1, 0, 'none', 'none'),)),
        10703: (0, ((1, 1, 0, 'none', 'none'),)),
        10704: (0, ((1, 1, 0, 'none', 'none'),)),
        10705: (0, ((1, 1, 0, 'none', 'none'),)),
        10706: (0, ((1, 1, 0, 'none', 'none'),)),
        10707: (0, ((1, 1, 0, 'none', 'none'),)),
        10708: (0, ((1, 1, 0, 'none', 'none'),))}),
 108: (10,
       (10809, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        10801: (140, ((1, 1, -140, 'none', 'none'),)),
        10802: (140, ((1, 1, -140, 'none', 'none'),)),
        10803: (140, ((1, 1, -140, 'none', 'none'),)),
        10804: (140, ((1, 1, -140, 'none', 'none'),)),
        10805: (140, ((1, 1, -140, 'none', 'none'),)),
        10806: (140, ((1, 1, -140, 'none', 'none'),)),
        10807: (140, ((1, 1, -140, 'none', 'none'),)),
        10808: (90, ((1, 1, -90, 'none', 'none'),)),
        10809: (0, ((1, 1, 30, 'none', 'none'),))}),
 109: (4,
       (10901, 10902, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        10901: (0, ((1, 2, 100, 'none', 'none'), (1, 2, 0, 'common', 'party'))),
        10902: (120, ((1, 1, -120, 'regular', 'party'),))}),
 110: (2,
       (11001, 11002, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        11001: (0, ((1, 2, 120, 'none', 'none'), (1, 2, 0, 'rare', 'party'))),
        11002: (160, ((1, 1, -160, 'rare', 'party'),))}),
 111: (4,
       (11101, 11102, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        11101: (0, ((1, 1, 0, 'none', 'none'),)),
        11102: (0, ((1, 1, 0, 'common', 'party'),))}),
 112: (4,
       (11201, 11202, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        11201: (0, ((1, 1, 0, 'none', 'none'),)),
        11202: (0, ((1, 1, 0, 'regular', 'party'),))}),
 113: (1,
       (11301, 11302, 11303, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        11301: (0, ((1, 1, 0, 'none', 'none'),)),
        11302: (0, ((1, 1, 0, 'regular', 'party'),)),
        11303: (0, ((1, 1, 0, 'rare', 'party'),))}),
 114: (10,
       (11403, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        11401: (0, ((1, 1, 0, 'none', 'none'),)),
        11402: (0, ((1, 1, 0, 'none', 'none'),)),
        11403: (0, ((1, 1, 0, 'none', 'none'),)),
        11404: (0, ((1, 1, 0, 'none', 'none'),)),
        11405: (0, ((1, 1, 0, 'none', 'none'),))}),
 115: (5,
       (11503, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        11501: (0, ((1, 1, 0, 'none', 'none'),)),
        11502: (0, ((1, 1, 0, 'none', 'none'),)),
        11503: (0, ((1, 1, 0, 'regular', 'party'),)),
        11504: (0, ((1, 1, 0, 'none', 'none'),)),
        11505: (0, ((1, 1, 0, 'none', 'none'),))}),
 116: (1,
       (11603, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        11601: (0, ((1, 1, 0, 'none', 'none'),)),
        11602: (0, ((1, 1, 0, 'none', 'none'),)),
        11603: (0, ((1, 1, 0, 'rare', 'party'),)),
        11604: (0, ((1, 1, 0, 'none', 'none'),)),
        11605: (0, ((1, 1, 0, 'none', 'none'),))}),
 117: (30,
       (11701, 11702, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        11701: (0, ((1, 4, 100, 'none', 'none'), (1, 4, -100, 'none', 'none'), (1, 2, 0, 'none', 'none'))),
        11702: (0, ((1, 1, 30, 'none', 'none'),))}),
 118: (4,
       (11801, 11802, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        11801: (0,
                ((1, 3, 100, 'none', 'none'),
                 (1, 3, -100, 'none', 'none'),
                 (1, 6, 0, 'none', 'none'),
                 (1, 6, 0, 'regular', 'party'))),
        11802: (0, ((1, 1, 30, 'none', 'none'),))}),
 119: (1,
       (11901, 11902, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        11901: (0, ((1, 3, 120, 'none', 'none'), (1, 3, -120, 'none', 'none'), (1, 3, 0, 'rare', 'party'))),
        11902: (0, ((1, 1, 30, 'none', 'none'),))}),
 120: (10,
       (12001, 12002, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        12001: (50, ((1, 1, -50, 'none', 'none'),)),
        12002: (0, ((1, 1, 0, 'none', 'none'),))}),
 121: (4,
       (12101, 12102, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        12101: (100, ((1, 1, -100, 'regular', 'party'),)),
        12102: (0, ((1, 2, 0, 'none', 'none'), (1, 2, 0, 'regular', 'party')))}),
 122: (1,
       (12201, 12202, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        12201: (180, ((1, 1, -180, 'rare', 'party'),)),
        12202: (0, ((1, 2, 0, 'none', 'none'), (1, 2, 0, 'rare', 'party')))}),
 123: (4,
       (12301, 12302, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        12301: (50, ((1, 1, -50, 'none', 'none'),)),
        12302: (0, ((1, 1, 0, 'regular', 'party'),))}),
 124: (4,
       (12401, 12402, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        12401: (0, ((1, 3, 0, 'regular', 'party'), (2, 3, 0, 'none', 'none'))),
        12402: (0, ((1, 1, 30, 'none', 'none'),))}),
 125: (2,
       (12501, 12502, 12503, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        12501: (0, ((1, 3, 0, 'common', 'party'), (2, 3, 0, 'none', 'none'))),
        12502: (0, ((1, 3, 0, 'rare', 'party'), (2, 3, 0, 'none', 'none'))),
        12503: (0, ((1, 1, 30, 'none', 'none'),))}),
 126: (4,
       (12601, 12602, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        12601: (0, ((1, 1, 0, 'common', 'support_random'),)),
        12602: (0, ((1, 1, 0, 'none', 'none'),))}),
 127: (20,
       (12701, 12702, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        12701: (0, ((1, 1, 0, 'regular', 'support_random'),)),
        12702: (0, ((1, 1, 0, 'none', 'none'),))}),
 128: (2,
       (12801, 12802, 0),
       {0: (0, ((1, 1, 0, 'none', 'none'),)),
        12801: (0, ((1, 1, 0, 'rare', 'support_random'),)),
        12802: (0, ((1, 1, 30, 'none', 'none'),))})}

DOMINANCE = {
 101:((10103,0),),102:((10204,0),),103:((10301,0),(10301,10302)),
 104:((10403,0),(10401,10402)),105:((10503,0),),106:((10603,0),),
 108:((10809,0),),111:((11101,0),),112:((11201,0),),113:((11301,0),),
 114:((11403,0),),117:((11702,0),),118:((11802,0),),119:((11902,0),),
 120:((12002,0),),124:((12402,0),),125:((12503,0),),
 126:((12602,0),),127:((12702,0),),128:((12802,0),),
}


def initial_843() -> PublicState:
    """Explicit historical fixture; never the implicit user preset."""
    return initial_from_pools(((6,0,6),(6,6,0,0,6,6,6,0,6),
                               (6,0,6),(6,6,0,0,0,0,0,0,0),
                               (6,0,0),(0,6,6,0,0,0,0,0,0)))


def initial_from_pools(pools: Iterable[Iterable[int]]) -> PublicState:
    """Create an initial state from ALL 36 ordered non-core target levels.

    Pool order is role1 rare/common, role2 rare/common, role3 rare/common.
    Each role retains its complete 3-rare/9-common frozen catalogue; target
    zero means non-preset, not removed. Any 0..12 selected targets per role
    and any integer target level 1..6 are supported.
    """
    raw = tuple(tuple(p) for p in pools)
    if tuple(map(len,raw)) != INITIAL_POOL_LENGTHS:
        raise InvalidState("INITIAL_CATALOGUE_INCOMPLETE", "six complete pools of lengths 3,9,3,9,3,9 required; retain target-zero IDs")
    if any(not isinstance(x,Real) or isinstance(x,bool) or not isfinite(x)
           or x!=int(x) or not 0<=x<=6 for p in raw for x in p):
        raise InvalidState("six ordered target vectors with values 0..6 required")
    ps = tuple(tuple(int(x) for x in p) for p in raw)
    state=PublicState(pools=ps,cap=sum(map(sum,ps)))
    validate_state(state)
    return state


def score(s: PublicState) -> int:
    return _inventory_score(s.H)


@lru_cache(maxsize=8192)
def _inventory_score(inventory) -> int:
    return sum(min(t, level) * n for _, _, t, level, n in inventory)


def total_score(s: PublicState) -> int:
    return score(s) + 6


def is_terminal(s: PublicState) -> bool:
    return s.phase == "END"


def state_key(s: PublicState) -> PublicState:
    """Exact immutable public-state key; contains no hidden latent particle."""
    return s


def action_key(a: Action) -> str:
    if a.kind == "SELECT":
        return a.key
    if a.kind == "REROLL":
        return "02_REROLL"
    if a.kind == "ENHANCE":
        return "03_FREE_ENHANCE" if a.price == 0 else f"04_PAID_ENHANCE_{a.price:04d}"
    if a.kind == "BUY":
        return f"05_BUY_{a.type:02d}_{a.price:04d}"
    if a.kind == "RESTOCK":
        return "06_RESTOCK"
    if a.kind == "EVENT_OPTION":
        return f"07_EVENT_{a.type:03d}_{a.index:06d}"
    if a.kind == "CONTINUE":
        return "08_01_CONTINUE"
    if a.kind == "LEAVE":
        return "08_02_LEAVE"
    if a.kind == "SKIP_EVENT":
        return "08_03_SKIP_EVENT"
    return f"09_FORCED_{a.kind}_{a.type:02d}_{a.price:04d}_{a.index:06d}"


def legal_actions(s: PublicState) -> tuple[Action, ...]:
    out: list[Action] = []
    if s.phase in ("PAGE", "ENHANCE_PAGE"):
        seen = set()
        for i, c in enumerate(s.offer, 1):
            if c[0] == 0:
                semantic = ("O", c[1:6])
                key = "01_SELECT_0_" + "".join(f"{x:02d}_" for x in c[1:6])
            else:
                pi = 2*(c[1]-1) + c[2]-1
                pool = list(s.pools[pi]); j = c[6]-1
                if not 0 <= j < len(pool) or pool[j] != c[3]:
                    raise InvalidState("new offer does not match its post-display pool")
                pool[j] = pool[-1]; pool.pop()
                semantic = ("N", c[1], c[2], c[3], c[5], tuple(pool))
                key = "01_SELECT_1_" + "".join(f"{x:02d}_" for x in (c[1],c[2],c[6],c[3],c[5]))
            if semantic not in seen:
                seen.add(semantic); out.append(Action("SELECT", i, key=key))
        if s.phase == "PAGE" and s.coins >= 40 and (s.source is None or s.source.rerollAllowed):
            out.append(Action("REROLL", price=40))
    elif s.phase == "SHOP":
        for t in range(2):
            j = next((j for j, n in enumerate(s.stock[t]) if n > 0), None)
            if j is not None and s.coins >= DRINK_PRICES[j]:
                out.append(Action("BUY", j+1, DRINK_PRICES[j], t+1))
        if any(row[3] < 6 for row in s.H):
            cost = 0 if s.freeEnhance else PRICES[min(s.paidIndex, 6)]
            if s.coins >= cost:
                out.append(Action("ENHANCE", price=cost))
        if s.restocks > 0 and s.coins >= 100:
            out.append(Action("RESTOCK", price=100))
        out.append(Action("LEAVE"))
    elif s.phase == "EVENT":
        if s.eventId not in EVENT_DATA:
            raise ModelGap("UNKNOWN_EVENT", s.eventId)
        _, predictive, opts = EVENT_DATA[s.eventId]
        for oid in predictive:
            if oid not in s.eventOptions:
                continue
            cost, _ = opts[oid]
            if s.coins >= cost:
                out.append(Action("SKIP_EVENT", type=s.eventId) if oid == 0
                           else Action("EVENT_OPTION", oid, cost, s.eventId))
        ids = {a.index for a in out}
        removed = {b for a, b in DOMINANCE.get(s.eventId, ()) if a in ids}
        out = [a for a in out if a.index not in removed]
        if not out:
            raise ModelGap("NO_MODELED_VISIBLE_EVENT_ACTION", s.eventId)
    elif s.phase == "END":
        pass
    else:
        if s.phase not in FORCED_ACTIONS:
            raise ModelGap("UNKNOWN_PHASE", s.phase)
        out.append(FORCED_ACTIONS[s.phase])
    return tuple(sorted(out, key=action_key))


def _bern(rng: Random, num: int, den: int) -> bool:
    return num >= den or (num > 0 and rng.randrange(den) < num)


def _weighted_int(rng: Random, weights: Iterable[int]) -> int:
    w = tuple(weights); total = sum(w)
    if total <= 0:
        raise InvalidState("nonpositive categorical total")
    x = rng.randrange(total)
    for i, a in enumerate(w):
        if x < a:
            return i
        x -= a
    raise AssertionError("unreachable categorical tail")


@lru_cache(maxsize=20000)
def _fraction_weights(probabilities: tuple[Fraction, ...]) -> tuple[int, ...]:
    den = 1
    for p in probabilities:
        if p < 0:
            raise InvalidState("negative probability")
        den = den * p.denominator // gcd(den, p.denominator)
    if sum(probabilities, ZERO) != ONE:
        raise InvalidState("probability mass must sum exactly to one")
    return tuple(p.numerator * (den // p.denominator) for p in probabilities)


def _weighted_fraction(rng: Random, probabilities: tuple[Fraction, ...]) -> int:
    return _weighted_int(rng, _fraction_weights(probabilities))


def sample_latent(s: PublicState, rng: Random) -> Latent:
    e = s.eBel[_weighted_fraction(rng, tuple(row[2] for row in s.eBel))]
    c = s.cBel[_weighted_fraction(rng, tuple(row[3] for row in s.cBel))]
    return Latent(e[0], e[1], c[0], c[1], c[2])


def _hist_change(H: tuple, cat: tuple, delta: int) -> tuple:
    d = {tuple(row[:4]): row[4] for row in H}
    n = d.get(cat, 0) + delta
    if n < 0:
        raise InvalidState("negative learned category count")
    if n:
        d[cat] = n
    else:
        d.pop(cat, None)
    return tuple((*c, n) for c, n in sorted(d.items()))


def _uniform_old(H: tuple, k: int, rng: Random) -> tuple:
    expanded = [row[:4] for row in H for _ in range(row[4])]
    if k > len(expanded):
        raise ModelGap("OLD_POOL_EXHAUSTED", (k, len(expanded)))
    picked = sorted(rng.sample(expanded, k)) if k else []
    return tuple((0, *row, row[3]+1, 0) for row in picked)


def _new_draw(pool: tuple, k: int, rng: Random) -> tuple[tuple, tuple]:
    if k < 0 or k > len(pool):
        raise ModelGap("NEW_POOL_EXHAUSTED", (k, len(pool)))
    a = list(pool)
    if 0 < k < len(a):
        for j in range(k):
            v = rng.randrange(j, len(a)); a[j], a[v] = a[v], a[j]
    return tuple(a), tuple(a[:k])


def _quality_count(k: int, nr: int, nc: int, rarity: str, rng: Random) -> int:
    if k > nr + nc:
        raise ModelGap("QUALITY_SUBPOOL_EXHAUSTED", (k, nr, nc))
    if rarity == "rare":
        raw = k
    elif rarity == "common":
        raw = 0
    elif rarity == "regular":
        raw = sum(_bern(rng, 1, 5) for _ in range(k))
    else:
        raise InvalidState("unknown rarity")
    return min(min(k, nr), max(max(0, k-nc), raw))


def _regular_page(s: PublicState, rng: Random) -> PublicState:
    src = s.source
    if src is None:
        raise InvalidState("potential source missing")
    if src.roleMode == "main":
        role = 1
    elif src.roleMode == "bound":
        role = src.boundRole
    elif src.roleMode == "any":
        weights = []
        for role in (1, 2, 3):
            c = sum(row[4] for row in s.H if row[0] == role)
            tau = 6 if role == 1 else 5
            w = max(1, min(100, 30*min(4, max(0, tau-c))-10))
            weights.append((4 if role == 1 else 3)*w)
        role = _weighted_int(rng, weights)+1
    else:
        raise InvalidState("unknown role source")
    owned = tuple(row for row in s.H if row[0] == role)
    eligible = tuple(row for row in owned if row[3] < 6)
    c, u = sum(r[4] for r in owned), sum(r[4] for r in eligible)
    tau = 6 if role == 1 else 5
    qn, qd = ((0, 1) if c <= 1 else (51, 100) if c <= 3
              else (3, 4) if c < tau else (1, 1))
    old_n = min(u, sum(_bern(rng, qn, qd) for _ in range(3)))
    new_n = 3-old_n
    nr, nc = s.pools[2*role-2], s.pools[2*role-1]
    old_r = tuple(row for row in eligible if row[1] == 1)
    old_c = tuple(row for row in eligible if row[1] == 2)
    kr = _quality_count(new_n, len(nr), len(nc), src.rarity, rng)
    ko = _quality_count(old_n, sum(r[4] for r in old_r), sum(r[4] for r in old_c), src.rarity, rng)
    pr, cr = _new_draw(nr, kr, rng)
    pc, cc = _new_draw(nc, new_n-kr, rng)
    pools = list(s.pools); pools[2*role-2] = pr; pools[2*role-1] = pc
    prefix = [(1, role, 1, target, 0, 0, j) for j, target in enumerate(cr, 1)]
    prefix += [(1, role, 2, target, 0, 0, j) for j, target in enumerate(cc, 1)]
    P, Q = s.P, s.Q
    offer = []
    for row in prefix:
        if not P and _bern(rng, 49, 100):
            grade = 1; P = True
        elif s.h >= 10:
            grade = 2; P = False; Q = False
        elif Q:
            grade = 3; P = False; Q = False
        elif _bern(rng, 1, 3):
            grade = 2; P = False; Q = True
        else:
            grade = 3; P = False; Q = False
        offer.append((*row[:5], grade, row[6]))
    if s.h >= 10:
        Q = False
    offer.extend(_uniform_old(old_r, ko, rng))
    offer.extend(_uniform_old(old_c, old_n-ko, rng))
    return replace(s, phase="PAGE", pools=tuple(pools), offer=tuple(offer), P=P, Q=Q)


def _select(s: PublicState, index: int) -> PublicState:
    if not 1 <= index <= len(s.offer) or s.source is None:
        raise InvalidState("invalid selection")
    c = s.offer[index-1]
    pools = s.pools; h = s.h; Q = s.Q
    if c[0]:
        pi = 2*(c[1]-1)+c[2]-1
        p = list(pools[pi]); j = c[6]-1
        if not 0 <= j < len(p) or p[j] != c[3]:
            raise InvalidState("new candidate mismatch")
        p[j] = p[-1]; p.pop()
        pp = list(pools); pp[pi] = tuple(p); pools = tuple(pp)
        H = _hist_change(s.H, (c[1], c[2], c[3], c[5]), 1)
        if c[5] == 3:
            h += 1
        if h >= 10:
            Q = False
    else:
        H = _hist_change(s.H, (c[1],c[2],c[3],c[4]), -1)
        H = _hist_change(H, (c[1],c[2],c[3],c[5]), 1)
    return replace(s, H=H, pools=pools, h=h, Q=Q, phase=s.source.returnPhase, source=None, offer=())


def _enhance_prob(g: int, f: int) -> Fraction:
    return ZERO if g >= 5 else F(min(3*(f+1), 10), 10)


def _enhance_latent(g: int, f: int, bonus: bool) -> tuple[int, int]:
    ng, nf = (g+1, 0) if bonus else (g, min(f+1, 3))
    return (5, 0) if ng >= 5 else (ng, nf)


@lru_cache(maxsize=20000)
def _enhance_posterior(belief: tuple, observed_bonus: bool | None) -> tuple:
    mass = defaultdict(F)
    for g, f, prior in belief:
        p = _enhance_prob(g, f)
        for b, q in ((False, ONE-p), (True, p)):
            if q and (observed_bonus is None or b == observed_bonus):
                mass[_enhance_latent(g, f, b)] += prior*q
    total = sum(mass.values(), ZERO)
    if total <= 0:
        raise InvalidState("impossible enhancement observation")
    return tuple((g, f, p/total) for (g, f), p in sorted(mass.items()))


def _enhance(s: PublicState, rng: Random, latent: Latent) -> tuple[PublicState, Latent]:
    eligible = [row[:4] for row in s.H if row[3] < 6 for _ in range(row[4])]
    if not eligible:
        raise InvalidState("no enhancement candidates")
    picked = sorted(rng.sample(eligible, min(3, len(eligible))))
    p = _enhance_prob(latent.gE, latent.fE)
    b = _bern(rng, p.numerator, p.denominator)
    ng, nf = _enhance_latent(latent.gE, latent.fE, b)
    visible_b = None if all(row[3] == 5 for row in picked) else b
    eBel = _enhance_posterior(s.eBel, visible_b)
    offer = tuple((0, *row, min(6, row[3]+1+int(b)), 0) for row in picked)
    return (replace(s, phase="ENHANCE_PAGE", offer=offer, eBel=eBel),
            replace(latent, gE=ng, fE=nf))


def _shelf(s: PublicState, rng: Random, first: bool, *, initial: bool = False) -> PublicState:
    u, v = rng.sample(range(10), 2)
    y, bonus = int(u < 6), int(v < 8)
    x = min(2, sum(_bern(rng, 65-5*y, 100) for _ in range(4)))
    b = int(s.shopOrdinal == 1)
    ng, nm = 2-b+x, b+y
    middle_initial = initial and s.shopOrdinal in (2, 3)
    half_n = min(2, sum(_bern(rng, 3 if middle_initial else 7, 4 if middle_initial else 10) for _ in range(3)))
    discounted = rng.sample(range(8), 4)
    half = frozenset(rng.sample(discounted, half_n))
    discount_set = frozenset(discounted)
    stock = [[0,0,0],[0,0,0]]
    for i in range(ng+nm):
        role = 0 if i < ng else 1
        price = 0 if i in half else 1 if i in discount_set else 2
        stock[role][price] += 1
    return replace(s, stock=tuple(map(tuple, stock)), phase="SHOP", firstInitial=False, offer=(), source=None)


# A coin path descriptor records only what is observable at the next boundary.
# node,floor,k,A,phase,coin_delta,shop_ordinal(0 unless entering a shop)
@lru_cache(maxsize=30000)
def _coin_paths(node: int, k: int, A: bool, g: int, f: int, g1: int, npc3_event_reward: bool = False) -> tuple:
    paths = [(ONE, node, k, A, g, f, g1, 0)]
    result = defaultdict(F)
    while paths:
        p, pos, kk, aa, gg, ff, first, money = paths.pop()
        if pos > len(ROUTE):
            # At this boundary the last real floor remains 20.
            desc = (pos, 20, kk, aa, "END", money, 0)
            result[(desc, gg, ff, first)] += p
            continue
        kind, floor, arg = ROUTE[pos-1]; pos += 1
        if kind == "BATTLE":
            kk = min(4, kk+1)
            pb = ZERO if gg >= 5 else F(min(30*(ff+1), 100), 101)
            if pb:
                paths.append((p*pb,pos,kk,aa,gg+1,0,1 if floor==1 else first,money+arg+50))
            if pb < ONE:
                nf = 0 if gg >= 5 else min(3,ff+1)
                paths.append((p*(ONE-pb),pos,kk,aa,gg,nf,0 if floor==1 else first,money+arg))
        elif kind == "EXTRA":
            pn = 50-30*int(aa)+30*first*(1-int(npc3_event_reward)) if floor==3 else min(30*kk,100)
            pe = F(pn,101)
            if pe:
                na = True if floor==1 else False if floor==3 else aa
                nfirst = 0 if floor==3 else first
                desc = (pos,floor,0,na,"EVENT_DRAW",money,0)
                result[(desc,gg,ff,nfirst)] += p*pe
            if pe < ONE:
                na = False if floor in (1,3) else aa
                nfirst = 0 if floor==3 else first
                paths.append((p*(ONE-pe),pos,kk,na,gg,ff,nfirst,money))
        else:
            phase = {"CORE":"CORE","DRAW":"DRAW","EVENT":"EVENT_DRAW","SHOP":"SHOP_UNSEEN"}[kind]
            desc = (pos,floor,kk,aa,phase,money,arg if kind=="SHOP" else 0)
            result[(desc,gg,ff,first)] += p
    return tuple((p, desc, (gg,ff,first)) for (desc,gg,ff,first),p in sorted(result.items()))


@lru_cache(maxsize=40000)
def _coin_belief_outputs(node: int, k: int, A: bool, belief: tuple, npc3_event_reward: bool = False) -> dict:
    groups = defaultdict(lambda: defaultdict(F))
    for g, f, g1, prior in belief:
        for p, desc, nxt in _coin_paths(node,k,A,g,f,g1,npc3_event_reward):
            groups[desc][nxt] += prior*p
    out = {}
    for desc, masses in groups.items():
        total = sum(masses.values(),ZERO)
        out[desc] = tuple((*latent, p/total) for latent,p in sorted(masses.items()))
    return out


def _advance(s: PublicState, rng: Random, latent: Latent) -> tuple[PublicState, Latent]:
    paths = _coin_paths(s.node,s.k,s.A,latent.gC,latent.fC,latent.G1,s.npc3EventReward)
    p, desc, nxt = paths[_weighted_fraction(rng, tuple(row[0] for row in paths))]
    belief = _coin_belief_outputs(s.node,s.k,s.A,s.cBel,s.npc3EventReward).get(desc)
    if belief is None:
        raise InvalidState("latent coin particle inconsistent with public belief")
    node,floor,k,A,phase,delta,shop = desc
    kw = dict(npc3EventReward=s.npc3EventReward and node <= 7, node=node,floor=floor,k=k,A=A,phase=phase,coins=s.coins+delta,cBel=belief)
    if phase == "DRAW":
        kw["source"] = Source()
    elif phase == "SHOP_UNSEEN":
        kw.update(shopOrdinal=shop,firstInitial=shop==1,paidIndex=0,stock=EMPTY_STOCK,shopReturnPhase="ADVANCE")
    return replace(s,**kw), replace(latent,gC=nxt[0],fC=nxt[1],G1=nxt[2])


def _event_draw(s: PublicState, rng: Random) -> PublicState:
    eligible = [(eid,row[0]) for eid,row in sorted(EVENT_DATA.items()) if row[0]>0 and eid not in s.seen]
    if not eligible:
        raise ModelGap("EVENT_POOL_EXHAUSTED")
    if s.floor < 5 and 105 not in s.seen and _bern(rng,7,10):
        eid = 105
    else:
        eid = eligible[_weighted_int(rng,tuple(w for _,w in eligible))][0]
    return replace(s,phase="EVENT",eventId=eid,eventOptions=EVENT_DATA[eid][1],seen=tuple(sorted((*s.seen,eid))))


def visible_event(s: PublicState, event_id: int, visible_ids: Iterable[int], disabled_ids: Iterable[int]=()) -> PublicState:
    if event_id not in EVENT_DATA:
        raise ModelGap("UNKNOWN_EVENT",event_id)
    _,pred,opts = EVENT_DATA[event_id]
    v = set(visible_ids)
    if not v.issubset(opts):
        raise ModelGap("UNKNOWN_EVENT_OPTION",tuple(v-set(opts)))
    ids = (v.intersection(pred)-set(disabled_ids)) | {0}
    return replace(s,phase="EVENT",eventId=event_id,eventOptions=tuple(sorted(ids)),seen=tuple(sorted(set(s.seen)|{event_id})))


def _event_option(s: PublicState, a: Action, rng: Random) -> PublicState:
    if s.eventId not in s.seen:
        raise InvalidState("displayed event must already count as seen")
    t = replace(s,phase="ADVANCE",eventId=0,eventOptions=())
    if a.kind == "SKIP_EVENT":
        return t
    cost,outcomes = EVENT_DATA[s.eventId][2][a.index]
    # The exact R12 action is undefined if ANY positive-probability branch
    # reaches this boundary. Never let favorable sampling hide that gap.
    if any(n and s.coins+delta<0 for n,d,delta,kind,scope in outcomes):
        raise ModelGap("INSUFFICIENT_RANDOM_LOSS",{"event":s.eventId,"option":a.index,"coins":s.coins})
    probabilities = tuple(F(row[0],row[1]) for row in outcomes)
    n,d,delta,kind,scope = outcomes[_weighted_fraction(rng,probabilities)]
    t = replace(t,coins=t.coins+delta)
    if kind == "none":
        return t
    if scope == "support_random":
        src = Source(kind,"bound",2+rng.randrange(2))
    else:
        src = Source(kind,"main" if scope=="main" else "any")
    return replace(t,phase="DRAW",source=src,
                   npc3EventReward=t.npc3EventReward or 3 <= s.node <= 6)


def _single_step(s: PublicState, a: Action, rng: Random, latent: Latent) -> tuple[PublicState, Latent]:
    if a.kind == "SELECT":
        return _select(s,a.index), latent
    if a.kind == "REROLL":
        return replace(s,coins=s.coins-40,phase="DRAW",offer=()), latent
    if a.kind == "BUY":
        stock = [list(row) for row in s.stock]; stock[a.type-1][a.index-1] -= 1
        t = replace(s,coins=s.coins-a.price,stock=tuple(map(tuple,stock)),phase="DRAW",offer=(),
                    source=Source("regular","main" if a.type==2 else "any",0,"SHOP"))
        return t, latent
    if a.kind == "ENHANCE":
        t = replace(s,coins=s.coins-a.price,freeEnhance=False,
                    paidIndex=s.paidIndex if s.freeEnhance else min(6,s.paidIndex+1),
                    source=Source("regular","any",0,"SHOP",False))
        return _enhance(t,rng,latent)
    if a.kind == "RESTOCK":
        return _shelf(replace(s,coins=s.coins-100,restocks=s.restocks-1),rng,False), latent
    if a.kind == "LEAVE":
        return replace(s,phase=s.shopReturnPhase,stock=EMPTY_STOCK,offer=(),source=None,paidIndex=0,firstInitial=False),latent
    if a.kind in ("EVENT_OPTION","SKIP_EVENT"):
        return _event_option(s,a,rng),latent
    if a.kind == "ADVANCE":
        return _advance(s,rng,latent)
    if a.kind == "CONFIRM_CORE":
        return replace(s,phase="ADVANCE"),latent
    if a.kind == "GENERATE_PAGE":
        return _regular_page(s,rng),latent
    if a.kind == "GENERATE_EVENT":
        return _event_draw(s,rng),latent
    if a.kind == "OPEN_SHOP":
        return _shelf(s,rng,s.firstInitial,initial=True),latent
    raise InvalidState("unrecognized action")


def _drain_forced(s: PublicState, rng: Random, latent: Latent) -> tuple[PublicState,Latent]:
    while s.phase in FORCED_PHASES:
        s,latent = _single_step(s,FORCED_ACTIONS[s.phase],rng,latent)
    return s,latent


def advance_to_decision(s: PublicState, rng: Random, latent: Latent | None=None) -> Transition:
    """Collapse forced UI actions, conditioning beliefs at EVERY boundary.

    CORE balances remain observation boundaries even though no choice is
    made there. Battles 7/8 and (when no NPC) 14/15 are observed jointly, as
    in R12. EVENT/PAGE/ENHANCE_PAGE/SHOP are never automatically decided.
    """
    latent = sample_latent(s,rng) if latent is None else latent
    s,latent = _drain_forced(s,rng,latent)
    # Forced transitions display offers or move between interfaces; none
    # confirms an ordinary potential. The six cores are a fixed constant.
    return Transition(s,s,latent,0,is_terminal(s))


def step(s: PublicState, a: Action, rng: Random, latent: Latent | None=None,
         collapse_forced: bool=True, validate_action: bool=True) -> Transition:
    """Apply a public action and sample its frozen outcomes.

    validate_action=False is ONLY for trusted planning loops that obtained
    `a` from legal_actions(the SAME immutable state). External/software calls
    retain the default validation; do not use this flag to accept UI input.
    """
    if validate_action and a not in legal_actions(s):
        raise InvalidState(f"illegal action {a} in phase {s.phase}")
    before = score(s)
    latent = sample_latent(s,rng) if latent is None else latent
    t,latent = _single_step(s,a,rng,latent)
    if t.coins < 0 or t.restocks < 0 or any(x<0 for row in t.stock for x in row):
        raise InvalidState("negative balance, refresh count or inventory")
    if collapse_forced:
        t,latent = _drain_forced(t,rng,latent)
    return Transition(t,t,latent,score(t)-before,is_terminal(t))


def late_state(coins: int=188) -> PublicState:
    if coins not in (188,308):
        raise InvalidState("recorded late fixture supports only 188 or 308 coins")
    H = [(1,2,6,3,1),(1,2,6,4,1),(1,2,6,6,4),
         (2,1,6,4,1),(2,1,6,6,1),(2,2,0,3,1),(2,2,6,6,1),
         (3,1,6,6,1),(3,2,0,2,1)]
    if coins == 188:
        H.append((3,2,6,5,2)); belief=((5,0,ONE),); paid=2
    else:
        H.extend(((3,2,6,3,1),(3,2,6,5,1))); belief=((4,1,ONE),); paid=1
    return replace(initial_843(),phase="SHOP",node=46,floor=20,shopOrdinal=4,shopReturnPhase="END",
                   coins=coins,freeEnhance=False,restocks=0,firstInitial=False,
                   stock=((0,0,1),(0,0,0)),P=False,Q=False,h=10,
                   pools=((6,0,6),(0,0,0),(0,),(6,0,0,0,0,0,0),(0,0),(0,0,0,0,0,0)),
                   H=tuple(sorted(H)),eBel=belief,paidIndex=paid)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj,Fraction):
        return {"numerator":obj.numerator,"denominator":obj.denominator}
    if is_dataclass(obj):
        return {f.name:to_jsonable(getattr(obj,f.name)) for f in fields(obj)}
    if isinstance(obj,(tuple,list)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj,dict):
        return {str(k):to_jsonable(v) for k,v in obj.items()}
    return obj


def state_from_dict(data: dict) -> PublicState:
    d = dict(data)
    if "modelVersion" in d:
        if "model" in d and d["model"]!=d["modelVersion"]:
            raise InvalidState("MODEL_VERSION_CONFLICT")
        d["model"]=d.pop("modelVersion")
    required={f.name for f in fields(PublicState)}
    if required-set(d):
        raise InvalidState("INPUT_INCOMPLETE",sorted(required-set(d)))
    if set(d)-required:
        raise InvalidState("INPUT_UNKNOWN_FIELDS",sorted(set(d)-required))
    def integer(x):
        if not isinstance(x,Real) or isinstance(x,bool) or not isfinite(x) or x!=int(x):
            raise InvalidState("NONINTEGER_STATE_FIELD",repr(x))
        return int(x)
    for key in ("node","floor","cap","h","coins","paidIndex","restocks","shopOrdinal","k","eventId"):
        d[key]=integer(d[key])
    for key in ("P","Q","freeEnhance","firstInitial","A","npc3EventReward"):
        if d[key] not in (False,True,0,1):raise InvalidState("NONBOOLEAN_STATE_FIELD",key)
        d[key]=bool(d[key])
    for key in ("H","pools","stock","offer"):
        if key in d:
            d[key] = tuple(tuple(integer(v) for v in row) for row in d[key])
    for key in ("seen","eventOptions"):
        if key in d:
            d[key] = tuple(integer(x) for x in d[key])
    for key in ("eBel","cBel"):
        if key in d:
            b=[]
            for row in d[key]:
                p=row[-1]
                if isinstance(p,dict):p=F(integer(p["numerator"]),integer(p["denominator"]))
                elif isinstance(p,(list,tuple)):p=F(*p)
                else:p=F(p)
                b.append((*map(integer,row[:-1]),p))
            d[key]=tuple(b)
    if isinstance(d.get("source"),dict):
        src=dict(d["source"])
        if set(src)!={f.name for f in fields(Source)}:raise InvalidState("SOURCE_INCOMPLETE_OR_UNKNOWN_FIELDS")
        src["boundRole"]=integer(src["boundRole"])
        if src["rerollAllowed"] not in (False,True,0,1):raise InvalidState("SOURCE_REROLL_FLAG")
        src["rerollAllowed"]=bool(src["rerollAllowed"])
        d["source"] = Source(**src)
    elif not d.get("source"):
        d["source"] = None
    state=PublicState(**d)
    validate_state(state)
    return state


def validate_state(s: PublicState) -> None:
    """Validate a COMPLETE software snapshot; never fill missing history."""
    if not isinstance(s,PublicState):
        raise InvalidState("INVALID_PUBLIC_STATE_TYPE")
    if not isinstance(s.model,str):
        raise InvalidState("UNSUPPORTED_MODEL_VERSION")
    if s.model not in MODEL_ALIASES:
        # R14 encoded its general-preset vector in the version string. It is
        # an explicitly known same-rule alias, not a wildcard future model.
        prefix="R12_frozen_general_v1|"
        if not s.model.startswith(prefix) or len(s.model)>4096:
            raise InvalidState("UNSUPPORTED_MODEL_VERSION",s.model)
        try:
            old_pools=literal_eval(s.model[len(prefix):])
            valid=(isinstance(old_pools,tuple)
                   and all(isinstance(row,tuple) for row in old_pools)
                   and tuple(map(len,old_pools))==INITIAL_POOL_LENGTHS
                   and all(type(x) is int and 0<=x<=6 for row in old_pools for x in row))
        except (TypeError,ValueError,SyntaxError,RecursionError):
            valid=False
        if not valid:
            raise InvalidState("UNSUPPORTED_MODEL_VERSION",s.model)
    integer_fields=("node","floor","cap","h","coins","paidIndex","restocks","shopOrdinal","k","eventId")
    for name in integer_fields:
        if type(getattr(s,name)) is not int:
            raise InvalidState("DIRECT_STATE_REQUIRES_INTEGER",name)
    for name in ("P","Q","freeEnhance","firstInitial","A","npc3EventReward"):
        if type(getattr(s,name)) is not bool:
            raise InvalidState("DIRECT_STATE_REQUIRES_BOOLEAN",name)
    for name in ("H","pools","stock","offer"):
        rows=getattr(s,name)
        if not isinstance(rows,tuple) or any(not isinstance(row,tuple) for row in rows):
            raise InvalidState("DIRECT_STATE_REQUIRES_IMMUTABLE_TUPLES",name)
        if any(type(v) is not int for row in rows for v in row):
            raise InvalidState("DIRECT_STATE_REQUIRES_INTEGER",name)
    for name in ("seen","eventOptions"):
        values=getattr(s,name)
        if not isinstance(values,tuple) or any(type(v) is not int for v in values):
            raise InvalidState("DIRECT_STATE_REQUIRES_INTEGER_TUPLE",name)
    for name in ("eBel","cBel"):
        rows=getattr(s,name)
        if not isinstance(rows,tuple) or any(not isinstance(row,tuple) for row in rows):
            raise InvalidState("DIRECT_STATE_REQUIRES_IMMUTABLE_TUPLES",name)
        if any(type(v) is not int for row in rows for v in row[:-1]):
            raise InvalidState("DIRECT_STATE_REQUIRES_INTEGER",name)
    phases=FORCED_PHASES|frozenset(("PAGE","ENHANCE_PAGE","EVENT","SHOP","END"))
    if not isinstance(s.model,str) or not s.model or s.phase not in phases:
        raise InvalidState("UNKNOWN_STATE_MODEL_OR_PHASE")
    if not 1<=s.node<=len(ROUTE)+1 or not 0<=s.floor<=20:
        raise InvalidState("ROUTE_POSITION_OUT_OF_RANGE")
    if s.floor!=(0 if s.node==1 else ROUTE[s.node-2][1]):
        raise InvalidState("FLOOR_NODE_MISMATCH")
    if s.coins<0 or s.cap<0 or not 0<=s.h<=10 or not 0<=s.k<=4:
        raise InvalidState("NEGATIVE_OR_OUT_OF_RANGE_RESOURCE")
    if not 0<=s.paidIndex<=6 or not 0<=s.restocks<=2 or not 0<=s.shopOrdinal<=4:
        raise InvalidState("SHOP_COUNTER_OUT_OF_RANGE")
    if s.freeEnhance and s.paidIndex!=0:raise InvalidState("FREE_ENHANCEMENT_WITH_PAID_HISTORY")
    if s.h>=10 and s.Q:raise InvalidState("NONCANONICAL_Q_AT_H10")
    if len(s.stock)!=2 or any(len(row)!=3 or any(n<0 for n in row) for row in s.stock):
        raise InvalidState("INVALID_STOCK")
    if sum(map(sum,s.stock))>8:raise InvalidState("STOCK_EXCEEDS_EIGHT_SLOTS")
    if len(s.pools)!=6 or any(not 0<=x<=6 for p in s.pools for x in p):
        raise InvalidState("INVALID_ORDERED_POOLS")
    if any(len(row)!=5 for row in s.H):raise InvalidState("INVALID_H_SHAPE")
    if any(r not in (1,2,3) or q not in (1,2) or not 0<=t<=6 or not 1<=l<=6 or n<1 for r,q,t,l,n in s.H):
        raise InvalidState("INVALID_LEARNED_CATEGORY")
    if len({row[:4] for row in s.H})!=len(s.H) or tuple(sorted(s.H))!=s.H:
        raise InvalidState("NONCANONICAL_H")
    if sum(row[4] for row in s.H)>36:
        raise InvalidState("LEARNED_COUNT_EXCEEDS_36")
    if s.cap!=sum(row[2]*row[4] for row in s.H)+sum(map(sum,s.pools)):
        raise InvalidState("TARGET_CAP_POOL_H_MISMATCH")
    for role in (1,2,3):
        for quality,expected in ((1,3),(2,9)):
            total=sum(row[4] for row in s.H if row[0]==role and row[1]==quality)+len(s.pools[2*(role-1)+quality-1])
            if total!=expected:
                raise InvalidState("ROLE_QUALITY_CATALOGUE_MISMATCH",(role,quality,total,expected))
    for name,width in (("eBel",3),("cBel",4)):
        bel=getattr(s,name)
        if not bel or any(len(row)!=width for row in bel):raise InvalidState("INVALID_BELIEF_SHAPE",name)
        if any(not isinstance(row[-1],F) or row[-1]<=0 for row in bel) or sum((row[-1] for row in bel),ZERO)!=ONE:
            raise InvalidState("INVALID_BELIEF_MASS",name)
        if any(not 0<=row[0]<=5 or not 0<=row[1]<=3 or (row[0]==5 and row[1]!=0) for row in bel):
            raise InvalidState("INVALID_BELIEF_COUNTER",name)
        if name=="cBel" and any(row[2] not in (0,1) for row in bel):raise InvalidState("INVALID_FIRST_BATTLE_BELIEF")
        if len({row[:-1] for row in bel})!=len(bel):raise InvalidState("DUPLICATE_BELIEF_ROWS",name)
    if s.source is not None:
        src=s.source
        if not isinstance(src,Source) or src.rarity not in ("regular","common","rare") or src.roleMode not in ("any","main","bound"):
            raise InvalidState("INVALID_POTENTIAL_SOURCE")
        if type(src.boundRole) is not int or type(src.rerollAllowed) is not bool:
            raise InvalidState("DIRECT_SOURCE_REQUIRES_CANONICAL_TYPES")
        if src.returnPhase not in ("ADVANCE","SHOP","END") or (src.roleMode=="bound" and src.boundRole not in (1,2,3)):
            raise InvalidState("INVALID_POTENTIAL_SOURCE_BINDING")
    if s.phase in ("DRAW","PAGE","ENHANCE_PAGE") and s.source is None:
        raise InvalidState("SOURCE_REQUIRED_FOR_POTENTIAL_PHASE")
    if s.phase in ("PAGE","ENHANCE_PAGE"):
        if not 1<=len(s.offer)<=3 or any(len(c)!=7 for c in s.offer):raise InvalidState("INVALID_VISIBLE_OFFER")
        old_counts=Counter()
        for c in s.offer:
            new,r,q,t,old,level,idx=c
            if r not in (1,2,3) or q not in (1,2) or not 0<=t<=6 or not 1<=level<=6:
                raise InvalidState("INVALID_VISIBLE_CANDIDATE")
            if new==1:
                if s.phase!="PAGE" or old!=0 or level>3:raise InvalidState("INVALID_NEW_CANDIDATE")
                pool=s.pools[2*(r-1)+q-1]
                if not 1<=idx<=len(pool) or pool[idx-1]!=t:raise InvalidState("VISIBLE_NEW_POOL_MISMATCH")
            elif new==0:
                old_counts[(r,q,t,old)]+=1
                if idx!=0 or not 1<=old<6 or level not in ({old+1} if s.phase=="PAGE" else {min(6,old+1),min(6,old+2)}):
                    raise InvalidState("INVALID_OLD_CANDIDATE")
            else:raise InvalidState("INVALID_CANDIDATE_NEW_FLAG")
        counts={row[:4]:row[4] for row in s.H}
        if any(n>counts.get(cat,0) for cat,n in old_counts.items()):raise InvalidState("VISIBLE_OLD_H_MISMATCH")
    elif s.offer:
        raise InvalidState("OFFER_OUTSIDE_SELECTION_PHASE")
    if len(set(s.seen))!=len(s.seen) or any(e not in EVENT_DATA for e in s.seen):raise InvalidState("INVALID_EVENT_SEEN_SET")
    if s.phase=="EVENT":
        if s.eventId not in s.seen or s.eventId not in EVENT_DATA:raise InvalidState("EVENT_MUST_ALREADY_BE_SEEN")
        if any(e not in EVENT_DATA[s.eventId][2] for e in s.eventOptions) or 0 not in s.eventOptions:
            raise InvalidState("INVALID_CURRENT_EVENT_OPTIONS")
