"""Rules shared by the live tower readers and model adapters."""
from .vendor.stella_sim import ROUTE


BATTLE_FLOORS = frozenset(floor for kind, floor, _ in ROUTE if kind == "BATTLE")
LAST_FLOOR = max(floor for _, floor, _ in ROUTE)

# The frozen simulator enforces these limits internally.
MAX_POTENTIAL_LEVEL = 6
MAX_SHOP_RESTOCKS = 2
