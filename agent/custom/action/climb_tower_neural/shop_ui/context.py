"""Per-visit neural data; the native shop's configuration tables are read-only."""
from dataclasses import dataclass, field

from custom.action.climb_tower_shop.context import (
    ITEM_STANDARD_PRICES, Item as NativeItem, ShopParams,
)


@dataclass
class Item(NativeItem):
    missing: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ShopContext:
    params: ShopParams
    shop_type: str = ""
    current_coin: int = -1
    refresh_remaining: int = -1
    refresh_remaining_observed: int = -1
    refresh_remaining_basis: str = "unconfirmed"
    refresh_cost: int = -1
    current_enhancement_cost: int = -1
    items: list[Item] = field(default_factory=list)
    _item_reverse_maps: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self):
        self._item_reverse_maps = {
            name: internal_name
            for internal_name, languages in self.params.item_translations.items()
            for name in languages.get(self.params.lang, ())
        }

    def parse_item_name(self, text):
        if text in self._item_reverse_maps:
            return self._item_reverse_maps[text]
        for name, internal_name in self._item_reverse_maps.items():
            if name in text:
                return internal_name
        return ""

    @property
    def goods_missing(self):
        by_slot = {item.grid_num: item for item in self.items}
        return [f"shop_slot_{slot}_{name}"
                for slot in range(1, len(self.params.item_rois) + 1)
                for name in (by_slot[slot].missing if slot in by_slot else ["unread"])]
