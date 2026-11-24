"""Connect potential and shop readers to optional research observations."""
import uuid

from .shop_ui.handler import ShopHandler
from .sampling_data import snapshot
from .sampling_evidence import raw_ocr
from .observation import PotentialObservationIncomplete


def run_observed_operation(collector, event, handler, function, *args, before=False, offer_id=None, operation_id=None):
    if not collector.ensure_run(event):
        return function(*args)
    data = handler.data
    initial = snapshot(data)
    common = dict(operation_id=operation_id or uuid.uuid4().hex, offer_id=offer_id,
                  reward_chain_id=collector.reward["reward_chain_id"] if collector.reward and not collector.reward["closed"] else None,
                  stock_scan_id=collector.stock_scan_id,
                  source_type=getattr(data.params, "potential_source", None))
    if event == "selection_result" and args:
        common["selected_candidate_index"] = args[0].index
    if offer_id and collector.reward and not collector.reward["closed"]:
        common.update(collector.reward["source_fields"], source_raw=data.params.potential_source)
    if before:
        collector.emit(event + "_attempt", phase="before", coin_before=initial.get("current_coin"),
                  refresh_price=initial.get("refresh_cost"), **common)
    with collector.action_scope(**common), collector.capture(event, **common) as read:
        result = function(*args)
    next_offer_id = (collector.refresh_succeeded(offer_id, common["operation_id"])
                     if event == "refresh_result" and result else None)
    outcome = dict(result=snapshot(result),
                   Maa_reported_success=result if isinstance(result, bool) else None,
                   next_offer_id=next_offer_id, coin_before=initial.get("current_coin"),
                   read_id=read.read_id, **common)
    if event == "shop_refresh_result" and result is True and not handler.interactor.context.tasker.stopping:
        # The adapter publishes the final confirmed frame before completing this operation.
        outcome["refresh_remaining_after_action"] = data.refresh_remaining
        outcome["previous_state"] = initial
        handler.pending_refresh = outcome
    else:
        collector.emit(event, phase="after",
                  coin_after=None if event == "shop_refresh_result" else getattr(data, "current_coin", None),
                  **outcome)
    return result


class ObservedPotentialHandler:
    """Delegate the official handler without adding attributes to its instance."""
    def __init__(self, handler, collector, notes=None):
        self.handler = handler
        self.collector = collector
        self.notes = notes
        self.offer_id = None
        self.selection_id = None
        self.page_read = None
        read = collector.current_read
        self.preload_records = list(read.recognitions) if read else []

    def __getattr__(self, name):
        return getattr(self.handler, name)

    def initialize_potentials(self):
        if not self.collector.enabled or not self.collector.run_confirmed:
            return self.handler.initialize_potentials()
        page = self.collector.prepare_offer(self.handler.data)
        source = self.collector.reward.get("source_fields", {})
        if self.notes is not None:
            restored, _ = self.notes.complete_potential_reward_inventory(self.handler.screen,
                source_event_id=source.get("source_event_id") if source.get("source") == "event" else None)
            if not restored:
                raise PotentialObservationIncomplete("event inventory navigation did not return to the potential page")
        self.page_read = self.collector.new_read("potential_page", offer_id=page["offer_id"],
            reward_chain_id=self.collector.reward["reward_chain_id"])
        self.page_read.recognitions.extend(self.preload_records)
        self.preload_records = []
        with self.collector.capture(read=self.page_read):
            return self.handler.initialize_potentials()

    def read_potentials_info(self):
        if not self.collector.enabled or not self.collector.run_confirmed:
            return self.handler.read_potentials_info()
        with self.collector.capture(read=self.page_read):
            result = self.handler.read_potentials_info()
        self.offer_id = self.collector.observe_offer(self.handler.data, self.page_read)
        return result

    def choose_potential(self):
        return self.handler.choose_potential()

    def pick(self, potential):
        self.selection_id = uuid.uuid4().hex
        return run_observed_operation(
            self.collector, "selection_result", self.handler, self.handler.pick, potential,
            before=True, offer_id=self.offer_id, operation_id=self.selection_id,
        )

    def cache_potential_data(self, potential):
        result = self.handler.cache_potential_data(potential)
        if self.collector.enabled and self.collector.run_confirmed:
            self.collector.cache_selection(potential, self.offer_id, self.selection_id)
        return result

    def refresh(self):
        return run_observed_operation(
            self.collector, "refresh_result", self.handler, self.handler.refresh, before=True, offer_id=self.offer_id
        )


class ObservedShopHandler(ShopHandler):
    def __init__(self, interactor, data, collector):
        super().__init__(interactor, data)
        self.collector = collector
        self.shop_visit_id = uuid.uuid4().hex
        self.stock_generations = {}
        self.stock_generation_id = None
        self.observed_stock_signature = None
        self.observed_stock_scan_id = None
        self.pending_refresh = None

    def read_goods_page_info(self, *, enter_shopping=True, refresh_exhausted=False):
        if not self.collector.ensure_run("shop_page"):
            return super().read_goods_page_info(enter_shopping=enter_shopping, refresh_exhausted=refresh_exhausted)
        self.collector.stock_scan_id = uuid.uuid4().hex
        with self.collector.capture("shop_stock", stock_scan_id=self.collector.stock_scan_id,
                                    shop_visit_id=self.shop_visit_id) as read:
            result = super().read_goods_page_info(enter_shopping=enter_shopping, refresh_exhausted=refresh_exhausted)
        self._record_stock(read)
        self._complete_refresh_record(read)
        return result

    def refresh(self):
        if not self.collector.ensure_run("shop_refresh_result"):
            return run_observed_operation(self.collector, "shop_refresh_result", self, super().refresh, before=True)
        stock_read = None

        def capture_frame(observe, image, *, new_scan):
            nonlocal stock_read
            action_read = self.collector.current_read
            if stock_read is None or new_scan:
                stock_read = self.collector.new_read("shop_refresh_frame", stock_scan_id=None,
                                                    shop_visit_id=self.shop_visit_id)
            previous_count = len(stock_read.recognitions)
            with self.collector.capture(read=stock_read) as read:
                result = observe(image)
            action_read.recognitions.extend(read.recognitions[previous_count:])
            return result

        result = run_observed_operation(self.collector, "shop_refresh_result", self,
            lambda: super(ObservedShopHandler, self).refresh(frame_capture=capture_frame), before=True)
        if self.pending_refresh is not None:
            self.pending_refresh.update(
                refresh_remaining_after_action_basis=self.data.refresh_remaining_basis,
                refresh_remaining_after_action_observed=self.data.refresh_remaining_observed)
        if result and not self.interactor.context.tasker.stopping:
            self.collector.stock_scan_id = uuid.uuid4().hex
            self._record_stock(stock_read)
            self._complete_refresh_record(stock_read)
        return result

    def _enhance(self):
        return run_observed_operation(self.collector, "enhancement_result", self, super()._enhance, before=True)

    def _record_stock(self, read):
        if not self.collector.ensure_run("shop_page"):
            return
        data = self.data
        remaining = data.refresh_remaining
        previous_generation = self.stock_generation_id
        previous_remaining = next((count for count, generation in self.stock_generations.items()
                                   if generation == previous_generation), None)
        pending = self.pending_refresh
        refreshed = bool(pending and pending.get("result") is True)
        advanced = remaining >= 0 and previous_remaining is not None and remaining < previous_remaining
        generation = (uuid.uuid4().hex if refreshed or advanced or
                      (previous_generation is None and remaining >= 0) else previous_generation)
        generation_remaining = pending.get("refresh_remaining_after_action", remaining) if refreshed else remaining
        if generation is not None and generation_remaining >= 0 and (refreshed or advanced or previous_generation is None):
            self.stock_generations[generation_remaining] = generation
        repeated = generation is not None and generation == previous_generation
        self.stock_generation_id = generation
        items = snapshot(data.items)
        by_slot = {p["grid_num"]-1: p for p in items}
        slots = []
        for slot, rois in enumerate(data.params.item_rois):
            name = read.matching("星塔_节点_商店_购物_识别物品内容_agent", rois["name_roi"])
            price = read.matching(("星塔_神经网络_商店_识别物品价格",
                                   "星塔_节点_商店_购物_识别物品价格_agent"), rois["price_roi"])
            item = by_slot.get(slot)
            observed_item = {key: item[key] for key in (
                "internal_name", "quantity", "price", "display_name", "trekker_specified", "missing")
                if key in item} if item else None
            if observed_item:
                missing = observed_item.get("missing", [])
                for field, column in (("name", "internal_name"), ("quantity", "quantity"), ("price", "price")):
                    if field in missing:
                        observed_item[column] = None
                if "name" in missing:
                    observed_item["trekker_specified"] = None
            effective_price = observed_item.get("price") if observed_item else None
            slots.append(dict(slot=slot,
                item_instance_id=f"{generation}:{slot}" if generation else None, item=observed_item,
                effective_price=effective_price if effective_price is not None and effective_price >= 0 else None))
            if observed_item is None or observed_item.get("missing"):
                slots[-1]["unresolved_values"] = {key: list(dict.fromkeys(row.get("text", "") for row in raw_ocr(records)))
                    for key, records in (("name", name), ("price", price)) if records}
        signature = (generation, slots, data.current_coin, remaining,
                     getattr(data, "refresh_remaining_basis", "ocr"), data.refresh_cost)
        if repeated and getattr(self, "observed_stock_signature", None) == signature:
            self.collector.stock_scan_id = self.observed_stock_scan_id
            return
        self.observed_stock_signature = signature
        self.observed_stock_scan_id = self.collector.stock_scan_id
        self.collector.emit("shop_observed", phase="after", shop_event_type="stock_snapshot",
                  stock_scan_id=self.collector.stock_scan_id, shop_visit_id=self.shop_visit_id,
                  shop_entry_id=self.shop_visit_id, stock_generation_id=generation,
                  shelf_generation_id=generation, previous_shelf_generation_id=previous_generation,
                  shelf_generation_basis=("confirmed_refresh_operation" if refreshed else
                      "confirmed_global_refresh_counter" if generation and not repeated
                      and getattr(data, "refresh_remaining_basis", "ocr") == "global_confirmed_refreshes_exhausted" else
                      "visible_remaining_refresh_count" if generation and not repeated else
                      "same_visit_without_confirmed_refresh" if repeated else "refresh_count_not_observed"),
                  shelf_generation_complete=generation is not None,
                  repeated_scan=repeated, refresh_remaining=remaining,
                  refresh_remaining_observed=getattr(data, "refresh_remaining_observed", remaining),
                  refresh_remaining_basis=getattr(data, "refresh_remaining_basis", "ocr"),
                  refresh_controls_expected_visible=(getattr(data, "refresh_remaining_basis", "ocr")
                                                     != "global_confirmed_refreshes_exhausted"),
                  payload=dict(slots=slots), coin=data.current_coin,
                  refresh_price=data.refresh_cost, read_id=read.read_id,
                  missing=data.goods_missing)

    def _complete_refresh_record(self, read):
        if not self.collector.enabled or not self.collector.run_confirmed:
            return
        pending = self.pending_refresh
        if pending is None:
            return
        self.pending_refresh = None
        before, data = pending["previous_state"], self.data
        wallet = self.collector.last_wallet
        coin_after = (wallet["coin"] if wallet and wallet["coin_complete"]
                      and wallet["recognition_record_id"] in read.record_ids else None)
        coin_before, price = before.get("current_coin"), before.get("refresh_cost")
        debit_confirmed = (coin_before is not None and coin_before >= 0 and price is not None
                           and price > 0 and coin_after is not None and coin_after >= 0
                           and coin_before - coin_after == price)
        remaining_before, remaining_after = before.get("refresh_remaining"), data.refresh_remaining
        action_remaining = pending.get("refresh_remaining_after_action")
        remaining_decreased = (remaining_before is not None and remaining_before > 0
                               and (remaining_after == remaining_before - 1
                                    or action_remaining == remaining_before - 1))
        items = snapshot(data.items)
        stock_complete = (len(items) == len(data.params.item_rois)
                          and {item["grid_num"] for item in items} == set(range(1, len(items) + 1)))
        shelf_changed = items != before.get("items")
        success = True if debit_confirmed and stock_complete and (remaining_decreased or shelf_changed) else None
        pending.update(result=success, Maa_return_value=pending["result"])
        pending.pop("previous_state")
        self.collector.emit("shop_refresh_result", phase="after", coin_after=coin_after,
                  success=success, operation_confirmed=success is True,
                  confirmation_basis="next_shelf_wallet_and_refresh_count_or_stock_change",
                  confirmation_status="confirmed" if success else "incomplete",
                  debit_confirmed=debit_confirmed, refresh_remaining_before=remaining_before,
                  refresh_remaining_after=remaining_after, refresh_remaining_decreased=remaining_decreased,
                  stock_read_complete=stock_complete, shelf_changed=shelf_changed,
                  next_stock_scan_id=self.collector.stock_scan_id, stock_read_id=read.read_id,
                  wallet_observation_id=wallet["record_id"] if coin_after is not None else None,
                  **pending)
