"""Model economics with neural-only shop observations and confirmed operations."""
import uuid
from .shop_ui.interactor import ShopInteractor
from .shop_ui.purchase import purchase_override

from .tower_sampling_collector import ObservedShopHandler, snapshot
from .purchase_evidence import purchase_wallet_evidence
from .pages import BY_NAME, classify
from .ui_flow import read_until
from .value_model import Action, sim, replace


class ShopNavigationFailed(RuntimeError):
    """A model shop action cannot continue on the current page."""

    def __init__(self, action, *, observed_page=None):
        super().__init__(action)
        self.observed_page = observed_page


class ModelShopInteractor(ShopInteractor):
    def __init__(self, context):
        super().__init__(context)
        self.last_shop_page = None

    def is_shopping(self):
        shopping = classify(self.context, self.image, candidates={"shop"}) == "shop"
        self.last_shop_page = "shopping" if shopping else None
        return shopping

    def is_main_page(self):
        main = classify(self.context, self.image, candidates={"shop_main"}) == "shop_main"
        self.last_shop_page = "main" if main else None
        return main

    # The upstream handler calls these internally and ignores False.
    def enter_shopping(self):
        if not super().enter_shopping():
            if self.current_shop_page(required="shopping") == "shopping":
                return True
            raise ShopNavigationFailed("enter_shopping", observed_page=self.last_shop_page)
        return True

    def back_to_main_page(self):
        if not super().back_to_main_page():
            if self.current_shop_page(required="main") == "main":
                return True
            raise ShopNavigationFailed("back_to_main_page", observed_page=self.last_shop_page)
        return True

    def current_shop_page(self, *, required):
        """Confirm a required page in one read window, without further input."""
        def observe(image):
            self.image = image
            page = classify(self.context, image, candidates={"shop", "shop_main"})
            self.last_shop_page = {"shop": "shopping", "shop_main": "main"}.get(page)
            return self.last_shop_page if self.last_shop_page == required else None

        return read_until(self.context, observe, initial_image=self.image)


def run_model_shop(action_handler, context, argv, runtime):
    tracker = runtime.require_tracker()
    data = action_handler._init_data(context, argv.node_name)
    interactor = ModelShopInteractor(context)
    handler = ObservedShopHandler(interactor, data, runtime.collector)
    shopping_page = False
    refresh_before = None

    def stop_for_missing(missing, *, reason="shop_observation_incomplete", **observed):
        if not context.tasker.stopping:
            runtime.stop_model(context, reason, phase="shop", missing=missing,
                observed_coins=data.current_coin if data.current_coin >= 0 else None,
                observed_refresh_remaining=data.refresh_remaining if data.refresh_remaining >= 0 else None,
                observed_enhancement_cost=(data.current_enhancement_cost
                                           if data.current_enhancement_cost >= 0 else None), **observed)
        return False

    def stop_after_failed_action(action, *, navigation=None):
        if context.tasker.stopping:
            return False
        # The failed operation already used its read window. Keep its last
        # observation instead of starting another page or wallet wait.
        page = navigation.observed_page if navigation is not None else interactor.last_shop_page
        if page not in {"shopping", "main"}:
            page = None
        if (action == "RESTOCK" and handler.refresh_attempted is True
                and refresh_before is not None and data.current_coin >= 0
                and refresh_before[0] - data.current_coin == refresh_before[1]):
            tracker.state = replace(tracker.state, restocks=max(0, refresh_before[2] - 1),
                                    coins=data.current_coin)
        missing = [action.lower() + "_confirmation"]
        if action == "RESTOCK":
            missing += data.goods_missing
            if data.current_coin < 0:
                missing.append("current_coin")
            if data.refresh_remaining_observed > 0 and data.refresh_cost <= 0:
                missing.append("refresh_cost")
        return stop_for_missing(missing,
                                reason="shop_action_unconfirmed", observed_shop_page=page)

    handler.read_main_page_info(valid_costs=(0, *sim.PRICES))
    main_missing = (["current_enhancement_cost"] if data.current_enhancement_cost not in (0, *sim.PRICES) else [])
    if not data.shop_type:
        main_missing.append("shop_type")
    if main_missing or context.tasker.stopping:
        return stop_for_missing(main_missing)

    def read_goods():
        controls = {"refresh_exhausted": True} if tracker.state.restocks == 0 else {}
        handler.read_goods_page_info(**controls)
        return ((["current_coin"] if data.current_coin < 0 else []) + data.goods_missing
                + (["refresh_remaining"] if not 0 <= data.refresh_remaining <= 2 else [])
                + (["refresh_cost"] if data.refresh_remaining > 0 and data.refresh_cost <= 0 else []))

    try:
        missing = read_goods()
    except ShopNavigationFailed as error:
        return stop_after_failed_action("OPEN_SHOP", navigation=error)
    shopping_page = True
    if missing or context.tasker.stopping:
        return stop_for_missing(missing)
    try:
        tracker.observe_route("SHOP_UNSEEN", data.current_coin)
    except ValueError as error:
        return stop_for_missing(["route"], reason="shop_route_unresolved", observed_error=str(error))
    while not context.tasker.stopping:
        if not runtime.model_ready(context):
            return stop_for_missing(["model_state"])
        state = tracker.state
        cost = data.current_enhancement_cost
        missing = ((["current_coin"] if data.current_coin < 0 else []) + data.goods_missing
                   + (["current_enhancement_cost"] if cost not in (0, *sim.PRICES) else [])
                   + (["refresh_remaining"] if not 0 <= data.refresh_remaining <= 2 else [])
                   + (["refresh_cost"] if data.refresh_remaining > 0 and data.refresh_cost <= 0 else []))
        if missing:
            return stop_for_missing(missing)
        goods, candidates, slots = {}, [], []
        for slot, item in enumerate(data.items, 1):
            slot = getattr(item, "grid_num", slot)
            goods[slot] = item
            slots.append(dict(slot=slot, internal_name=item.internal_name, quantity=getattr(item, "quantity", 1),
                              price=item.price, bought=item.bought, trekker_specified=item.trekker_specified))
            if not item.bought and 0 < item.price <= data.current_coin:
                candidates.append(Action("BUY", index=slot, price=item.price, key=f"shop_slot_{slot}"))
        if cost <= data.current_coin and any(row[3] < 6 for row in state.H):
            candidates.append(Action("ENHANCE", price=cost))
        if data.refresh_remaining > 0 and 0 < data.refresh_cost <= data.current_coin:
            candidates.append(Action("RESTOCK", price=data.refresh_cost))
        candidates.append(Action("LEAVE"))
        restocks = data.refresh_remaining
        tracker.state = replace(state, coins=data.current_coin,
                                freeEnhance=cost == 0, restocks=restocks,
                                source=None, offer=())
        decision_context = runtime.decision_context("shop", {
            "candidates": sim.to_jsonable(candidates), "visible_items": slots,
            "stock_read_complete": not data.goods_missing,
            "enhancement_cost": cost, "refresh_cost": data.refresh_cost,
            "refresh_remaining": data.refresh_remaining,
            "effective_global_restocks": restocks,
            "refresh_remaining_basis": getattr(data, "refresh_remaining_basis", "ocr"),
        })
        if not tracker.decision_route_ready:
            return stop_for_missing(["confirmed_route_position"], reason="shop_model_inputs_unconfirmed")
        if not tracker.identity_pools.available:
            return stop_for_missing(["complete_pool_catalogue"], reason="shop_model_inputs_unconfirmed")
        actions = runtime.model.prepare_actions(tracker.state, candidates, decision_context)
        tracker.state = decision_context.pop("model_state", tracker.state)
        paid_index = tracker.state.paidIndex
        action = tracker.choose(tuple(actions), decision_context=decision_context)
        if action.kind == "LEAVE":
            if shopping_page:
                interactor.back_to_main_page()
            # The simulator advances directly to END when no route nodes remain.
            next_phase = "END" if tracker.state.node > len(sim.ROUTE) else tracker.state.shopReturnPhase
            tracker.state = replace(tracker.state, phase=next_phase,
                                    source=None, stock=sim.EMPTY_STOCK,
                                    paidIndex=0, firstInitial=False)
            return True
        if action.kind == "RESTOCK":
            refresh_before = (data.current_coin, data.refresh_cost, tracker.state.restocks)
            try:
                refreshed = handler.refresh()
            except ShopNavigationFailed as error:
                return stop_after_failed_action("RESTOCK", navigation=error)
            if not refreshed or context.tasker.stopping:
                return stop_after_failed_action("RESTOCK")
            tracker.state = replace(tracker.state, restocks=max(0, tracker.state.restocks - 1),
                                    firstInitial=False)
            # refresh() retains the complete shelf from its two matching frames.
            shopping_page = True
            continue
        if action.kind == "ENHANCE":
            before_enhance = tracker.state
            staged_enhance = replace(before_enhance, freeEnhance=False,
                                    paidIndex=paid_index if cost == 0 else min(6, paid_index+1),
                                    source=sim.Source("regular", "any", 0, "SHOP", False))
            tracker.state = staged_enhance
            try:
                enhanced = handler._enhance()
            except ShopNavigationFailed as error:
                if tracker.state == staged_enhance:
                    tracker.state = before_enhance
                return stop_after_failed_action("ENHANCE_NAVIGATION", navigation=error)
            if not enhanced:
                if tracker.state == staged_enhance:
                    tracker.state = before_enhance
                return stop_after_failed_action("ENHANCE")
            # Enhancement keeps the shelf unchanged and updates coin/cost on the main page.
            shopping_page = False
            continue
        if not shopping_page:
            try:
                interactor.enter_shopping()
            except ShopNavigationFailed as error:
                return stop_after_failed_action("OPEN_SHOP", navigation=error)
            shopping_page = True
        slot = decision_context.get("action_slots", {}).get(action.key or sim.action_key(action), action.index)
        item = goods[slot]
        source = "specified_drink" if item.trekker_specified else "drink"
        transaction = dict(operation_id=uuid.uuid4().hex, stock_scan_id=runtime.collector.stock_scan_id,
                           item=snapshot(item), source_type=source)
        before_coin = data.current_coin
        runtime.collector.emit("purchase_attempt", phase="before", coins=before_coin,
                               owned=runtime.collector.owned(), **transaction)
        purchase = dict(operation_id=transaction["operation_id"], source_type=source,
                        item=item, committed=False)
        tracker.pending_shop_purchase = purchase
        override = {
            **purchase_override(item),
            # Wait for the reward instead of accepting the shelf behind it.
            "星塔_节点_商店_购物_购买道具_确认购买_agent": {
                "next": ["星塔_节点_进入潜能选择_agent"]},
            "星塔_节点_选择潜能_agent": {"attach": {"reserved_coin": 0, "potential_source": source}},
        }
        with runtime.collector.purchase_wallet_capture(transaction["operation_id"]) as wallet_evidence:
            with runtime.collector.action_scope(**transaction), runtime.collector.capture("purchase", **transaction):
                try:
                    result = context.run_task("星塔_节点_商店_购物_购买道具_agent", override)
                finally:
                    tracker.pending_shop_purchase = None
                if context.tasker.stopping:
                    return False
                pipeline_succeeded = bool(result and result.status.succeeded)
                succeeded = pipeline_succeeded and purchase["committed"]
                if succeeded:
                    interactor.update_image()
                    shopping_page = purchase.get("return_page") != BY_NAME["shop_main"].node
                    if shopping_page:
                        handler.read_goods_page_info(enter_shopping=False,
                            refresh_exhausted=tracker.state.restocks == 0)
                    else:
                        handler.read_main_page_info(read_coin=True)
            fields = purchase_wallet_evidence(transaction["operation_id"], item.price,
                wallet_evidence["wallets"], before_wallet=wallet_evidence["before_wallet"],
                pipeline_wallet=runtime.collector.last_wallet)
            runtime.collector.emit("purchase_result", phase="after", success=succeeded,
                pipeline_succeeded=pipeline_succeeded, owned=runtime.collector.owned(), **fields, **transaction)
            if not succeeded:
                return stop_for_missing(["purchase_confirmation"], reason="shop_action_unconfirmed")
            if purchase.get("return_page") == BY_NAME["shop_main"].node:
                shopping_page = False
    return False
