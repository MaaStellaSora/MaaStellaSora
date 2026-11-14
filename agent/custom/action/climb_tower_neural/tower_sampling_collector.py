"""Local adapters around official readers; all run state belongs to the collector."""
import uuid

from .shop_ui.handler import ShopHandler
from .sampling_data import SamplingCollector, snapshot
from .observation import PotentialObservationIncomplete


class SamplingContext:
    """A view passed only to our callbacks; the actual Maa Context is untouched."""
    __slots__ = ("_context", "_collector", "_language")

    def __init__(self, context, collector):
        self._context = context
        self._collector = collector
        node = context.get_node_data("星塔_节点_商店_商店主流程_agent") or {}
        self._language = node.get("attach", {}).get("lang", "unknown")

    def __getattr__(self, name):
        return getattr(self._context, name)

    def run_recognition(self, entry, *args, **kwargs):
        result = self._context.run_recognition(entry, *args, **kwargs)
        if isinstance(entry, str) and "星塔" in entry:
            self._collector.record_recognition(entry, args, kwargs, result, self._language)
        return result

    def run_task(self, entry, *args, **kwargs):
        observed = isinstance(entry, str) and "星塔" in entry
        common = dict(node_name=entry, lang_type=self._language, method="run_task",
                      arguments=snapshot(args), keyword_arguments=snapshot(kwargs))
        read = self._collector.current_read
        if read:
            common.update(read_id=read.read_id, **read.links)
        if observed:
            self._collector.emit("context_run_task", phase="before", **common)
        result = self._context.run_task(entry, *args, **kwargs)
        if observed:
            self._collector.emit("context_run_task", phase="after", raw_result=snapshot(result), **common)
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
        page = self.collector.prepare_offer(self.handler)
        pending = self.notes.ledger.pending if self.notes is not None and self.notes.ledger is not None else None
        source = self.collector.reward.get("source_fields", {})
        if (pending and pending["source_type"] == "event" and source.get("source") == "event"
                and source.get("source_event_id") == pending["operation_id"]):
            restored, _ = self.notes.complete_potential_reward_inventory(self.handler.screen)
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
        self.offer_id = self.collector.observe_offer(self.handler, self.page_read)
        return result

    def choose_potential(self):
        potential = self.handler.choose_potential()
        if self.collector.enabled and self.collector.run_confirmed:
            candidate_index = potential.index if potential is not None else None
            self.collector.emit("policy_decision", offer_id=self.offer_id,
                reward_chain_id=self.collector.reward["reward_chain_id"],
                selected_candidate_index=candidate_index,
                selected=(self.collector.observed_offer_candidate(self.offer_id, candidate_index)
                          if potential is not None else None),
                refresh_requested=potential is None, operation_confirmed=False,
                decision_method=self.handler.decision_method,
                policy_mode=self.collector.policy_mode)
        return potential

    def pick(self, potential):
        self.selection_id = uuid.uuid4().hex
        return self.collector.operation(
            "selection_result", self.handler, self.handler.pick, potential,
            before=True, offer_id=self.offer_id, operation_id=self.selection_id,
        )

    def cache_potential_data(self, potential):
        result = self.handler.cache_potential_data(potential)
        if self.collector.enabled and self.collector.run_confirmed:
            self.collector.cache_selection(potential, self.offer_id, self.selection_id)
        return result

    def refresh(self):
        return self.collector.operation(
            "refresh_result", self.handler, self.handler.refresh, before=True, offer_id=self.offer_id
        )


class ObservedShopHandler(ShopHandler):
    def __init__(self, interactor, data, collector):
        super().__init__(interactor, data)
        self.collector = collector
        self.shop_visit_id = uuid.uuid4().hex
        self.stock_generations = {}
        self.stock_generation_id = None
        self.pending_refresh = None

    def read_goods_page_info(self, *, enter_shopping=True, refresh_exhausted=False):
        if not self.collector.ensure_run("shop_page"):
            return super().read_goods_page_info(enter_shopping=enter_shopping, refresh_exhausted=refresh_exhausted)
        previous = snapshot(self.data)
        self.collector.stock_scan_id = uuid.uuid4().hex
        with self.collector.capture("shop_stock", stock_scan_id=self.collector.stock_scan_id,
                                    shop_visit_id=self.shop_visit_id) as read:
            result = super().read_goods_page_info(enter_shopping=enter_shopping, refresh_exhausted=refresh_exhausted)
        self.collector.observe_shop(self, read, previous)
        self.collector.complete_shop_refresh(self, read)
        return result

    def refresh(self):
        if not self.collector.ensure_run("shop_refresh_result"):
            return super().refresh()
        previous = snapshot(self.data)
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

        result = self.collector.operation("shop_refresh_result", self,
            lambda: super(ObservedShopHandler, self).refresh(frame_capture=capture_frame), before=True)
        if self.pending_refresh is not None:
            self.pending_refresh.update(
                refresh_remaining_after_action_basis=self.data.refresh_remaining_basis,
                refresh_remaining_after_action_observed=self.data.refresh_remaining_observed)
        if result and not self.interactor.context.tasker.stopping:
            self.collector.stock_scan_id = uuid.uuid4().hex
            self.collector.observe_shop(self, stock_read, previous)
            self.collector.complete_shop_refresh(self, stock_read)
        return result

    def _enhance(self):
        return self.collector.operation("enhancement_result", self, super()._enhance, before=True)
