"""Observed run facts shared by callbacks, independent of research recording."""
from contextlib import contextmanager
from copy import deepcopy
import threading
import uuid


LINK_FIELDS = {"operation_id", "origin_operation_id", "offer_id", "reward_chain_id",
               "stock_scan_id", "shop_visit_id", "event_id", "source_type"}


class RunSession:
    def __init__(self):
        self.lock = threading.RLock()
        self.reset()

    def reset(self):
        with self.lock:
            self.run_confirmed = False
            self.current_floor_observation = None
            self.last_confirmed_floor = None
            self.current_wallet_observation = None
            self._operations = []
            self._purchase_wallets = {}

    def begin_run(self):
        """A real tower screen has confirmed the pending start click."""
        self.reset()
        self.run_confirmed = True

    def observe_floor(self, page, observation):
        """Keep this read, including unknowns, separately from the last clear HUD.

        A hidden plaque is not a new exact floor observation. Its previous
        confirmed value can still bound the route after a reward page closes.
        Recording IDs and write success do not determine either fact.
        """
        current = dict(deepcopy(observation), page=page)
        self.current_floor_observation = current
        if current.get("complete") and current.get("source") == "floor_hud" and current.get("floor") is not None:
            self.last_confirmed_floor = current
        return current

    @property
    def current_action_links(self):
        with self.lock:
            links = dict(self._operations[-1]) if self._operations else {}
            links.setdefault("operation_id", None)
            return links

    @contextmanager
    def action_scope(self, **links):
        # Nested Maa run_task callbacks can execute on another Python thread.
        links = {key: value for key, value in links.items() if key in LINK_FIELDS}
        with self.lock:
            operations = self._operations
            operations.append(links)
        try:
            yield
        finally:
            with self.lock:
                operations.remove(links)

    @contextmanager
    def purchase_wallet_capture(self, operation_id):
        with self.lock:
            evidence = dict(before_wallet=deepcopy(self.current_wallet_observation), wallets=[])
            captures = self._purchase_wallets
            captures[operation_id] = evidence
        try:
            yield evidence
        finally:
            with self.lock:
                captures.pop(operation_id)

    def observe_wallet(self, reading):
        """Accept one completed UI reading, including missing or conflicting amounts."""
        with self.lock:
            current = deepcopy(reading)
            current.update(self.current_action_links)
            current["observation_id"] = uuid.uuid4().hex
            self.current_wallet_observation = current
            purchase = self._purchase_wallets.get(current["operation_id"])
            if purchase is not None:
                purchase["wallets"].append(current)
            return current
