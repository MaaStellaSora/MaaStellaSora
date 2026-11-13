"""Public observations and per-run state; independent of Maa class registration."""
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from .decision_log import forecast_rows, log_forecast
from .name_matching import clean_name as _clean, identity_key, is_anonymous_non_target, match_name
from .pool_identity import IdentityPools, pool_contract
from .value_model import Action, sim, replace


class PotentialObservationIncomplete(RuntimeError):
    """The current card frame needs another OCR read before any state update."""

    def __init__(self, message, *, reason="potential_observation_incomplete"):
        super().__init__(message)
        self.reason = reason


def potential_levels_valid(old, new, source):
    if old == 0:
        return source != "enhance" and 1 <= new <= 3
    if source == "enhance":
        return 1 <= old < 6 and new in (old + 1, min(6, old + 2))
    return 1 <= old < 6 and new == old + 1


class ObservationTracker:
    def __init__(self, root, preset_data, logger, model, collector=None):
        if "pool_order_evidence" not in preset_data:
            pools, identities, evidence = pool_contract(preset_data["party"], preset_data.get("language", "jp"))
            if tuple(map(len, pools)) == (3, 9, 3, 9, 3, 9):
                preset_data = dict(preset_data, pools=pools, pool_identity_order=identities, pool_order_evidence=evidence)
        self.root, self.logger = Path(root), logger
        self.collector = collector
        self.model = model
        self.preset = self.model.preset(preset_data)
        self.preset_data = preset_data
        self.name_field = "name_zh_cn" if preset_data.get("language") == "cn" else "name_ja_jp"
        self.local_roles = {int(member["role"]): member for member in preset_data["party"]
                            if member.get("local_potential_catalogue")}
        self.names, self.local_names, self.catalogue = defaultdict(list), defaultdict(list), {}
        for i, member in enumerate(preset_data["party"], 1):
            role = int(member.get("role", i))
            for item in member["targets"]:
                item = dict(item, role=role)
                item["quality"] = 1 if item["rarity_class"] == "rare" else 2
                identity = identity_key(item)
                if identity is None:
                    continue
                self.catalogue[identity] = item
                item["identity"] = identity
                self._index_names(item)
        for item in preset_data.get("recognition_catalog", []):
            item = dict(item)
            item["quality"] = 1 if item["rarity_class"] == "rare" else 2
            item["identity"] = identity_key(item)
            if item["identity"] is None:
                continue
            self.catalogue[item["identity"]] = item
            self._index_names(item)
        self.reset()

    def _index_names(self, item):
        for field in ("name_ja_jp", "name_zh_cn", "name_en_us", "name_zh_tw"):
            if item.get(field):
                key = _clean(item[field])
                if item not in self.names[key]:
                    self.names[key].append(item)
                if field == self.name_field and item not in self.local_names[key]:
                    self.local_names[key].append(item)

    def reset(self):
        self.run_id = (getattr(self.collector, "run", None) or {}).get("run_id")
        self.decision_index = 0
        self.last_decision = None
        self.state = self.model.initial(self.preset)
        self.identity_pools = IdentityPools(self.preset_data)
        self.identity_levels = {}
        self.acquired = {}
        self.anonymous_held = Counter()
        self.trekkers = {}
        self.pending_event = None
        self.pending_source = None
        self.source_observation = None
        self.pending_source_observation = None
        self.coins_uncertain = False
        self.route_estimated = False
        self.route_candidates = ((self.state.node, self.state.k, self.state.A,
                                  self.state.cBel, Fraction(1)),)
        self._current_cards = None
        self._page_key = None
        self.selected_page_index = None
        self.event_page_cache = None
        self._event_transaction = None
        self.expected_total = None
        if self.pool_order_ready:
            self.expected_total = self.model.objective_current_value(self.state) + float(max(
                self.model.values(self.state, self.model.actions(self.state), self.preset)))
        self.logger.debug("[神经网络] 开始新一局，模型=%s，预设=%s，普通目标=%d",
                         self.model.model_id, self.preset.name, self.preset.total)

    @property
    def decision_route_ready(self):
        """Whether the current observations identify one route position."""
        positions = {(node, k, a) for node, k, a, _belief, _probability in self.route_candidates}
        observed = getattr(self.collector, "last_observed_floor_context", None)
        anchored = isinstance(observed, dict) and observed.get("floor") == self.state.floor
        return len(positions) == 1 and (not self.route_estimated or anchored)

    def event_location(self):
        floors = sorted({sim.ROUTE[node - 2][1] for node, *_ in self.route_candidates})
        confirmed = self.decision_route_ready
        return dict(floor=self.state.floor if confirmed else None, floor_confirmed=confirmed,
                    floor_candidates=floors)

    @property
    def event_reward_page(self):
        return (self.state.phase == "PAGE" and (self.source_observation or {}).get("basis")
                == "recognized_event_option_and_reward_page")

    @property
    def pool_order_ready(self):
        return self.identity_pools.ready

    def name_candidates(self, name, *, role=None, rarity=None):
        return [item for item in match_name(name, self.names, role=role, rarity=rarity,
                preferred=self.local_names, party_bound=True).items
                if rarity is None or item["rarity_class"] == rarity]

    def identify(self, name, *, role=None, rarity=None):
        items = self.name_candidates(name, role=role, rarity=rarity)
        if len(items) == 1:
            return items[0]
        raise PotentialObservationIncomplete(f"本帧潜能名称归属未确认：{name!r}，等待下一帧识别")

    def initialize_bag_inventory(self):
        """Build the fresh run's H and pools from its complete bag observation."""
        acquired, anonymous = {}, Counter()
        counts = Counter()
        for row in self.preset_data.get("initial_inventory", []):
            if row["rarity_class"] == "core" or row["level"] <= 0:
                continue
            identity = identity_key(row)
            quality = 1 if row["rarity_class"] == "rare" else 2
            if identity is None:
                raise PotentialObservationIncomplete("初始持有潜能身份未确认")
            self.identity_levels[row["role"], identity] = row["level"]
            self.identity_pools.remove(2 * (row["role"] - 1) + quality - 1, identity)
            if is_anonymous_non_target(row):
                anonymous[row["role"], quality, 0, row["level"]] += row.get("count", 1)
            else:
                item = self.catalogue.get(identity)
                if item is None:
                    item = self.identify(row.get("name", ""), role=row["role"], rarity=row["rarity_class"])
                acquired[identity] = (item, row["level"])
        counts.update(anonymous)
        for item, level in acquired.values():
            counts[item["role"], item["quality"], item["target_level"], level] += 1
        if counts and self.preset_data.get("bag_party"):
            # A bag can establish holdings, but cannot recreate earlier draws.
            self.identity_pools.forget_order()
        self.acquired, self.anonymous_held = acquired, anonymous
        self.state = replace(self.state, H=tuple((*key, n) for key, n in sorted(counts.items()) if n),
            pools=self.identity_pools.projection())
        self.expected_total = None
        if self.pool_order_ready:
            self.expected_total = self.model.objective_current_value(self.state) + float(max(
                self.model.values(self.state, self.model.actions(self.state), self.preset)))

    def _event_coin_deltas(self):
        if self.pending_event is None:
            return ((0, Fraction(1)),)
        eid, oid = self.pending_event
        rows = sim.EVENT_DATA[eid][2][oid][1]
        result = defaultdict(Fraction)
        for n, d, delta, _kind, _scope in rows:
            result[delta] += Fraction(n, d)
        return tuple(result.items())

    def _route_observations(self, phase, coins, observed_floor=None, observed_event=None):
        """Match the page, crossing missed boundaries only when needed."""
        observed = getattr(self.collector, "last_observed_floor_context", None)
        floor_minimum = observed.get("floor") if isinstance(observed, dict) else None
        floor_exact = observed_floor.get("floor") if observed_floor else None
        frontier = defaultdict(Fraction)
        for node, k, a, c_belief, route_probability in self.route_candidates:
            for g, f, first, prior in c_belief:
                frontier[node, k, a, g, f, first] += route_probability * prior
        skipped = 0
        cold_event = phase == "EVENT_DRAW" and self.state.floor == 0 and self.coins_uncertain
        event_nodes = None
        if cold_event and floor_minimum is not None:
            # A real result HUD bounds the next NPC to this battle's optional
            # event or the following fixed event, before another battle.
            battle = next((i for i, (kind, floor, _arg) in enumerate(sim.ROUTE)
                           if kind == "BATTLE" and floor == floor_minimum), None)
            if battle is not None:
                stop = next((i for i in range(battle + 1, len(sim.ROUTE))
                             if sim.ROUTE[i][0] == "BATTLE"), len(sim.ROUTE))
                event_nodes = {i + 2 for i in range(battle + 1, stop)
                               if sim.ROUTE[i][0] in {"EVENT", "EXTRA"}}
        cold_positions = defaultdict(lambda: defaultdict(Fraction))
        cold_skipped = None
        while frontier:
            exact = defaultdict(lambda: defaultdict(Fraction))
            matching = defaultdict(lambda: defaultdict(Fraction))
            following = defaultdict(Fraction)
            for (node, k, a, g, f, first), prior in frontier.items():
                for probability, desc, latent in sim._coin_paths(
                        node, k, a, g, f, first, self.state.npc3EventReward and node <= 7):
                    mass = prior * probability
                    # The zero-weight 650 gamble can only be generated before
                    # floor 5. Its observed menu rules out all later events.
                    if observed_event == 105 and desc[1] >= 5:
                        continue
                    wrong_floor = (desc[1] != floor_exact if floor_exact is not None else
                                   floor_minimum is not None and desc[1] < floor_minimum)
                    wrong_floor |= event_nodes is not None and desc[4] == phase and desc[0] not in event_nodes
                    if desc[4] != phase or wrong_floor:
                        if desc[4] != "END":
                            following[(desc[0], desc[2], desc[3], *latent)] += mass
                        continue
                    matching[desc][latent] += mass
                    if cold_event:
                        cold_positions[desc][latent] += mass
                        following[(desc[0], desc[2], desc[3], *latent)] += mass
                    if skipped == 0 and not self.coins_uncertain:
                        for delta, event_probability in self._event_coin_deltas():
                            if self.state.coins + delta + desc[5] == coins:
                                exact[desc][latent] += mass * event_probability
            if exact:
                return exact, None, 0
            if matching:
                if cold_event:
                    if cold_skipped is None:
                        cold_skipped = skipped
                    frontier = following
                    skipped += 1
                    continue
                reason = "route_observation_gap" if skipped else "wallet_delta_unattributed"
                return matching, reason, skipped
            frontier = following
            skipped += 1
        if cold_positions:
            return cold_positions, "route_observation_gap", cold_skipped
        return {}, None, skipped

    def _route_at_observed_floor(self, phase, observed_floor=None):
        """Recover an over-advanced tracker using the latest real floor HUD."""
        observed = observed_floor or getattr(self.collector, "last_observed_floor_context", None)
        if not isinstance(observed, dict) or observed.get("floor") is None:
            return {}, None
        kinds = {"CORE": {"CORE"}, "DRAW": {"DRAW"},
                 "EVENT_DRAW": {"EVENT", "EXTRA"}, "SHOP_UNSEEN": {"SHOP"}}
        positions = [(i + 2, kind, floor, arg) for i, (kind, floor, arg) in enumerate(sim.ROUTE)
                     if kind in kinds.get(phase, ()) and floor == observed["floor"]
                     and i + 2 <= self.state.node]
        if not positions:
            return {}, None
        masses = {(g, f, first): prior for g, f, first, prior in self.state.cBel}
        result = {}
        for node, kind, floor, arg in positions:
            k = 0 if kind == "EXTRA" else self.state.k
            a = (floor == 1) if kind == "EXTRA" and floor in (1, 3) else self.state.A
            desc = (node, floor, k, a, phase, 0, arg if kind == "SHOP" else 0)
            result[desc] = masses
        return result, observed

    def observe_route(self, phase, coins, *, observed_floor=None, observed_event=None):
        """Align an observed page; None keeps the wallet explicitly unknown."""
        if phase == "SHOP_UNSEEN" and self.state.phase == "SHOP":
            if coins is None:
                self.coins_uncertain = True
            else:
                self.state = replace(self.state, coins=int(coins))
                self.coins_uncertain = False
            return
        if not (observed_floor and observed_floor.get("complete") and observed_floor.get("source") == "floor_hud"):
            observed_floor = None
        current = self.state
        candidates, recovery_reason, skipped = self._route_observations(phase, coins, observed_floor, observed_event)
        floor_anchor = None
        if not candidates:
            candidates, floor_anchor = self._route_at_observed_floor(phase, observed_floor)
            if candidates:
                recovery_reason, skipped = "route_observation_gap", None
        if not candidates:
            raise ValueError(f"神经网络已尝试重新定位路线，仍无法对应当前页面：{phase}，route node={current.node}")
        if coins is None and recovery_reason == "wallet_delta_unattributed":
            recovery_reason = "wallet_not_observed"
        interval_record = (self._record_event_wallet_interval(coins, candidates)
                           if not recovery_reason and self.pending_event is not None else None)
        desc, masses = max(candidates.items(), key=lambda item: sum(item[1].values()))
        # An optional floor-1 event and the floor-2 event can have identical
        # wallets. Preserve both positions until later screens distinguish them.
        total_probability = sum(sum(row.values()) for row in candidates.values())
        self.route_candidates = tuple(
            (position[0], position[2], position[3],
             tuple((*latent, p/sum(rows.values())) for latent, p in sorted(rows.items())),
             sum(rows.values())/total_probability)
            for position, rows in sorted(candidates.items())
        )
        positions = {(row[0], row[1], row[2]) for row in self.route_candidates}
        self.route_estimated = recovery_reason is not None or len(positions) != 1
        total = sum(masses.values())
        belief = tuple((*key, p/total) for key, p in sorted(masses.items()))
        node, floor, k, a, _phase, _delta, shop = desc
        kw = dict(node=node, floor=floor, k=k, A=a, phase=phase,
                  coins=current.coins if coins is None else int(coins), cBel=belief,
                  npc3EventReward=current.npc3EventReward and node <= 7,
                  source=sim.Source() if phase == "DRAW" else None, offer=())
        if phase == "SHOP_UNSEEN":
            kw.update(shopOrdinal=shop, firstInitial=shop == 1, paidIndex=0, shopReturnPhase="ADVANCE")
        self.state = replace(current, **kw)
        self.pending_event = None
        # Reaching a new route boundary also ends the preceding event's
        # possible card reward. In particular, a core page can follow the
        # coin-only outcome of a random event before the next normal draw.
        self.pending_source = None
        self.pending_source_observation = None
        self.source_observation = (dict(status="estimated" if recovery_reason else "confirmed",
            complete=not bool(recovery_reason),
            basis="route_recovery_estimate" if recovery_reason else "observed_natural_reward_route")
            if phase == "DRAW" else None)
        if self.source_observation is not None and interval_record is not None:
            self.source_observation["wallet_interval_record_id"] = interval_record
        self.coins_uncertain = coins is None
        if recovery_reason:
            self.route_estimated = True
            if recovery_reason == "route_observation_gap":
                self.logger.debug("Route observation: %(before)s -> %(after)s floor=%(floor)s coins=%(coins)s", {
                    "before": current.node, "after": node, "floor": floor, "coins": coins,
                })
            elif coins is not None:
                self.logger.debug("Wallet synchronized to %(coins)s; previous source unresolved", {"coins": coins})
            if self.collector and self.collector.enabled:
                self.collector.emit("runtime_recovery", reason=recovery_reason,
                    observed_phase=phase, observed_coins=int(coins) if coins is not None else None,
                    wallet_observation_complete=coins is not None,
                    previous_node=current.node, previous_floor=current.floor, previous_coins=current.coins,
                    recovered_node=node, recovered_floor=floor, route_basis="model_route_inference",
                    skipped_observation_count=skipped, continued=True,
                    observed_floor_record_id=floor_anchor.get("record_id") if floor_anchor else None)

    def _record_event_wallet_interval(self, coins, candidates):
        """Record the observed total; component amounts remain rule inferences."""
        if not self.collector or not self.collector.enabled:
            return None
        event = self.collector.event or {}
        before = event.get("wallet_before") or {}
        after = self.collector.last_wallet or {}
        if not (event.get("clicked") and before.get("coin_complete") and after.get("coin_complete")
                and before.get("coin") == self.state.coins and after.get("coin") == coins):
            return None
        paths = set()
        for node, k, a, belief, _route_probability in self.route_candidates:
            for g, f, first, _prior in belief:
                for _probability, desc, _latent in sim._coin_paths(
                        node, k, a, g, f, first, self.state.npc3EventReward and node <= 7):
                    if desc not in candidates:
                        continue
                    battles = tuple((floor, base) for kind, floor, base in sim.ROUTE[node-1:desc[0]-1]
                                    if kind == "BATTLE")
                    for delta, _event_probability in self._event_coin_deltas():
                        if battles and self.state.coins + delta + desc[5] == coins:
                            base = sum(amount for _floor, amount in battles)
                            paths.add((delta, battles, base, desc[5] - base))
        if not paths:
            return None
        return self.collector.emit("wallet_interval_observed", source_type="combined_interval",
            interval_basis="event_then_battle_without_intermediate_wallet_capture",
            complete=True, independently_observed=False, analysis_eligible=False, probability_eligible=False,
            analysis_exclusion_reason="combined_sources_not_separable",
            coins_before=self.state.coins, coins_after=coins, delta=coins-self.state.coins,
            before_wallet_observation_id=before.get("record_id"),
            after_wallet_observation_id=after.get("record_id"),
            event_choice_record_id=event.get("choice_record_id"),
            event_click_record_id=event.get("click_record_id"),
            event_id=event.get("event_id"),
            model_mapping=dict(event=self.pending_event[0], option=self.pending_event[1]),
            component_amounts_basis="rule_constrained_inference_not_individual_UI_observations",
            candidate_components=[dict(event_delta=delta,
                battles=[dict(floor=floor, base_coins=amount) for floor, amount in battles],
                battle_base=base, battle_bonus=bonus) for delta, battles, base, bonus in sorted(paths)])

    def open_page(self, data, *, observed_floor=None):
        source_name = data.params.potential_source
        coins = int(data.current_coin)
        if coins < 0:
            raise PotentialObservationIncomplete("神经网络未读到当前金币")
        normalized = []
        for potential in data.potentials:
            if not _clean(potential.name):
                raise PotentialObservationIncomplete("本帧潜能名称为空")
            old, new = int(potential.old_level), int(potential.new_level)
            if not data.core_potential:
                if not potential_levels_valid(old, new, source_name):
                    raise PotentialObservationIncomplete(
                        f"潜能等级与来源规则不一致：{potential.name} {old}→{new}")
            normalized.append(self.identify(
                potential.name, role=potential.trekker.index + 1, rarity=potential.rarity))
        identities = [(item["role"], item["identity"]) for item in normalized]
        if len(set(identities)) != len(identities):
            raise PotentialObservationIncomplete("同页候选包含重复潜能身份")
        if not data.core_potential:
            if source_name != "enhance" and len({item["role"] for item in normalized}) > 1:
                raise PotentialObservationIncomplete("普通潜能页出现多个角色的候选")
            if source_name == "enhance" and len({int(p.new_level) - int(p.old_level)
                    for p in data.potentials if int(p.old_level) < 5}) > 1:
                raise PotentialObservationIncomplete("强化页候选的共同升级增量冲突")
        page_key = (source_name, bool(data.core_potential), coins, tuple(
            (item["identity"], int(p.old_level), int(p.new_level))
            for item, p in zip(normalized, data.potentials)))
        if page_key == self._page_key:
            return
        if data.core_potential:
            self.event_page_cache = None
            self.observe_route("CORE", coins, observed_floor=observed_floor)
            self.state = replace(self.state, phase="ADVANCE")
            self._current_cards = normalized
            self._page_key, self.selected_page_index = page_key, None
            return
        # Reject a bad frame before consuming its route boundary or event link.
        pools = self.identity_pools.copy()
        groups = defaultdict(list)
        for potential, item in zip(data.potentials, normalized):
            held = self.identity_levels.get((item["role"], item["identity"]), 0)
            if held != int(potential.old_level):
                raise PotentialObservationIncomplete("候选新旧或等级与该潜能的已确认持有不一致")
            if potential.old_level == 0:
                groups[2 * (item["role"] - 1) + item["quality"] - 1].append(item["identity"])
        if not pools.available:
            raise PotentialObservationIncomplete("完整潜能身份目录未确认")
        try:
            pools.observe(groups)
        except ValueError as error:
            raise PotentialObservationIncomplete(str(error)) from error
        page_already_open = self.state.phase in ("PAGE", "ENHANCE_PAGE")
        self.event_page_cache = None
        shop_page = source_name in ("drink", "specified_drink", "enhance")
        event_page = self.state.phase == "EVENT" or self.pending_source is not None
        if data.level_upped and not shop_page:
            event_page = False
            self.pending_source = None
            self.pending_source_observation = None
        elif event_page and self.pending_event is not None:
            eid, oid = self.pending_event
            outcomes = [row for row in sim.EVENT_DATA[eid][2][oid][1] if row[0]]
            # A guaranteed card reward is still an event reward when the
            # preceding wallet OCR was wrong. Do not consume a natural draw.
            event_page = (all(kind != "none" for _n, _d, _delta, kind, _scope in outcomes)
                          or any(kind != "none" and self.state.coins + delta == coins
                                 for _n, _d, delta, kind, _scope in outcomes))
            if not event_page:
                self.pending_source = None
                self.pending_source_observation = None
        if data.refresh_count == 0 and not shop_page and not event_page and not page_already_open:
            self.observe_route("DRAW", coins, observed_floor=observed_floor)
        else:
            self.state = replace(self.state, coins=coins)
            self.coins_uncertain = False
        if source_name == "enhance":
            source = sim.Source("regular", "any", 0, "SHOP", False)
            source_observation = dict(status="confirmed", complete=True, basis="observed_shop_reward")
            phase = "ENHANCE_PAGE"
        elif shop_page:
            source = sim.Source("regular", "main" if source_name == "specified_drink" else "any", 0, "SHOP")
            source_observation = dict(status="confirmed", complete=True, basis="observed_shop_reward")
            phase = "PAGE"
        else:
            source = self.pending_source or self.state.source or sim.Source()
            source_observation = self.pending_source_observation or self.source_observation
            if event_page and self.pending_event is not None:
                source_observation = dict(status="confirmed", complete=True,
                    basis="recognized_event_option_and_reward_page",
                    event_id=self.pending_event[0], option_id=self.pending_event[1])
            if source_observation is None:
                source_observation = dict(status="estimated", complete=False, basis="reward_source_unconfirmed")
            phase = "PAGE"
        # Only an observed reward page confirms a probabilistic event reward.
        npc3_event_reward = self.state.npc3EventReward or (
            event_page and not shop_page and 3 <= self.state.node <= 6)
        cards, indices = [], Counter()
        for potential, item in zip(data.potentials, normalized):
            role, quality, target = item["role"], item["quality"], item["target_level"]
            old, new = int(potential.old_level), int(potential.new_level)
            pool_index = 0
            if old == 0:
                pi = 2*(role-1)+quality-1
                offset = indices[pi]
                indices[pi] += 1
                pool_index = offset+1
            cards.append((int(old == 0), role, quality, target, old, new, pool_index))
        if source.roleMode == "bound" and cards:
            source = replace(source, boundRole=cards[0][1])
        p, q = self.state.P, self.state.Q
        for is_new, _r, _quality, _target, _old, level, _index in cards:
            if is_new:
                if level == 1:
                    p = True
                else:
                    p, q = False, level == 2 and self.state.h < 10
        if self.state.h >= 10:
            q = False
        belief = self.state.eBel
        if phase == "ENHANCE_PAGE":
            observable = [card for card in cards if card[4] < 5]
            bonus = observable[0][5] - observable[0][4] == 2 if observable else None
            belief = sim._enhance_posterior(belief, bonus)
        purchase = getattr(self, "pending_shop_purchase", None)
        if source_name in ("drink", "specified_drink") and purchase is not None:
            if purchase["source_type"] != source_name:
                raise PotentialObservationIncomplete("饮品奖励来源与本次购买不一致")
            from .potential_adapter import confirm_drink_purchase
            self.state = confirm_drink_purchase(self.state, purchase)
            source_observation["purchase_operation_id"] = purchase["operation_id"]
        self.state = replace(self.state, phase=phase, source=source, offer=tuple(cards),
                             pools=pools.projection(), P=p, Q=q, eBel=belief,
                             eventId=0, eventOptions=(), npc3EventReward=npc3_event_reward)
        self._current_cards = normalized
        self.identity_pools = pools
        self._page_key, self.selected_page_index = page_key, None
        self.pending_source = None
        self.source_observation = source_observation
        self.pending_source_observation = None
        self.pending_event = None

    def choose(self, actions, labels=None, *, decision_context=None):
        # Snapshot exactly the public input used for this returned action.
        state = self.state
        event = state.phase == "EVENT" and (decision_context or {}).get("kind") == "event"
        event_related = event or self.event_reward_page
        if not event_related and not self.decision_route_ready:
            raise PotentialObservationIncomplete("当前路线位置未确认，无法进行模型决策",
                reason="route_position_unconfirmed")
        if not self.identity_pools.available:
            raise PotentialObservationIncomplete("完整潜能身份目录未确认", reason="pool_catalogue_incomplete")
        actions = tuple(actions)
        inputs = {"decision_context": decision_context} if decision_context is not None else {}
        if event_related and len(self.route_candidates) > 1:
            # For a reward hand the early-NPC flag belongs to the actual route,
            # not the representative node used to store the other card fields.
            reward_flag = self._event_transaction["base"]["state"].npc3EventReward if event else state.npc3EventReward
            inputs["routes"] = tuple((replace(state, floor=sim.ROUTE[node - 2][1], node=node, k=k, A=a,
                cBel=belief, npc3EventReward=node <= 7 and (reward_flag or (not event and 3 <= node <= 6))),
                probability) for node, k, a, belief, probability in self.route_candidates)
        decision = self.model.decide(state, actions, self.preset, **inputs)
        action = Action(**decision["action_record"])
        self.logger.debug("[神经网络状态] phase=%s floor=%d coins=%d score=%d H=%s", state.phase, state.floor, state.coins, decision["ordinary_score"], state.H)
        for candidate in decision["values"]:
            self.logger.debug("[神经网络候选] %s | Q=%.6f", candidate["action"],
                              candidate.get("predicted_value", candidate.get("expected_remaining_ordinary_return")))
        previous_total = self.expected_total
        selected_total = decision["expected_total"]
        rows = forecast_rows(decision, actions, decision["ordinary_score"],
                             selected_total if previous_total is None else previous_total, labels)
        log_forecast(self.logger, selected_total, rows)
        self.expected_total = selected_total
        record = dict(decision, time=datetime.now().astimezone().isoformat(), preset=self.preset.name,
                      phase=self.state.phase, floor=self.state.floor, node=self.state.node,
                      coins=self.state.coins, wallet_route_sync_estimated=self.route_estimated)
        if event_related:
            record.update(self.event_location())
            if not record["floor_confirmed"]:
                record["node"] = None
                record["public_state_basis"] = "representative_route_for_event_evaluation"
        self.decision_index += 1
        record.update(
            log_schema_version="maa_public_decision_v3", run_id=self.run_id,
            previous_expected_total=previous_total, expected_total=selected_total,
            option_forecasts=rows, change_baseline=("first_model_forecast" if previous_total is None else
                                                  "preceding_decision_forecast_or_initial_state"),
            decision_index=self.decision_index, capture_timing="before_return_before_action_execution",
            public_state=sim.to_jsonable(state),
            original_preset=sim.to_jsonable(self.preset_data),
            original_target_pools=sim.to_jsonable(self.preset.pools),
            original_target_vector=list(self.preset.flat),
            preset_fingerprint=self.preset.fingerprint,
            pool_identity_observation=self.identity_pools.snapshot(),
            evaluation_pool_order_basis=("unique_target_projection" if self.pool_order_ready
                                         else "estimated_from_preset_catalogue"),
            legal_actions=sim.to_jsonable(actions),
            tracker_public_observation={
                "route_candidates": sim.to_jsonable(self.route_candidates),
                "coins_uncertain": self.coins_uncertain,
                "route_estimated": self.route_estimated,
                "pending_event": sim.to_jsonable(self.pending_event),
                "pending_source": sim.to_jsonable(self.pending_source),
                "source_observation": self.source_observation,
                "pending_source_observation": self.pending_source_observation,
            },
            versions=self.model.versions,
        )
        if self.collector and self.collector.enabled:
            self.collector.model_decision(record)
        self.last_decision = record
        return action

    def selected(self, potential):
        item = self.identify(potential.name, role=potential.trekker.index + 1, rarity=potential.rarity)
        index = next(i for i, row in enumerate(self._current_cards, 1)
                     if (row["role"], row["identity"]) == (item["role"], item["identity"]))
        if self.selected_page_index == index:
            return
        if item["rarity_class"] == "core":
            self.selected_page_index = index
            return
        self.state = sim._select(self.state, index)
        if potential.old_level == 0:
            self.identity_pools.remove(2 * (item["role"] - 1) + item["quality"] - 1, item["identity"])
            self.state = replace(self.state, pools=self.identity_pools.projection())
        self.identity_levels[item["role"], item["identity"]] = int(potential.new_level)
        self.selected_page_index = index
        bucket = (item["role"], item["quality"], item["target_level"], int(potential.old_level))
        if is_anonymous_non_target(item):
            if potential.old_level > 0:
                self.anonymous_held[bucket] -= 1
            self.anonymous_held[item["role"], item["quality"], 0, int(potential.new_level)] += 1
            return
        if item["identity"] not in self.acquired and potential.old_level > 0 and self.anonymous_held[bucket]:
            self.anonymous_held[bucket] -= 1
        self.acquired[item["identity"]] = (item, int(potential.new_level))

    def _event_snapshot(self):
        fields = ("state", "route_candidates", "pending_event", "pending_source", "source_observation",
                  "pending_source_observation", "coins_uncertain", "route_estimated", "expected_total", "last_decision")
        return {field: getattr(self, field) for field in fields}

    def _restore_event_snapshot(self, snapshot):
        for field, value in snapshot.items():
            setattr(self, field, value)

    @contextmanager
    def event_evaluation(self, coins, *, event_id=None):
        """Evaluate this menu from one route boundary without committing its action."""
        transaction = getattr(self, "_event_transaction", None)
        if transaction is None:
            base = self._event_snapshot()
            try:
                self.observe_route("EVENT_DRAW", coins, observed_event=event_id)
                evaluation = self._event_snapshot()
            finally:
                self._restore_event_snapshot(base)
            transaction = self._event_transaction = dict(base=base, evaluation=evaluation, selected=None)
        self._restore_event_snapshot(transaction["evaluation"])
        if coins is None:
            self.coins_uncertain = True
            self.route_estimated = True
        else:
            if self.state.coins != coins:
                self.state = replace(self.state, coins=int(coins))
                self.route_estimated = True
            self.coins_uncertain = False
        transaction["selected"] = None
        self.event_page_cache = None
        try:
            yield
        finally:
            self._restore_event_snapshot(transaction["base"])

    def stage_event_result(self):
        """Keep only the latest returned choice until the menu click succeeds."""
        self._event_transaction["selected"] = self._event_snapshot()

    def event_selected(self, eid, oid):
        """Build the selected afterstate inside event_evaluation."""
        self.pending_event = (eid, oid)
        self.pending_source = None
        self.pending_source_observation = None
        outcomes = sim.EVENT_DATA[eid][2][oid][1]
        kinds = {(kind, scope) for n, _d, _delta, kind, scope in outcomes if n and kind != "none"}
        if len(kinds) == 1:
            kind, scope = next(iter(kinds))
            self.pending_source = sim.Source(kind, "main" if scope == "main" else "bound" if scope == "support_random" else "any",
                                             2 if scope == "support_random" else 0)
            self.pending_source_observation = dict(status="pending", complete=False,
                basis="recognized_event_option", event_id=eid, option_id=oid, action_confirmed=False)
        self.state = replace(self.state, phase="ADVANCE", eventId=0, eventOptions=(),
                             seen=tuple(sorted({*self.state.seen, eid})), source=None, offer=())
        self.source_observation = None

    def event_action_confirmed(self):
        """Commit the latest staged result once the selected menu visibly closes."""
        transaction = getattr(self, "_event_transaction", None)
        if transaction is not None and transaction["selected"] is not None:
            self._restore_event_snapshot(transaction["selected"])
        self._event_transaction = None
        self.event_page_cache = None
        if self.pending_source_observation is not None:
            self.pending_source_observation = dict(self.pending_source_observation, action_confirmed=True)
