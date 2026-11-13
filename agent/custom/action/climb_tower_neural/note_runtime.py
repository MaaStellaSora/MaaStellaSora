"""Connect natural tower screens to the note inventory ledger."""
from copy import deepcopy
import re
import uuid

import cv2

from .interactor import Interactor

from .note_floor import read_floor_observation
from .note_bag import OPEN_SKILLS, SKILLS_READY, _retain_conflicts, _run, read_full_inventory
from .note_ledger import NoteLedger
from .note_ocr import TOWER_ELEMENTS, normalize_image, read_inventory, text_rules
from .note_setup_reader import load_rules, read_bag_skill_page, read_setup, read_settlement
from .critic.vendor.stella_sim import ROUTE
from .ui_flow import read_until


class NoteRuntime:
    def __init__(self, runtime):
        self.runtime = runtime
        self.collector = runtime.collector
        self.ledger = None
        self.setup = None
        self.floor = None
        self.mode = "field"
        self.snapshot_index = 0
        self.last_floor_observation = None
        self.bag_inventory = None
        self.bag_inventory_revision = None

    @property
    def observing(self):
        return self.collector.enabled and self.collector.run_confirmed

    def _record(self, event, **fields):
        if self.collector.enabled:
            return self.collector.emit(event, **fields)

    def _run_id(self):
        run = getattr(self.collector, "run", None) or {}
        return run.get("run_id") or getattr(getattr(self.runtime, "tracker", None), "run_id", None)

    def reset(self):
        self.finish("task_restart")
        self.ledger, self.setup, self.floor = None, None, None
        self.mode, self.snapshot_index = "field", 0
        self.last_floor_observation = None
        self.bag_inventory = None
        self.bag_inventory_revision = None

    def current_inventory(self):
        if self.ledger is None:
            return None
        if self.bag_inventory is not None and self.bag_inventory_revision == self.ledger.revision:
            return deepcopy(self.bag_inventory)
        return self.ledger.current_observation

    def _emit(self, rows):
        if not rows or not self.collector.enabled:
            return
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            row = dict(row)
            event = row.pop("event_type")
            if event in {"note_delta", "note_inventory_observed"}:
                row["probability_eligible"] = (bool(row.get("complete", False)) and row.get("floor") is not None
                    and row.get("floor_complete", True) and row.get("independently_observed", True)
                    and row.get("analysis_eligible", True))
            self.collector.emit(event, **row)
            if event == "note_delta":
                self.runtime.logger.debug("Note delta: source=%(source)s floor=%(floor)s total=%(total)s status=%(status)s", {
                    "source": row["source_type"], "floor": row.get("floor", "?"),
                    "total": row.get("total"), "status": row.get("status"),
                })

    def _tower(self, context):
        node = context.get_node_data("星塔_属性塔选择_agent") or {}
        elements = tuple(node.get("attach", {}).get("active_melodies", ()))
        return next(name for name, pair in TOWER_ELEMENTS.items() if set(pair) == set(elements))

    def _image(self, image):
        return image if image.shape[:2] == (720, 1280) else normalize_image(image)

    def _save_image(self, image, label):
        if not self.collector.enabled or not self.collector.config.get("capture_note_screenshots", False):
            return None
        self.snapshot_index += 1
        run_id = self.ledger.run_id
        folder = self.collector.directory / "screenshots" / str(run_id)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{self.snapshot_index:04d}_{label}.png"
            cv2.imencode(".png", image)[1].tofile(target)
        except OSError as error:
            self.collector.logger.error("$agent.custom.action.climb_tower_neural.sampling_record_failed", {"error": str(error)})
            return None
        return target.relative_to(self.runtime.app_dir).as_posix()

    def read_start_setup(self, context, image, node, box):
        """Read supply while the recognized start button still holds this page."""
        image = self._image(image)
        observed, first = None, True
        current_box = list(box)

        def observe(frame):
            nonlocal image, observed, first, current_box
            if not first:
                button = context.run_recognition(node, frame)
                current_box = list(button.best_result.box) if button and button.hit else None
                if current_box is None:
                    return None
            first = False
            image = self._image(frame)
            current = read_setup(context, image, read_discs=False)
            if observed and observed["conflicts"]:
                supply = _retain_conflicts(
                    dict(counts=observed["starting_supply"], readings=observed["readings"],
                         conflicts=observed["conflicts"]),
                    dict(counts=current["starting_supply"], readings=current["readings"],
                         conflicts=current["conflicts"], complete=current["supply_complete"]))
                current.update(starting_supply=supply["counts"], readings=supply["readings"],
                               conflicts=supply["conflicts"], supply_complete=False, complete=False)
                current["raw"] = dict(current.get("raw", {}), starting_supply=current["readings"])
            observed = current
            return observed if observed["supply_complete"] else None

        read_until(context, observe, initial_image=image)
        return image, observed, current_box

    def start(self, context, image, *, setup):
        if not self.observing or setup is None:
            return None
        self.ledger = NoteLedger()
        self.mode, self.floor, self.snapshot_index = "field", 1, 0
        image = self._image(image)
        self.bag_inventory = None
        self.bag_inventory_revision = None
        self.setup = setup
        supply = dict(counts=self.setup["starting_supply"], readings=self.setup["readings"],
                      conflicts=self.setup["conflicts"], complete=self.setup["supply_complete"])
        self._emit(self.ledger.begin_run(supply, self._tower(context),
            run_id=self._run_id(),
            main_discs=self.setup["main_discs"], setup_complete=self.setup["complete"]))
        self._record("note_setup_observed", **self.setup,
                            image_path=self._save_image(image, "setup"),
                            probability_eligible=self.setup["complete"])
        self._emit(self.ledger.begin_battle("battle_1", floor=1,
            floor_complete=True, floor_source="new_run_start"))
        self.runtime.logger.debug("Initial note supply: %(starting_supply)s", {
            "starting_supply": self.setup["starting_supply"],
        })
        return self.setup

    def read_bag_setup(self, context):
        """Read an already open bag; leave closing it to the caller.

        The first core offer precedes the first note-result animation. This
        snapshot may inform the model, but must not replace the frozen supply
        baseline or advance the pending battle's accounting revision.
        """
        if not self.observing or self.ledger is None:
            return None
        if not _run(context, OPEN_SKILLS):
            raise ValueError("背包秘纹技能页未打开")
        image = self._image(Interactor.capture(context))
        observed = bar_inventory = None
        first = True

        def observe(frame):
            nonlocal image, observed, bar_inventory, first
            if not first:
                ready = context.run_recognition(SKILLS_READY, frame)
                if not ready or not ready.hit:
                    return None
            first = False
            image = self._image(frame)
            observed = read_bag_skill_page(context, image)
            bar_inventory = _retain_conflicts(bar_inventory,
                read_inventory(context, image, "bag", tower=self._tower(context)))
            if observed["complete"] and bar_inventory["complete"]:
                return True

        read_until(context, observe, initial_image=image)
        if context.tasker.stopping or observed is None:
            return None
        requirements = observed.pop("requirement_readings")
        # The first battle adds tower drops only. A confirmed off-tower supply
        # remains valid; it is not an extra row that must be found in the bag.
        scope = (*self.ledger.visible_melodies, *self.ledger.unresolved_spend_extras())
        inventory, restored = read_full_inventory(context, scope,
            bag_open=True, close_bag=False, requirement_readings=requirements,
            bar_observation=bar_inventory, reread_skill_page=False)
        if not restored:
            raise ValueError("背包音符明细未关闭")
        observed["inventory"] = inventory
        observed["inventory_complete"] = bool(inventory and inventory["complete"])
        if inventory is not None:
            self.ledger.cache_spend_extras(inventory, page=self._bag_page(inventory))
        self.bag_inventory = inventory
        self.bag_inventory_revision = self.ledger.revision
        self.setup = dict(self.setup or {}, **{key: value for key, value in observed.items()
                                              if key not in {"inventory", "complete"}})
        self.setup["bag_setup_complete"] = observed["complete"]
        self.setup["complete"] = observed["complete"] and self.setup.get("supply_complete", True)
        self._record("note_bag_setup_observed", **observed,
                     image_path=self._save_image(image, "bag_setup"))
        return observed

    def _expected_floors(self, page):
        """Use the shared tower route as a prior, never as an OCR replacement."""
        expected = []
        inferred = getattr(self.collector, "inferred_state", lambda: {})() or {}
        hint = inferred.get("floor")
        if hint and 1 <= hint <= 20 and (self.floor is None or hint >= self.floor):
            expected.append(hint)
        pending = self.ledger.pending if self.ledger else None
        if page in {"field", "result"} and pending and pending["source_type"] in {"event", "combined_interval"}:
            expected.extend(pending["metadata"].get("event_floor_candidates", ()))
        if self.floor is not None:
            has_result = pending and self.ledger.revision > pending["start_revision"]
            next_battle = next((floor for kind, floor, _ in ROUTE
                                if kind == "BATTLE" and floor > self.floor), None)
            if page == "potential" and self.mode == "field" and has_result and next_battle:
                expected.append(next_battle)
            expected.append(self.floor)
            if page == "event":
                next_event = next((floor for kind, floor, _ in ROUTE
                                   if kind == "EVENT" and floor > self.floor), None)
                if next_event:
                    expected.append(next_event)
            elif self.mode != "shop" and next_battle:
                expected.append(next_battle)
        return tuple(dict.fromkeys(expected))

    def _floor_observation(self, context, image, page, provided=None):
        expected = self._expected_floors(page)
        if provided is not None:
            observation = dict(floor=provided, complete=True, source="provided_context", raw=[])
        elif page == "potential":
            # Potential selection blurs the scene, including the floor plaque.
            # Read it from the clear reward screen before attributing its stock.
            observation = dict(floor=None, observed_floor=None, observed_floors=[],
                complete=False, source="floor_hud", reason="floor_hud_obscured",
                alternative="result_floor_hud", raw=[], conflicts=[])
        elif page in {"event", "settlement"}:
            # These full-screen pages replace the floor plaque. The same
            # coordinates can contain a concerto requirement such as 15/15.
            # Read the floor when the operation's result exposes it again.
            observation = dict(floor=None, observed_floor=None, observed_floors=[],
                complete=False, source="floor_hud", reason="floor_hud_not_displayed",
                raw=[], conflicts=[])
        else:
            observation = read_floor_observation(context, image, expected_floors=expected)
        observation.setdefault("expected_floors", list(expected))
        observation.setdefault("sequence_consistent",
            observation["floor"] in expected if observation["floor"] is not None and expected else None)
        return observation

    def _read_floor(self, context, image, page, provided=None, *, observation=None):
        if observation is None:
            observation = self._floor_observation(context, image, page, provided)
        floor_id = self._record("note_floor_observed", page=page, **observation)
        if floor_id:
            observation = dict(observation, floor_observation_id=floor_id)
        self.last_floor_observation = observation
        return observation

    def _battle(self, floor, observation=None):
        pending = self.ledger.pending
        metadata = dict(floor=floor, floor_complete=floor is not None,
                        floor_source=(observation or {}).get("source", "floor_hud"),
                        floor_observation=observation)
        if pending and pending["source_type"] in {"event", "combined_interval"}:
            self._emit(self.ledger.combine_event_battle(
                f"battle_{floor}" if floor is not None else f"battle_stable_{uuid.uuid4().hex}",
                **metadata, source_basis="natural_reward_page_after_event",
                source_observation_id=(observation or {}).get("floor_observation_id")))
            self.floor = floor
            self.mode = "combined"
            return
        if pending and pending["source_type"] == "battle" and pending["metadata"].get("floor") is None:
            if floor is not None:
                self.ledger.update_pending_metadata(**metadata)
                self._record("note_source_floor_confirmed",
                    operation_id=pending["operation_id"], source_type="battle", **metadata)
        elif self.floor != floor or pending is None:
            operation_id = f"battle_{floor}" if floor is not None else f"battle_unknown_{uuid.uuid4().hex}"
            self._emit(self.ledger.begin_battle(operation_id, **metadata))
            self.mode = "field"
        if floor is not None:
            self.floor = floor

    def capture(self, context, image, layout="result", floor=None, observation=None, *, continue_box=None):
        """Read inventory; a result-page reread replaces the caller's click box."""
        if not self.observing or self.ledger is None:
            return None, True, False
        image = self._image(image)
        supplied = observation is not None
        observation = observation or read_inventory(context, image, layout)
        floor_observation = (self._floor_observation(context, image, layout, floor)
                             if observation.get("bar_present") else None)
        anchors = {"result": "星塔_点击空白处关闭_agent", "event": "星塔_节点_进入对话选择_agent"}
        needs_floor = layout == "result" and floor is None
        if (observation.get("bar_present") and layout in anchors
                and ((not supplied and not observation["complete"])
                     or (needs_floor and not floor_observation["complete"]))):
            if continue_box is not None:
                continue_box.clear()

            def observe(frame):
                nonlocal image, observation, floor_observation
                anchor = context.run_recognition(anchors[layout], frame)
                if continue_box is not None:
                    continue_box[:] = anchor.best_result.box if anchor and anchor.hit else ()
                if not anchor or not anchor.hit:
                    return None
                image = self._image(frame)
                observation = _retain_conflicts(observation, read_inventory(context, image, layout))
                floor_observation = (self._floor_observation(context, image, layout, floor)
                                     if observation.get("bar_present") else None)
                return observation if (observation["complete"]
                    and (not needs_floor or (floor_observation and floor_observation["complete"]))) else None

            read_until(context, observe)
        if context.tasker.stopping:
            return None, True, False
        restored = continue_box is None or bool(continue_box)
        if not observation.get("bar_present", False):
            self._record("note_inventory_skipped", page=layout,
                reason=observation.get("bar_reason"), floor=None,
                image_path=self._save_image(image, f"{layout}_skipped") if self.ledger else None)
            return None, restored, False
        floor_observation = self._read_floor(context, image, layout, floor, observation=floor_observation)
        floor = floor_observation["floor"]
        pending = self.ledger.pending
        if layout == "event":
            if floor is not None:
                self.floor = floor
        elif pending and pending["source_type"] == "event" and layout == "result":
            candidates = pending["metadata"].get("event_floor_candidates", [])
            if floor is not None and (floor == self.floor or floor in candidates):
                self.floor = floor
                self.ledger.update_pending_metadata(stable_event_result_confirmed=True,
                    floor=floor, floor_complete=True, floor_source="stable_event_reward_page",
                    floor_observation=floor_observation, independently_observed=True,
                    analysis_eligible=True, analysis_exclusion_reason=None)
            elif (floor is not None and (self.floor is None or floor > self.floor)
                    and any(kind == "BATTLE" and value == floor for kind, value, _ in ROUTE)):
                self._battle(floor, floor_observation)
            else:
                self.ledger.update_pending_metadata(independently_observed=False,
                    analysis_eligible=False, probability_eligible=False,
                    analysis_exclusion_reason="source_identity_unconfirmed")
        elif pending and pending["source_type"] == "combined_interval":
            if (floor is not None and (self.floor is None or floor > self.floor)
                    and not any(source["source_type"] == "battle" for source in pending["mixed_sources"])
                    and any(kind == "BATTLE" and value == floor for kind, value, _ in ROUTE)):
                self._battle(floor, floor_observation)
            if floor is not None:
                self.floor = floor
        elif self.mode == "combined_complete" and self.floor is None:
            self.floor = floor
        elif floor is not None and (floor != self.floor or (pending is None and self.mode not in {
                "shop", "event_complete", "combined_complete"})):
            self._battle(floor, floor_observation)
        elif floor is None and pending is None and layout == "result" and self.mode != "combined_complete":
            self._battle(None, floor_observation)
        row = self.ledger.observe(observation, page=layout)
        row.update(floor=floor, floor_complete=floor_observation["complete"],
            floor_observation=floor_observation, last_confirmed_floor=self.floor,
            image_path=self._save_image(image, layout), derived_concertos=self.derived_concertos())
        self._emit(row)
        pending = self.ledger.pending
        if restored and layout == "result" and pending and pending["source_type"] in {"event", "combined_interval"}:
            restored, navigated = self.complete_event_inventory(context, page="stable_battle_reward",
                close_interval=(pending["source_type"] == "combined_interval"
                                or pending["metadata"].get("stable_event_result_confirmed", False)))
            return observation, restored, navigated
        return observation, restored, False

    def before_potential(self, context, image, source="default"):
        if source != "default" or not self.observing or self.ledger is None:
            return True, False
        floor_observation = self._read_floor(context, self._image(image), "potential")
        floor = floor_observation["floor"]
        pending = self.ledger.pending
        if pending and pending["source_type"] == "combined_interval":
            last_event = self._pending_event(pending)
            if (not any(source["source_type"] == "battle" for source in pending["mixed_sources"])
                    and not re.search(text_rules(context, "event")["potential"],
                                      last_event.get("consequence", ""), re.I)):
                page_id = self._record("note_stable_page_observed", page="potential",
                    source_basis="main_pipeline_confirmed_potential_page",
                    image_path=self._save_image(self._image(image), "stable_potential"))
                self._emit(self.ledger.combine_event_battle(
                    f"battle_{floor}" if floor is not None else f"battle_stable_{uuid.uuid4().hex}",
                    floor=floor, floor_complete=floor is not None, floor_observation=floor_observation,
                    source_basis="stable_potential_page_after_event", source_observation_id=page_id))
                self.floor = floor
            return self.complete_event_inventory(context, page="stable_event_potential")
        if pending and pending["source_type"] == "event":
            consequence = pending["metadata"].get("consequence", "")
            # An event's own potential reward is not evidence of a new battle.
            if re.search(text_rules(context, "event")["potential"], consequence, re.I) and (
                    floor is None or floor in pending["metadata"].get("event_floor_candidates", [])):
                return True, False
            page_id = self._record("note_stable_page_observed", page="potential",
                source_basis="main_pipeline_confirmed_potential_page",
                image_path=self._save_image(self._image(image), "stable_potential"))
            self._emit(self.ledger.combine_event_battle(
                f"battle_{floor}" if floor is not None else f"battle_stable_{uuid.uuid4().hex}",
                floor=floor, floor_complete=floor is not None, floor_observation=floor_observation,
                source_basis="stable_potential_page_after_event", source_observation_id=page_id))
            self.floor = floor
            self.mode = "combined"
            return self.complete_event_inventory(context, page="stable_event_potential")
        elif floor is not None and floor != self.floor:
            self._battle(floor, floor_observation)
        return True, False

    def observe_event(self, context, image):
        """Read the already visible event inventory before the model scores it."""
        if not self.observing:
            return
        self.capture(context, image, "event")

    def before_event(self, context, image, *, event_id=None, choice="", consequence="",
                     note_effect=None, allow_text_effect=True, model_event_id=None, option_id=None):
        if not self.observing or self.ledger is None:
            return
        rules = text_rules(context, "event")
        note_text = bool(re.search(rules["note"], choice + consequence, re.I))
        spends = (note_effect["operation"] == "spend" if note_effect else
                  bool(allow_text_effect and note_text and re.search(rules["spend"], consequence, re.I)))
        event = getattr(self.collector, "event", None) or {}
        selection = dict(choice=choice, consequence=consequence, note_effect=note_effect, spends_notes=spends,
            event_choice_record_id=event.get("choice_record_id"), model_event_id=model_event_id, option_id=option_id)
        pending = self.ledger.pending
        current_event = self._pending_event(pending) if pending else {}
        current_event_id = (pending["operation_id"] if pending and pending["source_type"] == "event" else
                            current_event.get("operation_id"))
        same_event = (current_event_id is not None and event_id is not None
                      and str(current_event_id) == str(event_id))
        consecutive = bool(pending and pending["source_type"] in {"event", "combined_interval"} and not same_event)
        if consecutive:
            clicked = current_event.get("event_click_inventory_revision")
            boundary = (self.ledger.observed_in_scope(self._event_scope(pending), page="event",
                        after_revision=clicked) if clicked is not None else None)
            if boundary is not None:
                self._emit(self.ledger.finish_pending(reason=("stable_page_combined_observation"
                    if pending["source_type"] == "combined_interval" else "next_event_menu_observation")))
                pending = None
                consecutive = False
        if not same_event and not consecutive:
            self._emit(self.ledger.finish_pending(reason="event_menu_before_click"))
        self.mode = "event"
        if same_event or consecutive or note_effect or note_text:
            if same_event:
                self.ledger.update_event_metadata(event_id, **selection)
            else:
                floor_observation = self.last_floor_observation or {}
                candidates = self._event_floor_candidates()
                preceding = getattr(self.collector, "last_observed_floor_context", None) or {}
                if (not floor_observation.get("complete") and len(candidates) == 1
                        and preceding.get("floor") in candidates):
                    floor_observation = dict(preceding, complete=True, source="previous_clear_floor_hud",
                        floor_observation_id=preceding.get("record_id"))
                metadata = dict(
                    floor=floor_observation.get("floor"),
                    floor_complete=bool(floor_observation.get("complete")),
                    floor_source=floor_observation.get("source"), floor_observation=floor_observation,
                    independently_observed=True, analysis_eligible=bool(floor_observation.get("complete")),
                    probability_eligible=bool(floor_observation.get("complete")),
                    analysis_exclusion_reason=None if floor_observation.get("complete") else "source_identity_unconfirmed",
                    event_floor_candidates=candidates, **selection)
                operation_id = event_id or uuid.uuid4().hex
                self._emit(self.ledger.combine_events(operation_id, **metadata) if consecutive else
                           self.ledger.begin_operation("event", operation_id, **metadata))

    @staticmethod
    def _pending_event(pending):
        if pending and pending["source_type"] == "combined_interval":
            return pending["metadata"]["event_components"][-1]
        return pending["metadata"] if pending and pending["source_type"] == "event" else {}

    def _event_floor_candidates(self):
        """Limit a hidden event to the route segment before the next battle."""
        if self.floor is None:
            return []
        next_battle = next((floor for kind, floor, _ in ROUTE
                            if kind == "BATTLE" and floor > self.floor), 21)
        return sorted({floor for kind, floor, _ in ROUTE
                       if kind in {"EVENT", "EXTRA"} and self.floor <= floor < next_battle})

    @staticmethod
    def _bag_page(observation):
        return "bag_skill" if observation.get("layout") == "bag_skill" else "bag_detail"

    def _event_scope(self, pending):
        if "required_scope" in pending:
            return pending["required_scope"]
        events = pending["metadata"].get("event_components") or [pending["metadata"]]
        melodies = [(event.get("note_effect") or {}).get("melody") for event in events]
        return tuple(dict.fromkeys((*self.ledger.visible_melodies, *(melody for melody in melodies if melody))))

    def _ensure_event_inventory(self, context, pending, page):
        """Reuse inventory for this boundary; navigate only for a required unread count.

        Return (observation, page restored, bag navigation attempted).
        """
        ledger = self.ledger
        clicked = pending["metadata"].get("event_click_inventory_revision")
        if page == "stable_event_menu":
            phase = "before"
            scope = ledger.spend_scope()
            # Off-tower components may be carried before a spend, never after it.
            snapshot = (None if ledger.unresolved_spend_extras() else
                        ledger.observed_in_scope(ledger.visible_melodies, after_revision=clicked))
        else:
            phase = "after"
            scope = self._event_scope(pending)
            snapshot = ledger.observed_in_scope(scope, page="result",
                after_revision=clicked if clicked is not None else pending["start_revision"])
        if snapshot is not None:
            return snapshot, True, False
        if page == "stable_battle_reward" and pending.get("required_scope"):
            # A spend result that cannot settle the scope waits for a stable offer.
            return None, True, False
        observation, restored = read_full_inventory(context, scope,
            capture_skill_page=lambda image: self._save_image(image, f"event_{phase}_bag"))
        if observation is not None:
            if phase == "after":
                before_scope = pending["before"]["scope_melodies"]
                after_scope = observation["scope_melodies"]
                self.ledger.update_pending_metadata(before_scope_melodies=list(before_scope),
                    after_scope_melodies=list(after_scope),
                    unobserved_before_melodies=[key for key in after_scope if pending["before"]["counts"][key] is None])
            row = self.ledger.observe(observation, page=self._bag_page(observation))
            row.update(floor=None, floor_complete=False, last_confirmed_floor=self.floor,
                       image_path=observation.get("skill_image_path"),
                       derived_concertos=self.derived_concertos())
            self._emit(row)
        self._record("note_bag_inventory_read", phase=phase, operation_id=pending["operation_id"],
            complete=bool(observation and observation["complete"]), bag_closed=restored, page=page,
            inventory_page=self._bag_page(observation) if observation else None)
        return observation, restored, True

    def prepare_event_inventory(self, context):
        """Return (page restored, navigated) so the click uses the correct frame."""
        pending = self.ledger.pending if self.ledger is not None else None
        if not pending or not self._pending_event(pending).get("spends_notes"):
            return True, False
        _, restored, navigated = self._ensure_event_inventory(context, pending, "stable_event_menu")
        if pending["source_type"] == "combined_interval" and self.ledger.observed_in_scope(
                self._event_scope(pending), after_revision=pending["metadata"].get("event_click_inventory_revision")):
            self._emit(self.ledger.separate_event_boundary())
            pending = self.ledger.pending
        if self.ledger.revision > pending["start_revision"]:
            self._emit(self.ledger.refresh_pending_baseline())
        self._emit(self.ledger.prepare_spend_baseline())
        return restored, navigated

    def confirm_event_choice(self):
        pending = self.ledger.pending if self.ledger is not None else None
        if pending and pending["source_type"] in {"event", "combined_interval"}:
            event = self._pending_event(pending)
            event_id = pending["operation_id"] if pending["source_type"] == "event" else event["operation_id"]
            self.ledger.update_event_metadata(event_id, event_click_inventory_revision=self.ledger.revision)
            self.ledger.update_pending_metadata(event_click_inventory_revision=self.ledger.revision)
            if event.get("spends_notes"):
                self.ledger.invalidate_spend_extras()

    def complete_event_inventory(self, context, *, page, close_interval=True):
        """Close a pending observation when possible; report restoration and navigation."""
        pending = self.ledger.pending if self.ledger else None
        if not pending or pending["source_type"] not in {"event", "combined_interval"}:
            return True, False
        observation, restored, navigated = self._ensure_event_inventory(context, pending, page)
        if close_interval and (observation is not None or page == "stable_event_potential"):
            combined = pending["source_type"] == "combined_interval"
            self.finish("stable_event_potential_unread" if observation is None else
                        "stable_page_combined_observation" if combined else "stable_event_reward_observation")
            self.mode = "combined_complete" if combined else "event_complete"
        return restored, navigated

    def complete_potential_reward_inventory(self, screen):
        """Only a bag visit requires re-locating the reward page and its controls."""
        restored, navigated = self.complete_event_inventory(screen.context, page="stable_event_potential")
        if restored and navigated:
            screen.screenshot()
            result = screen.context.run_recognition("星塔_节点_进入潜能选择_agent", screen.image)
            restored = bool(result and result.hit)
        return restored, navigated

    def enter_shop(self):
        self.finish("shop_entry")
        self.mode = "shop"

    def after_shop(self):
        self.finish("shop_exit")
        self.mode = "field"

    def finish(self, reason="boundary"):
        if self.ledger is not None:
            self._emit(self.ledger.finish_pending(reason=reason))

    def settlement(self, context, image, inventory=None):
        if not self.observing or self.ledger is None:
            return None
        image = self._image(image)
        # A missed tab switch leaves the potential list at these same ROIs.
        # Without the skill-page inventory, those words/digits are not results.
        order = self.setup.get("concertos") if self.setup else None
        actual = None
        first = True

        def observe(frame):
            nonlocal image, inventory, actual, first
            if not first:
                ready = context.run_recognition("星塔_记录界面_读取数值_agent", frame)
                if not ready or not ready.hit:
                    return None
            image = self._image(frame)
            current = inventory if first and inventory is not None else read_inventory(context, image, "settlement")
            first = False
            inventory = _retain_conflicts(inventory, current)
            actual = (read_settlement(context, image, order) if inventory.get("bar_present") else
                dict(actual_concertos=[], score=None, score_raw=[], complete=False,
                     reason="settlement_page_not_confirmed"))
            if inventory["complete"] and actual["complete"]:
                return True

        read_until(context, observe, initial_image=image)
        if context.tasker.stopping or actual is None:
            return None
        inventory, _, _ = self.capture(context, image, "settlement", observation=inventory)
        actual["score_and_concertos_complete"] = actual["complete"]
        actual["inventory_complete"] = bool(inventory and inventory["complete"] and self.ledger
                                             and self.ledger.current_observation["complete"])
        actual["complete"] = actual["score_and_concertos_complete"] and actual["inventory_complete"]
        self.finish("settlement")
        self._record("note_settlement_observed", **actual,
            counts=inventory["counts"] if inventory else None,
            scope_melodies=self.ledger.current_observation["scope_melodies"] if inventory else [],
            derived_concertos=self.derived_concertos())
        return actual

    def derived_concertos(self):
        inventory = self.current_inventory()
        if not self.setup or inventory is None:
            return []
        counts = dict(inventory["counts"])
        unread = set(inventory.get("missing", ())) | set(inventory.get("incomplete", ())) | set(inventory.get("conflicts", ()))
        carried = {key: value for key, value in self.ledger.known_spend_extras().items()
                   if counts.get(key) is None and key not in unread}
        counts.update(carried)
        groups = load_rules()["concerto_groups"]
        cards = self.setup.get("concertos")
        if cards is None:
            cards = [dict(slot=slot, group_id=group, disc_slot=disc["slot"], mapping_status="catalogue")
                     for slot, (disc, group) in enumerate(
                         ((disc, group) for disc in self.setup["main_discs"]
                          for group in disc["secondary_group_ids"]), 1)]
        rows = []
        for card in cards:
            group = card.get("group_id")
            if group is None:
                continue
            levels = groups[str(group)]
            complete = bool(levels) and all(counts.get(key) is not None
                for entry in levels for key in entry["need"])
            level = max((entry["level"] for entry in levels
                         if complete and all(counts[key] >= need
                                             for key, need in entry["need"].items())), default=0)
            row = dict(slot=card["slot"], group_id=group, disc_slot=card.get("disc_slot"),
                mapping_status=card.get("mapping_status", "position"), level=level if complete else None,
                source="inventory_and_local_rules", basis="local_rules_derivation",
                complete=complete)
            used = sorted({key for entry in levels for key in entry["need"] if key in carried})
            if used:
                row["carried_melodies"] = used
            rows.append(row)
        return rows
