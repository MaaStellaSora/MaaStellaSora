"""One complete-visible-menu policy, independent of the model objective.

Event models receive every executable visible option through decision_context;
they own objective-specific valuation. Quiz answers use Maa's answer table.
"""
from dataclasses import dataclass, replace

from .event_mapping import RULES, choice_complete, match_event, note_descriptor
from .critic.environment import Action, sim


EVENT_POLICY_ID = RULES["policy"]["id"]
EVENT_VALUE_SAMPLES = RULES["policy"]["value_samples"]


def select_last_best(values):
    """Exact numerical ties go to the last candidate in visible UI order."""
    return max(range(len(values)), key=lambda index: (float(values[index]), index))


@dataclass
class EventMenu:
    event_id: int | None
    option_indices: dict
    visible_options: list
    state: object = None
    actions: tuple = ()
    full_visible_menu: bool = True

    def mapping(self, model, event_policy_id, *, decision_method, selected_ui_index,
                decision=None):
        values = {}
        for row in (decision or {}).get("values", ()):
            action = row["action_record"]
            oid = 0 if action["kind"] == "SKIP_EVENT" else action["index"]
            values[oid] = row.get("predicted_value", row.get("expected_remaining_ordinary_return"))
        options = [dict(row) for row in self.visible_options]
        for row in options:
            value = values.get(row["model_option_id"])
            if value is not None:
                row.update(evaluation_status="evaluated", reason="model_inference", predicted_value=value)
        mapping = dict(policy_mode=model.versions["policy_mode"], model_id=model.model_id,
                    event_policy_id=event_policy_id, decision_method=decision_method,
                    event=self.event_id, option=self.option_indices.get(selected_ui_index),
                    selected_ui_index=selected_ui_index, visible_options=options,
                    full_visible_menu=self.full_visible_menu, tie_break="last_ui_index")
        for key, value in (decision or {}).items():
            if key.startswith("evaluation_"):
                mapping[key] = value
        return mapping


class EventPolicy:
    """Keep the observed menu; restrict only executable resource boundaries."""
    event_policy_id = EVENT_POLICY_ID

    def menu(self, question, choices, lang):
        def field(choice, key, default=None):
            return choice.get(key, default) if isinstance(choice, dict) else getattr(choice, key, default)

        complete = bool(choices) and all(choice_complete(choice) for choice in choices)
        options = [dict(ui_index=index, model_option_id=None,
                        evaluation_status="not_evaluated", reason="official_rule_menu",
                        predicted_value=None, available=field(choice, "available"),
                        observations=field(choice, "observations", {}))
                   for index, choice in enumerate(choices)]
        matched = match_event(question, choices, lang)
        if matched is None:
            return EventMenu(None, {}, options, full_visible_menu=bool(complete))
        eid, indices = matched
        for row in options:
            oid = indices[row["ui_index"]]
            row.update(model_option_id=oid, reason="model_value_not_recorded",
                       note_effect=note_descriptor(eid, oid))
        return EventMenu(eid, indices, options, full_visible_menu=bool(complete))

    def prepare(self, menu, state, coins):
        state = replace(state, phase="EVENT", eventId=menu.event_id,
                        coins=int(coins),
                        eventOptions=tuple(sorted({0, *menu.option_indices.values()})),
                        seen=tuple(sorted({*state.seen, menu.event_id})))
        definitions = sim.EVENT_DATA[menu.event_id][2]
        legal = []
        for row in menu.visible_options:
            oid = row["model_option_id"]
            cost, outcomes = definitions[oid]
            if row.get("available") is False:
                row.update(evaluation_status="unavailable", reason="observed_disabled")
            elif coins < cost:
                row.update(evaluation_status="unavailable", reason="insufficient_coins")
            elif any(n and coins + delta < 0 for n, _d, delta, _kind, _scope in outcomes):
                row.update(evaluation_status="not_evaluated", reason="model_gap_random_loss")
            else:
                legal.append(Action("SKIP_EVENT", type=menu.event_id) if oid == 0
                             else Action("EVENT_OPTION", oid, cost, menu.event_id))
        if not legal:
            raise ValueError(f"事件 {menu.event_id} 未找到可执行选项")
        menu.state = state
        menu.actions = tuple(legal)
        return menu

    def decision_context(self, menu):
        return dict(kind="event", event_id=menu.event_id,
                    event_policy_id=self.event_policy_id, samples=EVENT_VALUE_SAMPLES,
                    tie_break="last_ui_index", full_visible_menu=menu.full_visible_menu,
                    visible_options=menu.visible_options)
