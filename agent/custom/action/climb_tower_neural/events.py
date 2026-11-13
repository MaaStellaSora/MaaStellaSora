"""Model event choices and matched official quiz answers."""
from dataclasses import dataclass, field

from maa.custom_recognition import CustomRecognition
from .interactor import Interactor
from .ui_flow import read_until

from .event_mapping import note_descriptor, quiz_correct_option, quiz_event_id
from .event_reader import EventRecognition


def _read_event_coins(context, image):
    # The icon moves with the wallet when the note bar expands.
    node = "星塔_神经网络_定位事件钱包"
    icon = context.run_recognition(node, image)
    if not icon or not icon.hit:
        return None
    dx, dy, width, height = context.get_node_data(node)["attach"]["amount_roi_offset"]
    x, y, _, _ = icon.best_result.box
    screen = Interactor(context)
    screen.image = image
    amount = screen.get_current_coin(roi=[x + dx, y + dy, width, height])
    return amount if amount >= 0 else None


def event_page_key(question, choices):
    return question, tuple((choice.text, choice.consequence, getattr(choice, "available", None))
                           for choice in choices)


@dataclass
class EventMenuRead:
    image: object = None
    question: str = ""
    choices: list = field(default_factory=list)
    coins: int | None = None
    missing: list = field(default_factory=list)
    attempts: int = 0
    menu: object = None
    quiz_id: int | None = None
    answer: object = None

    @property
    def required_missing(self):
        # A matched quiz answer does not depend on the wallet.
        return [name for name in self.missing if self.quiz_id is None or name != "coin"]


def _read_event_menu(recognizer, context, argv, runtime):
    node = context.get_node_data(argv.node_name) or {}
    lang = node.get("attach", {}).get("lang_type", "cn")
    latest = EventMenuRead(image=argv.image, missing=["question", "choices", "coin"])

    def observe(image):
        nonlocal latest
        if context.tasker.stopping:
            return None
        question = recognizer._get_question_text(context, image, lang)
        choices = recognizer._get_choice_texts(context, image, lang)
        with runtime.collector.capture("event_wallet"):
            coins = _read_event_coins(context, image)
        missing = ([] if question else ["question"]) + ([] if choices else ["choices"])
        if coins is None:
            missing.append("coin")
        for field, label in (("text", "choice_text"), ("consequence", "choice_consequence")):
            if any(not getattr(choice, field) or (
                    field in getattr(choice, "observations", {})
                    and not choice.observations[field].get("complete")) for choice in choices):
                missing.append(label)
        latest = EventMenuRead(image, question, choices, coins, missing, latest.attempts + 1,
                               quiz_id=quiz_event_id(choices))
        if latest.required_missing:
            return None
        latest.menu = runtime.event_policy.menu(question, choices, lang)
        if latest.quiz_id is not None:
            latest.answer = EventRecognition.match_from_menu(recognizer, context, argv, question, choices)
            if latest.answer.box is None:
                latest.missing.append("official_answer_rule")
        elif latest.menu.event_id is None:
            latest.missing.append("event_mapping")
        return latest if not latest.required_missing else None

    read_until(context, observe, initial_image=argv.image)
    return latest


def _finish_event_result(result, runtime, question, choices, missing, attempts, coins):
    detail = dict(result.detail) if isinstance(result.detail, dict) else {}
    detail.update(ocr_missing=list(missing), ocr_attempts=attempts, ocr_complete=not missing,
                  observed_coins=coins)
    if result.box is not None:
        detail["selected_box"] = list(result.box)
        index = next((i for i, choice in enumerate(choices) if list(choice.box) == list(result.box)), None)
        if index is not None:
            runtime.event_page_cache = (event_page_key(question, choices), index, detail)
    return CustomRecognition.AnalyzeResult(box=result.box, detail=detail)


def _same_event_read(cached, key, coins, missing):
    return (cached is not None and cached[0] == key and not missing
            and not cached[2].get("ocr_missing") and cached[2].get("observed_coins") == coins)


def _stop_event(context, runtime, reason, missing, **observed):
    runtime.event_page_cache = None
    runtime.stop_model(context, reason, phase="event", missing=missing, **observed)
    return CustomRecognition.AnalyzeResult(box=None, detail={
        "policy_mode": "model", "model_decision_recorded": False,
        "stopped": True, "reason": reason, "missing": list(missing)})


def analyze_model_event(recognizer, context, argv, runtime, *, reading=None):
    tracker = runtime.require_tracker()
    reading = reading if reading is not None else _read_event_menu(recognizer, context, argv, runtime)
    if context.tasker.stopping:
        return CustomRecognition.AnalyzeResult(box=None, detail={})
    question, choices, coins = reading.question, reading.choices, reading.coins
    missing, attempts, quiz_id = reading.missing, reading.attempts, reading.quiz_id
    if reading.required_missing:
        tracker.event_page_cache = None
        reason = ("quiz_answer_unmatched" if "official_answer_rule" in missing else
                  "event_mapping_unavailable" if "event_mapping" in missing else
                  "event_observation_incomplete")
        result = _stop_event(context, runtime, reason, reading.required_missing,
                             ocr_attempts=attempts, question=question)
        return _finish_event_result(result, runtime, question, choices, missing, attempts, coins)
    page_key = event_page_key(question, choices)
    if _same_event_read(tracker.event_page_cache, page_key, coins, missing):
        _key, index, detail = tracker.event_page_cache
        detail = dict(detail, selected_box=list(choices[index].box), ocr_attempts=attempts)
        tracker.event_page_cache = (page_key, index, detail)
        runtime.event_page_cache = (page_key, index, detail)
        return CustomRecognition.AnalyzeResult(box=choices[index].box, detail=detail)
    menu = reading.menu
    try:
        with tracker.event_evaluation(coins, event_id=menu.event_id or quiz_id):
            result = _evaluate_model_menu(recognizer, context, argv, runtime, tracker, menu,
                                          question, choices, page_key, quiz_id, answer=reading.answer)
            if result.box is not None:
                result.detail.update(tracker.event_location())
                tracker.stage_event_result()
            result = _finish_event_result(result, runtime, question, choices, missing, attempts, coins)
            if tracker.event_page_cache is not None:
                tracker.event_page_cache = (tracker.event_page_cache[0], tracker.event_page_cache[1], result.detail)
            return result
    except ValueError as error:
        tracker.event_page_cache = None
        result = _stop_event(context, runtime, "event_model_inputs_unconfirmed", ["model_state"],
                             observed_error=str(error))
        return _finish_event_result(result, runtime, question, choices, missing, attempts, coins)


def _select_quiz_answer(recognizer, context, argv, runtime, tracker, menu, question, choices, page_key, quiz_id,
                        answer=None):
    result = answer if answer is not None else EventRecognition.match_from_menu(
        recognizer, context, argv, question, choices)
    if result.box is None:
        return _stop_event(context, runtime, "quiz_answer_unmatched", ["official_answer_rule"],
                           question=question)
    index = next(i for i, choice in enumerate(choices) if list(choice.box) == list(result.box))
    choice = choices[index]
    context.override_pipeline({argv.node_name: {"attach": {
        "last_question": question, "last_choice": choice.text, "last_consequence": choice.consequence}}})
    oid = quiz_correct_option(quiz_id)
    menu.event_id = quiz_id
    menu.option_indices[index] = oid
    menu.visible_options[index].update(model_option_id=oid, reason="official_answer_rule",
                                       note_effect=note_descriptor(quiz_id, oid))
    tracker.event_selected(quiz_id, oid)
    detail = dict(result.detail)
    detail.update(menu.mapping(tracker.model, runtime.event_policy.event_policy_id,
        decision_method="official_answer_rule", selected_ui_index=index))
    detail.update(model_decision_recorded=False, selection_basis="official_answer_rule",
                  quiz_event_id=quiz_id, note_effect=note_descriptor(quiz_id, oid))
    tracker.event_page_cache = (page_key, index, detail)
    return CustomRecognition.AnalyzeResult(box=result.box, detail=detail)


def _evaluate_model_menu(recognizer, context, argv, runtime, tracker, menu, question, choices, page_key, quiz_id,
                         *, answer=None):
    policy = runtime.event_policy
    if quiz_id is not None:
        return _select_quiz_answer(recognizer, context, argv, runtime, tracker, menu,
                                  question, choices, page_key, quiz_id, answer=answer)
    if menu.event_id is None:
        return _stop_event(context, runtime, "event_mapping_unavailable", ["event_mapping"],
                           question=question)
    if not tracker.identity_pools.available:
        return _stop_event(context, runtime, "event_model_inputs_unconfirmed", ["complete_pool_catalogue"])
    menu = policy.prepare(menu, tracker.state, tracker.state.coins)
    tracker.state = menu.state
    by_option = {oid: index for index, oid in menu.option_indices.items()}
    labels = {a: choices[by_option[0 if a.kind == "SKIP_EVENT" else a.index]].text for a in menu.actions}
    decision_context = runtime.decision_context("event", dict(policy.decision_context(menu),
        question=question, choices=[dict(text=choice.text, consequence=choice.consequence,
                                        box=list(choice.box)) for choice in choices]))
    action = tracker.choose(menu.actions, labels, decision_context=decision_context)
    oid = 0 if action.kind == "SKIP_EVENT" else action.index
    choice = choices[by_option[oid]]
    context.override_pipeline({argv.node_name: {"attach": {
        "last_question": question, "last_choice": choice.text, "last_consequence": choice.consequence
    }}})
    tracker.event_selected(menu.event_id, oid)
    detail = menu.mapping(tracker.model, policy.event_policy_id,
        decision_method="model_argmax", selected_ui_index=by_option[oid], decision=tracker.last_decision)
    detail["value_model"] = tracker.model.model_id
    detail["note_effect"] = note_descriptor(menu.event_id, oid)
    tracker.event_page_cache = (page_key, by_option[oid], detail)
    return CustomRecognition.AnalyzeResult(box=choice.box, detail=detail)
