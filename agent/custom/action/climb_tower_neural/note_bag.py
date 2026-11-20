"""Read the actual note-detail list through Maa's existing bag navigation."""
from .interactor import Interactor

from .note_bag_reader import merge_bag_note_pages, read_bag_note_page
from .note_cross_checks import collect_cross_checks
from .note_ocr import OCR_RULES, normalize_image, read_inventory
from .pages import BAG_RETURN_PAGES, classify
from .ui_flow import read_until


OPEN_BAG = "星塔_背包_扫描音符_打开背包界面_agent"
OPEN_SKILLS = "星塔_背包_打开秘纹技能_agent"
SKILLS_READY = "星塔_背包_秘纹技能界面_agent"
CLOSE_BAG = "星塔_关闭背包界面_agent"
CLOSE_DETAIL = "星塔_背包_关闭音符效果窗口_agent"
OPEN_DETAIL = "星塔_神经网络_打开音符详情"
DETAIL_READY = "星塔_神经网络_音符详情已打开"
SCROLL_TOP = "星塔_神经网络_音符详情置顶"
SCROLL_DOWN = "星塔_神经网络_音符详情向下"


def _run(context, node):
    result = context.run_task(node, {node: {"timeout": 3000}})
    return bool(result and result.status.succeeded)


def _returned(context, image):
    return classify(context, image, candidates=BAG_RETURN_PAGES) in BAG_RETURN_PAGES


def _retain_conflicts(previous, current):
    """Keep explicit conflicts when rereading the same unchanged page."""
    conflicts = set((previous or {}).get("conflicts", ()))
    if not conflicts:
        return current
    current = dict(current, counts=dict(current["counts"]), readings=list(current["readings"]))
    for row in previous.get("readings", ()):
        if row.get("melody") in conflicts and row not in current["readings"]:
            current["readings"].append(row)
    current["conflicts"] = sorted(conflicts | set(current.get("conflicts", ())))
    for melody in current["conflicts"]:
        current["counts"][melody] = None
    current["complete"] = False
    return current


def _skill_counts_complete(bar, requirements, required):
    return bool(bar and bar.get("complete")) and not any(
        row.get("kind") != "skipped" and (row.get("melody") is None
            or row.get("melody") in required and (row.get("current") is None
                or row.get("conflicting_values"))) for row in requirements)


def _inventory_observation(pages, expected_melodies, requirements, bar_observation, skill_image_path):
    repeated = [dict(row, source=f"bag_requirement:{index}")
                for index, row in enumerate(requirements)]
    if bar_observation is not None:
        repeated.extend(dict(row, source="bag_inventory_bar")
                        for row in bar_observation.get("readings", ()))
        if not bar_observation.get("bar_present", False):
            repeated.append(dict(melody=None, current=None, kind="skipped", source="bag_inventory_bar",
                reason=bar_observation.get("bar_reason"), alternative="bag_note_details_and_requirements", raw=[]))
    if not (pages or repeated):
        return None
    observation = merge_bag_note_pages(pages, expected_melodies, repeated=repeated)
    if not pages:
        observation.update(layout="bag_skill", visible_melodies=list(dict.fromkeys(
            row["melody"] for row in repeated if row.get("melody") and row.get("kind") != "skipped")))
        if bar_observation is not None:
            observation.update(bar_present=bar_observation.get("bar_present", False),
                               bar_reason=bar_observation.get("bar_reason"))
    observation.update(requirement_readings=requirements, bar_observation=bar_observation,
                       skill_image_path=skill_image_path)
    return observation


def read_full_inventory(context, expected_melodies, *, bag_open=False, close_bag=True,
                        requirement_readings=None, bar_observation=None, capture_skill_page=None,
                        reread_skill_page=True):
    """Read skills first, opening details only for missing/conflicting required counts.

    Supplied skill-page observations are reused. With ``bag_open=True`` the
    caller already opened that page; ``close_bag=False`` leaves it open.
    """
    expected_melodies = tuple(dict.fromkeys(expected_melodies))
    required = set(expected_melodies)
    pages = []
    bag_closed = False
    requirements = list(requirement_readings or ())
    skill_image_path = None
    opened, detail_attempted = bag_open, False
    image = observation = None
    try:
        opened = bag_open or _run(context, OPEN_BAG)
        if opened and not context.tasker.stopping:
            if requirement_readings is None or bar_observation is None or capture_skill_page is not None:
                image = Interactor.capture(context)
                if image is not None:
                    image = normalize_image(image) if image.shape[:2] != (720, 1280) else image
                    if requirement_readings is None:
                        requirements = collect_cross_checks(context, image, "bag")
                    if bar_observation is None:
                        bar_observation = read_inventory(context, image, "bag")
            if (reread_skill_page and (image is not None or bar_observation is not None)
                    and not _skill_counts_complete(bar_observation, requirements, required)):
                def reread(frame):
                    nonlocal image, requirements, bar_observation
                    ready = context.run_recognition(SKILLS_READY, frame)
                    if not ready or not ready.hit:
                        return None
                    image = normalize_image(frame) if frame.shape[:2] != (720, 1280) else frame
                    new_requirements = collect_cross_checks(context, image, "bag")
                    # Retain already observed independent disagreements.
                    conflicts = set((_inventory_observation([], expected_melodies, requirements,
                        bar_observation, None) or {}).get("conflicts", ()))
                    for row in requirements:
                        if row.get("melody") in conflicts and row not in new_requirements:
                            new_requirements.append(row)
                    requirements = new_requirements
                    bar_observation = _retain_conflicts(bar_observation, read_inventory(context, image, "bag"))
                    if _skill_counts_complete(bar_observation, requirements, required):
                        return True

                read_until(context, reread)
            if capture_skill_page is not None and image is not None:
                skill_image_path = capture_skill_page(image)
            observation = _inventory_observation(pages, expected_melodies, requirements,
                                                 bar_observation, skill_image_path)
        outer_conflicts = set(observation["conflicts"]) & required if observation is not None else set()
        needs_detail = observation is not None and (observation["missing"] or outer_conflicts)
        can_read_detail = (image is not None or (requirement_readings is not None and bar_observation is not None))
        ready = False
        if needs_detail and can_read_detail and not context.tasker.stopping:
            detail_attempted = True
            ready = _run(context, OPEN_DETAIL) and _run(context, SCROLL_TOP)
        previous = None
        detail_checked = set()
        for _ in range(OCR_RULES["bag_detail"]["maximum_pages"] if ready else 0):
            if context.tasker.stopping:
                break
            image = Interactor.capture(context)
            if image is None:
                break
            image = normalize_image(image) if image.shape[:2] != (720, 1280) else image
            page = read_bag_note_page(context, image)
            if not page.get("complete", True):
                def reread_detail(frame):
                    nonlocal image, page
                    detail = context.run_recognition(DETAIL_READY, frame)
                    if not detail or not detail.hit:
                        return None
                    image = normalize_image(frame) if frame.shape[:2] != (720, 1280) else frame
                    page = _retain_conflicts(page, read_bag_note_page(context, image))
                    return page if page["complete"] else None

                read_until(context, reread_detail)
            pages.append(page)
            observation = _inventory_observation(pages, expected_melodies, requirements,
                                                 bar_observation, skill_image_path)
            detail_checked.update(row["melody"] for row in page["readings"]
                                  if row.get("melody") and row.get("current") is not None)
            unresolved = set(observation["missing"]) | (set(observation["conflicts"]) & required)
            # More scrolling cannot reconcile two already clear outer values.
            # Keep that conflict after one independent detail reading.
            if not unresolved - (outer_conflicts & detail_checked):
                break
            # The list can bounce a few pixels at its bottom. Counts and note
            # identities establish an unchanged page; retain the real boxes
            # in the observations rather than rescanning the same bottom rows.
            signature = tuple((item.get("melody"), item.get("current"), item.get("kind"))
                              for item in page["readings"])
            if signature == previous:
                break
            previous = signature
            if not _run(context, SCROLL_DOWN):
                break
    finally:
        if not context.tasker.stopping:
            detail_closed = True
            if detail_attempted:
                image = Interactor.capture(context)
                # The title can be unreadable while the existing close-icon
                # anchor still identifies the actual detail window.
                detail = context.run_recognition(CLOSE_DETAIL, image) if image is not None else None
                if detail and detail.hit:
                    detail_closed = _run(context, CLOSE_DETAIL)
                    image = Interactor.capture(context)
            if close_bag:
                if image is None:
                    image = Interactor.capture(context)
                bag_closed = image is not None and _returned(context, image)
            else:
                bag_closed = opened and detail_closed
            if close_bag and not bag_closed and not context.tasker.stopping:
                bag_closed = _run(context, CLOSE_BAG)
                if not bag_closed and not context.tasker.stopping:
                    image = Interactor.capture(context)
                    bag_closed = image is not None and _returned(context, image)
    return observation, bag_closed
