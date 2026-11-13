"""Frame analysis on Maa's bounded Pipeline recognition loop."""
from time import perf_counter

from .interactor import Interactor


READ_TIMEOUT = 10.0


def read_until(context, observe, *, initial_image=None):
    """Read a stable page without input. Only None means its fields are incomplete."""
    if context.tasker.stopping:
        return None
    if initial_image is not None:
        result = observe(initial_image)
        if context.tasker.stopping or result is not None:
            return None if context.tasker.stopping else result
    return Interactor.wait_for(context, observe, timeout=READ_TIMEOUT)


def wait_for(context, observe, *, timeout=3.0, stable_frames=1, retry=None, retry_after=1.5):
    """Confirm a UI transition; an ignored input may be retried on a recognized target."""
    previous, consecutive = None, 0

    def stable(image):
        nonlocal previous, consecutive
        result = observe(image)
        if context.tasker.stopping:
            return None
        if result:
            consecutive = consecutive + 1 if result == previous else 1
            previous = result
            return result if consecutive >= stable_frames else None
        previous, consecutive = result, 0
        return None

    if retry is None or retry_after >= timeout:
        return Interactor.wait_for(context, stable, timeout=timeout)
    checkpoint = object()
    deadline = perf_counter() + retry_after

    def before_retry(image):
        result = stable(image)
        if result is not None:
            return result
        # Finish the expected retry checkpoint successfully. Only the actual
        # transition deadline should enter Maa's failure/screenshot path.
        return checkpoint if perf_counter() >= deadline and not context.tasker.stopping else None

    result = Interactor.wait_for(context, before_retry, timeout=timeout)
    if context.tasker.stopping:
        return None
    if result is not checkpoint:
        return result

    # One fresh framework frame both checks the transition and verifies the
    # original target before the caller may retry its one ignored input.
    image = Interactor.capture(context)
    if context.tasker.stopping:
        return None
    result = stable(image)
    if result is not None:
        return result
    sent = retry(image)
    if sent:
        previous, consecutive = None, 0
    remaining = timeout if sent else timeout - retry_after
    return Interactor.wait_for(context, stable, timeout=remaining)
