"""Hide legacy strategy chatter in the console during model callbacks only."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging


class ModelLogFilter(logging.Filter):
    PREPARATION_IMPORTED_KEY = "agent.custom.action.climb_tower_preparation.run.imported"
    LEGACY_LOGGERS = (
        "custom.reco.climb_tower_potential.",
        "custom.action.climb_tower_shop.",
        "custom.reco.climb_tower_event.",
    )

    def __init__(self):
        super().__init__()
        self._active = ContextVar("stella_value_model_log_scope", default=False)
        self._console_handlers = []

    def attach(self, logger):
        # File handlers keep every original record, at its original severity.
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.addFilter(self)
                self._console_handlers.append(handler)

    def filter(self, record):
        if (self._active.get() and record.levelno == logging.INFO
                and record.name == "custom.action.climb_tower_preparation"
                and (getattr(record, "i18n_key", None) == self.PREPARATION_IMPORTED_KEY
                     or record.msg == f"${self.PREPARATION_IMPORTED_KEY}")):
            return False
        # Maa enables DEBUG_MODE after importing custom callbacks.
        if any(handler.level <= logging.DEBUG for handler in self._console_handlers):
            return True
        return not (self._active.get() and record.levelno == logging.INFO
                    and record.name.startswith(self.LEGACY_LOGGERS))

    @contextmanager
    def scope(self, value_model=True):
        token = self._active.set(value_model)
        try:
            yield
        finally:
            self._active.reset(token)
