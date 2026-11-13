"""Unrestricted greedy public Q policy, with batched legal-action inference."""
from __future__ import annotations

import numpy as np
from .environment import PublicState, legal_actions
from .features import encode_actions
from .network import QNetwork


class QPolicy:
    def __init__(self, model, model_id="model"):
        self.model = model
        self.model_id = str(model_id)

    @classmethod
    def load(cls, path, model_id="model"):
        return cls(QNetwork.load(path), model_id)

    def values(self, state, actions=None, preset=None):
        if not isinstance(state, PublicState):
            raise TypeError("Only public states may enter a policy")
        actions = legal_actions(state) if actions is None else tuple(actions)
        q = self.model.predict(encode_actions(state, actions, preset))
        if not np.isfinite(q).all():
            raise FloatingPointError("Network produced nonfinite action values")
        return q

    def choose(self, state, actions=None, preset=None):
        actions = legal_actions(state) if actions is None else tuple(actions)
        if len(actions) == 1:
            return actions[0]
        # No min-gain, incumbent lock, confidence test, or hard output clipping.
        return actions[int(np.argmax(self.values(state, actions, preset)))]
