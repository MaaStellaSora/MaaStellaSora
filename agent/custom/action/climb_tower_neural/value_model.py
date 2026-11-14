"""Portable NumPy action critic used by the Maa integration.

The fixed distribution checkpoint is loaded once per agent process.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np

from .critic.environment import PublicState, Action, legal_actions, sim
from .critic.features import FEATURE_VERSION, encode_actions
from .critic.network import MODEL_VERSION
from .critic.policy import QPolicy
from .critic.presets import Preset, canonical_seed
from .deployment_preset import DeploymentPreset
from .event_policy import EVENT_POLICY_ID, EVENT_VALUE_SAMPLES, select_last_best


EVENT_VALUE_METHOD = "exact_outcomes_next_decision_max_q_v1"
# Keep identical future draws across checkpoint upgrades for the same public menu.
EVENT_SEED_DOMAIN = "m144_exact_outcomes_next_decision_max_q_v1"


class ValueModel:
    def __init__(self, plugin_dir=None):
        self.root = Path(plugin_dir or Path(__file__).resolve().parent)
        self._load()

    def _load(self):
        selection = json.loads((self.root / "model.json").read_text(encoding="utf-8-sig"))
        checkpoint = Path(selection["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = self.root / checkpoint
        model_id = selection.get("model_id", checkpoint.stem)
        actor = QPolicy.load(checkpoint, model_id)
        checkpoint_rules = actor.model.metadata.get("rules_version")
        if checkpoint_rules and checkpoint_rules != sim.MODEL_VERSION:
            raise ValueError(f"模型规则 {checkpoint_rules} 与模块规则 {sim.MODEL_VERSION} 不一致。")
        self.actor = actor
        self.model_id = model_id
        self.checkpoint = checkpoint
        self.versions = {
            "policy_mode": "potential_only", "model_id": model_id,
            "event_policy_id": EVENT_POLICY_ID,
            "event_evaluation_method": EVENT_VALUE_METHOD,
            "state_schema_version": sim.MODEL_VERSION,
            "observation_adapter_version": "identity_ordered_pools_v1",
            "action_schema_version": "potential_actions_v1",
            "objective": {"id": "terminal_ordinary_potential_score_v1",
                          "direction": "maximize", "aggregation": "weighted_sum",
                          "weights": {"ordinary_potential_score": 1}},
            "value_semantics": actor.model.metadata.get("value_semantics"),
            "feature_version": FEATURE_VERSION, "network_version": MODEL_VERSION,
            "rules_version": sim.MODEL_VERSION, "policy": "direct_argmax",
            "checkpoint_rules_version": checkpoint_rules,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }

    @staticmethod
    def preset(value):
        if value is None or isinstance(value, Preset):
            return value
        if not isinstance(value, dict):
            value = {"name": "Maa preset", "pools": value}
        preset_type = DeploymentPreset if value.get("deployment_only", False) else Preset
        return preset_type.from_dict(value)

    @classmethod
    def initial(cls, preset):
        return sim.initial_from_pools(cls.preset(preset).pools)

    @staticmethod
    def state(value):
        return value if isinstance(value, PublicState) else sim.state_from_dict(value)

    @staticmethod
    def actions(state):
        return legal_actions(ValueModel.state(state))

    @staticmethod
    def prepare_actions(state, candidates, decision_context):
        from .potential_adapter import prepare_actions
        return prepare_actions(state, candidates, decision_context)

    @staticmethod
    def prepare_core(state, candidates, decision_context):
        from .potential_adapter import prepare_core
        return prepare_core(state, candidates, decision_context)

    def values(self, state, actions=None, preset=None):
        state = self.state(state)
        actions = legal_actions(state) if actions is None else tuple(actions)
        return self.actor.values(state, actions, self.preset(preset))

    def choose(self, state, actions=None, preset=None):
        return self.actor.choose(self.state(state), actions, self.preset(preset))

    @staticmethod
    def objective_current_value(state):
        return 6 + sim.score(state)

    def event_values(self, state, actions, preset=None, *, decision_context=None, routes=None):
        """Enumerate event effects, then value the next trained decision roots.

        Event candidate IDs never enter the frozen direct-Q network. Identical
        projected afterstates share one estimate. Outcome probabilities and the
        two possible bound support roles are exact; only forced future draws
        use the recorded common random samples.
        """
        started = perf_counter()
        context = decision_context or {}
        samples = int(context.get("samples", EVENT_VALUE_SAMPLES))
        routes = routes or ((state, Fraction(1)),)
        # A hidden floor is a distribution, not the representative route used by the tracker.
        if len(routes) == 1:
            seed_state = sim.to_jsonable(replace(routes[0][0], eventOptions=()))
        else:
            seed_state = sorted((canonical_seed(sim.to_jsonable(replace(route, eventOptions=()))),
                                 str(probability)) for route, probability in routes)
        seed = canonical_seed(EVENT_SEED_DOMAIN, seed_state)
        seeds = [canonical_seed(EVENT_SEED_DOMAIN, seed, index) for index in range(samples)]
        distributions = []
        afterstates = {}
        for action in actions:
            distribution = {}
            if action.kind == "SKIP_EVENT":
                outcomes = ((1, 1, 0, "none", "none"),)
            else:
                outcomes = sim.EVENT_DATA[state.eventId][2][action.index][1]
            for n, d, delta, rarity, scope in outcomes:
                if not n:
                    continue
                for route, route_probability in routes:
                    base = replace(route, phase="ADVANCE", eventId=0, eventOptions=(), coins=route.coins + delta)
                    probability = Fraction(n, d) * route_probability
                    if rarity == "none":
                        branches = ((base, probability),)
                    else:
                        roles = (2, 3) if scope == "support_random" else (0,)
                        branches = tuple((replace(base, phase="DRAW",
                            source=sim.Source(rarity, "bound" if role else "main" if scope == "main" else "any", role),
                            npc3EventReward=base.npc3EventReward or 3 <= route.node <= 6),
                            probability / len(roles)) for role in roles)
                    for after, mass in branches:
                        distribution[after] = distribution.get(after, Fraction()) + mass
                        afterstates.setdefault(after, [])
            distributions.append(distribution)

        roots = {}
        for after, observations in afterstates.items():
            for sample_seed in seeds:
                next_state = sim.advance_to_decision(after, Random(sample_seed)).state
                if next_state.phase == "END":
                    observations.append((None, sim.score(next_state) - sim.score(state)))
                else:
                    roots.setdefault(next_state, legal_actions(next_state))
                    observations.append((next_state, sim.score(next_state) - sim.score(state)))

        root_values = {}
        if roots:
            matrix = np.concatenate([encode_actions(root, root_actions, self.preset(preset))
                                     for root, root_actions in roots.items()])
            predictions = self.actor.model.predict(matrix)
            offset = 0
            for root, root_actions in roots.items():
                count = len(root_actions)
                root_values[root] = float(np.max(predictions[offset:offset + count]))
                offset += count
        continuation = {after: sum(reward + (root_values[root] if root is not None else 0.0)
                                    for root, reward in observations) / samples
                        for after, observations in afterstates.items()}
        values = np.asarray([sum(float(mass) * continuation[after] for after, mass in distribution.items())
                             for distribution in distributions], dtype=np.float64)
        return values, dict(evaluation_method=EVENT_VALUE_METHOD, evaluation_samples=samples,
                            evaluation_seed=f"{seed:016x}",
                            evaluation_sample_seeds=[f"{value:016x}" for value in seeds],
                            evaluation_outcomes="exact_probability_enumeration", evaluation_route_count=len(routes),
                            evaluation_afterstates=len(afterstates), evaluation_next_roots=len(roots),
                            evaluation_elapsed_ms=round((perf_counter() - started) * 1000, 3))

    def decide(self, state, actions=None, preset=None, *, decision_context=None, routes=None):
        state = self.state(state)
        actions = legal_actions(state) if actions is None else tuple(actions)
        if not actions:
            return {"action": None, "action_record": None, "terminal": True,
                    "ordinary_score": sim.score(state), "model_id": self.model_id,
                    "objective_current_value": self.objective_current_value(state),
                    "expected_total": self.objective_current_value(state), "values": []}
        event = decision_context is not None and decision_context.get("kind") == "event"
        if event:
            q, evaluation = self.event_values(state, actions, preset, decision_context=decision_context, routes=routes)
        elif routes:
            q = np.average([self.values(route, actions, preset) for route, _ in routes],
                           axis=0, weights=[float(p) for _, p in routes])
            evaluation = dict(evaluation_route_count=len(routes))
        else:
            q, evaluation = self.values(state, actions, preset), {}
        index = select_last_best(q) if event else int(np.argmax(q))
        selected = actions[index]
        current_value = self.objective_current_value(state)
        return {
            "action": sim.action_key(selected),
            "action_record": sim.to_jsonable(selected),
            "q": float(q[index]),
            "model_id": self.model_id,
            "checkpoint": self.checkpoint.name,
            "ordinary_score": sim.score(state),
            "objective_current_value": current_value,
            "expected_total": current_value + float(q[index]),
            "terminal": False,
            "decision_context": decision_context,
            "tie_break": "last_ui_index" if event else "first_action_index",
            **evaluation,
            "values": [
                {"action": sim.action_key(action), "action_record": sim.to_jsonable(action),
                 "predicted_value": float(value), "expected_total": current_value + float(value),
                 "expected_remaining_ordinary_return": float(value)}
                for action, value in zip(actions, q)
            ],
        }


__all__ = ["ValueModel", "PublicState", "Action", "Preset", "sim", "replace", "legal_actions"]
