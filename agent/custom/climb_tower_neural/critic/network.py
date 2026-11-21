"""Read-only FP32 NumPy inference for the frozen potential-only residual Q network."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .features import FEATURE_VERSION, TARGET_SCALE

MODEL_VERSION = "numpy_residual_direct_q_v1"


def _silu(x):
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
    return x * sigmoid


class QNetwork:
    """Direct Q^teacher(s,a) in ordinary effective level units.

    ``predict`` returns the raw estimate used for direct argmax. Parameters
    are loaded from the release checkpoint and remain read-only.
    """

    def __init__(self, input_dim, hidden, params, metadata):
        self.input_dim = int(input_dim)
        self.hidden = tuple(int(width) for width in hidden)
        self.params = params
        self.metadata = metadata
        for array in self.params.values():
            array.flags.writeable = False

    @property
    def parameter_count(self) -> int:
        return sum(array.size for array in self.params.values())

    def _inputs(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected [batch, {self.input_dim}] public/action features")
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite neural input")
        return x

    def forward_normalized(self, x):
        x = self._inputs(x)
        h = x
        for i in range(len(self.hidden)):
            z = h @ self.params[f"W{i}"] + self.params[f"b{i}"]
            h = _silu(z)
        y = h @ self.params["head_W"] + self.params["head_b"][0] + x @ self.params["skip_W"]
        return y

    def predict(self, x) -> np.ndarray:
        result = self.forward_normalized(x) * TARGET_SCALE
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite neural action value")
        return result

    predict_batch = predict

    def checksum(self) -> str:
        digest = hashlib.sha256()
        for key in sorted(self.params):
            array = self.params[key]
            digest.update(key.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.astype("<f4", copy=False).tobytes(order="C"))
        return digest.hexdigest()

    @classmethod
    def load(cls, path) -> "QNetwork":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__metadata__"].item()))
            if meta.get("model_version") != MODEL_VERSION or meta.get("feature_version") != FEATURE_VERSION:
                raise ValueError("Unsupported model/feature version")
            if meta.get("target_scale") != TARGET_SCALE:
                raise ValueError("Checkpoint target units disagree")
            shapes = {}
            previous = int(meta["input_dim"])
            for index, width in enumerate(meta["hidden"]):
                shapes[f"W{index}"] = (previous, width)
                shapes[f"b{index}"] = (width,)
                previous = width
            shapes.update(head_W=(previous,), head_b=(1,), skip_W=(meta["input_dim"],))
            if set(data.files) != {*shapes, "__metadata__"}:
                raise ValueError("Checkpoint parameter names disagree")
            params = {}
            for key, shape in shapes.items():
                value = np.asarray(data[key], dtype=np.float32)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid checkpoint parameter: {key}")
                params[key] = value.copy()
        obj = cls(meta["input_dim"], meta["hidden"], params, meta)
        if obj.checksum() != meta.get("parameter_sha256"):
            raise ValueError("Checkpoint parameter checksum mismatch")
        return obj
