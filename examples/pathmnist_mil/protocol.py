"""Validated, hashable development protocol for the small PathMNIST MIL example."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class EvaluationProtocol:
    schema_version: int = 1
    seeds: tuple[int, ...] = (7,)
    noise_rates: tuple[float, ...] = (0.0, 0.15)
    max_steps: int = 160
    check_every: int = 20
    patience: int | None = 3
    min_delta: float = 0.001
    max_fit_seconds: float = 20.0
    max_parameters: int = 20000
    batch_size: int = 32
    learning_rate: float | None = None
    label_smoothing: float | None = None
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 5.0
    num_threads: int = 2

    def __post_init__(self):
        for name in ("schema_version", "max_steps", "check_every", "max_parameters", "batch_size", "num_threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.schema_version != 1:
            raise ValueError("Unsupported protocol schema_version")
        if self.patience is not None and (type(self.patience) is not int or self.patience < 1):
            raise ValueError("patience must be a positive integer or null (disabled)")
        if not isinstance(self.seeds, (list, tuple)) or not self.seeds:
            raise ValueError("seeds must be a nonempty array")
        if any(type(seed) is not int or not 0 <= seed < 2**32 - 1000 for seed in self.seeds):
            raise ValueError("seeds must be integers in [0, 2**32 - 1000)")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if not isinstance(self.noise_rates, (list, tuple)) or not self.noise_rates:
            raise ValueError("noise_rates must be a nonempty array")
        for value in self.noise_rates:
            _bounded("noise_rates", value, 0.0, 0.5)
        if len(set(self.noise_rates)) != len(self.noise_rates):
            raise ValueError("noise_rates must be unique")
        _bounded("min_delta", self.min_delta, 0.0, 1.0)
        _bounded("weight_decay", self.weight_decay, 0.0, None)
        for name in ("max_fit_seconds", "gradient_clip_norm"):
            _bounded(name, getattr(self, name), 0.0, None, strict_lower=True)
        if self.learning_rate is not None:
            _bounded("learning_rate", self.learning_rate, 0.0001, 0.01)
        if self.label_smoothing is not None:
            _bounded("label_smoothing", self.label_smoothing, 0.0, 0.2)
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "noise_rates", tuple(float(x) for x in self.noise_rates))

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @property
    def sha256(self):
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode()).hexdigest()


def _bounded(name, value, lower, upper, *, strict_lower=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if value < lower or (strict_lower and value == lower) or (upper is not None and value > upper):
        raise ValueError(f"{name} outside the declared bounds")


def load_protocol(path: str | Path | None = None, *, seeds=None) -> EvaluationProtocol:
    values = {} if path is None else json.loads(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError("Evaluation protocol must be a JSON object")
    unknown = set(values) - {field.name for field in fields(EvaluationProtocol)}
    if unknown:
        raise ValueError(f"Unknown protocol fields: {', '.join(sorted(unknown))}")
    protocol = EvaluationProtocol(**values)
    if seeds is not None:
        if path is not None:
            raise ValueError("Do not override seeds from a protocol file; create a new protocol")
        protocol = replace(protocol, seeds=tuple(seeds))
    return protocol
