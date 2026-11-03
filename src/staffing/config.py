"""Model parameters and experiment configuration, loaded from ``configs/default.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass(frozen=True)
class StaffingParams:
    """Rules of the staffing problem (see the case brief)."""

    capacity: int = 10
    support_days: int = 60
    ramp_days: int = 30
    notice_days: int = 30
    fire_penalty: float = 0.4
    uplift: float = 0.135
    max_wait: int = 60
    days_per_year: int = 365

    def __post_init__(self) -> None:
        if self.capacity < 1 or self.support_days < 1 or self.max_wait < 1:
            raise ValueError("capacity, support_days and max_wait must be positive")
        if not 0 <= self.ramp_days <= self.support_days:
            raise ValueError("ramp_days must lie in [0, support_days]")


@dataclass(frozen=True)
class Config:
    params: StaffingParams = field(default_factory=StaffingParams)
    country: str = "USA"
    train_year: int = 2023
    test_year: int = 2024
    baseline: dict[str, Any] = field(default_factory=dict)
    lookahead: dict[str, Any] = field(default_factory=dict)
    forecasters: dict[str, dict[str, Any]] = field(default_factory=dict)
    seed: int = 7

    def baseline_grid(self) -> dict[str, list[int]]:
        return {name: _int_range(self.baseline[name]) for name in ("hire", "fire")}

    def lookahead_grid(self) -> dict[str, list[float]]:
        # beta is the outer loop, matching the original notebook's search order.
        return {name: _linspace(self.lookahead[name]) for name in ("beta", "alpha")}


def load_config(path: str | Path | None = None) -> Config:
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    known = {f.name for f in fields(StaffingParams)}
    unknown = set(raw.get("params", {})) - known
    if unknown:
        raise ValueError(f"unknown params in config: {sorted(unknown)}")
    experiment = raw.get("experiment", {})
    policies = raw.get("policies", {})
    return Config(
        params=StaffingParams(**raw.get("params", {})),
        country=experiment.get("country", "USA"),
        train_year=int(experiment.get("train_year", 2023)),
        test_year=int(experiment.get("test_year", 2024)),
        baseline=policies.get("baseline", {}),
        lookahead=policies.get("lookahead", {}),
        forecasters=raw.get("forecasters", {}),
        seed=int(raw.get("synthetic", {}).get("seed", 7)),
    )


def _int_range(spec: list[int]) -> list[int]:
    lo, hi = spec
    return list(range(int(lo), int(hi) + 1))


def _linspace(spec: dict[str, float] | list[float]) -> list[float]:
    if isinstance(spec, list):
        return [float(v) for v in spec]
    return [float(v) for v in np.linspace(spec["start"], spec["stop"], int(spec["num"]))]
