"""Daily hire/fire policies. See ``simulation.Policy`` for the interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaselinePolicy:
    """Hire ``hire`` agents when mean forecast demand over today and the next ``window`` days
    exceeds ``threshold`` (the training-period mean); otherwise fire up to ``fire``."""

    hire: int
    fire: int
    threshold: float
    window: int = 60

    def decide(self, t: int, pool: np.ndarray, forecast: np.ndarray) -> tuple[int, int]:
        upcoming = forecast[t : t + self.window + 1]
        if len(upcoming) and upcoming.mean() > self.threshold:
            return self.hire, 0
        return 0, self.fire


@dataclass
class LookaheadPolicy:
    """Match capacity to the next ``horizon`` days of forecast demand.

    D = forecast demand over today and the next ``horizon`` days
    S = slots that free up within ``horizon`` days
    If D > beta * S, hire (D - beta * S) / capacity agents.
    Else if S > alpha * D, fire (S - alpha * D) / capacity agents.

    ``alpha`` tolerates spare capacity (firing is expensive); ``beta`` sets how eagerly to hire.

    ``legacy=True`` reproduces a bug in the original firing loop: when the computed number to fire
    rounded down to 0, the loop's ``count == fire`` break never triggered, so every fireable agent
    was fired.
    """

    alpha: float
    beta: float = 1.0
    horizon: int = 30
    capacity: int = 10
    legacy: bool = False

    def decide(self, t: int, pool: np.ndarray, forecast: np.ndarray) -> tuple[int, int]:
        demand = forecast[t : t + self.horizon + 1].sum()
        slots = int(np.count_nonzero(pool < self.horizon))
        if demand > self.beta * slots:
            return int((demand - self.beta * slots) / self.capacity), 0
        if slots > self.alpha * demand:
            fire = int((slots - self.alpha * demand) / self.capacity)
            if self.legacy and fire == 0:
                fire = pool.shape[0]
            return 0, fire
        return 0, 0
