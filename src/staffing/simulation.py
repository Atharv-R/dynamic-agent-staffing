"""Day-by-day simulation of the agent pool and the advertiser waiting pool.

Agent pool: an ``n_agents x capacity`` integer matrix. Each cell is the number of days until that
slot is free: 0 is free, ``support_days`` was just assigned, and a new hire's slots start at
``ramp_days``.

Advertiser pool: budgets and days waited, sorted by budget (highest first) and assigned greedily
to free slots.

Each day:
  1. today's sign-ups join the advertiser pool
  2. the policy decides hires and fires
  3. salaries are paid (fires cost ``notice_days`` of salary plus ``fire_penalty`` x salary)
  4. advertisers are assigned to free slots, earning ``uplift`` x budget
  5. occupied slots count down a day
  6. waiting advertisers age, and those who have waited ``max_wait`` days leave

A fired agent must have no slot with more than ``notice_days`` left, so they can finish their
current advertisers during the paid notice period. They take no new advertisers, so they leave
the matrix immediately (revenue is booked at assignment).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .config import StaffingParams


class Policy(Protocol):
    def decide(self, t: int, pool: np.ndarray, forecast: np.ndarray) -> tuple[int, int]:
        """Return ``(hires, fires)`` for day ``t``.

        ``forecast[t:]`` is forecast demand from today onwards; ``pool`` is the agent matrix.
        """
        ...


@dataclass
class SimulationResult:
    profit: float
    revenue: float
    cost: float
    daily: pd.DataFrame


def simulate(
    arrivals: Sequence[np.ndarray],
    dates: pd.DatetimeIndex,
    policy: Policy,
    forecast: pd.Series,
    initial_agents: int,
    salary: float,
    params: StaffingParams | None = None,
) -> SimulationResult:
    """Run ``policy`` over ``dates``.

    ``arrivals[t]`` holds the budgets of advertisers signing up on ``dates[t]``. ``forecast`` is
    the demand series the policy sees. It must start on or before ``dates[0]`` and be daily; it
    may run past the last date (lookahead windows are cut off where it ends).
    """
    if len(arrivals) != len(dates):
        raise ValueError("arrivals and dates must have the same length")
    params = params or StaffingParams()
    fc = _align_forecast(forecast, dates[0])

    cap = params.capacity
    daily_salary = salary / params.days_per_year
    fire_cost_each = params.notice_days * daily_salary + params.fire_penalty * salary

    pool = np.zeros((initial_agents, cap), dtype=np.int64)
    budgets = np.empty(0)
    waited = np.empty(0, dtype=np.int64)
    rows = []

    for t, new in enumerate(arrivals):
        budgets = np.concatenate([budgets, new])
        waited = np.concatenate([waited, np.zeros(len(new), dtype=np.int64)])
        order = np.argsort(-budgets, kind="stable")
        budgets, waited = budgets[order], waited[order]

        hires, fires = policy.decide(t, pool, fc)
        fired = 0
        if fires > 0:
            # Only agents whose current advertisers all finish within the notice period.
            can_fire = np.flatnonzero((pool > params.notice_days).sum(axis=1) == 0)[:fires]
            fired = len(can_fire)
            pool = np.delete(pool, can_fire, axis=0)
        if hires > 0:
            pool = np.vstack([pool, np.full((hires, cap), params.ramp_days, dtype=np.int64)])

        salary_cost = pool.shape[0] * daily_salary
        firing_cost = fired * fire_cost_each

        flat = pool.reshape(-1)
        free = np.flatnonzero(flat == 0)
        assigned = min(len(free), len(budgets))
        flat[free[:assigned]] = params.support_days
        revenue = params.uplift * budgets[:assigned].sum()
        budgets, waited = budgets[assigned:], waited[assigned:]

        np.subtract(pool, 1, out=pool, where=pool > 0)
        waited += 1
        stay = waited < params.max_wait
        expired = int((~stay).sum())
        budgets, waited = budgets[stay], waited[stay]

        rows.append(
            (dates[t], len(new), hires, fired, pool.shape[0],
             int((pool.sum(axis=1) == 0).sum()), assigned, expired, len(budgets),
             revenue, salary_cost, firing_cost)
        )

    daily = pd.DataFrame(
        rows,
        columns=["date", "signups", "hires", "fires", "agents", "idle_agents", "assigned",
                 "expired", "waiting", "revenue", "salary_cost", "firing_cost"],
    ).set_index("date")
    daily["cost"] = daily["salary_cost"] + daily["firing_cost"]
    daily["profit"] = daily["revenue"] - daily["cost"]
    revenue, cost = float(daily["revenue"].sum()), float(daily["cost"].sum())
    return SimulationResult(profit=revenue - cost, revenue=revenue, cost=cost, daily=daily)


def _align_forecast(forecast: pd.Series, start: pd.Timestamp) -> np.ndarray:
    idx = forecast.index
    if len(idx) == 0 or idx[0] > start:
        raise ValueError(f"forecast must start on or before {start.date()}")
    fc = forecast.loc[start:]
    if len(fc) > 1 and not (fc.index[1:] - fc.index[:-1] == pd.Timedelta(days=1)).all():
        raise ValueError("forecast must be a consecutive daily series")
    return fc.to_numpy(dtype=float)
