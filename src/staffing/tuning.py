"""Grid search over policy parameters."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from .simulation import Policy, SimulationResult


@dataclass
class TuningResult:
    best: dict[str, float]
    profit: float
    grid: pd.DataFrame  # one row per parameter combination
    at_edge: list[str]  # parameters whose best value is the smallest or largest tried


def grid_search(
    make_policy: Callable[..., Policy],
    grid: Mapping[str, Sequence[float]],
    run: Callable[[Policy], SimulationResult],
) -> TuningResult:
    """Try every combination in ``grid`` (first key is the outer loop).

    Ties keep the first combination found, as in the original notebooks.
    """
    names = list(grid)
    rows = []
    best, best_profit = None, float("-inf")
    for values in itertools.product(*(grid[n] for n in names)):
        combo = dict(zip(names, values, strict=True))
        profit = run(make_policy(**combo)).profit
        rows.append({**combo, "profit": profit})
        if profit > best_profit:
            best, best_profit = combo, profit
    at_edge = [
        n for n in names
        if len(grid[n]) > 1 and best[n] in (min(grid[n]), max(grid[n]))
    ]
    return TuningResult(best=best, profit=best_profit, grid=pd.DataFrame(rows), at_edge=at_edge)
