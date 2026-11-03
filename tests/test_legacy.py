"""The legacy model must reproduce the team's original simulator exactly."""

import numpy as np
import pandas as pd
import pytest
from legacy_reference import run_strategy

from staffing.evaluate import LEGACY, Scenario
from staffing.policies import BaselinePolicy, LookaheadPolicy


@pytest.fixture(scope="module")
def legacy(dataset, cfg):
    sc = Scenario(dataset, "USA", LEGACY, cfg)
    budget_demand = pd.concat(
        [sc.demand.rename("Demand"), dataset.daily_budget("USA").rename("Budget")], axis=1
    )
    return sc, budget_demand


def last_year(budget_demand):
    """The notebook's 2024 forecast: 2023 with Feb 28 duplicated for the leap day."""
    fc = budget_demand[:"2023-12-31"]
    fc = pd.concat([fc[:"2023-02-28"], fc[fc.index == "2023-02-28"], fc["2023-03-01":]])
    fc.index = budget_demand["2024-01-01":].index
    return fc


@pytest.mark.parametrize(
    "kind, kwargs, year",
    [
        ("baseline", {"H": 4, "F": 2}, 2023),
        ("baseline", {"H": 1, "F": 1}, 2024),
        ("main", {"alpha": 2.16, "beta": 1.0}, 2023),
        ("main", {"alpha": 2.16, "beta": 1.0}, 2024),
        ("main", {"alpha": 1.3, "beta": 0.8}, 2023),
    ],
)
def test_matches_original_run_strategy(legacy, kind, kwargs, year):
    sc, bd = legacy
    threshold = bd["Demand"][:365].mean()
    forecast = bd if year == 2023 else last_year(bd)
    profit, agents, idle = run_strategy(
        threshold, bd, forecast, sc.info.existing_agents, sc.info.salary,
        strategy_type=kind, start_date=f"{year}-01-01", **kwargs,
    )
    if kind == "baseline":
        policy = BaselinePolicy(kwargs["H"], kwargs["F"], threshold)
    else:
        policy = LookaheadPolicy(kwargs["alpha"], kwargs["beta"], legacy=True)
    res = sc.run(policy, forecast["Demand"], year)
    assert res.profit == pytest.approx(profit, rel=1e-12)
    np.testing.assert_array_equal(res.daily["agents"], agents[1:])
    np.testing.assert_array_equal(res.daily["idle_agents"], idle[1:])


def test_firing_bug_changes_outcomes(legacy):
    sc, bd = legacy
    buggy = sc.run(LookaheadPolicy(1.3, 0.8, legacy=True), bd["Demand"], 2023)
    fixed = sc.run(LookaheadPolicy(1.3, 0.8), bd["Demand"], 2023)
    assert buggy.daily["fires"].max() > fixed.daily["fires"].max()
