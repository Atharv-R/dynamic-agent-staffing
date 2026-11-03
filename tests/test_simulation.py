import numpy as np
import pandas as pd
import pytest

from staffing.config import StaffingParams
from staffing.policies import BaselinePolicy, LookaheadPolicy
from staffing.simulation import simulate

PARAMS = StaffingParams()


class Fixed:
    """Policy that hires/fires a fixed amount on given days."""

    def __init__(self, hires=None, fires=None):
        self.hires, self.fires = hires or {}, fires or {}

    def decide(self, t, pool, forecast):
        assert pool.size == 0 or (pool.min() >= 0 and pool.max() <= PARAMS.support_days)
        return self.hires.get(t, 0), self.fires.get(t, 0)


def run(arrivals, policy, agents=1, salary=36_500.0, days=None):
    days = days or len(arrivals)
    dates = pd.date_range("2023-01-01", periods=days)
    arrivals = arrivals + [np.empty(0)] * (days - len(arrivals))
    forecast = pd.Series(1.0, index=dates)
    return simulate(arrivals, dates, policy, forecast, agents, salary, PARAMS)


def test_no_staff_no_revenue_and_waiting_advertisers_expire():
    res = run([np.array([100.0, 200.0])], Fixed(), agents=0, days=70)
    assert res.revenue == 0 and res.cost == 0
    d = res.daily
    assert d["waiting"].iloc[58] == 2 and d["waiting"].iloc[59] == 0
    assert d["expired"].sum() == 2


def test_capacity_limits_assignment_and_highest_budgets_go_first():
    budgets = np.arange(1.0, 13.0)  # 12 advertisers, one agent with 10 slots
    res = run([budgets], Fixed(), days=1)
    assert res.daily["assigned"].iloc[0] == 10
    assert res.revenue == pytest.approx(PARAMS.uplift * budgets[2:].sum())


def test_slots_free_after_support_period():
    res = run([np.ones(10), np.ones(1)], Fixed(), days=62)
    assigned = res.daily["assigned"]
    # The 11th advertiser waits until the agent's slots free up after 60 days.
    assert assigned.iloc[0] == 10 and assigned.iloc[1:60].sum() == 0 and assigned.iloc[60] == 1


def test_new_hires_ramp_up_before_taking_advertisers():
    arrivals = [np.ones(5)]
    res = run(arrivals, Fixed(hires={0: 1}), agents=0, days=32)
    assigned = res.daily["assigned"]
    assert assigned.iloc[:30].sum() == 0 and assigned.iloc[30] == 5


def test_cost_accounting():
    salary = 36_500.0  # 100 a day
    res = run([], Fixed(hires={0: 2}, fires={5: 1}), agents=1, salary=salary, days=10)
    d = res.daily
    assert d["salary_cost"].iloc[0] == pytest.approx(300)  # hires are paid from day one
    assert d["salary_cost"].iloc[5] == pytest.approx(200)
    fire_cost = PARAMS.notice_days * 100 + PARAMS.fire_penalty * salary
    assert d["firing_cost"].sum() == pytest.approx(fire_cost)
    assert res.cost == pytest.approx(3 * 5 * 100 + 2 * 5 * 100 + fire_cost)
    assert res.profit == pytest.approx(res.revenue - res.cost)


def test_cannot_fire_agent_with_long_commitments():
    res = run([np.ones(1)], Fixed(fires={1: 1, 31: 1}), agents=1, days=35)
    fires = res.daily["fires"]
    assert fires.iloc[1] == 0  # slot has 58 days left, more than the notice period
    assert fires.iloc[31] == 1  # 29 days left: can work out the notice period


@pytest.mark.parametrize(
    "policy",
    [BaselinePolicy(hire=5, fire=3, threshold=20), LookaheadPolicy(alpha=1.5, beta=1.0)],
)
def test_invariants_on_synthetic_year(dataset, policy):
    dates = pd.date_range("2023-01-01", "2023-12-31")
    info = dataset.country_info("USA")
    res = simulate(dataset.arrivals("USA", dates), dates, policy, dataset.daily_demand("USA"),
                   info.existing_agents, info.salary, PARAMS)
    d = res.daily
    assert (d[["agents", "idle_agents", "assigned", "waiting", "expired"]] >= 0).all().all()
    assert (d["idle_agents"] <= d["agents"]).all()
    assert d["assigned"].sum() + d["expired"].sum() + d["waiting"].iloc[-1] == d["signups"].sum()
    assert (d["agents"].diff().iloc[1:] == (d["hires"] - d["fires"]).iloc[1:]).all()
    assert res.revenue == pytest.approx(d["revenue"].sum())


def test_deterministic(dataset):
    dates = pd.date_range("2023-01-01", "2023-06-30")
    info = dataset.country_info("USA")
    args = (dataset.arrivals("USA", dates), dates, LookaheadPolicy(2.0),
            dataset.daily_demand("USA"), info.existing_agents, info.salary)
    pd.testing.assert_frame_equal(simulate(*args).daily, simulate(*args).daily)


def test_forecast_must_cover_start():
    dates = pd.date_range("2023-01-10", periods=3)
    late = pd.Series(1.0, index=pd.date_range("2023-01-11", periods=5))
    with pytest.raises(ValueError, match="start"):
        simulate([np.empty(0)] * 3, dates, Fixed(), late, 1, 1.0)
