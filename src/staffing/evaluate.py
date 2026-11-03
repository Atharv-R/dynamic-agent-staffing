"""Experiments: tune on the training year, test on the next year with each forecaster.

Two model specifications are provided:

``LEGACY`` reproduces the original notebooks: demand counts every sign-up, all of a day's
advertisers share the day's mean eligible budget, the lookahead firing bug is kept, 365 days are
simulated and forecasts stop at the end of the test year.

``CORRECTED`` counts eligible sign-ups only, gives each advertiser their own budget, fixes the
firing bug, simulates the whole year and extends forecasts past year end so the last lookahead
windows are complete.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .config import Config
from .data import ArrivalMode, Dataset
from .forecast import (
    SeasonalNaive,
    calendar_profile,
    forecast_errors,
    make_forecaster,
    seasonal_blend,
)
from .policies import BaselinePolicy, LookaheadPolicy
from .simulation import Policy, SimulationResult, simulate
from .tuning import TuningResult, grid_search

PERFECT_FORESIGHT = "perfect_foresight"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    arrivals: ArrivalMode
    legacy_fire_bug: bool
    full_year: bool
    extend_forecast: bool

    @property
    def eligible_demand(self) -> bool:
        return self.arrivals != "legacy"


LEGACY = ModelSpec("legacy", "legacy", legacy_fire_bug=True, full_year=False, extend_forecast=False)
CORRECTED = ModelSpec(
    "corrected", "individual", legacy_fire_bug=False, full_year=True, extend_forecast=True
)

# Apply the fixes one at a time, from the original model to the corrected one.
ABLATION = (
    LEGACY,
    replace(LEGACY, name="+ firing bug fixed", legacy_fire_bug=False),
    replace(LEGACY, name="+ eligible demand only", legacy_fire_bug=False, arrivals="averaged"),
    replace(LEGACY, name="+ individual budgets", legacy_fire_bug=False, arrivals="individual"),
    replace(CORRECTED, name="+ full year, extended forecasts (corrected)"),
)


class Scenario:
    """One country's data under one model specification."""

    def __init__(self, dataset: Dataset, country: str, spec: ModelSpec, cfg: Config) -> None:
        self.dataset = dataset
        self.country = country
        self.spec = spec
        self.cfg = cfg
        self.info = dataset.country_info(country)
        self.demand = dataset.daily_demand(country, eligible_only=spec.eligible_demand)
        self._arrivals: dict[int, list[np.ndarray]] = {}

    def year_dates(self, year: int) -> pd.DatetimeIndex:
        return pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")

    def sim_dates(self, year: int) -> pd.DatetimeIndex:
        dates = self.year_dates(year)
        return dates if self.spec.full_year else dates[:365]

    def run(self, policy: Policy, forecast: pd.Series, year: int) -> SimulationResult:
        dates = self.sim_dates(year)
        if year not in self._arrivals:
            self._arrivals[year] = self.dataset.arrivals(self.country, dates, self.spec.arrivals)
        return simulate(
            self._arrivals[year], dates, policy, forecast,
            self.info.existing_agents, self.info.salary, self.cfg.params,
        )

    def threshold(self, year: int) -> float:
        return float(self.demand[self.year_dates(year)].mean())

    def baseline(self, threshold: float):
        window = int(self.cfg.baseline.get("window", 60))
        return lambda hire, fire: BaselinePolicy(int(hire), int(fire), threshold, window)

    def lookahead(self):
        horizon = int(self.cfg.lookahead.get("horizon", 30))
        return lambda alpha, beta: LookaheadPolicy(
            alpha, beta, horizon, self.cfg.params.capacity, legacy=self.spec.legacy_fire_bug
        )

    def oracle(self, year: int) -> pd.Series:
        """Actual demand as a perfect forecast.

        With ``extend_forecast``, days past the end of the data are padded with last-year values.
        Otherwise lookahead windows near the data end shrink, and the policy fires staff into the
        year end, which is an artifact of where the data stops.
        """
        end = self.forecast_dates(year)[-1]
        last = self.demand.index[-1]
        if not self.spec.extend_forecast or end <= last:
            return self.demand
        pad_dates = pd.date_range(last + pd.Timedelta(days=1), end, freq="D")
        pad = SeasonalNaive().fit(self.demand).predict(pad_dates)
        return pd.concat([self.demand, pad])

    def forecast_dates(self, year: int) -> pd.DatetimeIndex:
        dates = self.year_dates(year)
        if self.spec.extend_forecast:
            extra = max(int(self.cfg.baseline.get("window", 60)),
                        int(self.cfg.lookahead.get("horizon", 30)))
            dates = pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=extra), freq="D")
        return dates


@dataclass
class ExperimentResult:
    spec: ModelSpec
    country: str
    results: pd.DataFrame
    forecast_errors: pd.DataFrame
    tuning: dict[str, TuningResult]
    forecasts: dict[str, pd.Series]
    daily: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)


def run_experiment(
    dataset: Dataset,
    cfg: Config,
    spec: ModelSpec = CORRECTED,
    country: str | None = None,
    policies: Iterable[str] = ("baseline", "lookahead"),
    forecasters: Iterable[str] | None = None,
) -> ExperimentResult:
    """Tune each policy on the training year with perfect foresight (an upper bound), then test
    the tuned parameters on the test year with each forecaster fit only on earlier data."""
    country = country or cfg.country
    sc = Scenario(dataset, country, spec, cfg)
    train, test = cfg.train_year, cfg.test_year
    names = list(forecasters if forecasters is not None else cfg.forecasters)

    history = sc.demand[: pd.Timestamp(f"{test}-01-01") - pd.Timedelta(days=1)]
    actual_test = sc.demand[sc.year_dates(test)]
    forecasts: dict[str, pd.Series] = {}
    error_rows = []
    for name in names:
        fc = make_forecaster(name, **cfg.forecasters.get(name, {})).fit(history)
        forecasts[name] = fc.predict(sc.forecast_dates(test))
        error_rows.append({"forecaster": name, **forecast_errors(actual_test, forecasts[name])})

    threshold = sc.threshold(train)
    makers = {"baseline": sc.baseline(threshold), "lookahead": sc.lookahead()}
    grids = {"baseline": cfg.baseline_grid(), "lookahead": cfg.lookahead_grid()}

    train_oracle = sc.oracle(train)
    rows, tuning, daily = [], {}, {}
    for policy in policies:
        make = makers[policy]
        tuned = grid_search(make, grids[policy], lambda p: sc.run(p, train_oracle, train))
        tuning[policy] = tuned
        best = make(**tuned.best)
        label = _format_params(tuned.best)
        edge = ", ".join(tuned.at_edge)

        runs = [(f"train {train}", PERFECT_FORESIGHT, train, train_oracle),
                (f"test {test}", PERFECT_FORESIGHT, test, sc.oracle(test))]
        runs += [(f"test {test}", name, test, forecasts[name]) for name in names]
        for period, fc_name, year, fc in runs:
            res = sc.run(best, fc, year)
            if year == test:
                daily[(policy, fc_name)] = res.daily
            rows.append({
                "policy": policy, "period": period, "forecaster": fc_name, "params": label,
                "params_at_grid_edge": edge, "profit": res.profit, "revenue": res.revenue,
                "cost": res.cost, "final_agents": int(res.daily["agents"].iloc[-1]),
                "expired": int(res.daily["expired"].sum()),
            })

    return ExperimentResult(
        spec=spec, country=country, results=pd.DataFrame(rows),
        forecast_errors=pd.DataFrame(error_rows), tuning=tuning, forecasts=forecasts, daily=daily,
    )


def run_ablation(
    dataset: Dataset, cfg: Config, forecaster: str = "seasonal_naive", country: str | None = None
) -> pd.DataFrame:
    """Test-year profit as each fix is applied in turn (each step is re-tuned)."""
    rows = []
    for spec in ABLATION:
        exp = run_experiment(dataset, cfg, spec, country, forecasters=[forecaster])
        test = exp.results[exp.results["forecaster"] == forecaster]
        for _, r in test.iterrows():
            rows.append({"model": spec.name, "policy": r["policy"], "params": r["params"],
                         "profit": r["profit"]})
    return pd.DataFrame(rows)


# Results reported in the team's Main.ipynb (real data, USA).
ORIGINAL_RESULTS = {
    "baseline_train_pf": 205_952_346.40,
    "lookahead_train_pf": 301_364_815.17,
    "baseline_test_last_year": 206_450_371.42,
    "lookahead_test_last_year": 282_370_015.50,
    "baseline_test_leaky_blend": 292_709_463.71,
    "lookahead_test_leaky_blend": 294_466_597.12,
}


def reproduce_original(dataset: Dataset, cfg: Config, country: str = "USA") -> pd.DataFrame:
    """Re-run the original notebook's experiments with its grids and choices.

    The last two rows reproduce the leaky setup: the "ARIMA" blend averages 2023 *and 2024*
    actuals per calendar day, and alpha is tuned directly on the 2024 test year.
    """
    cfg = replace(cfg, baseline={**cfg.baseline, "window": 60},
                  lookahead={**cfg.lookahead, "horizon": 30})
    sc = Scenario(dataset, country, LEGACY, cfg)
    train, test = 2023, 2024
    threshold = sc.threshold(train)
    last_year = make_forecaster("seasonal_naive").fit(sc.demand[sc.year_dates(train)])
    last_year_fc = last_year.predict(sc.year_dates(test))
    # Deliberately leaky (see docstring); not available through the Forecaster interface.
    leaky_fc = pd.Series(
        seasonal_blend(calendar_profile(sc.demand), sc.year_dates(test)), index=sc.year_dates(test)
    )

    base = grid_search(sc.baseline(threshold), {"hire": range(1, 20), "fire": range(1, 5)},
                       lambda p: sc.run(p, sc.demand, train))
    look = grid_search(sc.lookahead(),
                       {"beta": np.linspace(0.5, 2, 4), "alpha": np.linspace(1, 20, 50)},
                       lambda p: sc.run(p, sc.demand, train))
    leaky_threshold = float(leaky_fc.iloc[:365].mean())
    leaky_look = grid_search(sc.lookahead(), {"beta": [1.0], "alpha": np.linspace(1.1, 30, 100)},
                             lambda p: sc.run(p, leaky_fc, test))

    look_policy = sc.lookahead()
    rows = [
        ("Baseline, tuned on 2023 (perfect foresight)", "baseline_train_pf", base.best,
         base.profit),
        ("Lookahead, tuned on 2023 (perfect foresight)", "lookahead_train_pf", look.best,
         look.profit),
        ("Baseline, 2024, last year as forecast", "baseline_test_last_year", base.best,
         sc.run(sc.baseline(threshold)(**base.best), last_year_fc, test).profit),
        ("Lookahead, 2024, last year as forecast", "lookahead_test_last_year",
         {"alpha": 2.16, "beta": 1.0},
         sc.run(look_policy(alpha=2.16, beta=1.0), last_year_fc, test).profit),
        ("Baseline, 2024, leaky blend forecast", "baseline_test_leaky_blend",
         {"hire": 3, "fire": 1},
         sc.run(sc.baseline(leaky_threshold)(hire=3, fire=1), leaky_fc, test).profit),
        ("Lookahead, 2024, leaky blend, alpha tuned on 2024", "lookahead_test_leaky_blend",
         leaky_look.best, leaky_look.profit),
    ]
    return pd.DataFrame(
        [{"setting": s, "params": _format_params(p), "original_real_data": ORIGINAL_RESULTS[k],
          "reproduced": v} for s, k, p, v in rows]
    )


def _format_params(params: dict[str, float]) -> str:
    return ", ".join(
        f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()
    )
