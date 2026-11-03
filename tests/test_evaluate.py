import pandas as pd
import pytest

from staffing.config import load_config
from staffing.evaluate import CORRECTED, LEGACY, Scenario, run_experiment
from staffing.tuning import grid_search


@pytest.fixture(scope="module")
def experiment(dataset, cfg):
    return run_experiment(dataset, cfg, CORRECTED, forecasters=["seasonal_naive", "seasonal_blend"])


def test_experiment_table(experiment):
    r = experiment.results
    assert set(r["policy"]) == {"baseline", "lookahead"}
    assert set(r["forecaster"]) == {"perfect_foresight", "seasonal_naive", "seasonal_blend"}
    assert (r["profit"] == r["revenue"] - r["cost"]).all()
    assert len(experiment.forecast_errors) == 2


def test_forecasts_extend_past_test_year(experiment):
    fc = experiment.forecasts["seasonal_naive"]
    assert fc.index[0] == pd.Timestamp("2024-01-01")
    assert fc.index[-1] > pd.Timestamp("2024-12-31")


def test_corrected_runs_whole_year_and_legacy_365_days(dataset, cfg):
    assert len(Scenario(dataset, "USA", CORRECTED, cfg).sim_dates(2024)) == 366
    assert len(Scenario(dataset, "USA", LEGACY, cfg).sim_dates(2024)) == 365


def test_experiment_is_deterministic(dataset, cfg, experiment):
    again = run_experiment(dataset, cfg, CORRECTED, forecasters=list(experiment.forecasts))
    pd.testing.assert_frame_equal(experiment.results, again.results)


def test_grid_search_reports_edges(dataset, cfg):
    sc = Scenario(dataset, "USA", CORRECTED, cfg)
    res = grid_search(sc.lookahead(), {"beta": [1.0], "alpha": [50.0, 60.0]},
                      lambda p: sc.run(p, sc.demand, 2023))
    assert res.at_edge == ["alpha"] and len(res.grid) == 2
    assert res.profit == res.grid["profit"].max()


def test_default_config_loads():
    cfg = load_config()
    assert cfg.params.capacity == 10 and cfg.params.uplift == 0.135
    assert list(cfg.lookahead_grid()) == ["beta", "alpha"]
    assert cfg.baseline_grid()["hire"][0] == 0
