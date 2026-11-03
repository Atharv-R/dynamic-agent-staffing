import numpy as np
import pandas as pd
import pytest

from staffing.forecast import (
    SARIMA,
    SeasonalBlend,
    SeasonalNaive,
    forecast_errors,
    make_forecaster,
)

TRAIN = pd.date_range("2023-01-01", "2023-12-31")
TEST = pd.date_range("2024-01-01", "2024-12-31")


@pytest.fixture
def series(dataset):
    return dataset.daily_demand("USA")


@pytest.mark.parametrize("name", ["seasonal_naive", "seasonal_blend", "sarima"])
def test_forecast_ignores_test_period_actuals(series, cfg, name):
    history = series[TRAIN]
    tampered = series.copy()
    tampered[TEST] = 0.0
    a = make_forecaster(name, **cfg.forecasters[name]).fit(history).predict(TEST)
    b = make_forecaster(name, **cfg.forecasters[name]).fit(tampered[TRAIN]).predict(TEST)
    pd.testing.assert_series_equal(a, b)
    assert a.index.equals(TEST) and np.isfinite(a).all()


@pytest.mark.parametrize("cls", [SeasonalNaive, SeasonalBlend])
def test_cannot_predict_inside_history(series, cls):
    model = cls().fit(series[:"2024-03-31"])
    with pytest.raises(ValueError, match="out of sample"):
        model.predict(TEST)


def test_seasonal_naive_is_last_year_with_leap_day_from_feb_28(series):
    fc = SeasonalNaive().fit(series[TRAIN]).predict(TEST)
    assert fc["2024-07-04"] == series["2023-07-04"]
    assert fc["2024-02-29"] == series["2023-02-28"]
    assert fc["2024-03-01"] == series["2023-03-01"]


def test_seasonal_naive_uses_most_recent_year(series):
    fc = SeasonalNaive().fit(series).predict(pd.date_range("2025-01-01", periods=3))
    assert (fc.to_numpy() == series["2024-01-01":"2024-01-03"].to_numpy()).all()


def test_seasonal_blend_recursion():
    history = pd.Series(10.0, index=TRAIN)
    history["2023-01-02"] = 40.0
    fc = SeasonalBlend(weight=0.8, window=30).fit(history).predict(TEST[:3])
    assert fc.iloc[0] == pytest.approx(10.0)
    assert fc.iloc[1] == pytest.approx(0.8 * 40 + 0.2 * 10)
    assert fc.iloc[2] == pytest.approx(0.8 * 10 + 0.2 * np.mean([10, 34]))


def test_sarima_picks_an_order_and_is_nonnegative(series, cfg):
    model = SARIMA(**cfg.forecasters["sarima"]).fit(series[TRAIN])
    fc = model.predict(pd.date_range("2024-02-01", periods=10))
    assert len(model.order) == 3 and (fc >= 0).all()


def test_forecast_errors():
    idx = pd.date_range("2024-01-01", periods=3)
    errs = forecast_errors(pd.Series([10.0, 0.0, 20.0], idx), pd.Series([12.0, 1.0, 15.0], idx))
    assert errs["mae"] == pytest.approx(8 / 3)
    assert errs["mape"] == pytest.approx((20 + 25) / 2)
