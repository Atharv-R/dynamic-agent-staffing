"""Daily demand forecasters.

Every forecaster is fit on a history series and can only predict dates strictly after the end of
that history, so test-period actuals cannot leak into a forecast.
"""

from __future__ import annotations

import itertools
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Forecaster(ABC):
    name: str = "forecaster"

    def __init__(self) -> None:
        self.history_end: pd.Timestamp | None = None

    def fit(self, history: pd.Series) -> Forecaster:
        _check_daily(history)
        if history.isna().any():
            raise ValueError("history contains NaN")
        self.history_end = history.index[-1]
        self._fit(history)
        return self

    def predict(self, dates: pd.DatetimeIndex) -> pd.Series:
        if self.history_end is None:
            raise RuntimeError(f"{self.name} must be fit before predicting")
        dates = pd.DatetimeIndex(dates)
        if len(dates) and dates.min() <= self.history_end:
            raise ValueError(
                f"{self.name} was fit on data up to {self.history_end.date()} and cannot "
                f"predict {dates.min().date()} (forecasts must be out of sample)"
            )
        return pd.Series(self._predict(dates), index=dates, name="forecast").rename_axis("date")

    @abstractmethod
    def _fit(self, history: pd.Series) -> None: ...

    @abstractmethod
    def _predict(self, dates: pd.DatetimeIndex) -> np.ndarray: ...


class SeasonalNaive(Forecaster):
    """Same calendar day in the most recent year of history ("last year as forecast").

    Feb 29 takes Feb 28's value when the history year has no leap day, which is how the original
    analysis aligned 2023 to 2024.
    """

    name = "seasonal_naive"

    def _fit(self, history: pd.Series) -> None:
        last_year = history[history.index > history.index[-1] - pd.DateOffset(years=1)]
        self.profile = calendar_profile(last_year)

    def _predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return _lookup(self.profile, dates)


class SeasonalBlend(Forecaster):
    """The original "ARIMA" forecast, which is actually a seasonal blend.

    forecast[d] = weight * (historical mean for d's calendar day)
                + (1 - weight) * (mean of the previous ``window`` forecasts)
    """

    name = "seasonal_blend"

    def __init__(self, weight: float = 0.8, window: int = 30) -> None:
        super().__init__()
        self.weight = weight
        self.window = window

    def _fit(self, history: pd.Series) -> None:
        self.profile = calendar_profile(history)

    def _predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return seasonal_blend(self.profile, dates, self.weight, self.window)


class SARIMA(Forecaster):
    """Seasonal ARIMA with weekly seasonality and Fourier terms for the annual cycle.

    Orders are chosen by AIC over a small grid. With a single year of history the annual cycle
    can't be estimated as a seasonal lag (s=365), so it enters as ``fourier_order`` sine/cosine
    regressors in day-of-year.
    """

    name = "sarima"

    def __init__(
        self,
        seasonal_period: int = 7,
        fourier_order: int = 3,
        max_p: int = 2,
        max_d: int = 1,
        max_q: int = 2,
        max_P: int = 1,
        max_Q: int = 1,
    ) -> None:
        super().__init__()
        self.seasonal_period = seasonal_period
        self.fourier_order = fourier_order
        self.grid = list(
            itertools.product(
                range(max_p + 1), range(max_d + 1), range(max_q + 1),
                range(max_P + 1), range(max_Q + 1),
            )
        )

    def _fit(self, history: pd.Series) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = history.to_numpy(dtype=float)
        exog = fourier_terms(history.index, self.fourier_order)
        best = None
        for p, d, q, P, Q in self.grid:
            model = SARIMAX(
                y,
                exog=exog,
                order=(p, d, q),
                seasonal_order=(P, 0, Q, self.seasonal_period),
                trend="c" if d == 0 else "n",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = model.fit(disp=False)
                except (np.linalg.LinAlgError, ValueError):
                    continue
            if np.isfinite(res.aic) and (best is None or res.aic < best.aic):
                best = res
                self.order = (p, d, q)
                self.seasonal_order = (P, 0, Q, self.seasonal_period)
        if best is None:
            raise RuntimeError("no SARIMA order could be fit")
        self.result = best

    def _predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        steps = (dates.max() - self.history_end).days
        future = pd.date_range(self.history_end + pd.Timedelta(days=1), periods=steps, freq="D")
        mean = self.result.forecast(steps, exog=fourier_terms(future, self.fourier_order))
        path = pd.Series(np.clip(mean, 0, None), index=future)
        return path.reindex(dates).to_numpy()


FORECASTERS = {cls.name: cls for cls in (SeasonalNaive, SeasonalBlend, SARIMA)}


def make_forecaster(name: str, **kwargs) -> Forecaster:
    if name not in FORECASTERS:
        raise KeyError(f"unknown forecaster {name!r}; choose from {sorted(FORECASTERS)}")
    return FORECASTERS[name](**kwargs)


def calendar_profile(series: pd.Series) -> pd.Series:
    """Mean value for each calendar day (``MM-DD``) across the years in ``series``."""
    return series.groupby(series.index.strftime("%m-%d")).mean()


def seasonal_blend(
    profile: pd.Series, dates: pd.DatetimeIndex, weight: float = 0.8, window: int = 30
) -> np.ndarray:
    """The blend recursion from the original notebook, given a calendar-day profile."""
    hist = _lookup(profile, dates)
    out = np.empty(len(dates))
    for i, h in enumerate(hist):
        prev = out[max(0, i - window) : i].mean() if i else h
        out[i] = weight * h + (1 - weight) * prev
    return out


def fourier_terms(dates: pd.DatetimeIndex, order: int, period: float = 365.25) -> np.ndarray:
    t = dates.day_of_year.to_numpy()[:, None] * np.arange(1, order + 1)[None, :]
    angle = 2 * np.pi * t / period
    return np.hstack([np.sin(angle), np.cos(angle)])


def forecast_errors(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    """MAE and MAPE over the dates the two series share (MAPE skips zero-demand days)."""
    a, f = actual.align(forecast, join="inner")
    err = (f - a).abs()
    nonzero = a > 0
    return {
        "mae": float(err.mean()),
        "mape": float((err[nonzero] / a[nonzero]).mean() * 100),
    }


def _lookup(profile: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    keys = dates.strftime("%m-%d")
    if "02-29" not in profile.index and "02-28" in profile.index:
        profile = pd.concat([profile, pd.Series({"02-29": profile["02-28"]})])
    values = profile.reindex(keys).to_numpy(dtype=float)
    if np.isnan(values).any():
        missing = sorted(set(keys[np.isnan(values)]))
        raise ValueError(f"history has no data for calendar days {missing[:5]}")
    return values


def _check_daily(series: pd.Series) -> None:
    idx = series.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0:
        raise ValueError("expected a non-empty series with a DatetimeIndex")
    if len(idx) > 1 and not (idx[1:] - idx[:-1] == pd.Timedelta(days=1)).all():
        raise ValueError("expected consecutive daily dates")
