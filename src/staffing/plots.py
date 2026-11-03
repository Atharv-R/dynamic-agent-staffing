"""PNG figures: staffing over time, idle agents, waiting pool, forecasts vs actual."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# Categorical slots in fixed order (validated for colour-vision deficiency), plus text and grid.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
ACTUAL = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

LABELS = {
    "perfect_foresight": "Perfect foresight",
    "seasonal_naive": "Last year",
    "seasonal_blend": "Seasonal blend",
    "sarima": "SARIMA",
    "baseline": "Baseline",
    "lookahead": "Lookahead",
}


def label(name: str) -> str:
    return LABELS.get(name, name)


def plot_lines(
    series: Mapping[str, pd.Series],
    title: str,
    ylabel: str,
    path: str | Path | None = None,
    actual: pd.Series | None = None,
) -> plt.Figure:
    """One line per entry (at most four), labelled at the line end and in a legend."""
    if len(series) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} series per chart")
    fig, ax = _axes(title, ylabel)
    if actual is not None:
        ax.plot(actual.index, actual.to_numpy(), color=ACTUAL, lw=1.5, alpha=0.6, label="Actual")
    for (name, s), color in zip(series.items(), SERIES, strict=False):
        ax.plot(s.index, s.to_numpy(), color=color, lw=2, label=label(name))
    _end_labels(ax, series)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    return _finish(fig, path)


def _end_labels(ax: plt.Axes, series: Mapping[str, pd.Series]) -> None:
    """Label each line at its right end, nudging labels apart so they don't overlap."""
    lo, hi = ax.get_ylim()
    height_pt = ax.get_window_extent().height * 72 / ax.figure.dpi
    pt_per_unit = height_pt / (hi - lo)
    gap = 12  # points between label centres
    ends = sorted((float(s.iloc[-1]), name, s.index[-1]) for name, s in series.items())
    placed: list[float] = []
    for y, name, x in ends:
        y_pt = y * pt_per_unit
        if placed:
            y_pt = max(y_pt, placed[-1] + gap)
        placed.append(y_pt)
        ax.annotate(label(name), (x, y), xytext=(6, y_pt - y * pt_per_unit),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK_SECONDARY, annotation_clip=False)


def plot_daily(
    daily: Mapping[str, pd.DataFrame], column: str, title: str, ylabel: str,
    path: str | Path | None = None,
) -> plt.Figure:
    """Plot one column of several simulations' daily metrics."""
    return plot_lines({k: df[column] for k, df in daily.items()}, title, ylabel, path)


def plot_forecasts(
    actual: pd.Series, forecasts: Mapping[str, pd.Series], path: str | Path | None = None,
    smooth: int = 7,
) -> plt.Figure:
    """Forecasts against actual demand over the actual's dates, as ``smooth``-day means."""
    def roll(s: pd.Series) -> pd.Series:
        return s.reindex(actual.index).rolling(smooth, min_periods=1).mean()

    return plot_lines(
        {k: roll(v) for k, v in forecasts.items()},
        f"Forecast vs actual daily demand ({smooth}-day mean)",
        "Eligible sign-ups per day",
        path,
        actual=roll(actual),
    )


def _axes(title: str, ylabel: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    return fig, ax


def _finish(fig: plt.Figure, path: str | Path | None) -> plt.Figure:
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
    return fig
