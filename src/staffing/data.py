"""Load the case-study data (real CSVs or synthetic) and build per-country daily series."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from . import synthetic

SIGNUPS_FILE = "Advertiser Sign-ups.csv"
COUNTRIES_FILE = "Agent Salary and Eligibility Threshold.csv"
STAFFING_FILE = "Existing Agent Staffing.csv"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

ArrivalMode = Literal["individual", "averaged", "legacy"]


@dataclass(frozen=True)
class CountryInfo:
    country: str
    salary: float
    threshold: float
    existing_agents: int


@dataclass
class Dataset:
    """Sign-ups with an ``Eligible`` flag, plus one row of country parameters per country."""

    signups: pd.DataFrame
    countries: pd.DataFrame
    source: str
    _eligible_by_day: dict[str, pd.Series] = field(default_factory=dict, repr=False)

    @property
    def dates(self) -> pd.DatetimeIndex:
        d = self.signups["Sign_Up_Date"]
        return pd.date_range(d.min(), d.max(), freq="D")

    def country_info(self, country: str) -> CountryInfo:
        rows = self.countries.loc[self.countries["Country"] == country]
        if rows.empty:
            raise KeyError(f"no country parameters for {country!r}")
        row = rows.iloc[0]
        return CountryInfo(
            country=country,
            salary=float(row["Annual_Agent_Salary_USD"]),
            threshold=float(row["Advertiser_Eligibility_Threshold_USD"]),
            existing_agents=int(row["Existing_Agent_Count"]),
        )

    def daily_demand(self, country: str, eligible_only: bool = True) -> pd.Series:
        """Sign-ups per day (eligible only by default), zero-filled over the full date range."""
        df = self._country(country)
        if eligible_only:
            df = df[df["Eligible"]]
        counts = df.groupby("Sign_Up_Date").size().reindex(self.dates, fill_value=0)
        return counts.astype(float).rename("demand").rename_axis("date")

    def daily_budget(self, country: str) -> pd.Series:
        """Mean projected budget of the day's eligible sign-ups (NaN on days with none)."""
        df = self._country(country)
        means = df[df["Eligible"]].groupby("Sign_Up_Date")["Projected_Annual_Budget"].mean()
        return means.reindex(self.dates).rename("budget").rename_axis("date")

    def arrivals(
        self, country: str, dates: pd.DatetimeIndex, mode: ArrivalMode = "individual"
    ) -> list[np.ndarray]:
        """Budgets of the advertisers who join the waiting pool on each date.

        ``individual``: each eligible advertiser with their own budget.
        ``averaged``: the day's eligible count, all at the day's mean eligible budget.
        ``legacy``: the original notebooks' input, every sign-up (eligible or not) at the day's
        mean eligible budget, or 0 when nobody was eligible.
        """
        if mode == "individual":
            by_day = self._eligible_budgets(country)
            empty = np.empty(0)
            return [by_day.get(d, empty) for d in dates]
        if mode not in ("averaged", "legacy"):
            raise ValueError(f"unknown arrival mode {mode!r}")
        counts = self.daily_demand(country, eligible_only=mode == "averaged").reindex(dates)
        budget = self.daily_budget(country).reindex(dates).fillna(0.0)
        return [np.full(int(n), b) for n, b in zip(counts.fillna(0), budget, strict=True)]

    def _country(self, country: str) -> pd.DataFrame:
        df = self.signups[self.signups["Country"] == country]
        if df.empty:
            raise KeyError(f"no sign-ups for {country!r}")
        return df

    def _eligible_budgets(self, country: str) -> pd.Series:
        if country not in self._eligible_by_day:
            df = self._country(country)
            df = df[df["Eligible"]]
            self._eligible_by_day[country] = df.groupby("Sign_Up_Date")[
                "Projected_Annual_Budget"
            ].agg(lambda s: s.to_numpy(dtype=float))
        return self._eligible_by_day[country]


def prepare(
    signups: pd.DataFrame, salary_threshold: pd.DataFrame, staffing: pd.DataFrame, source: str
) -> Dataset:
    countries = salary_threshold.merge(staffing, on="Country", how="left")
    if countries["Existing_Agent_Count"].isna().any():
        missing = countries.loc[countries["Existing_Agent_Count"].isna(), "Country"].tolist()
        raise ValueError(f"no existing agent count for {missing}")

    signups = signups.copy()
    signups["Sign_Up_Date"] = pd.to_datetime(signups["Sign_Up_Date"])
    thresholds = signups["Country"].map(
        countries.set_index("Country")["Advertiser_Eligibility_Threshold_USD"]
    )
    # Eligible means strictly above the country threshold, as in the original analysis.
    signups["Eligible"] = signups["Projected_Annual_Budget"] > thresholds
    return Dataset(signups=signups, countries=countries, source=source)


def load_real(data_dir: str | Path = DEFAULT_DATA_DIR) -> Dataset:
    data_dir = Path(data_dir)
    paths = [data_dir / name for name in (SIGNUPS_FILE, COUNTRIES_FILE, STAFFING_FILE)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"real data not found in {data_dir} (missing {missing}). The case-study data is not "
            "public; use the synthetic data instead."
        )
    return prepare(*(pd.read_csv(p) for p in paths), source="real")


def load_synthetic(seed: int = 7, **kwargs) -> Dataset:
    return prepare(*synthetic.generate(seed=seed, **kwargs), source="synthetic")


def load_dataset(
    source: Literal["synthetic", "real"] = "synthetic",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    seed: int = 7,
) -> Dataset:
    if source == "real":
        return load_real(data_dir)
    if source == "synthetic":
        return load_synthetic(seed)
    raise ValueError(f"unknown data source {source!r}")
