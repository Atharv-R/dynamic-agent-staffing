"""Seeded synthetic data with the same schema as the (confidential) case-study CSVs.

Volumes are calibrated to the real data's aggregates: about 250k sign-ups a year, about 42% of
them eligible, and about 100 USA sign-ups a day. The USA salary and headcount are the case values;
every other number is invented. Daily counts build through the year with Q2 and Q4 peaks, a
weekly cycle and persistent noise. Each year gets its own jitter on peak timing and size, so "same
as last year" is a reasonable but imperfect forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Standard normal quantile at 0.58, so P(budget > threshold) = 0.42 for a lognormal budget.
_Z_ELIGIBLE = 0.2019


@dataclass(frozen=True)
class SyntheticCountry:
    country: str
    salary: int
    threshold: int
    existing_agents: int
    share: float  # share of all sign-ups
    budget_sigma: float = 1.0


COUNTRIES = (
    SyntheticCountry("USA", 77_721, 35_000, 652, 0.146),
    SyntheticCountry("UK", 54_000, 25_000, 210, 0.11),
    SyntheticCountry("Germany", 56_000, 25_000, 190, 0.10),
    SyntheticCountry("India", 15_000, 4_000, 280, 0.21),
    SyntheticCountry("Brazil", 26_000, 8_000, 160, 0.14),
    SyntheticCountry("Japan", 48_000, 20_000, 150, 0.10),
    SyntheticCountry("Australia", 62_000, 22_000, 70, 0.05),
    SyntheticCountry("Canada", 60_000, 25_000, 90, 0.144),
)


def generate(
    seed: int = 7,
    years: tuple[int, ...] = (2023, 2024),
    signups_per_year: float = 250_000,
    scale: float = 1.0,
    countries: tuple[SyntheticCountry, ...] = COUNTRIES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(sign_ups, salary_threshold, staffing)`` shaped like the three case CSVs.

    ``scale`` shrinks sign-up volumes and existing headcount together (handy for fast tests).
    """
    rng = np.random.default_rng(seed)
    frames = []
    for c in countries:
        mu = np.log(c.threshold) - _Z_ELIGIBLE * c.budget_sigma
        for year in years:
            dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            rate = scale * signups_per_year * c.share / 365 * _daily_profile(dates, rng)
            # Gamma-Poisson: real daily counts vary far more than Poisson (about 25% day to day).
            counts = rng.poisson(rate * rng.gamma(16, 1 / 16, len(dates)))
            budgets = rng.lognormal(mu, c.budget_sigma, counts.sum()).round()
            frames.append(
                pd.DataFrame(
                    {
                        "Country": c.country,
                        "Sign_Up_Date": np.repeat(dates.strftime("%Y-%m-%d"), counts),
                        "Projected_Annual_Budget": budgets.astype(np.int64),
                    }
                )
            )

    sign_ups = pd.concat(frames, ignore_index=True)
    sign_ups = sign_ups.sort_values("Sign_Up_Date", kind="stable", ignore_index=True)
    sign_ups.insert(0, "Advertiser_ID", [f"A{i:07d}" for i in range(1, len(sign_ups) + 1)])

    salary_threshold = pd.DataFrame(
        {
            "Country": [c.country for c in countries],
            "Annual_Agent_Salary_USD": [c.salary for c in countries],
            "Advertiser_Eligibility_Threshold_USD": [c.threshold for c in countries],
        }
    )
    staffing = pd.DataFrame(
        {
            "Country": [c.country for c in countries],
            "Existing_Agent_Count": [max(1, round(c.existing_agents * scale)) for c in countries],
        }
    )
    return sign_ups, salary_threshold, staffing


def _daily_profile(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative demand profile (mean about 1) for one country-year."""
    doy = dates.day_of_year.to_numpy()

    def bump(center: float, width: float, height: float) -> np.ndarray:
        return height * np.exp(-(((doy - center) / width) ** 2))

    annual = (
        0.85 + 0.30 * doy / 365  # demand builds through the year
        + bump(135 + rng.normal(0, 7), 28, 0.22 * rng.uniform(0.8, 1.2))  # Q2 peak
        + bump(318 + rng.normal(0, 7), 32, 0.30 * rng.uniform(0.8, 1.2))  # Q4 peak
        - bump(205, 30, 0.12)  # summer lull
        - bump(358, 6, 0.25)  # holidays
    )
    weekly = np.array([1.06, 1.10, 1.09, 1.07, 1.02, 0.83, 0.83])[dates.day_of_week]
    level = rng.uniform(0.95, 1.03)

    # AR(1) noise on the log scale gives runs of busy and quiet weeks.
    noise = np.empty(len(dates))
    noise[0] = rng.normal(0, 0.06)
    for t in range(1, len(dates)):
        noise[t] = 0.9 * noise[t - 1] + rng.normal(0, 0.06 * np.sqrt(1 - 0.81))
    return level * annual * weekly * np.exp(noise)
