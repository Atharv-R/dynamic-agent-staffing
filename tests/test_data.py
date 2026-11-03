import numpy as np
import pandas as pd
import pytest

from staffing import synthetic
from staffing.data import load_real, load_synthetic


def test_synthetic_is_deterministic():
    a, b = synthetic.generate(seed=11, scale=0.05), synthetic.generate(seed=11, scale=0.05)
    for x, y in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(x, y)
    other = synthetic.generate(seed=12, scale=0.05)[0]
    assert not a[0].equals(other)


def test_synthetic_matches_case_schema():
    signups, salary_threshold, staffing = synthetic.generate(seed=1, scale=0.05)
    assert list(signups.columns) == [
        "Advertiser_ID", "Country", "Sign_Up_Date", "Projected_Annual_Budget"
    ]
    assert list(salary_threshold.columns) == [
        "Country", "Annual_Agent_Salary_USD", "Advertiser_Eligibility_Threshold_USD"
    ]
    assert list(staffing.columns) == ["Country", "Existing_Agent_Count"]
    assert signups["Advertiser_ID"].is_unique


def test_synthetic_calibration():
    ds = load_synthetic(seed=7)
    per_year = ds.signups.groupby(ds.signups["Sign_Up_Date"].dt.year).size()
    assert ((per_year > 200_000) & (per_year < 300_000)).all()
    assert ds.signups["Eligible"].mean() == pytest.approx(0.42, abs=0.02)
    usa = ds.country_info("USA")
    assert (usa.salary, usa.existing_agents) == (77_721, 652)


def test_country_lookup_is_by_name(dataset):
    shuffled = dataset.countries.sample(frac=1, random_state=0).reset_index(drop=True)
    ds = type(dataset)(dataset.signups, shuffled, "synthetic")
    assert ds.country_info("USA") == dataset.country_info("USA")
    with pytest.raises(KeyError):
        ds.country_info("Atlantis")


def test_demand_and_arrival_modes_agree(dataset):
    dates = pd.date_range("2023-03-01", periods=20)
    eligible = dataset.daily_demand("USA")[dates].to_numpy()
    everyone = dataset.daily_demand("USA", eligible_only=False)[dates].to_numpy()
    assert (eligible <= everyone).all() and eligible.sum() < everyone.sum()

    individual = dataset.arrivals("USA", dates, "individual")
    averaged = dataset.arrivals("USA", dates, "averaged")
    legacy = dataset.arrivals("USA", dates, "legacy")
    assert [len(a) for a in individual] == list(eligible)
    assert [len(a) for a in averaged] == list(eligible)
    assert [len(a) for a in legacy] == list(everyone)
    for ind, avg in zip(individual, averaged, strict=True):
        assert ind.sum() == pytest.approx(avg.sum())
    threshold = dataset.country_info("USA").threshold
    assert all((a > threshold).all() for a in individual)


def test_missing_real_data_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not public"):
        load_real(tmp_path)


def test_daily_demand_covers_every_day(dataset):
    d = dataset.daily_demand("USA")
    assert d.index.equals(pd.date_range("2023-01-01", "2024-12-31"))
    assert not np.isnan(d.to_numpy()).any()
