import numpy as np

from staffing.policies import BaselinePolicy, LookaheadPolicy


def test_baseline_hires_above_threshold_else_fires():
    pool = np.zeros((5, 10), dtype=int)
    policy = BaselinePolicy(hire=4, fire=2, threshold=10, window=2)
    assert policy.decide(0, pool, np.array([12.0, 12, 12, 0])) == (4, 0)
    assert policy.decide(0, pool, np.array([8.0, 8, 8])) == (0, 2)


def test_lookahead_hires_for_shortfall():
    pool = np.zeros((2, 10), dtype=int)  # 20 free slots
    forecast = np.full(31, 2.0)  # 62 advertisers in the window
    assert LookaheadPolicy(alpha=2.0, horizon=30).decide(0, pool, forecast) == (4, 0)


def test_lookahead_fires_surplus_beyond_alpha():
    pool = np.zeros((10, 10), dtype=int)  # 100 free slots
    forecast = np.full(31, 1.0)  # 31 advertisers; alpha * 31 = 62 slots tolerated
    assert LookaheadPolicy(alpha=2.0).decide(0, pool, forecast) == (0, 3)


def test_lookahead_small_surplus_fires_nobody_unless_legacy():
    # 100 slots vs alpha * demand = 93: surplus of 7 rounds down to zero agents.
    pool = np.zeros((10, 10), dtype=int)
    forecast = np.full(31, 1.0)
    assert LookaheadPolicy(alpha=3.0).decide(0, pool, forecast) == (0, 0)
    # The original loop never matched count == 0, so it fired everyone it could.
    assert LookaheadPolicy(alpha=3.0, legacy=True).decide(0, pool, forecast) == (0, 10)


def test_lookahead_counts_slots_freeing_within_horizon():
    pool = np.full((3, 10), 45)
    pool[0] = 0
    pool[1] = 29
    forecast = np.full(31, 20 / 31)
    # 20 slots free within 30 days, 20 demand: no change.
    assert LookaheadPolicy(alpha=1.0, beta=1.0).decide(0, pool, forecast) == (0, 0)
