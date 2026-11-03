from dataclasses import replace

import pytest

from staffing.config import load_config
from staffing.data import load_synthetic


@pytest.fixture(scope="session")
def dataset():
    # A small synthetic world keeps the verbatim reference simulator fast.
    return load_synthetic(seed=3, scale=0.25)


@pytest.fixture(scope="session")
def cfg():
    base = load_config()
    return replace(
        base,
        baseline={**base.baseline, "hire": [0, 3], "fire": [0, 2]},
        lookahead={**base.lookahead, "alpha": [1.5, 2.5, 4.0], "beta": [1.0]},
        forecasters={
            "seasonal_naive": {},
            "seasonal_blend": {"weight": 0.8, "window": 30},
            "sarima": {"fourier_order": 2, "max_p": 1, "max_d": 0, "max_q": 1,
                       "max_P": 0, "max_Q": 0},
        },
    )
