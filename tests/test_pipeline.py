import numpy as np
import pandas as pd
import pytest

from src import clustering as cl
from src import data, features
from src.profiling import bh_adjust


@pytest.fixture(scope="module")
def cleaned():
    c, p = data.load_raw()
    c, _ = data.clean_clients(c)
    p, _ = data.clean_properties(p)
    return c, p


def test_parse_dob_handles_three_formats():
    s = pd.Series(["5/1/1968", "05-11-1968", "11/26/1962"])
    out = data.parse_dob(s)
    assert out.notna().all()
    assert list(out.dt.month) == [5, 5, 11]


def test_parse_price():
    assert data.parse_price(pd.Series(["$300,385.62"])).iloc[0] == pytest.approx(300385.62)


def test_row_counts_and_validation(cleaned):
    c, p = cleaned
    assert len(c) == 2000 and len(p) == 10000
    assert all(data.validate(c, p).values())


def test_no_pii_after_cleaning(cleaned):
    c, _ = cleaned
    assert "first_name" not in c.columns and "last_name" not in c.columns


def test_features_cover_every_client(cleaned):
    c, p = cleaned
    df = features.build_client_features(c, p)
    assert len(df) == 2000
    assert (df["units_bought"] >= 1).all()
    assert df["tower_hhi"].between(0, 1).all()
    assert df["office_share"].between(0, 1).all()


def test_bh_adjust_monotone_and_bounded():
    adj = bh_adjust(np.array([0.01, 0.04, 0.03, 0.5]))
    assert (adj >= np.array([0.01, 0.04, 0.03, 0.5])).all() and (adj <= 1).all()


def test_permutation_null_detects_real_clusters():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, .3, (100, 2)), rng.normal(5, .3, (100, 2))])
    res = cl.permutation_null(X, 2, n_perm=20)
    assert res["p_value"] < 0.1
