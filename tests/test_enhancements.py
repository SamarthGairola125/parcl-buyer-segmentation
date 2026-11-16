import pandas as pd
import pytest

from src import config as C
from src.insights import invest_title, segment_contrast_title, share_title
from src.uncertainty import icon_counts, rate_summary, wilson_ci


@pytest.fixture(scope="module")
def seg():
    return pd.read_csv(C.DATA_PROCESSED / "client_segments.csv")


def test_wilson_known_value():
    lo, hi = wilson_ci(16, 44)  # Portfolio Builders investment rate
    assert 0.23 < lo < 0.25 and 0.50 < hi < 0.52


def test_wilson_edges():
    assert wilson_ci(0, 10)[0] == 0.0 and wilson_ci(10, 10)[1] == 1.0


def test_rate_summary_flags_small_sample():
    assert rate_summary([1] * 10 + [0] * 34)["small_sample"]
    assert not rate_summary([1] * 50 + [0] * 150)["small_sample"]


def test_icon_counts():
    assert icon_counts(0.364) == 36 and icon_counts(0.0) == 0 and icon_counts(1.0) == 100


def test_share_title_matches_data(seg):
    t = share_title(seg)
    share = (seg["segment"] == "Diversified Repeat Buyers").mean() * 100
    assert t.startswith("Diversified Repeat Buyers are the largest group")
    assert f"{share:.0f}%" in t and "4.4% of spend" in t


def test_invest_title_matches_data(seg):
    r = seg.groupby("segment")["is_investment"].mean() * 100
    t = invest_title(seg)
    assert f"{r.min():.0f}%" in t and f"{r.max():.0f}%" in t


def test_contrast_title_portfolio(seg):
    pb = seg[seg["segment"] == "Portfolio Builders"]
    assert "units bought" in segment_contrast_title(pb, seg, "Portfolio Builders")


def test_embedding_matches_frozen_segments(seg):
    emb = pd.read_csv(C.RESULTS / "embedding_3d_v1.csv")
    assert (emb["segment"].values == seg["segment"].values).all()
    assert (emb["pc1"] - seg["pc1"]).abs().max() < 1e-9
