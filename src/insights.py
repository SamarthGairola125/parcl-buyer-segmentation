"""Action titles: chart headlines that state a finding, generated from data.

Every title is computed from the data passed in, never typed by hand, so it
stays correct when filters change. Tests check each template against the data.
"""
from __future__ import annotations

import pandas as pd

SEG_ORDER = ["Portfolio Builders", "Diversified Repeat Buyers",
             "Focused Residential Buyers", "Mixed-Use Office Buyers"]


def share_title(df: pd.DataFrame) -> str:
    """Overview: the largest segment and the value-concentration contrast."""
    n = df["segment"].value_counts()
    spend = df.groupby("segment")["total_spend"].sum()
    top = n.idxmax()
    out = f"{top} are the largest group: {n[top] / n.sum() * 100:.0f}% of buyers"
    if "Portfolio Builders" in n.index and n["Portfolio Builders"] > 0:
        pc = n["Portfolio Builders"] / n.sum() * 100
        ps = spend["Portfolio Builders"] / spend.sum() * 100
        out += f". Portfolio Builders are {pc:.1f}% of buyers but {ps:.1f}% of spend"
    return out


def invest_title(df: pd.DataFrame) -> str:
    """Investor page: spread of investment intent across segments."""
    r = df.groupby("segment")["is_investment"].mean() * 100
    if len(r) < 2:
        return f"{r.iloc[0]:.0f}% of these buyers bought to invest"
    lo, hi = r.min(), r.max()
    if hi - lo < 10:
        return (f"Investment intent barely differs by segment ({lo:.0f}%–{hi:.0f}%): "
                "segments are defined by how much and what people buy, not why")
    return f"Investment intent ranges from {lo:.0f}% ({r.idxmin()}) to {hi:.0f}% ({r.idxmax()})"


def segment_contrast_title(seg_df: pd.DataFrame, all_df: pd.DataFrame, segment: str) -> str:
    """Insights page: the trait on which this segment differs most from average."""
    traits = {"units_bought": "units bought", "office_share": "office share",
              "total_spend": "total spend", "recency_months": "months since last purchase",
              "age": "age"}
    best, best_z = None, 0.0
    for col in traits:
        sd = all_df[col].std()
        if sd > 0:
            z = (seg_df[col].mean() - all_df[col].mean()) / sd
            if abs(z) > abs(best_z):
                best, best_z = col, z
    if best is None:
        return f"{segment} look like the average buyer"
    seg_v, avg_v = seg_df[best].mean(), all_df[best].mean()
    word = "higher" if best_z > 0 else "lower"
    if best == "office_share":
        vals = f"{seg_v * 100:.0f}% vs {avg_v * 100:.0f}% on average"
    elif best == "total_spend":
        vals = f"${seg_v / 1e6:.2f}M vs ${avg_v / 1e6:.2f}M on average"
    else:
        vals = f"{seg_v:.1f} vs {avg_v:.1f} on average"
    return f"{segment} stand out most on {traits[best]}: {word} than average ({vals})"
