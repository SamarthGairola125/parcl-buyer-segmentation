"""Cluster profiling, hypothesis testing and geographic concentration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr

PROFILE_NUMERIC = ["age", "satisfaction_score", "units_bought", "total_spend",
                   "avg_unit_price", "premium_index", "office_share", "tower_hhi",
                   "recency_months"]
PROFILE_RATES = ["is_company", "is_investment", "has_loan", "is_international",
                 "is_female", "ref_agency", "ref_client", "portfolio_builder",
                 "bought_both_years"]


def profile(df: pd.DataFrame, seg: str = "segment") -> pd.DataFrame:
    g = df.groupby(seg)
    out = pd.concat([
        g.size().rename("n"),
        (g.size() / len(df) * 100).rename("share_pct"),
        g[PROFILE_NUMERIC].mean(),
        g[PROFILE_RATES].mean() * 100,
    ], axis=1)
    overall = pd.concat([
        pd.Series({"n": len(df), "share_pct": 100.0}),
        df[PROFILE_NUMERIC].mean(), df[PROFILE_RATES].mean() * 100])
    out.loc["Overall"] = overall
    return out


def lift(df: pd.DataFrame, seg: str = "segment") -> pd.DataFrame:
    """Segment rate / overall rate for each binary trait (1.0 = average)."""
    rates = df.groupby(seg)[PROFILE_RATES].mean()
    return rates / df[PROFILE_RATES].mean()


def cramers_v(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    t = pd.crosstab(a, b)
    if min(t.shape) < 2:
        return 0.0, 1.0
    chi2, p, _, _ = chi2_contingency(t)
    v = np.sqrt(chi2 / t.values.sum() / (min(t.shape) - 1))
    return float(v), float(p)


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg false discovery rate adjustment."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1)
    return out


def _mw(df, val, grp, a, b):
    x, y = df.loc[df[grp] == a, val], df.loc[df[grp] == b, val]
    return mannwhitneyu(x, y).pvalue, f"{x.mean():.2f} vs {y.mean():.2f} (n={len(x)} vs {len(y)})"


def _chi(df, a, b, level_a, level_b):
    t = pd.crosstab(df[a], df[b])
    p = chi2_contingency(t)[1]
    r1 = (df.loc[df[a] == level_a, b] == level_b).mean() * 100
    r0 = (df.loc[df[a] != level_a, b] == level_b).mean() * 100
    n1, n0 = (df[a] == level_a).sum(), (df[a] != level_a).sum()
    return p, f"{r1:.1f}% vs {r0:.1f}% (n={n1} vs {n0})"


def hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["intl"] = np.where(d["is_international"] == 1, "International", "USA")
    d["repeat"] = np.where(d["bought_both_years"] == 1, "Both years", "One year")
    rows = []

    def add(hid, text, test, p, effect, expected_direction=None):
        rows.append({"id": hid, "hypothesis": text, "test": test, "p_value": p, "evidence": effect})

    p, e = _mw(d, "age", "acquisition_purpose", "Investment", "Home")
    add("H1", "Investment buyers are younger than Home buyers", "Mann-Whitney U", p, e)
    p, e = _chi(d, "acquisition_purpose", "loan_applied", "Investment", "Yes")
    add("H2", "Investment buyers apply for loans more often", "Chi-square", p, e)
    p, e = _chi(d, "intl", "acquisition_purpose", "International", "Investment")
    add("H3", "International buyers are more investment-oriented", "Chi-square", p, e)
    p, e = _mw(d, "units_bought", "client_type", "Company", "Individual")
    add("H4", "Companies buy more units (brief C3)", "Mann-Whitney U", p, e)
    p, e = _mw(d, "age", "client_type", "Company", "Individual")
    add("H5", "Company representatives differ in age from individuals", "Mann-Whitney U", p, e)
    groups = [g["satisfaction_score"].values for _, g in d.groupby("referral_channel")]
    means = d.groupby("referral_channel")["satisfaction_score"].mean().round(2).to_dict()
    add("H6", "Satisfaction differs by referral channel", "Kruskal-Wallis",
        kruskal(*groups).pvalue, str(means))
    rho, p = spearmanr(d["total_spend"], d["satisfaction_score"])
    add("H7", "Higher-spending clients are more satisfied (brief C4)", "Spearman rho", p,
        f"rho = {rho:.3f}")
    p, e = _mw(d, "office_share", "acquisition_purpose", "Investment", "Home")
    add("H8a", "Investment buyers hold a higher share of office units", "Mann-Whitney U", p, e)
    p, e = _mw(d, "office_share", "client_type", "Company", "Individual")
    add("H8b", "Companies hold a higher share of office units", "Mann-Whitney U", p, e)
    p, e = _mw(d, "avg_unit_price", "loan_applied", "Yes", "No")
    add("H9", "Loan applicants buy cheaper units", "Mann-Whitney U", p, e)
    p, e = _mw(d, "total_spend", "repeat", "Both years", "One year")
    add("H10", "Cross-year repeat buyers spend more", "Mann-Whitney U", p, e)
    p, e = _mw(d, "total_spend", "intl", "International", "USA")
    add("H14", "International buyers spend differently from US buyers", "Mann-Whitney U", p, e)
    p, e = _mw(d, "age", "loan_applied", "Yes", "No")
    add("H15", "Loan applicants are younger (brief C2)", "Mann-Whitney U", p, e)

    out = pd.DataFrame(rows)
    out["p_bh"] = bh_adjust(out["p_value"].values)
    out["result"] = np.where(out["p_bh"] < 0.05, "Supported", "Not supported")
    return out


def independence_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v, p = cramers_v(df[a], df[b])
            rows.append({"var_a": a, "var_b": b, "cramers_v": v, "p_value": p})
    out = pd.DataFrame(rows)
    out["p_bh"] = bh_adjust(out["p_value"].values)
    return out


def location_quotient(df: pd.DataFrame, geo: str, seg: str = "segment",
                      B: int = 200, seed: int = 42, min_n: int = 30) -> pd.DataFrame:
    """LQ = (segment share within area) / (segment share overall).
    Borrowed from regional economics. Bootstrap 90% CI by resampling clients."""
    rng = np.random.default_rng(seed)

    def lq(d):
        ct = pd.crosstab(d[geo], d[seg], normalize="index")
        overall = d[seg].value_counts(normalize=True)
        return ct / overall

    base = lq(df)
    boots = []
    for _ in range(B):
        sample = df.sample(len(df), replace=True,
                           random_state=int(rng.integers(1e9))).reset_index(drop=True)
        boots.append(lq(sample)
                     .reindex(index=base.index, columns=base.columns))
    stack = np.stack([b.values for b in boots])
    lo = pd.DataFrame(np.nanpercentile(stack, 5, axis=0), index=base.index, columns=base.columns)
    hi = pd.DataFrame(np.nanpercentile(stack, 95, axis=0), index=base.index, columns=base.columns)
    n = df[geo].value_counts()
    rows = []
    for area in base.index:
        for s in base.columns:
            rows.append({geo: area, seg: s, "n_area": int(n[area]), "lq": base.loc[area, s],
                         "ci_low": lo.loc[area, s], "ci_high": hi.loc[area, s],
                         "reliable": bool(n[area] >= min_n),
                         "significant": bool(lo.loc[area, s] > 1 or hi.loc[area, s] < 1)})
    return pd.DataFrame(rows)


def decision_usefulness(df: pd.DataFrame, labels: np.ndarray, targets: list[str],
                        n_random: int = 200, seed: int = 42) -> pd.DataFrame:
    """How strongly does segment membership separate each actionable variable,
    compared with random segmentations of identical sizes?"""
    rng = np.random.default_rng(seed)
    rows = []
    for t in targets:
        v_obs, _ = cramers_v(pd.Series(labels), df[t].reset_index(drop=True))
        rand = [cramers_v(pd.Series(rng.permutation(labels)), df[t].reset_index(drop=True))[0]
                for _ in range(n_random)]
        rows.append({"target": t, "cramers_v": v_obs, "random_mean": float(np.mean(rand)),
                     "random_p95": float(np.percentile(rand, 95)),
                     "beats_random": bool(v_obs > np.percentile(rand, 95))})
    return pd.DataFrame(rows)
