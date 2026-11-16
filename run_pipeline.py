"""End-to-end pipeline: clean -> features -> cluster -> validate -> profile.

Run:  python run_pipeline.py
Outputs go to data/processed/, results/ and reports/figures/.
"""
from __future__ import annotations

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kmodes.kprototypes import KPrototypes
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from src import clustering as cl
from src import config as C
from src import data, features
from src import profiling as P

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = ["#2a6f97", "#e07a5f", "#81b29a", "#9c6ade", "#f2cc8f", "#3d405b"]
for d in (C.DATA_PROCESSED, C.FIGURES, C.RESULTS):
    d.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(C.FIGURES / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def name_segments(prof: pd.DataFrame) -> dict:
    """Assign business names from profiles with explicit rules, so names do not
    depend on the arbitrary label order K-Means returns."""
    p = prof.drop(index="Overall")
    names = {}
    pb = p["units_bought"].idxmax()
    names[pb] = "Portfolio Builders"
    rest = p.drop(index=pb)
    office = rest["office_share"].idxmax()
    names[office] = "Mixed-Use Office Buyers"
    rest = rest.drop(index=office)
    hi, lo = rest["units_bought"].idxmax(), rest["units_bought"].idxmin()
    names[hi] = "Diversified Repeat Buyers"
    names[lo] = "Focused Residential Buyers"
    return names


def name_brief_segments(prof: pd.DataFrame) -> dict:
    p = prof.drop(index="Overall")
    age_mid, sat_mid = p["age"].median(), p["satisfaction_score"].median()
    out = {}
    for s, r in p.iterrows():
        a = "Younger" if r["age"] < age_mid else "Older"
        h = "Satisfied" if r["satisfaction_score"] >= sat_mid else "Dissatisfied"
        out[s] = f"{a} {h}"
    return out


def main():
    log = {}
    # 1. Clean -------------------------------------------------------------
    raw_c, raw_p = data.load_raw()
    clients, log["clean_clients"] = data.clean_clients(raw_c)
    props, log["clean_properties"] = data.clean_properties(raw_p)
    log["validation"] = data.validate(clients, props)
    clients.to_csv(C.DATA_PROCESSED / "clients_clean.csv", index=False)

    # 2. Features ----------------------------------------------------------
    df = features.build_client_features(clients, props)

    # 3. EDA figures -------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    df["country"].value_counts().plot.barh(ax=ax[0, 0], color=PALETTE[0])
    ax[0, 0].set_title("Clients by country"); ax[0, 0].invert_yaxis()
    for a, col, t in [(ax[0, 1], "acquisition_purpose", "Acquisition purpose"),
                      (ax[1, 0], "loan_applied", "Loan applied"),
                      (ax[1, 1], "referral_channel", "Referral channel")]:
        df[col].value_counts().plot.bar(ax=a, color=PALETTE[1:4], rot=0); a.set_title(t)
    save(fig, "fig01_categorical_overview.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df["age"], bins=30, ax=ax[0], color=PALETTE[0]); ax[0].set_title("Age (at Dec 2025)")
    df["satisfaction_score"].value_counts().sort_index().plot.bar(ax=ax[1], color=PALETTE[2], rot=0)
    ax[1].set_title("Satisfaction score")
    df["units_bought"].value_counts().sort_index().plot.bar(ax=ax[2], color=PALETTE[3], rot=0)
    ax[2].set_title("Units bought per client")
    save(fig, "fig02_numeric_distributions.png")

    sold = props[props["listing_status"] == "Sold"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(data=sold, x="sale_price", hue="unit_category", bins=40, ax=ax[0], palette=PALETTE[:2])
    ax[0].set_title("Sale price by unit category")
    m = sold.groupby(sold["transaction_date"].dt.to_period("M")).agg(
        units=("listing_id", "count"), median_price=("sale_price", "median"))
    m.index = m.index.to_timestamp()
    ax[1].bar(m.index, m["units"], width=20, color=PALETTE[0], alpha=.7); ax[1].set_ylabel("Units sold")
    ax2 = ax[1].twinx(); ax2.plot(m.index, m["median_price"], color=PALETTE[1], marker="o")
    ax2.set_ylabel("Median price ($)"); ax[1].set_title("Monthly sales volume and median price")
    save(fig, "fig03_property_market.png")
    m.reset_index().rename(columns={"transaction_date": "month"}).to_csv(C.RESULTS / "monthly_sales.csv", index=False)

    # Independence audit
    cat_cols = ["client_type", "gender", "is_international", "acquisition_purpose",
                "loan_applied", "referral_channel", "age_band", "satisfaction_score"]
    ind = P.independence_matrix(df, cat_cols)
    ind.to_csv(C.RESULTS / "independence_matrix.csv", index=False)
    mat = pd.DataFrame(0.0, index=cat_cols, columns=cat_cols)
    for _, r in ind.iterrows():
        mat.loc[r.var_a, r.var_b] = mat.loc[r.var_b, r.var_a] = r.cramers_v
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=0.3, ax=ax, cbar_kws={"label": "Cramér's V"})
    ax.set_title("Association between client attributes (0 = independent)")
    save(fig, "fig04_independence_heatmap.png")

    # Hypothesis tests
    hyp = P.hypothesis_tests(df)
    hyp.to_csv(C.RESULTS / "hypothesis_tests.csv", index=False)

    # 4. Clustering --------------------------------------------------------
    views = {
        "brief": (C.NUMERIC_BRIEF, C.BINARY_FEATURES),
        "behaviour": (C.NUMERIC_BEHAVIOUR, []),
        "full": (C.NUMERIC_BRIEF + C.NUMERIC_BEHAVIOUR, C.BINARY_FEATURES),
    }
    scans, matrices = {}, {}
    for v, (num, bins) in views.items():
        X, cols, scaler = features.model_matrix(df, num, bins)
        matrices[v] = (X, cols, scaler)
        s = cl.kmeans_scan(X); s["view"] = v
        scans[v] = s
    scan_df = pd.concat(scans.values())
    scan_df.to_csv(C.RESULTS / "kmeans_scan.csv", index=False)

    # Scaler comparison on the primary view (brief asks for Standard or MinMax)
    num_full = C.NUMERIC_BRIEF + C.NUMERIC_BEHAVIOUR
    Xmm = np.hstack([MinMaxScaler().fit_transform(df[num_full]), df[C.BINARY_FEATURES].to_numpy(float)])
    mm = cl.kmeans_scan(Xmm, range(2, 9)); mm["view"] = "full_minmax"
    mm.to_csv(C.RESULTS / "kmeans_scan_minmax.csv", index=False)

    K = 4
    X, cols, scaler = matrices["full"]
    km = cl.fit_kmeans(X, K)
    df["segment_id"] = km.labels_

    # Elbow + silhouette figure (primary view, with brief view for comparison)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    s = scans["full"]
    ax[0].plot(s.k, s.inertia, marker="o", color=PALETTE[0]); ax[0].axvline(K, ls="--", c="grey")
    ax[0].set(title="Elbow method (primary feature set)", xlabel="k", ylabel="Inertia (within-cluster SS)")
    for v, c in zip(["brief", "behaviour", "full"], PALETTE):
        ax[1].plot(scans[v].k, scans[v].silhouette, marker="o", label=v, color=c)
    ax[1].axhline(0.25, ls=":", c="grey"); ax[1].text(8.5, 0.255, "weak-structure line (0.25)", fontsize=8)
    ax[1].axvline(K, ls="--", c="grey"); ax[1].legend(title="feature set")
    ax[1].set(title="Silhouette score by k", xlabel="k", ylabel="Silhouette")
    save(fig, "fig05_elbow_silhouette.png")

    # Hierarchical (Ward)
    Z = cl.ward_linkage(X)
    hc_scan = cl.hierarchical_scan(X, Z); hc_scan.to_csv(C.RESULTS / "hierarchical_scan.csv", index=False)
    hc = cl.hierarchical_labels(Z, K)
    df["hc_segment_id"] = hc
    fig, ax = plt.subplots(figsize=(12, 4.5))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, color_threshold=Z[-(K - 1), 2],
               above_threshold_color="grey")
    ax.set(title=f"Hierarchical clustering (Ward) - dendrogram cut at k = {K}",
           xlabel="Merged groups (size)", ylabel="Ward distance")
    save(fig, "fig06_dendrogram.png")

    # 5. Validation --------------------------------------------------------
    gap = cl.gap_statistic(X, k_max=10, B=20)
    gap.to_csv(C.RESULTS / "gap_statistic.csv", index=False)
    gap_brief = cl.gap_statistic(matrices["brief"][0], k_max=10, B=20)
    gap_brief.to_csv(C.RESULTS / "gap_statistic_brief.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for a, g, t in [(ax[0], gap, "Primary (brief + behaviour)"), (ax[1], gap_brief, "Brief fields only")]:
        a.errorbar(g.k, g.gap, yerr=g.s_k, marker="o", capsize=3, color=PALETTE[0])
        a.axvline(g.attrs["selected_k"], ls="--", c=PALETTE[1])
        a.set(title=f"Gap statistic - {t} (selected k = {g.attrs['selected_k']})", xlabel="k", ylabel="Gap")
    save(fig, "fig07_gap_statistic.png")

    perm_full = cl.permutation_null(X, K, n_perm=200)
    perm_brief = cl.permutation_null(matrices["brief"][0], K, n_perm=200)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for a, pr, t in [(ax[0], perm_brief, "Brief fields only"), (ax[1], perm_full, "Primary (brief + behaviour)")]:
        sns.histplot(pr["null_values"], bins=25, ax=a, color="lightgrey", edgecolor="grey")
        a.axvline(pr["observed"], color=PALETTE[1], lw=2.5, label=f"observed = {pr['observed']:.3f}")
        a.set(title=f"{t}: permutation p = {pr['p_value']:.3f}", xlabel="Silhouette on column-shuffled data")
        a.legend()
    save(fig, "fig08_permutation_null.png")

    stab = cl.bootstrap_stability(X, km.labels_, K, B=100)
    stab_brief_km = cl.fit_kmeans(matrices["brief"][0], K)
    stab_brief = cl.bootstrap_stability(matrices["brief"][0], stab_brief_km.labels_, K, B=100)

    ari_km_hc = cl.agreement(km.labels_, hc)

    # K-Prototypes on the brief's raw mixed-type fields (Huang, 1998)
    kp_df = df[["age", "satisfaction_score", "client_type", "acquisition_purpose", "loan_applied",
                "referral_channel", "country_group", "gender"]].copy()
    kp_df[["age", "satisfaction_score"]] = (kp_df[["age", "satisfaction_score"]]
                                            - kp_df[["age", "satisfaction_score"]].mean()) / kp_df[["age", "satisfaction_score"]].std()
    kp = KPrototypes(n_clusters=K, init="Cao", n_init=3, random_state=C.SEED)
    kp_labels = kp.fit_predict(kp_df.to_numpy(), categorical=list(range(2, 8)))
    ari_kp_brief = cl.agreement(kp_labels, stab_brief_km.labels_)
    ari_kp_full = cl.agreement(kp_labels, km.labels_)

    # 6. Profiling & naming ------------------------------------------------
    prof = P.profile(df, "segment_id")
    names = name_segments(prof)
    df["segment"] = df["segment_id"].map(names)
    stab["segment"] = stab["cluster"].map(names)
    prof = P.profile(df, "segment")
    prof.to_csv(C.RESULTS / "segment_profiles.csv")
    lift = P.lift(df, "segment"); lift.to_csv(C.RESULTS / "segment_lift.csv")
    stab.to_csv(C.RESULTS / "stability_primary.csv", index=False)

    df["brief_segment_id"] = stab_brief_km.labels_
    bprof = P.profile(df, "brief_segment_id")
    bnames = name_brief_segments(bprof)
    df["brief_segment"] = df["brief_segment_id"].map(bnames)
    stab_brief["segment"] = stab_brief["cluster"].map(bnames)
    stab_brief.to_csv(C.RESULTS / "stability_brief.csv", index=False)
    P.profile(df, "brief_segment").to_csv(C.RESULTS / "brief_segment_profiles.csv")

    order = ["Portfolio Builders", "Diversified Repeat Buyers", "Focused Residential Buyers",
             "Mixed-Use Office Buyers"]
    colors = dict(zip(order, PALETTE))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for a, st, t in [(ax[0], stab_brief, "Brief fields only"), (ax[1], stab, "Primary segments")]:
        a.barh(st["segment"], st["mean_jaccard"], color=PALETTE[0])
        a.axvline(0.75, ls="--", c=PALETTE[1]); a.axvline(0.6, ls=":", c="grey")
        a.set(xlim=(0, 1), title=f"Bootstrap stability - {t}", xlabel="Mean Jaccard (100 resamples)")
    save(fig, "fig09_stability.png")

    pca = PCA(n_components=2, random_state=C.SEED).fit(X)
    pcs = pca.transform(X)
    df["pc1"], df["pc2"] = pcs[:, 0], pcs[:, 1]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for sname in order:
        mk = df["segment"] == sname
        ax.scatter(df.loc[mk, "pc1"], df.loc[mk, "pc2"], s=12, alpha=.6, label=f"{sname} (n={mk.sum()})", color=colors[sname])
    ax.set(title="Primary segments in 2-D (PCA projection)",
           xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.legend(fontsize=8, loc="best")
    save(fig, "fig10_pca_segments.png")

    show = ["units_bought", "total_spend", "office_share", "tower_hhi", "recency_months",
            "age", "satisfaction_score"]
    z = prof.loc[order, show]
    z = (z - prof.loc["Overall", show]) / df[show].std()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    annot = prof.loc[order, show].copy().astype(object)
    for c in show:
        annot[c] = prof.loc[order, c].map(
            (lambda v: f"${v/1e6:.2f}M") if c == "total_spend" else (lambda v: f"{v:.2f}"))
    sns.heatmap(z, annot=annot, fmt="", cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "difference from average (SD units)"})
    ax.set_title("Segment profiles (cell = segment mean; colour = distance from overall average)")
    save(fig, "fig11_segment_profile_heatmap.png")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    sns.heatmap(lift.loc[order], annot=True, fmt=".2f", cmap="RdBu_r", center=1, ax=ax,
                cbar_kws={"label": "lift vs overall (1 = average)"})
    ax.set_title("Segment lift on binary traits")
    save(fig, "fig12_segment_lift.png")

    # Geography: location quotients (regional economics)
    lq_country = P.location_quotient(df, "country", "segment")
    lq_country.to_csv(C.RESULTS / "lq_country.csv", index=False)
    lq_mat = lq_country.pivot(index="country", columns="segment", values="lq")[order]
    nmap = df["country"].value_counts()
    lq_mat.index = [f"{c} (n={nmap[c]})" for c in lq_mat.index]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(lq_mat, annot=True, fmt=".2f", cmap="RdBu_r", center=1, vmin=0, vmax=2, ax=ax,
                cbar_kws={"label": "Location quotient"})
    ax.set_title("Where each segment is over/under-represented (LQ > 1 = more common than average)")
    save(fig, "fig13_location_quotient.png")

    # Brief-view quadrant figure
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (sname, g), c in zip(df.groupby("brief_segment"), PALETTE):
        ax.scatter(g["age"], g["satisfaction_score"] + np.random.default_rng(1).uniform(-.25, .25, len(g)),
                   s=8, alpha=.5, label=f"{sname} (n={len(g)})", color=c)
    ax.set(title="Brief-field clustering splits age x satisfaction into quadrants", xlabel="Age",
           ylabel="Satisfaction (jittered)")
    ax.legend(fontsize=8)
    save(fig, "fig14_brief_quadrants.png")

    # Hypothesis figure
    h = hyp.sort_values("p_bh")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(h["id"] + "  " + h["hypothesis"].str.slice(0, 48), -np.log10(h["p_bh"].clip(lower=1e-12)),
            color=[PALETTE[2] if r == "Supported" else "lightgrey" for r in h["result"]])
    ax.axvline(-np.log10(0.05), ls="--", c=PALETTE[1]); ax.invert_yaxis()
    ax.set(title="Pre-registered hypotheses (BH-adjusted)", xlabel="-log10(adjusted p)   [right of dashed line = significant]")
    save(fig, "fig15_hypotheses.png")

    # Decision usefulness (non-circular): do behaviour-only segments separate
    # marketing-relevant attributes that were NOT used to build them?
    Xb = matrices["behaviour"][0]
    kb = cl.fit_kmeans(Xb, K)
    du = P.decision_usefulness(df, kb.labels_, ["acquisition_purpose", "loan_applied", "referral_channel",
                                               "is_international", "client_type", "gender", "age_band"])
    du.to_csv(C.RESULTS / "decision_usefulness.csv", index=False)

    # 7. Persist -----------------------------------------------------------
    keep = ["client_id", "client_type", "gender", "country", "region", "country_group", "age", "age_band",
            "acquisition_purpose", "loan_applied", "referral_channel", "satisfaction_score",
            "units_bought", "total_spend", "avg_unit_price", "avg_price_per_sqft", "premium_index",
            "office_units", "office_share", "n_towers", "tower_hhi", "recency_months", "bought_both_years",
            "portfolio_builder", "is_company", "is_investment", "has_loan", "is_international", "is_female",
            "ref_agency", "ref_client", "segment_id", "segment", "hc_segment_id", "brief_segment",
            "pc1", "pc2"]
    df[keep].to_csv(C.DATA_PROCESSED / "client_segments.csv", index=False)
    props.merge(df[["client_id", "segment"]], left_on="client_ref", right_on="client_id", how="left") \
        .drop(columns="client_id").to_csv(C.DATA_PROCESSED / "properties_clean.csv", index=False)
    joblib.dump({"scaler": scaler, "kmeans": km, "features": cols, "names": names},
                C.ROOT / "results" / "segmentation_model.joblib")

    metrics = {
        "cleaning": log,
        "n_clients": int(len(df)), "n_units_sold": int(len(sold)),
        "chosen_k": K, "primary_features": cols,
        "silhouette_primary": float(scans["full"].set_index("k").loc[K, "silhouette"]),
        "silhouette_brief": float(scans["brief"].set_index("k").loc[K, "silhouette"]),
        "silhouette_behaviour_k3": float(scans["behaviour"].set_index("k").loc[3, "silhouette"]),
        "silhouette_hierarchical": float(hc_scan.set_index("k").loc[K, "silhouette"]),
        "gap_selected_k_primary": int(gap.attrs["selected_k"]),
        "gap_selected_k_brief": int(gap_brief.attrs["selected_k"]),
        "permutation_primary": {k: v for k, v in perm_full.items() if k != "null_values"},
        "permutation_brief": {k: v for k, v in perm_brief.items() if k != "null_values"},
        "ari_kmeans_vs_hierarchical": ari_km_hc,
        "ari_kprototypes_vs_brief_kmeans": ari_kp_brief,
        "ari_kprototypes_vs_primary": ari_kp_full,
        "pca_variance_2d": float(pca.explained_variance_ratio_[:2].sum()),
        "segment_names": {int(k): v for k, v in names.items()},
        "significant_independence_pairs_bh": int((ind["p_bh"] < 0.05).sum()),
        "independence_pairs_tested": int(len(ind)),
        "max_cramers_v": float(ind["cramers_v"].max()),
    }
    with open(C.RESULTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in metrics.items() if k != "cleaning"}, indent=2, default=str))


if __name__ == "__main__":
    main()
