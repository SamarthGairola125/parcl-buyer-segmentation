"""Parcl Buyer Intelligence - Streamlit dashboard.

Pages follow the project brief:
  1. Buyer Segmentation Overview   - cluster distribution
  2. Investor Behavior Dashboard   - investment patterns by cluster
  3. Geographic Buyer Analysis     - buyer segments by region
  4. Segment Insights Panel        - descriptive statistics per cluster
  +  Model Validation              - how far the segments can be trusted
Filters (brief): country, region, acquisition purpose, client type.

Run locally:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.insights import invest_title, segment_contrast_title, share_title  # noqa: E402
from src.uncertainty import icon_counts, rate_summary  # noqa: E402

SEG_ORDER = ["Portfolio Builders", "Diversified Repeat Buyers",
             "Focused Residential Buyers", "Mixed-Use Office Buyers"]
SEG_COLORS = dict(zip(SEG_ORDER, ["#1d4e89", "#e07a5f", "#3d9970", "#8e5ac8"]))
SEG_ICON = {"Portfolio Builders": "🏙️", "Diversified Repeat Buyers": "🔁",
            "Focused Residential Buyers": "🏠", "Mixed-Use Office Buyers": "🏢"}
SEG_TAGLINE = {
    "Portfolio Builders": "Buy many units across many towers",
    "Diversified Repeat Buyers": "Loyal core buying ~4 units over time",
    "Focused Residential Buyers": "Three apartments, no offices",
    "Mixed-Use Office Buyers": "Mix homes with office units",
}
PERSONAS = {
    "Portfolio Builders": {
        "who": "A small, high-value group buying 5-13 units (average 7.5) spread across many towers. "
               "The oldest and most satisfied buyers; every one of them bought in both 2024 and 2025.",
        "do": ["Assign relationship managers", "Early access to new towers",
               "Bulk-purchase pricing", "Portfolio performance reports"],
    },
    "Diversified Repeat Buyers": {
        "who": "Half of all clients. Typically 4 units in different towers, recently active, and "
               "85% bought in both years - the loyal core of the business.",
        "do": ["Loyalty & referral programme", "Cross-tower bundles",
               "Upgrade offers", "Nurture toward portfolio building"],
    },
    "Focused Residential Buyers": {
        "who": "Exactly three apartments, no offices, concentrated in fewer towers. "
               "They have gone longest without a new purchase.",
        "do": ["Re-engagement campaigns", "Residential lifestyle messaging",
               "Single-tower community perks", "Watch for lapsing"],
    },
    "Mixed-Use Office Buyers": {
        "who": "About three units with a high share of offices (~39%) and the highest average "
               "unit price. Membership is the least stable of the four.",
        "do": ["Commercial & mixed-use listings", "Office-leasing support",
               "Yield-focused messaging", "Test campaigns before scaling"],
    },
}
US_STATES = {"California": "CA", "Oregon": "OR", "Arizona": "AZ", "Nevada": "NV", "Colorado": "CO",
             "Utah": "UT", "Virginia": "VA", "Wyoming": "WY", "Kansas": "KS", "Ohio": "OH",
             "Washington": "WA", "Texas": "TX", "Florida": "FL", "New York": "NY", "Georgia": "GA"}
ISO3 = {"USA": "USA", "UK": "GBR", "Canada": "CAN", "Germany": "DEU", "France": "FRA",
        "Belgium": "BEL", "Mexico": "MEX", "Australia": "AUS", "Russia": "RUS", "Denmark": "DNK"}

st.set_page_config(page_title="Parcl Buyer Intelligence", page_icon="🏙️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {background: linear-gradient(120deg, #0f2d52 0%, #1d4e89 60%, #2a6f97 100%);
       color: #fff; padding: 22px 28px; border-radius: 16px; margin-bottom: 14px;}
.hero h1 {font-size: 2rem; margin: 0 0 4px 0; color: #fff;}
.hero p {margin: 0; opacity: .85; font-size: 1rem;}
.kpi {background: #fff; border: 1px solid #e6e9ef; border-radius: 14px; padding: 14px 16px;
      box-shadow: 0 1px 3px rgba(16,24,40,.06); height: 100%;}
.kpi .lbl {font-size: .78rem; color: #667085; text-transform: uppercase; letter-spacing: .04em;}
.kpi .val {font-size: 1.75rem; font-weight: 700; color: #101828; line-height: 1.2;}
.kpi .sub {font-size: .8rem; color: #667085;}
.seg {border-radius: 14px; padding: 16px; color: #fff; height: 100%;
      box-shadow: 0 2px 6px rgba(16,24,40,.12);}
.seg .ico {font-size: 1.9rem;}
.seg .nm {font-weight: 700; font-size: 1.02rem; margin-top: 2px;}
.seg .pct {font-size: 2.2rem; font-weight: 800; line-height: 1.1;}
.seg .tg {font-size: .84rem; opacity: .92; min-height: 2.4em;}
.seg .row {display: flex; justify-content: space-between; font-size: .82rem; margin-top: 8px;
           border-top: 1px solid rgba(255,255,255,.3); padding-top: 8px;}
.badge {display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: .72rem;
        background: rgba(255,255,255,.22); margin-top: 6px;}
.insight {background: #f0f6ff; border-left: 5px solid #1d4e89; padding: 12px 16px;
          border-radius: 8px; margin: 6px 0 14px 0; font-size: .95rem;}
.insight b {color: #0f2d52;}
.howto {color: #667085; font-size: .85rem; margin-top: -6px; margin-bottom: 8px;}
.waffle {display: grid; grid-template-columns: repeat(20, 1fr); gap: 4px; max-width: 560px;}
.waffle div {aspect-ratio: 1; border-radius: 3px;}
.bar-wrap {background: #eef1f6; border-radius: 6px; height: 12px; width: 100%; margin: 3px 0 9px 0;}
.bar {height: 12px; border-radius: 6px;}
.persona {border-radius: 16px; padding: 20px 22px; color: #fff;}
.persona h3 {color: #fff; margin: 0 0 6px 0;}
.chip {display: inline-block; background: rgba(255,255,255,.18); padding: 4px 10px;
       border-radius: 999px; margin: 4px 6px 0 0; font-size: .85rem;}
.stat {background: #fff; border: 1px solid #e6e9ef; border-radius: 12px; padding: 12px 14px;}
.stat .lbl {font-size: .76rem; color: #667085;}
.stat .val {font-size: 1.35rem; font-weight: 700; color: #101828;}
.up {color: #12805c; font-size: .8rem;} .down {color: #c4320a; font-size: .8rem;}
.flat {color: #667085; font-size: .8rem;}
html, body, [class*="css"], .kpi .val, .stat .val, .seg .pct {font-variant-numeric: tabular-nums;}
.action {font-size: 1.25rem; font-weight: 700; color: #0f2d52; border-left: 5px solid #d99a2b;
         padding: 4px 0 4px 12px; margin: 4px 0 10px 0;}
.chip-obs, .chip-mod, .chip-small {display:inline-block; font-size:.7rem; font-weight:600; padding:2px 8px;
         border-radius:999px; margin-left:6px; vertical-align:middle;}
.chip-obs {background:#e8f1fb; color:#1d4e89; border:1px solid #b9d0ea;}
.chip-mod {background:#fdf3e1; color:#8a5a00; border:1px solid #f0d29a;}
.chip-small {background:#fff1ec; color:#b4441f; border:1px solid #f5c3b0;}
.ci {font-size:.72rem; color:#667085; margin-top:-6px; margin-bottom:6px;}
.flow {display:flex; gap:8px; align-items:stretch; flex-wrap:wrap; margin:6px 0 14px 0;}
.flow .step {flex:1 1 140px; background:#fff; border:1px solid #e6e9ef; border-top:4px solid #1d4e89;
         border-radius:12px; padding:10px 12px;}
.flow .step .n {font-size:1.35rem; font-weight:800; color:#0f2d52;}
.flow .step .t {font-size:.78rem; color:#667085;}
.flow .arrow {align-self:center; color:#d99a2b; font-size:1.4rem; font-weight:700;}
.icons {display:grid; grid-template-columns:repeat(10, 16px); gap:3px; margin:6px 0;}
.icons span {font-size:14px; line-height:16px; text-align:center;}
.prov {font-size:.75rem; color:#667085; border-top:1px solid #e6e9ef; margin-top:24px; padding-top:10px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    clients = pd.read_csv(PROC / "client_segments.csv")
    props = pd.read_csv(PROC / "properties_clean.csv", parse_dates=["transaction_date"])
    metrics = json.loads((RES / "metrics.json").read_text())
    tables = {n: pd.read_csv(RES / f"{n}.csv") for n in
              ["kmeans_scan", "gap_statistic", "stability_primary", "stability_brief",
               "hypothesis_tests", "lq_country"]}
    emb = pd.read_csv(RES / "embedding_3d_v1.csv")
    emb_meta = json.loads((RES / "embedding_3d_v1.meta.json").read_text())
    model_hash = hashlib.sha256((RES / "segmentation_model.joblib").read_bytes()).hexdigest()
    checks = (ROOT / "data" / "raw" / "CHECKSUMS.sha256").read_text().split()
    data_hash = {checks[i + 1]: checks[i][:12] for i in range(0, len(checks), 2)}
    return clients, props, metrics, tables, emb, emb_meta, model_hash, data_hash


clients, props, M, T, EMB, EMB_META, MODEL_HASH, DATA_HASH = load()
STAB = T["stability_primary"].set_index("segment")["mean_jaccard"]


def money(v):
    return f"${v/1e9:,.2f}B" if v >= 1e9 else f"${v/1e6:,.2f}M" if v >= 1e6 else f"${v/1e3:,.0f}K"


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="lbl">{label}</div><div class="val">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def action(text, chips=("obs",)):
    lab = {"obs": '<span class="chip-obs">Observed in data</span>',
           "mod": '<span class="chip-mod">Model output</span>'}
    st.markdown(f'<div class="action">{text}{"".join(lab[c] for c in chips)}</div>',
                unsafe_allow_html=True)


def icon_array(rate, color, icon="●"):
    k = icon_counts(rate)
    cells = "".join(f'<span style="color:{color if i < k else "#d5dbe5"}">{icon}</span>'
                    for i in range(100))
    return f'<div class="icons" role="img" aria-label="{k} out of 100">{cells}</div>'


def howto(text):
    st.markdown(f'<div class="howto">How to read: {text}</div>', unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(size=13), legend_title_text="", title_font_size=16)
    return fig


# ---------------- Sidebar filters (brief: country, region, purpose, client type) ----------------
with st.sidebar:
    st.markdown("### 🔎 Filters")
    st.caption("Leave a filter empty to include everything.")
    sel_country = st.multiselect("Country", sorted(clients["country"].unique()), placeholder="All countries")
    pool = clients if not sel_country else clients[clients["country"].isin(sel_country)]
    sel_region = st.multiselect("Region", sorted(pool["region"].unique()), placeholder="All regions")
    sel_purpose = st.multiselect("Acquisition purpose", ["Home", "Investment"], placeholder="All purposes")
    sel_type = st.multiselect("Client type", ["Individual", "Company"], placeholder="All client types")
    st.divider()
    st.markdown("**Segment colours**")
    for s in SEG_ORDER:
        st.markdown(f'<span style="color:{SEG_COLORS[s]};font-size:1.1rem">■</span> {SEG_ICON[s]} {s}',
                    unsafe_allow_html=True)
    st.divider()
    st.caption("Segments were built once from all 2,000 clients with K-Means (k = 4) on client and "
               "purchase-behaviour features. Filters change the view, not the model.")

f = clients.copy()
for col, sel in [("country", sel_country), ("region", sel_region),
                 ("acquisition_purpose", sel_purpose), ("client_type", sel_type)]:
    if sel:
        f = f[f[col].isin(sel)]
fp = props[props["client_ref"].isin(f["client_id"])]

# ---------------- Header ----------------
st.markdown('<div class="hero"><h1>🏙️ Parcl Buyer Intelligence</h1>'
            '<p>Who buys Parcl properties, why they buy, how they pay and where they live - '
            'discovered with machine-learning clustering.</p></div>', unsafe_allow_html=True)
if f.empty:
    st.warning("No clients match these filters. Clear a filter in the sidebar.")
    st.stop()

k = st.columns(5)
kpi(k[0], "Clients", f"{len(f):,}", f"of {len(clients):,} in total")
kpi(k[1], "Units bought", f"{int(f['units_bought'].sum()):,}", f"{f['units_bought'].mean():.1f} per client")
kpi(k[2], "Total spend", money(f["total_spend"].sum()), f"{money(f['total_spend'].mean())} per client")
kpi(k[3], "Investment buyers", f"{f['is_investment'].mean()*100:.0f}%", "bought to invest")
kpi(k[4], "Used a loan", f"{f['has_loan'].mean()*100:.0f}%", "applied for financing")
st.write("")

tabs = st.tabs(["📊 Buyer Segmentation Overview", "💼 Investor Behavior Dashboard",
                "🌍 Geographic Buyer Analysis", "🔍 Segment Insights Panel", "✅ Model Validation"])
present = [s for s in SEG_ORDER if s in set(f["segment"])]
counts = f["segment"].value_counts().reindex(SEG_ORDER).fillna(0).astype(int)

# =============== 1. BUYER SEGMENTATION OVERVIEW ===============
with tabs[0]:
    action(share_title(f), chips=("mod",))
    insight(f"<b>{len(f):,} buyers fall into four segments defined by how they buy</b> - how many units, "
            "whether they buy offices, and how widely they spread across towers. "
            "Demographics alone did not produce real segments (see Model Validation).")
    cols = st.columns(4)
    for c, s in zip(cols, SEG_ORDER):
        g = f[f["segment"] == s]
        share = len(g) / len(f) * 100
        units = f"{g['units_bought'].mean():.1f}" if len(g) else "-"
        spend = money(g["total_spend"].mean()) if len(g) else "-"
        stable = "✓ Stable" if STAB[s] >= 0.75 else "◐ Less stable"
        if 0 < len(g) < 100:
            stable += f" · ⚠ small sample (n={len(g)})"
        c.markdown(f'<div class="seg" style="background:{SEG_COLORS[s]}">'
                   f'<div class="ico">{SEG_ICON[s]}</div><div class="nm">{s}</div>'
                   f'<div class="pct">{share:.0f}%</div><div class="tg">{SEG_TAGLINE[s]}</div>'
                   f'<div class="row"><span>👥 {len(g):,}</span><span>🏢 {units} units</span>'
                   f'<span>💰 {spend}</span></div><span class="badge">{stable}</span></div>',
                   unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([1, 1])
    with a:
        st.markdown("##### Cluster distribution - every square is 1% of buyers")
        raw = counts / counts.sum() * 100
        cells = raw.astype(int)
        for s in (raw - cells).sort_values(ascending=False).index[: 100 - cells.sum()]:
            cells[s] += 1
        squares = "".join(f'<div style="background:{SEG_COLORS[s]}" title="{s}"></div>'
                          for s in SEG_ORDER for _ in range(int(cells[s])))
        st.markdown(f'<div class="waffle">{squares}</div>', unsafe_allow_html=True)
        st.markdown(" &nbsp; ".join(f'<span style="color:{SEG_COLORS[s]}">■</span> {s} ({cells[s]}%)'
                                    for s in SEG_ORDER), unsafe_allow_html=True)
    with b:
        d = pd.DataFrame({"segment": SEG_ORDER, "clients": counts.values,
                          "spend": [f.loc[f.segment == s, "total_spend"].sum() for s in SEG_ORDER]})
        d["Share of clients"] = d["clients"] / d["clients"].sum() * 100
        d["Share of spend"] = d["spend"] / d["spend"].sum() * 100
        m = d.melt(id_vars="segment", value_vars=["Share of clients", "Share of spend"])
        fig = px.bar(m, y="variable", x="value", color="segment", orientation="h",
                     color_discrete_map=SEG_COLORS, text=m["value"].map(lambda v: f"{v:.0f}%" if v >= 4 else ""),
                     title="Share of clients vs share of spend")
        fig.update_layout(barmode="stack", xaxis_title="%", yaxis_title="", showlegend=False)
        st.plotly_chart(style(fig, 300), width="stretch")
        howto("Portfolio Builders are a sliver of clients but a bigger slice of spend.")
    st.markdown("##### 🌌 Buyer Galaxy - every dot is one real buyer")
    mode = st.radio("View", ["3D (rotate, zoom)", "2D (precise reading)"], horizontal=True,
                    key="galaxy_mode", label_visibility="collapsed")
    gal = EMB[EMB["client_id"].isin(f["client_id"])].merge(
        f[["client_id", "units_bought", "country", "total_spend"]], on="client_id")
    tl = EMB_META["top_loadings"]
    names = {"units_bought": "units", "tower_hhi": "tower spread", "total_spend": "spend",
             "premium_index": "price premium", "office_share": "office share", "age": "age",
             "recency_months": "recency", "satisfaction_score": "satisfaction"}
    ax = {k: f"{k.upper()}: " + ", ".join(names.get(v, v) for v in tl[k][:2]) for k in ["pc1", "pc2", "pc3"]}
    hover = {"client_id": True, "units_bought": True, "country": True,
             "pc1": False, "pc2": False, "pc3": False}
    if mode.startswith("3D"):
        fig = px.scatter_3d(gal, x="pc1", y="pc2", z="pc3", color="segment", color_discrete_map=SEG_COLORS,
                            category_orders={"segment": SEG_ORDER}, opacity=.7, hover_data=hover,
                            labels={"pc1": ax["pc1"], "pc2": ax["pc2"], "pc3": ax["pc3"]})
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(scene=dict(camera=dict(eye=dict(x=1.35, y=1.35, z=0.75)), aspectmode="cube"),
                          legend=dict(orientation="h", y=1.02, x=0))
        st.plotly_chart(style(fig, 560).update_layout(margin=dict(l=0, r=0, t=30, b=0)), width="stretch")
        howto("drag to rotate, scroll to zoom, double-click to reset the view. Use 2D for exact positions.")
    else:
        fig = px.scatter(gal, x="pc1", y="pc2", color="segment", color_discrete_map=SEG_COLORS,
                         category_orders={"segment": SEG_ORDER}, opacity=.6, hover_data=hover,
                         labels={"pc1": ax["pc1"], "pc2": ax["pc2"]})
        st.plotly_chart(style(fig, 480), width="stretch")
    ev = EMB_META["explained_variance"]
    st.caption(f"Positions come from a PCA projection of the clustering features "
               f"(3 axes explain {sum(ev)*100:.0f}% of the variation). Segments overlap at the edges - "
               "they are tendencies, not hard boxes. Showing all {:,} buyers in view; no sampling.".format(len(gal)))

# =============== 2. INVESTOR BEHAVIOR DASHBOARD ===============
with tabs[1]:
    action(invest_title(f), chips=("obs", "mod"))
    metrics_rows = [("💼 Buy to invest", "is_investment", 1), ("🏦 Applied for a loan", "has_loan", 1),
                    ("🏢 Units that are offices", "office_share", 1), ("🔁 Bought in both years", "bought_both_years", 1)]
    cols = st.columns(len(present))
    for c, s in zip(cols, present):
        g = f[f["segment"] == s]
        html = (f'<div class="kpi" style="border-top:5px solid {SEG_COLORS[s]}">'
                f'<div style="font-weight:700;margin-bottom:6px">{SEG_ICON[s]} {s}</div>')
        if len(g) < 100:
            html += f'<span class="chip-small" style="margin:0 0 6px 0">⚠ small sample, n={len(g)}</span>'
        for lbl, var, _ in metrics_rows:
            if var == "office_share":
                v = g[var].mean() * 100
                ci = f"average across {len(g)} buyers"
            else:
                r = rate_summary(g[var].astype(int).tolist())
                v = r["rate"] * 100
                ci = f'{r["k"]} of {r["n"]} · 95% range {r["ci_low"]*100:.0f}–{r["ci_high"]*100:.0f}%'
            html += (f'<div style="font-size:.82rem;display:flex;justify-content:space-between">'
                     f'<span>{lbl}</span><b>{v:.0f}%</b></div>'
                     f'<div class="bar-wrap"><div class="bar" style="width:{v:.0f}%;background:{SEG_COLORS[s]}"></div></div>'
                     f'<div class="ci">{ci}</div>')
        html += (f'<div style="font-size:.82rem;margin-top:4px">🏠 Avg units: <b>{g["units_bought"].mean():.1f}</b>'
                 f' &nbsp; 💰 Avg spend: <b>{money(g["total_spend"].mean())}</b></div></div>')
        c.markdown(html, unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    agg = f.groupby("segment").agg(clients=("client_id", "count"), units=("units_bought", "mean"),
                                   spend=("total_spend", "mean"), invest=("is_investment", "mean")).reindex(present).reset_index()
    fig = px.scatter(agg, x="units", y="spend", size="clients", color="segment", color_discrete_map=SEG_COLORS,
                     size_max=70, title="Units bought vs spend per client (bubble = number of clients)",
                     labels={"units": "Average units bought", "spend": "Average total spend ($)"})
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    a.plotly_chart(style(fig, 400), width="stretch")
    mix = (pd.crosstab(f["segment"], f["acquisition_purpose"], normalize="index") * 100).reindex(present).reset_index()
    mixm = mix.melt(id_vars="segment", var_name="Purpose", value_name="pct")
    fig = px.bar(mixm, y="segment", x="pct", color="Purpose", orientation="h",
                 color_discrete_map={"Home": "#9db4c0", "Investment": "#1d4e89"},
                 text=mixm["pct"].map(lambda v: f"{v:.0f}%"), title="Home vs investment purpose within each segment")
    fig.update_layout(xaxis_title="% of segment", yaxis_title="", barmode="stack")
    b.plotly_chart(style(fig, 400), width="stretch")
    a, b = st.columns(2)
    trend = fp.groupby([fp["transaction_date"].dt.to_period("M").dt.to_timestamp(), "segment"]).size().reset_index(name="units")
    fig = px.area(trend, x="transaction_date", y="units", color="segment", color_discrete_map=SEG_COLORS,
                  category_orders={"segment": SEG_ORDER}, title="Units purchased each month by segment")
    fig.update_layout(xaxis_title="", yaxis_title="Units")
    a.plotly_chart(style(fig, 380), width="stretch")
    cat = (pd.crosstab(fp["segment"], fp["unit_category"], normalize="index") * 100).reindex(present).reset_index()
    catm = cat.melt(id_vars="segment", var_name="Unit type", value_name="pct")
    fig = px.bar(catm, y="segment", x="pct", color="Unit type", orientation="h",
                 color_discrete_map={"Apartment": "#9db4c0", "Office": "#8e5ac8"},
                 text=catm["pct"].map(lambda v: f"{v:.0f}%" if v >= 3 else ""),
                 title="What each segment bought: apartments vs offices")
    fig.update_layout(barmode="stack", xaxis_title="% of units bought", yaxis_title="")
    b.plotly_chart(style(fig, 380), width="stretch")

# =============== 3. GEOGRAPHIC BUYER ANALYSIS ===============
with tabs[2]:
    st.subheader("Where each segment lives")
    insight("Raw counts mostly show that 77% of buyers are in the USA, so this page uses the "
            "<b>location quotient (LQ)</b>: a segment's share in a place ÷ its share overall. "
            "<b>LQ 1.5 = 50% more common than average</b>; LQ 0.5 = half as common. "
            "Clear examples: Focused Residential Buyers are 1.46× as common in Germany; "
            "Diversified Repeat Buyers 1.30× in Belgium.")
    seg_pick = st.radio("Segment to map", SEG_ORDER, horizontal=True,
                        format_func=lambda s: f"{SEG_ICON[s]} {s}")
    overall = (clients["segment"] == seg_pick).mean()
    a, b = st.columns(2)
    geo = f.groupby("country").agg(clients=("client_id", "count"),
                                   share=("segment", lambda s: (s == seg_pick).mean())).reset_index()
    geo["LQ"] = geo["share"] / overall
    geo["iso3"] = geo["country"].map(ISO3)
    fig = px.choropleth(geo, locations="iso3", color="LQ", hover_name="country",
                        hover_data={"clients": True, "LQ": ":.2f", "iso3": False},
                        color_continuous_scale="RdBu_r", range_color=(0, 2), projection="natural earth",
                        title=f"By country - {seg_pick}")
    fig.update_geos(showcountries=True, countrycolor="#bbb", showframe=False)
    a.plotly_chart(style(fig, 400), width="stretch")
    us = f[f["country"] == "USA"].groupby("region").agg(
        clients=("client_id", "count"), share=("segment", lambda s: (s == seg_pick).mean())).reset_index()
    if not us.empty:
        us["LQ"] = us["share"] / overall
        us["code"] = us["region"].map(US_STATES)
        fig = px.choropleth(us, locations="code", locationmode="USA-states", color="LQ", scope="usa",
                            hover_name="region", hover_data={"clients": True, "LQ": ":.2f", "code": False},
                            color_continuous_scale="RdBu_r", range_color=(0, 2),
                            title=f"By US state - {seg_pick}")
        b.plotly_chart(style(fig, 400), width="stretch")
    else:
        b.info("No US buyers under the current filters.")
    howto("red = segment more common than average, blue = less common, white = about average.")
    st.markdown("##### Segment mix by region")
    reg = f.groupby(["country", "region", "segment"]).size().reset_index(name="n")
    tot = reg.groupby(["country", "region"])["n"].transform("sum")
    reg["pct"] = reg["n"] / tot * 100
    reg["label"] = reg["region"] + " (" + reg["country"] + ", n=" + tot.astype(str) + ")"
    min_n = st.slider("Show regions with at least this many buyers", 5, 100, 30, 5)
    keep = reg[tot >= min_n]
    order = keep.groupby("label")["n"].sum().sort_values().index.tolist()
    fig = px.bar(keep, y="label", x="pct", color="segment", orientation="h", color_discrete_map=SEG_COLORS,
                 category_orders={"segment": SEG_ORDER, "label": order},
                 title=f"Share of each segment in regions with ≥ {min_n} buyers")
    fig.update_layout(barmode="stack", xaxis_title="% of buyers in region", yaxis_title="")
    st.plotly_chart(style(fig, max(320, 26 * len(order) + 120)), width="stretch")
    howto("regions with few buyers swing a lot by chance, so the default hides regions under 30 buyers.")
    with st.expander("Statistically clear country concentrations (90% bootstrap interval excludes 1, n ≥ 30)"):
        lq = T["lq_country"]
        sig = lq[lq["significant"] & lq["reliable"]].copy()
        sig["reading"] = sig["lq"].map(lambda v: f"{v:.2f}× average" if v > 0 else "none observed")
        st.dataframe(sig[["country", "segment", "n_area", "reading", "ci_low", "ci_high"]].rename(
            columns={"n_area": "buyers in country", "ci_low": "interval low", "ci_high": "interval high"})
            .style.format({"interval low": "{:.2f}", "interval high": "{:.2f}"}), width="stretch", hide_index=True)

# =============== 4. SEGMENT INSIGHTS PANEL ===============
with tabs[3]:
    pick = st.radio("Choose a segment", SEG_ORDER, horizontal=True, key="ins",
                    format_func=lambda s: f"{SEG_ICON[s]} {s}")
    s = f[f["segment"] == pick]
    if s.empty:
        st.info("No buyers in this segment under the current filters.")
    else:
        chips = "".join(f'<span class="chip">✔ {x}</span>' for x in PERSONAS[pick]["do"])
        st.markdown(f'<div class="persona" style="background:{SEG_COLORS[pick]}">'
                    f'<h3>{SEG_ICON[pick]} {pick} · {len(s):,} buyers ({len(s)/len(f)*100:.0f}%)</h3>'
                    f'<div>{PERSONAS[pick]["who"]}</div>'
                    f'<div style="margin-top:10px;font-weight:700">What Parcl should do</div>{chips}</div>',
                    unsafe_allow_html=True)
        st.write("")
        action(segment_contrast_title(s, clients, pick), chips=("mod",))
        ia, ib = st.columns(2)
        for col, var, lbl in [(ia, "is_investment", "bought to invest"), (ib, "has_loan", "applied for a loan")]:
            r = rate_summary(s[var].astype(int).tolist())
            small = (f'<span class="chip-small">⚠ small sample, n={r["n"]}</span>' if r["small_sample"] else "")
            col.markdown(f'<div class="kpi"><div style="font-weight:700">About <span style="color:{SEG_COLORS[pick]}">'
                         f'{icon_counts(r["rate"])} in 100</span> {pick} {lbl}{small}</div>'
                         f'{icon_array(r["rate"], SEG_COLORS[pick])}'
                         f'<div class="sub">{r["k"]} of {r["n"]} buyers · 95% range '
                         f'{r["ci_low"]*100:.0f}–{r["ci_high"]*100:.0f} in 100</div></div>',
                         unsafe_allow_html=True)
        st.write("")
        st.markdown("##### Compared with the average buyer")
        stats = [("Avg units bought", "units_bought", "{:.1f}"), ("Avg total spend", "total_spend", "money"),
                 ("Avg unit price", "avg_unit_price", "money"), ("Office share", "office_share", "pct"),
                 ("Avg age", "age", "{:.1f}"), ("Satisfaction (1-5)", "satisfaction_score", "{:.2f}"),
                 ("Months since last buy", "recency_months", "{:.1f}"), ("Towers bought in", "n_towers", "{:.1f}")]
        cols = st.columns(4)
        for i, (lbl, var, fmt) in enumerate(stats):
            v, o = s[var].mean(), clients[var].mean()
            show = money(v) if fmt == "money" else f"{v*100:.0f}%" if fmt == "pct" else fmt.format(v)
            diff = (v - o) / o * 100 if o else 0
            cls, arrow = ("up", "▲") if diff > 3 else ("down", "▼") if diff < -3 else ("flat", "●")
            cols[i % 4].markdown(f'<div class="stat"><div class="lbl">{lbl}</div><div class="val">{show}</div>'
                                 f'<div class="{cls}">{arrow} {diff:+.0f}% vs average</div></div>',
                                 unsafe_allow_html=True)
            if i == 3:
                st.write("")
                cols = st.columns(4)
        st.write("")
        a, b, c = st.columns(3)
        for col, var, title in [(a, "client_type", "Client type"), (b, "referral_channel", "How they found Parcl"),
                                (c, "age_band", "Age band")]:
            d = s[var].value_counts().reset_index()
            d.columns = [var, "n"]
            fig = px.pie(d, names=var, values="n", hole=.55, title=title,
                         color_discrete_sequence=[SEG_COLORS[pick], "#c9d6e3", "#8aa1b1", "#e3c9a8", "#b5c99a"])
            fig.update_traces(textinfo="percent+label", textposition="inside")
            col.plotly_chart(style(fig, 300).update_layout(showlegend=False), width="stretch")
        st.markdown("##### Descriptive statistics")
        num = {"age": "Age", "satisfaction_score": "Satisfaction", "units_bought": "Units bought",
               "total_spend": "Total spend ($)", "avg_unit_price": "Avg unit price ($)",
               "office_share": "Office share", "n_towers": "Towers", "recency_months": "Months since last buy"}
        desc = s[list(num)].describe().T.rename(index=num)[["count", "mean", "std", "min", "50%", "max"]]
        st.dataframe(desc.rename(columns={"50%": "median"}).style.format("{:,.2f}"), width="stretch")
        st.download_button("⬇️ Download this segment (client IDs and features, no names)",
                           s.to_csv(index=False), file_name=f"{pick.lower().replace(' ', '_')}.csv")
    st.divider()
    st.markdown("##### The brief's suggested segments vs what the data shows")
    brief = [("C1 Global Investors", "High income, investment purchases", "❌ Not found",
              "International buyers are not more investment-focused and spend ~3% less. No income data."),
             ("C2 First-Time Buyers", "Younger, loan dependent", "❌ Not found",
              "Loan users are not younger. Every client owns 3+ units, so first-time status can't be seen."),
             ("C3 Corporate Buyers", "Companies buying multiple units", "❌ Not as described",
              "Companies buy the same number of units; their representatives are ~8 years younger."),
             ("C4 Luxury Investors", "High satisfaction, large investments", "◐ Partly",
              "Portfolio Builders spend most and are more satisfied (3.39 vs 3.03).")]
    cols = st.columns(4)
    for col, (nm, exp, verdict, ev) in zip(cols, brief):
        col.markdown(f'<div class="kpi"><div style="font-weight:700">{nm}</div>'
                     f'<div class="sub">Expected: {exp}</div><div style="font-size:1.1rem;margin:8px 0">{verdict}</div>'
                     f'<div style="font-size:.84rem">{ev}</div></div>', unsafe_allow_html=True)

# =============== 5. MODEL VALIDATION ===============
with tabs[4]:
    st.subheader("How this analysis works")
    cl = M["cleaning"]
    steps = [(f'{cl["clean_clients"]["rows_out"]:,} + {cl["clean_properties"]["rows_out"]:,}', "clients + property listings loaded"),
             (f'{sum(cl["validation"].values())}/{len(cl["validation"])}', "data quality checks passed"),
             (f'{len(M["primary_features"])}', "features per buyer (client + purchase behaviour)"),
             (f'k = {M["chosen_k"]}', "K-Means, checked against hierarchical clustering"),
             ("3 tests", "shuffle test, gap statistic, bootstrap stability"),
             (f'{len(M["segment_names"])} segments', "named from their profiles")]
    flow = '<span class="arrow">→</span>'.join(
        f'<div class="step"><div class="n">{n}</div><div class="t">{t}</div></div>' for n, t in steps)
    st.markdown(f'<div class="flow">{flow}</div>', unsafe_allow_html=True)
    st.subheader("Can these segments be trusted?")
    insight("Clustering always returns clusters - even from random noise. So we tested the segments "
            "against <b>shuffled data</b> (where every real relationship is destroyed). "
            "Segments from the brief's client fields alone did <b>no better than shuffled data</b>; "
            "adding purchase behaviour produced segments that <b>clearly beat it</b>.")
    pm, pb = M["permutation_primary"], M["permutation_brief"]
    c = st.columns(4)
    kpi(c[0], "Brief fields only", "❌ Not real", f"shuffle test p = {pb['p_value']:.2f}")
    kpi(c[1], "Brief + behaviour", "✅ Real", f"shuffle test p = {pm['p_value']:.3f}")
    kpi(c[2], "K-Means vs Hierarchical", f"{M['ari_kmeans_vs_hierarchical']:.2f}", "agreement (1 = identical)")
    kpi(c[3], "Stable segments", f"{int((STAB >= .75).sum())} of 4", "survive 100 resamples")
    st.write("")
    a, b = st.columns(2)
    nulls = pd.DataFrame({"Model": ["Brief fields only", "Brief + behaviour"],
                          "Observed": [pb["observed"], pm["observed"]],
                          "Shuffled data (95th pct)": [pb["null_p95"], pm["null_p95"]]})
    fig = go.Figure()
    fig.add_bar(name="Shuffled data (95th percentile)", x=nulls["Model"], y=nulls["Shuffled data (95th pct)"],
                marker_color="#c9d6e3")
    fig.add_bar(name="Our segments", x=nulls["Model"], y=nulls["Observed"], marker_color="#1d4e89")
    fig.update_layout(barmode="group", title="Cluster quality: our segments vs shuffled data", yaxis_title="Silhouette")
    a.plotly_chart(style(fig, 360), width="stretch")
    st_df = T["stability_primary"].copy()
    fig = px.bar(st_df, x="mean_jaccard", y="segment", orientation="h", color="segment",
                 color_discrete_map=SEG_COLORS, range_x=[0, 1], text=st_df["mean_jaccard"].map(lambda v: f"{v:.2f}"),
                 title="Stability: how often each segment re-appears (100 resamples)")
    fig.update_traces(textposition="inside", insidetextanchor="start")
    fig.add_vline(x=0.75, line_dash="dash", annotation_text="stable ≥ 0.75", annotation_position="top left")
    fig.update_layout(showlegend=False, xaxis_title="Mean Jaccard similarity", yaxis_title="")
    b.plotly_chart(style(fig, 360), width="stretch")
    a, b = st.columns(2)
    scan = T["kmeans_scan"]
    full = scan[scan["view"] == "full"]
    fig = px.line(full, x="k", y="inertia", markers=True, title="Elbow method (primary features)")
    fig.add_vline(x=4, line_dash="dash", annotation_text="chosen k = 4")
    a.plotly_chart(style(fig, 340), width="stretch")
    sc = scan.replace({"view": {"brief": "Brief fields", "behaviour": "Behaviour only", "full": "Brief + behaviour"}})
    fig = px.line(sc, x="k", y="silhouette", color="view", markers=True, title="Silhouette score by number of clusters")
    fig.add_vline(x=4, line_dash="dash")
    b.plotly_chart(style(fig, 340), width="stretch")
    with st.expander("Pre-registered hypotheses (Benjamini-Hochberg adjusted)"):
        h = T["hypothesis_tests"][["id", "hypothesis", "evidence", "p_bh", "result"]].copy()
        h["result"] = h["result"].map({"Supported": "✅ Supported", "Not supported": "❌ Not supported"})
        st.dataframe(h.style.format({"p_bh": "{:.4f}"}), width="stretch", hide_index=True)
    st.caption("Full method and discussion: reports/research_paper.pdf in the project repository.")

# ---------------- Provenance footer (every page) ----------------
st.markdown(
    f'<div class="prov">🔎 <b>Provenance</b> · Model: K-Means k = {M["chosen_k"]}, seed 42, '
    f'file hash {MODEL_HASH[:12]} · Data: clients.csv {DATA_HASH.get("clients.csv", "?")}, '
    f'properties.csv {DATA_HASH.get("properties.csv", "?")} · 3D coordinates generated '
    f'{EMB_META["generated_at"][:10]} · '
    '<span class="chip-obs">Observed in data</span> = counted directly from the files; '
    '<span class="chip-mod">Model output</span> = depends on the clustering. '
    'Data-science internship project (Unified Mentor) - not an official Parcl product.</div>',
    unsafe_allow_html=True)
