"""Client-level feature engineering.

Demographic fields come from clients.csv. Behavioural fields are built by
aggregating each client's purchased units in properties.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import BINARY_FEATURES, REFERENCE_DATE


def _hhi(values: pd.Series) -> float:
    """Herfindahl-Hirschman index of a client's units across towers (0-1).
    1 = all units in one tower; lower = more diversified."""
    shares = values.value_counts(normalize=True)
    return float((shares ** 2).sum())


def build_client_features(clients: pd.DataFrame, properties: pd.DataFrame) -> pd.DataFrame:
    sold = properties[properties["listing_status"] == "Sold"].copy()

    # Premium index: price relative to the median of comparable units
    # (same tower, month and unit category). >1 means the buyer paid above peers.
    grp = sold.groupby(["tower_number", "transaction_date", "unit_category"])["price_per_sqft"]
    sold["peer_median_ppsf"] = grp.transform("median")
    sold["premium_ratio"] = sold["price_per_sqft"] / sold["peer_median_ppsf"]

    ref = pd.Timestamp(REFERENCE_DATE)
    agg = sold.groupby("client_ref").agg(
        units_bought=("listing_id", "count"),
        total_spend=("sale_price", "sum"),
        avg_unit_price=("sale_price", "mean"),
        total_sqft=("floor_area_sqft", "sum"),
        avg_price_per_sqft=("price_per_sqft", "mean"),
        premium_index=("premium_ratio", "mean"),
        office_units=("unit_category", lambda s: (s == "Office").sum()),
        n_towers=("tower_number", "nunique"),
        tower_hhi=("tower_number", _hhi),
        purchase_months=("transaction_date", "nunique"),
        first_purchase=("transaction_date", "min"),
        last_purchase=("transaction_date", "max"),
    )
    agg["office_share"] = agg["office_units"] / agg["units_bought"]
    agg["recency_months"] = ((ref.year - agg["last_purchase"].dt.year) * 12
                             + (ref.month - agg["last_purchase"].dt.month))
    agg["bought_both_years"] = ((agg["first_purchase"].dt.year == 2024)
                                & (agg["last_purchase"].dt.year == 2025)).astype(int)

    df = clients.merge(agg, left_on="client_id", right_index=True, how="left")
    df["units_bought"] = df["units_bought"].fillna(0).astype(int)

    # Binary flags (label-encoded binaries; nominal channel one-hot, Website = baseline)
    df["is_company"] = (df["client_type"] == "Company").astype(int)
    df["is_investment"] = (df["acquisition_purpose"] == "Investment").astype(int)
    df["has_loan"] = (df["loan_applied"] == "Yes").astype(int)
    df["is_international"] = (df["country"] != "USA").astype(int)
    df["is_female"] = (df["gender"] == "F").astype(int)
    df["ref_agency"] = (df["referral_channel"] == "Agency").astype(int)
    df["ref_client"] = (df["referral_channel"] == "Client").astype(int)
    df["portfolio_builder"] = (df["units_bought"] >= 5).astype(int)

    df["age_band"] = pd.cut(df["age"], bins=[0, 34, 44, 54, 64, 120],
                            labels=["25-34", "35-44", "45-54", "55-64", "65+"])
    df["country_group"] = np.where(df["country"] == "USA", "USA",
                          np.where(df["country"].isin(["Canada", "Mexico"]), "Rest of Americas",
                          np.where(df["country"] == "Australia", "Asia-Pacific", "Europe")))
    return df


def model_matrix(df: pd.DataFrame, numeric: list[str], binary: list[str] = BINARY_FEATURES,
                 scaler: StandardScaler | None = None):
    """Numerics are standardised (mean 0, sd 1); binaries stay 0/1 so that one
    category difference counts about as much as one standard deviation."""
    scaler = scaler or StandardScaler().fit(df[numeric])
    X_num = scaler.transform(df[numeric])
    X = np.hstack([X_num, df[binary].to_numpy(dtype=float)])
    return X, numeric + binary, scaler
