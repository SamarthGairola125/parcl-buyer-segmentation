"""Loading, cleaning and validating the raw client and property tables."""
from __future__ import annotations

import pandas as pd

from .config import DATA_RAW, REFERENCE_DATE

PII_COLUMNS = ["first_name", "last_name"]
CATEGORICAL_CLIENT = [
    "client_type", "gender", "country", "region",
    "acquisition_purpose", "loan_applied", "referral_channel",
]


def load_raw(path=DATA_RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    clients = pd.read_csv(path / "clients.csv", dtype=str)
    properties = pd.read_csv(path / "properties.csv", dtype=str)
    return clients, properties


def parse_dob(series: pd.Series) -> pd.Series:
    """Parse the three mixed DOB formats (M/D/YYYY, MM-DD-YYYY, MM/DD/YYYY).

    All three are month-first, so unifying the separator is enough.
    """
    unified = series.str.strip().str.replace("/", "-", regex=False)
    return pd.to_datetime(unified, format="%m-%d-%Y", errors="coerce")


def parse_price(series: pd.Series) -> pd.Series:
    """'$300,385.62' -> 300385.62"""
    return pd.to_numeric(series.str.replace(r"[$,\s]", "", regex=True), errors="coerce")


def clean_clients(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"rows_in": len(raw)}
    df = raw.drop(columns=[c for c in PII_COLUMNS if c in raw.columns]).copy()

    # Normalise labels: strip whitespace, consistent casing
    for col in CATEGORICAL_CLIENT:
        df[col] = df[col].astype(str).str.strip()
    df["gender"] = df["gender"].str.upper()
    for col in ["client_type", "acquisition_purpose", "loan_applied", "referral_channel"]:
        df[col] = df[col].str.title()
    df["client_id"] = df["client_id"].str.strip().str.upper()

    # Duplicates
    before = len(df)
    df = df.drop_duplicates()
    log["exact_duplicates_removed"] = before - len(df)
    before = len(df)
    df = df.drop_duplicates(subset="client_id", keep="first")
    log["duplicate_ids_removed"] = before - len(df)

    # Types
    df["date_of_birth"] = parse_dob(df["date_of_birth"])
    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")
    df["age"] = ((pd.Timestamp(REFERENCE_DATE) - df["date_of_birth"]).dt.days / 365.25).round(1)

    # Missing values: report, then impute (mode for categoricals, median for numerics)
    missing = df.isna().sum()
    log["missing_before_imputation"] = {k: int(v) for k, v in missing[missing > 0].items()}
    for col in CATEGORICAL_CLIENT:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    for col in ["age", "satisfaction_score"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    log["rows_out"] = len(df)
    return df.reset_index(drop=True), log


def clean_properties(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"rows_in": len(raw)}
    df = raw.copy()
    df["listing_id"] = pd.to_numeric(df["listing_id"])
    df["tower_number"] = pd.to_numeric(df["tower_number"])
    df["unit_number"] = pd.to_numeric(df["unit_number"])
    df["floor_area_sqft"] = pd.to_numeric(df["floor_area_sqft"])
    df["sale_price"] = parse_price(df["sale_price"])
    df["transaction_date"] = pd.to_datetime(df["transaction_date"].str.strip(), format="%m-%d-%Y")
    df["unit_category"] = df["unit_category"].str.strip().str.title()
    df["listing_status"] = df["listing_status"].str.strip().str.title()
    df["client_ref"] = df["client_ref"].str.strip().str.upper()
    df["price_per_sqft"] = df["sale_price"] / df["floor_area_sqft"]
    before = len(df)
    df = df.drop_duplicates(subset="listing_id")
    log["duplicate_listings_removed"] = before - len(df)
    log["sold"] = int((df["listing_status"] == "Sold").sum())
    log["available"] = int((df["listing_status"] == "Available").sum())
    log["rows_out"] = len(df)
    return df.reset_index(drop=True), log


def validate(clients: pd.DataFrame, properties: pd.DataFrame) -> dict:
    """Hard checks; raises AssertionError on failure."""
    sold = properties[properties["listing_status"] == "Sold"]
    checks = {
        "client_ids_unique": clients["client_id"].is_unique,
        "dob_all_parsed": clients["date_of_birth"].notna().all(),
        "age_in_range_18_100": clients["age"].between(18, 100).all(),
        "satisfaction_1_to_5": clients["satisfaction_score"].between(1, 5).all(),
        "prices_positive": (properties["sale_price"] > 0).all(),
        "sold_units_have_buyer": sold["client_ref"].notna().all(),
        "available_units_have_no_buyer": properties.loc[
            properties["listing_status"] == "Available", "client_ref"].isna().all(),
        "all_buyers_exist_in_clients": sold["client_ref"].isin(clients["client_id"]).all(),
        "region_nested_in_country": (clients.groupby("region")["country"].nunique() == 1).all(),
    }
    failed = [k for k, v in checks.items() if not v]
    assert not failed, f"Validation failed: {failed}"
    return {k: bool(v) for k, v in checks.items()}
