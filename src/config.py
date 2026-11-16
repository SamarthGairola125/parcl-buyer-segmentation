"""Project-wide constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
RESULTS = ROOT / "results"

SEED = 42
# Age is measured at the last transaction month in the data (Dec 2025),
# so results do not change with the date the code is run.
REFERENCE_DATE = "2025-12-31"

# Variables used for clustering (see docs/decision_log.md, D5/D10/D11)
NUMERIC_BRIEF = ["age", "satisfaction_score"]
NUMERIC_BEHAVIOUR = [
    "units_bought",
    "total_spend",
    "premium_index",
    "office_share",
    "tower_hhi",
    "recency_months",
]
BINARY_FEATURES = [
    "is_company",
    "is_investment",
    "has_loan",
    "is_international",
    "is_female",
    "ref_agency",
    "ref_client",
]
K_RANGE = range(2, 11)
