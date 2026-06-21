"""Shared constants and small helpers for the customer-segmentation project.

Fixed values that the rest of the code (notebooks, src modules, tests) imports, so the
analysis parameters live in exactly one place. See planning/docs for the reasoning behind each.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- paths
# This file is <root>/src/utils.py, so the project root is two levels up.
ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW: Path = ROOT / "data" / "raw"
DATA_PROCESSED: Path = ROOT / "data" / "processed"
REPORTS_FIGURES: Path = ROOT / "reports" / "figures"
RAW_XLSX: Path = DATA_RAW / "online_retail_II.xlsx"

# --------------------------------------------------------------------------- analysis constants
# "Today" for Recency / Tenure / CLV — anchored just after the data ends (doc 08).
ANCHOR_DATE: pd.Timestamp = pd.Timestamp("2012-01-01")

# Reproducibility — used everywhere a random_state is needed.
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- cleaning constants
# Non-product StockCodes to DROP (doc 16, Decision 2b). These are fees / postage /
# adjustments / tests — not real product purchases. NOTE: postage (POST/DOT/C2) IS dropped
# (Decision 2: postage is a fee, excluded so Monetary reflects product spend).
NON_PRODUCT_STOCKCODES: frozenset[str] = frozenset({
    "POST", "DOT", "C2",                 # postage / carriage
    "BANK CHARGES", "BANK CHARGE",       # bank charges
    "AMAZONFEE",                         # marketplace fee
    "ADJUST", "ADJUST2",                 # manual adjustments
    "M", "m",                            # manual
    "D",                                 # discount
    "S",                                 # sample
    "B",                                 # bad-debt adjustment
    "CRUK",                              # charity
    "TEST001", "TEST002",                # test rows
    "PADS",                              # PADS TO MATCH ALL CUSHIONS (catch-all)
})

# Gift-card StockCodes share this prefix (e.g. "gift_0001_20") — also non-product.
GIFT_STOCKCODE_PREFIX: str = "gift_0001"

# Cancellation invoices start with 'C' (doc 16, Decision 1 / 2).
CANCELLATION_PREFIX: str = "C"


def is_non_product(stockcode: object) -> bool:
    """True if a StockCode is a fee/adjustment/test/gift-card (not a real product)."""
    s = str(stockcode).strip()
    return s in NON_PRODUCT_STOCKCODES or s.lower().startswith(GIFT_STOCKCODE_PREFIX)
