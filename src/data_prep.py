"""Data loading and cleaning for the customer-segmentation project.

Implements the cleaning pipeline from planning/docs/16: Load & Merge -> Drop Invalid Records ->
Resolve Non-Purchases -> Clean Dataset. Each function is small and pure so it can be unit-tested.
"""
from __future__ import annotations

import pandas as pd

from . import utils

# The 8 raw columns of Online Retail II (doc 03).
RAW_COLUMNS = [
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "Price", "Customer ID", "Country",
]


def load_raw(path=None) -> pd.DataFrame:
    """Load BOTH year-sheets of Online Retail II and concatenate into one transaction table.

    One row per invoice line. Adds a ``SourceSheet`` column recording which sheet each row came
    from. No cleaning is applied here — that is ``build_clean_table`` (doc 16).

    Parameters
    ----------
    path : path-like, optional
        Location of the .xlsx. Defaults to ``utils.RAW_XLSX``.
    """
    path = path or utils.RAW_XLSX
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")  # {sheet_name: DataFrame}
    frames = []
    for name, df in sheets.items():
        df = df.copy()
        df["SourceSheet"] = name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    # Normalize dtypes so the table is consistent and Parquet-safe. StockCode and Invoice mix
    # numeric and alphanumeric values (e.g. '85123A', '79323P', 'C536379'), which pandas reads as
    # a mixed 'object' column -- Parquet needs one type per column. Coerce the string-like columns
    # to the nullable 'string' dtype (preserves missing values as <NA>), and Customer ID to a
    # nullable integer (it is a float with NaNs on load).
    string_cols = ["Invoice", "StockCode", "Description", "Country", "SourceSheet"]
    out[string_cols] = out[string_cols].astype("string")
    out["Customer ID"] = out["Customer ID"].astype("Int64")
    return out


def drop_invalid_records(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows that cannot be modelled as a real product purchase (doc 16, Decision 2).

    Applies rules a-e in order, dropping rows that are: 
    (a) missing a Customer ID,
    (b) a non-product StockCode (fees/postage/adjustments/tests/gift cards), 
    (c) priced <= 0,
    (d) zero quantity, or 
    (e) exact duplicates. Negative-quantity returns are deliberately KEPT
    here -- they are handled later by ``resolve_non_purchases`` so the return-rate can be captured first.

    Returns
    -------
    (clean_df, report)
        ``clean_df`` is the filtered frame; ``report`` is a per-rule count of how many rows each
        rule removed (sequentially, so the counts sum exactly to start - end) -- for the
        reconciliation audit trail.
    """
    report: dict[str, int] = {"start": len(df)}
    out = df

    # rule a -- missing Customer ID (~22.8%): can't attribute to a customer.
    before = len(out)
    out = out.loc[out["Customer ID"].notna()]
    report["missing_customer_id"] = before - len(out)

    # rule b -- non-product StockCodes: fees, postage, adjustments, tests, gift cards.
    before = len(out)
    out = out.loc[~out["StockCode"].map(utils.is_non_product)]
    report["non_product_stockcode"] = before - len(out)

    # rule c -- invalid price (<= 0): free items, adjustments, errors.
    before = len(out)
    out = out.loc[out["Price"] > 0]
    report["invalid_price"] = before - len(out)

    # rule d -- zero quantity. (Negative = returns; kept for resolve_non_purchases.)
    before = len(out)
    out = out.loc[out["Quantity"] != 0]
    report["zero_quantity"] = before - len(out)

    # rule e -- exact duplicate rows (all columns identical): data artifacts.
    before = len(out)
    out = out.drop_duplicates()
    report["exact_duplicates"] = before - len(out)

    report["end"] = len(out)
    return out.copy(), report


def resolve_non_purchases(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Split rows into completed purchases vs non-purchases (doc 16, Decision 1: DROP, not net).

    Cancellations (Invoice starts with 'C') and negative-quantity returns are removed from the
    modelling table -- but returned SEPARATELY as ``non_purchases`` so the per-customer
    return/cancellation rate (a supporting + validation variable) can be computed downstream.
    Run AFTER ``drop_invalid_records`` (assumes invalid rows are already gone).

    Returns
    -------
    (purchases, non_purchases, report)
        ``purchases`` = positive-quantity, non-cancelled lines (the modelling table).
        ``non_purchases`` = the removed cancellations/returns (kept for the return-rate).
        ``report`` = counts. NOTE: 'cancellations' and 'negative_qty_returns' can OVERLAP (a
        C-invoice usually has negative qty), so they need not sum to 'non_purchases_removed';
        the reconciliation number is 'non_purchases_removed' (= start - purchases).
    """
    invoice = df["Invoice"].astype(str)
    is_cancellation = invoice.str.startswith(utils.CANCELLATION_PREFIX)
    is_return = df["Quantity"] < 0
    non_purchase = is_cancellation | is_return  # either flavour is a non-purchase

    purchases = df.loc[~non_purchase].copy()
    non_purchases = df.loc[non_purchase].copy()

    report: dict[str, int] = {
        "start": len(df),
        "cancellations": int(is_cancellation.sum()),
        "negative_qty_returns": int(is_return.sum()),
        "non_purchases_removed": int(non_purchase.sum()),
        "purchases": len(purchases),
    }
    return purchases, non_purchases, report


def build_clean_table(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the full cleaning pipeline (doc 16) and produce the clean table + reconciliation.

    Chains: Load & Merge -> Drop Invalid Records -> Resolve Non-Purchases -> Clean Dataset.
    If ``df`` is None, loads the raw data via ``load_raw``; pass a frame to clean it in place
    (used by tests). Pure function -- it does NOT write to disk (the notebook saves the result),
    so it stays easy to test.

    Returns
    -------
    (purchases, non_purchases, reconciliation)
        ``purchases`` = the clean modelling table (completed purchases, identifiable customers).
        ``non_purchases`` = removed cancellations/returns (kept for the return-rate).
        ``reconciliation`` = the full audit trail: raw -> after invalid-drop -> clean, plus the
        detailed per-step reports. Every raw row is accounted for (see the invariant in tests).
    """
    raw = load_raw() if df is None else df
    n_raw = len(raw)

    cleaned, invalid_report = drop_invalid_records(raw)
    purchases, non_purchases, np_report = resolve_non_purchases(cleaned)

    reconciliation: dict = {
        "raw_rows": n_raw,
        "after_drop_invalid": invalid_report["end"],
        "clean_rows": len(purchases),
        "non_purchases_rows": len(non_purchases),
        "detail_invalid": invalid_report,
        "detail_non_purchases": np_report,
    }
    return purchases, non_purchases, reconciliation
