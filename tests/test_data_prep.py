"""Unit tests for src.data_prep (the cleaning logic, doc 16).

Strategy: tiny hand-built fixtures with KNOWN answers (fast), plus one slow data-contract test
that checks the real raw file against the documented facts.
"""
import pandas as pd
import pytest

from src import data_prep, utils


def _toy_transactions() -> pd.DataFrame:
    """8 rows, each crafted to trip exactly one cleaning rule (a-e), so answers are known.

    Rows: valid | POST (non-product) | missing ID | zero qty | price<=0 | dup | dup | negative-qty return
    """
    return pd.DataFrame({
        "Invoice":     ["1",      "2",    "3",     "4",     "5",     "6",      "6",      "C7"],
        "StockCode":   ["85123A", "POST", "22423", "21232", "84029", "84879",  "84879",  "84879"],
        "Description": ["bag",    "post", "clock", "lamp",  "knife", "hearts", "hearts", "hearts"],
        "Quantity":    [6,        1,      3,       0,       2,       5,        5,        -2],
        "InvoiceDate": pd.to_datetime(["2010-01-01"] * 8),
        "Price":       [2.5,      3.0,    1.0,     5.0,     -1.0,    4.0,      4.0,      4.0],
        "Customer ID": [111,      222,    None,    333,     444,     555,      555,      555],
        "Country":     ["UK"] * 8,
    })


def test_drop_invalid_records_counts_and_survivors():
    clean, report = data_prep.drop_invalid_records(_toy_transactions())

    # one row removed by each rule a-e (the fixture is designed so there's no overlap)
    assert report["start"] == 8
    assert report["missing_customer_id"] == 1     # the None Customer ID row
    assert report["non_product_stockcode"] == 1   # the POST row
    assert report["invalid_price"] == 1           # the Price = -1.0 row
    assert report["zero_quantity"] == 1           # the Quantity = 0 row
    assert report["exact_duplicates"] == 1        # the repeated invoice-6 line
    assert report["end"] == 3

    # sequential counts must sum exactly to (start - end) -- the reconciliation invariant
    removed = sum(report[k] for k in
                  ("missing_customer_id", "non_product_stockcode",
                   "invalid_price", "zero_quantity", "exact_duplicates"))
    assert removed == report["start"] - report["end"]

    # the negative-quantity return (C7) is KEPT here -- resolve_non_purchases handles it later
    assert (clean["Quantity"] < 0).sum() == 1
    assert "C7" in set(clean["Invoice"])


def test_resolve_non_purchases_split_and_counts():
    # 2 purchases, 1 cancellation (C, negative), 1 plain return (negative, not C)
    df = pd.DataFrame({
        "Invoice":     ["10",     "11",    "C12",    "13"],
        "StockCode":   ["85123A", "22423", "85123A", "22423"],
        "Description": ["bag",    "clock", "bag",    "clock"],
        "Quantity":    [5,        3,       -2,       -1],
        "InvoiceDate": pd.to_datetime(["2010-01-01"] * 4),
        "Price":       [2.5,      1.0,     2.5,      1.0],
        "Customer ID": [111,      222,     111,      222],
        "Country":     ["UK"] * 4,
    })
    purchases, non_purchases, report = data_prep.resolve_non_purchases(df)

    assert report["start"] == 4
    assert report["cancellations"] == 1            # C12
    assert report["negative_qty_returns"] == 2     # C12 (-2) and 13 (-1)
    assert report["non_purchases_removed"] == 2    # union of the two -> C12, 13
    assert report["purchases"] == 2                # 10, 11

    # the purchases table is all positive-quantity, non-cancelled
    assert (purchases["Quantity"] > 0).all()
    assert not purchases["Invoice"].astype(str).str.startswith("C").any()

    # reconciliation: purchases + removed == start, and nothing is lost in the split
    assert len(purchases) + report["non_purchases_removed"] == report["start"]
    assert len(purchases) + len(non_purchases) == report["start"]


def test_build_clean_table_end_to_end_reconciliation():
    # reuse the 8-row fixture: after invalid-drop 3 survive; one of those (C7) is a return
    purchases, non_purchases, recon = data_prep.build_clean_table(_toy_transactions())

    assert recon["raw_rows"] == 8
    assert recon["after_drop_invalid"] == 3      # valid + dup-survivor + the C7 return
    assert recon["clean_rows"] == 2              # C7 then removed as a non-purchase
    assert recon["non_purchases_rows"] == 1
    assert len(purchases) == 2

    # the clean table holds only positive-qty, identifiable, non-cancelled purchases
    assert (purchases["Quantity"] > 0).all()
    assert purchases["Customer ID"].notna().all()
    assert not purchases["Invoice"].astype(str).str.startswith("C").any()

    # FULL reconciliation invariant: every raw row is accounted for
    invalid_dropped = recon["raw_rows"] - recon["after_drop_invalid"]
    assert recon["raw_rows"] == recon["clean_rows"] + invalid_dropped + recon["non_purchases_rows"]


@pytest.mark.slow
def test_raw_data_matches_documentation():
    """Data contract: the real raw file matches doc 03/16. Slow -- reads the 45MB xlsx."""
    if not utils.RAW_XLSX.exists():
        pytest.skip(f"raw data not found at {utils.RAW_XLSX}")
    df = data_prep.load_raw()
    assert len(df) == 1_067_371
    assert df["Customer ID"].nunique() == 5_942
    assert df["Country"].nunique() == 43
    assert df["Customer ID"].isna().sum() == 243_007
