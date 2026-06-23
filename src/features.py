"""Feature engineering: collapse clean transactions into per-customer features (doc 17).

Three lanes (doc 16/17): CORE (clustering — R/F/M/Tenure, + AvgBasket profiling-only),
CLV inputs (raw, for BG/NBD + Gamma-Gamma), and SUPPORTING (wholesaler flag, return-rate,
country — explain/validate the segments, never define them). All time features are anchored to
2012-01-01 (doc 08). Logic lives here (unit-tested); the notebook runs the Feature-EDA checkpoint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import utils

CORE_FEATURES = ["Recency", "Frequency", "Monetary", "Tenure", "AvgBasket"]


def build_core_features(tx: pd.DataFrame, anchor: pd.Timestamp | None = None) -> pd.DataFrame:
    """Aggregate clean transactions into per-customer CORE features (doc 17, core lane).

    One row per customer, in RAW units (no log/scale yet):

    - ``Recency``   = days from the customer's LAST purchase to the anchor (lower = more recent)
    - ``Frequency`` = number of distinct purchase invoices
    - ``Monetary``  = total spend (sum of Quantity x Price)
    - ``Tenure``    = days from the customer's FIRST purchase to the anchor (customer age; doc 17 D1)
    - ``AvgBasket`` = Monetary / Frequency  (profiling-only; excluded from clustering, = M/F redundancy)

    For a one-time buyer first == last purchase, so Recency == Tenure (the doc-17 degeneracy).
    Time features are anchored to ``anchor`` (default ``utils.ANCHOR_DATE`` = 2012-01-01).
    """
    anchor = pd.Timestamp(anchor) if anchor is not None else utils.ANCHOR_DATE
    tx = tx.assign(LineRevenue=tx["Quantity"] * tx["Price"])

    feat = tx.groupby("Customer ID").agg(
        Frequency=("Invoice", "nunique"),
        Monetary=("LineRevenue", "sum"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
    )
    feat["Recency"] = (anchor - feat["last_purchase"]).dt.days
    feat["Tenure"] = (anchor - feat["first_purchase"]).dt.days
    feat["AvgBasket"] = feat["Monetary"] / feat["Frequency"]

    return feat[CORE_FEATURES]


def build_clv_inputs(tx: pd.DataFrame, anchor: pd.Timestamp | None = None) -> pd.DataFrame:
    """Build per-customer CLV inputs for BG/NBD + Gamma-Gamma (doc 15), via PyMC-Marketing.

    Uses ``pymc_marketing.clv.rfm_summary`` -- the model author's own data prep, so it's correct by
    construction (doc 18, decision B). One row per customer, the BTYD 'naming trap' columns:

    - ``frequency``      = number of repeat purchase OCCASIONS (distinct days) minus 1
    - ``recency``        = age at last purchase = (last - first) in days  (t_x)
    - ``T``              = customer age = (anchor - first) in days
    - ``monetary_value`` = average value of REPEAT transactions (Gamma-Gamma convention; 0 for one-timers)

    Occasions are DAY-based (literature standard), so this ``frequency`` differs slightly from the
    clustering ``Frequency`` (invoice-based) for same-day multi-invoice customers (~55, doc 18).
    """
    # Lazy import: pymc_marketing is heavy, so don't pay for it unless CLV inputs are actually built.
    from pymc_marketing.clv import rfm_summary

    anchor = pd.Timestamp(anchor) if anchor is not None else utils.ANCHOR_DATE
    tx = tx.assign(LineRevenue=tx["Quantity"] * tx["Price"])

    summary = rfm_summary(
        tx,
        customer_id_col="Customer ID",
        datetime_col="InvoiceDate",
        monetary_value_col="LineRevenue",
        observation_period_end=anchor,
        time_unit="D",
    )
    # rfm_summary returns the id as a 'customer_id' column -- index it and align the name with
    # build_core_features ('Customer ID') so the two lanes join cleanly later.
    summary = summary.set_index("customer_id")
    summary.index.name = "Customer ID"
    return summary


def build_supporting_features(
    tx: pd.DataFrame, non_purchases: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build per-customer SUPPORTING variables (doc 16/17).

    Supporting variables EXPLAIN and VALIDATE segments but never DEFINE them (never clustering
    inputs). One row per customer:

    - ``Country``    = the customer's most common country
    - ``BulkUnits``  = median units per invoice -- the *continuous* wholesaler signal. The binary
      wholesaler flag's cutoff is decided in the Feature-EDA checkpoint (doc 17), not hard-coded here.
    - ``ReturnRate`` = share of the customer's orders that were cancellations/returns, computed from
      the ``non_purchases`` table captured during cleaning (doc 16). 0 if no returns / not provided.
    """
    # Country: the customer's most frequent country (customers are ~always single-country).
    country = tx.groupby("Customer ID")["Country"].agg(lambda s: s.mode().iat[0])

    # Bulk signal: total units per INVOICE, then the median across the customer's invoices.
    units_per_invoice = tx.groupby(["Customer ID", "Invoice"])["Quantity"].sum()
    bulk_units = units_per_invoice.groupby("Customer ID").median()

    sup = pd.DataFrame({"Country": country, "BulkUnits": bulk_units})

    # Return-rate: (cancellation/return invoices) / (purchase + cancellation/return invoices).
    n_purchase_inv = tx.groupby("Customer ID")["Invoice"].nunique()
    if non_purchases is not None and len(non_purchases):
        n_return_inv = non_purchases.groupby("Customer ID")["Invoice"].nunique()
    else:
        n_return_inv = pd.Series(dtype="int64")
    n_return_inv = n_return_inv.reindex(sup.index, fill_value=0)
    sup["ReturnRate"] = n_return_inv / (n_purchase_inv.reindex(sup.index) + n_return_inv)

    return sup


def scale_features(df, log_cols, cluster_cols, scaler=None):
    """log1p the skewed columns, then scale -> the clustering-ready matrix (doc 09).

    Two steps, in order (log first reshapes; scaling then re-ranges -- doc 09):
      1. apply ``log1p`` to ``log_cols`` (tames right-skew)
      2. fit-transform ``scaler`` on ``cluster_cols``

    Returns
    -------
    (scaled, scaler)
        ``scaled`` = the clustering matrix (one column per ``cluster_cols``, same index).
        ``scaler`` = the FITTED scaler -- PERSIST it (doc 09) so new/held-out customers get the
        identical transform.

    ``scaler`` defaults to ``RobustScaler`` (doc 09 primary); pass ``StandardScaler()`` for the
    comparison experiment. ``log_cols`` / ``cluster_cols`` are chosen per the Feature-EDA checkpoint
    (e.g. log Frequency/Monetary/AvgBasket; cluster on R/F/M/Tenure with AvgBasket excluded).
    """
    from sklearn.preprocessing import RobustScaler

    df = df.copy()
    for c in log_cols:
        df[c] = np.log1p(df[c])

    scaler = scaler if scaler is not None else RobustScaler()
    cluster_cols = list(cluster_cols)
    scaled = pd.DataFrame(
        scaler.fit_transform(df[cluster_cols]),
        index=df.index,
        columns=cluster_cols,
    )
    return scaled, scaler
