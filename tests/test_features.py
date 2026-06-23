"""Unit tests for src.features (feature engineering, doc 17)."""
import pandas as pd
import pytest

from src import features


def _toy_tx() -> pd.DataFrame:
    """2 customers, known transactions. Anchor will be 2012-01-01.

    Customer 1: invoice A (2 lines, 2011-12-01) + invoice B (1 line, 2011-11-01).
    Customer 2: invoice C (1 line, 2011-12-15) -- a one-time buyer.
    """
    return pd.DataFrame({
        "Invoice":     ["A",   "A",   "B",   "C"],
        "Quantity":    [2,     1,     1,     3],
        "Price":       [10.0,  5.0,   5.0,   4.0],
        "InvoiceDate": pd.to_datetime(["2011-12-01", "2011-12-01", "2011-11-01", "2011-12-15"]),
        "Customer ID": [1,     1,     1,     2],
    })


def test_build_core_features_known_values():
    feat = features.build_core_features(_toy_tx(), anchor=pd.Timestamp("2012-01-01"))

    # one row per customer, columns in the documented order
    assert len(feat) == 2
    assert list(feat.columns) == ["Recency", "Frequency", "Monetary", "Tenure", "AvgBasket"]

    c1 = feat.loc[1]
    assert c1["Frequency"] == 2        # distinct invoices A, B (NOT 3 product lines)
    assert c1["Monetary"] == 30.0      # 2*10 + 1*5 + 1*5
    assert c1["Recency"] == 31         # 2012-01-01 - 2011-12-01 (last)
    assert c1["Tenure"] == 61          # 2012-01-01 - 2011-11-01 (first)
    assert c1["AvgBasket"] == 15.0     # 30 / 2

    c2 = feat.loc[2]
    assert c2["Frequency"] == 1
    assert c2["Monetary"] == 12.0      # 3 * 4
    assert c2["Recency"] == 17         # 2012-01-01 - 2011-12-15
    assert c2["Tenure"] == 17          # one-time buyer -> Recency == Tenure (doc 17 degeneracy)
    assert c2["AvgBasket"] == 12.0


def test_build_clv_inputs_naming_trap():
    """CLV inputs via rfm_summary -- the BTYD 'naming trap' (values verified vs rfm_summary)."""
    tx = pd.DataFrame({
        "Invoice":     ["A",   "B",   "C"],
        "Quantity":    [1,     2,     3],
        "Price":       [10.0,  10.0,  4.0],
        "InvoiceDate": pd.to_datetime(["2011-11-01", "2011-12-01", "2011-12-15"]),
        "Customer ID": [1,     1,     2],
    })
    clv = features.build_clv_inputs(tx, anchor=pd.Timestamp("2012-01-01"))

    assert clv.index.name == "Customer ID"
    assert list(clv.columns) == ["frequency", "recency", "T", "monetary_value"]

    c1 = clv.loc[1]
    assert c1["frequency"] == 1          # 2 occasions - 1 (NOT the clustering Frequency = invoices)
    assert c1["recency"] == 30           # last - first = 2011-12-01 - 2011-11-01
    assert c1["T"] == 61                 # anchor - first
    assert c1["monetary_value"] == 20.0  # average of REPEAT transactions (the Dec-01 occasion)

    c2 = clv.loc[2]                       # one-time buyer
    assert c2["frequency"] == 0          # BG/NBD frequency = 0
    assert c2["recency"] == 0
    assert c2["T"] == 17
    assert c2["monetary_value"] == 0.0   # no repeat transaction -> excluded from Gamma-Gamma


def test_build_supporting_features():
    tx = pd.DataFrame({
        "Invoice":     ["A", "A", "B", "C"],
        "Quantity":    [5,   5,   2,   100],
        "Customer ID": [1,   1,   1,   2],
        "Country":     ["United Kingdom"] * 4,
    })
    non_purchases = pd.DataFrame({
        "Invoice":     ["C12"],
        "Quantity":    [-3],
        "Customer ID": [1],
        "Country":     ["United Kingdom"],
    })
    sup = features.build_supporting_features(tx, non_purchases)

    assert sup.index.name == "Customer ID"
    assert list(sup.columns) == ["Country", "BulkUnits", "ReturnRate"]

    assert sup.loc[1, "BulkUnits"] == 6     # median of invoice-units {A:10, B:2}
    assert sup.loc[2, "BulkUnits"] == 100   # one invoice of 100 units (a bulk buyer)
    assert sup.loc[1, "Country"] == "United Kingdom"
    assert sup.loc[1, "ReturnRate"] == pytest.approx(1 / 3)   # 1 return invoice / (2 purchases + 1)
    assert sup.loc[2, "ReturnRate"] == 0.0                    # no returns

    # without a non_purchases table, ReturnRate is all 0
    sup_no_ret = features.build_supporting_features(tx)
    assert (sup_no_ret["ReturnRate"] == 0).all()


def test_scale_features_log_then_robustscale():
    df = pd.DataFrame({"Monetary": [1.0, 9.0, 99.0]}, index=[10, 20, 30])
    scaled, scaler = features.scale_features(df, log_cols=["Monetary"], cluster_cols=["Monetary"])

    # shape / columns / index preserved
    assert list(scaled.columns) == ["Monetary"]
    assert list(scaled.index) == [10, 20, 30]
    # RobustScaler centres the median to 0; order preserved (log + linear scaling are monotonic)
    import numpy as np
    assert np.isclose(scaled["Monetary"].median(), 0.0)
    assert scaled["Monetary"].is_monotonic_increasing
    # proof log1p was applied: extremes are near-symmetric (~1.4x); on RAW data they'd be ~11x apart
    lo, hi = abs(scaled["Monetary"].iloc[0]), abs(scaled["Monetary"].iloc[2])
    assert hi / lo < 3
    # the scaler comes back fitted (so it can be persisted and reused)
    assert hasattr(scaler, "center_") and hasattr(scaler, "scale_")


def test_scale_features_accepts_standardscaler():
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    df = pd.DataFrame({"M": [1.0, 9.0, 99.0, 4.0]})
    scaled, _ = features.scale_features(df, ["M"], ["M"], scaler=StandardScaler())
    assert np.isclose(scaled["M"].std(ddof=0), 1.0)   # StandardScaler -> unit std
