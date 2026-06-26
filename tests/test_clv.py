"""Unit tests for src.clv (CLV foundation + Gamma-Gamma validity gate, doc 15)."""
import numpy as np
import pandas as pd

from src import clv


def _toy_clv() -> pd.DataFrame:
    """4 customers: two one-timers (frequency 0) + two repeat buyers."""
    return pd.DataFrame({
        "frequency": [0, 0, 3, 5],
        "recency": [0, 0, 120, 300],
        "T": [200, 50, 400, 500],
        "monetary_value": [0.0, 0.0, 80.0, 120.0],
    })


def test_repeat_buyers_filters_frequency_zero():
    rep = clv.repeat_buyers(_toy_clv())
    assert len(rep) == 2
    assert (rep["frequency"] > 0).all()


def test_one_timer_share():
    assert clv.one_timer_share(_toy_clv()) == 0.5   # 2 of 4 have frequency 0


def test_gate_passes_when_independent():
    # frequency and monetary value drawn independently -> near-zero correlation -> passes.
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "frequency": rng.integers(1, 20, n),
        "recency": rng.integers(0, 300, n),
        "T": rng.integers(300, 700, n),
        "monetary_value": rng.gamma(2.0, 50.0, n),   # independent of frequency
    })
    gate = clv.gamma_gamma_gate(df)
    assert gate["n_repeat"] == n
    assert abs(gate["pearson"]) < 0.30
    assert gate["passes"] is True


def test_gate_fails_when_strongly_correlated():
    # monetary value built FROM frequency -> strong correlation -> gate fails (assumption violated).
    rng = np.random.default_rng(1)
    n = 500
    freq = rng.integers(1, 20, n)
    df = pd.DataFrame({
        "frequency": freq,
        "recency": rng.integers(0, 300, n),
        "T": rng.integers(300, 700, n),
        "monetary_value": freq * 10.0 + rng.normal(0, 2, n),   # spend rises with frequency
    })
    gate = clv.gamma_gamma_gate(df)
    assert gate["pearson"] > 0.30
    assert gate["passes"] is False


def test_prepare_clv_data_makes_customer_id_and_keeps_order():
    df = pd.DataFrame(
        {"frequency": [0, 3], "recency": [0, 100], "T": [200, 400], "monetary_value": [0.0, 80.0]},
        index=pd.Index([5, 9], name="Customer ID"),
    )
    out = clv.prepare_clv_data(df)
    assert "customer_id" in out.columns and "Customer ID" not in out.columns
    assert list(out["customer_id"]) == [5, 9]                 # index order preserved
    assert {"frequency", "recency", "T", "monetary_value"} <= set(out.columns)


def test_per_customer_averages_over_chain_and_draw():
    import xarray as xr
    da = xr.DataArray(np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
                      dims=("chain", "draw", "customer_id"))
    idx = pd.Index([10, 20, 30, 40], name="Customer ID")
    s = clv._per_customer(da, idx, "x")
    assert list(s.index) == [10, 20, 30, 40] and s.name == "x"
    assert np.allclose(s.values, da.mean(dim=("chain", "draw")).values)   # MAP/MCMC point estimate
