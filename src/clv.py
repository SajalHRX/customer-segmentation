"""Customer Lifetime Value (CLV) — the value track (docs 15, 12, 08).

A SEPARATE pipeline from clustering (no circularity, doc 12): raw frequency/recency/T/monetary →
BG/NBD + Gamma-Gamma → predicted future value + P(alive). This module holds the reusable CLV logic;
the notebooks (05a validation, 05b production) call it.

The "naming trap" (doc 15): these CLV columns share names with the clustering features but MEAN
different things — built in ``features.build_clv_inputs`` via PyMC-Marketing's ``rfm_summary``:

- ``frequency``      = number of REPEAT purchase occasions (so a one-time buyer has frequency = 0)
- ``recency``        = age at last purchase, t_x = (last − first) in days (NOT "days since")
- ``T``              = customer age = (anchor − first) in days
- ``monetary_value`` = average value of the customer's REPEAT transactions (0 for one-timers)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Gamma-Gamma's independence assumption is conventionally accepted when |Pearson| is small; 0.30 is
# the common rule-of-thumb ceiling (doc 15). Stricter analysts use 0.10.
GG_CORR_THRESHOLD: float = 0.30

# CLV horizon: predict value over the next 12 months. Our time unit is DAYS (T/recency in days), so
# the forecast window is 365 days (doc 12).
HORIZON_DAYS: int = 365


def repeat_buyers(clv: pd.DataFrame) -> pd.DataFrame:
    """Customers with at least one REPEAT purchase (``frequency > 0``).

    Gamma-Gamma models the value of REPEAT transactions, so it needs ``frequency > 0``; one-time
    buyers (``frequency == 0``) are excluded from it and fall back to the population mean (doc 15).
    BG/NBD, by contrast, KEEPS them (a frequency-0 customer is still informative about dropout).
    """
    return clv[clv["frequency"] > 0]


def one_timer_share(clv: pd.DataFrame) -> float:
    """Share of customers with ``frequency == 0`` — a CLV-reliability caveat to always report (doc 15)."""
    return float((clv["frequency"] == 0).mean())


def gamma_gamma_gate(clv: pd.DataFrame, threshold: float = GG_CORR_THRESHOLD) -> dict:
    """The Gamma-Gamma validity gate: frequency and monetary value must be ~uncorrelated (doc 15).

    Gamma-Gamma assumes a customer's average spend is INDEPENDENT of how often they buy. If frequent
    buyers systematically spend more/less per order, the model is misapplied — so we check this BEFORE
    fitting. A real validity gate, not a formality.

    The assumption is about LINEAR (product-moment) independence, so the pass/fail uses ``pearson``;
    ``spearman`` (monotonic) is reported alongside for honesty — a low Pearson with a higher Spearman
    means "no linear link, but a weak monotonic tendency", worth noting.

    Returns ``{n_repeat, pearson, spearman, threshold, passes}``.
    """
    rep = repeat_buyers(clv)
    pearson = float(rep["frequency"].corr(rep["monetary_value"]))
    spearman = float(rep["frequency"].corr(rep["monetary_value"], method="spearman"))
    return {
        "n_repeat": int(len(rep)),
        "pearson": pearson,
        "spearman": spearman,
        "threshold": threshold,
        "passes": bool(abs(pearson) < threshold),
    }


# --------------------------------------------------------------------------- BG/NBD (doc 15)
def prepare_clv_data(clv_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape ``clv_inputs`` (indexed by Customer ID) into PyMC-Marketing's expected layout.

    PyMC-Marketing wants a ``customer_id`` column (not an index) plus frequency/recency/T. We keep the
    original index order so predictions line up with ``clv_df.index``.
    """
    df = clv_df.reset_index()
    df = df.rename(columns={df.columns[0]: "customer_id"})
    return df


def fit_bg_nbd(clv_df: pd.DataFrame, method: str = "map", **kwargs):
    """Fit the BG/NBD model — the transaction-count + dropout half of CLV (doc 15).

    BG/NBD = "Beta-Geometric / Negative-Binomial-Distribution". Two hidden processes per customer:
    while ALIVE they buy as a Poisson process at rate λ (λ ~ Gamma(r, α) across the population → the
    NBD); after each purchase they drop out with probability p (p ~ Beta(a, b) → the BG). The four
    population parameters (r, α, a, b) are fit from everyone's (frequency, recency, T) — ALL customers,
    including frequency-0 one-timers (they inform the dropout process).

    ``method`` = "map" (fast point estimate — use while iterating) or "mcmc" (full posterior with
    uncertainty — the final deliverable). Returns the fitted ``BetaGeoModel`` (carries its idata).
    """
    from pymc_marketing.clv import BetaGeoModel

    df = prepare_clv_data(clv_df)[["customer_id", "frequency", "recency", "T"]]
    model = BetaGeoModel(data=df)
    model.fit(method=method, **kwargs)
    return model


def _per_customer(da, index: pd.Index, name: str) -> pd.Series:
    """Collapse a PyMC-Marketing prediction DataArray to one value per customer.

    For MAP there is a single posterior sample; for MCMC there are many — averaging over
    chain/draw gives the point estimate (MAP value) or the posterior mean (MCMC) respectively.
    """
    reduce_dims = [d for d in da.dims if d in ("chain", "draw")]
    vals = da.mean(dim=reduce_dims).values if reduce_dims else da.values
    return pd.Series(np.asarray(vals).ravel(), index=index, name=name)


def predict_alive(model, clv_df: pd.DataFrame) -> pd.Series:
    """P(alive) per customer — the posterior probability the customer is still active (doc 15).

    A recent last purchase relative to the customer's own rate → high P(alive); a long silence from a
    once-frequent buyer → low P(alive). This is the model's churn-risk read.
    """
    df = prepare_clv_data(clv_df)[["customer_id", "frequency", "recency", "T"]]
    return _per_customer(model.expected_probability_alive(data=df), clv_df.index, "p_alive")


def predict_purchases(model, clv_df: pd.DataFrame, future_t: int = HORIZON_DAYS) -> pd.Series:
    """Expected number of purchases each customer makes in the next ``future_t`` days (doc 15)."""
    df = prepare_clv_data(clv_df)[["customer_id", "frequency", "recency", "T"]]
    da = model.expected_purchases(data=df, future_t=future_t)
    return _per_customer(da, clv_df.index, "expected_purchases")


# --------------------------------------------------------------------------- Gamma-Gamma + CLV (doc 15)
def fit_gamma_gamma(clv_df: pd.DataFrame, method: str = "map", **kwargs):
    """Fit the Gamma-Gamma model — the £-per-transaction half of CLV (doc 15).

    Each customer has a true average spend; observed values scatter (Gamma) around it, and that
    per-customer mean itself varies across the population (also Gamma) → Gamma-Gamma. Fit ONLY on
    repeat buyers (``frequency > 0``); one-timers have no repeat transaction to learn a mean from and
    fall back to the population average (handled in :func:`predict_spend`). Returns the fitted model.
    """
    from pymc_marketing.clv import GammaGammaModel

    rep = prepare_clv_data(repeat_buyers(clv_df))[["customer_id", "monetary_value", "frequency"]]
    model = GammaGammaModel(data=rep)
    model.fit(method=method, **kwargs)
    return model


def predict_spend(gg_model, clv_df: pd.DataFrame) -> pd.Series:
    """Expected average spend per transaction, per customer (with Bayesian shrinkage, doc 15).

    Repeat buyers get a shrunk estimate (a blend of their own observed average and the population
    mean — thin evidence pulls toward the population); one-timers (excluded from the fit) get the
    population mean. Returns one value per customer in ``clv_df`` order.
    """
    rep = repeat_buyers(clv_df)
    rep_df = prepare_clv_data(rep)[["customer_id", "monetary_value", "frequency"]]
    shrunk = _per_customer(gg_model.expected_customer_spend(data=rep_df), rep.index, "spend")

    pop_mean = float(gg_model.expected_new_customer_spend().mean())
    out = pd.Series(pop_mean, index=clv_df.index, name="spend")   # default = population mean
    out.loc[shrunk.index] = shrunk.values                          # repeat buyers override
    return out


def predict_clv(gg_model, bg_nbd_model, clv_df: pd.DataFrame, future_t_months: int = 12,
                discount_rate: float = 0.0) -> pd.Series:
    """Predicted CLV per customer over ``future_t_months`` months (doc 15, default 12; doc 12).

    CLV = E[future transactions] (BG/NBD) × E[value per transaction] (Gamma-Gamma), optionally
    discounted (``discount_rate`` per year; default 0 = undiscounted v1, doc 12). Repeat buyers use
    PyMC-Marketing's combined ``expected_customer_lifetime_value`` (proper month-by-month DCF); one-
    timers use BG/NBD's expected purchases × the population-mean spend (their Gamma-Gamma fallback).
    """
    out = pd.Series(0.0, index=clv_df.index, name="clv")

    rep = repeat_buyers(clv_df)
    rep_df = prepare_clv_data(rep)[["customer_id", "frequency", "recency", "T", "monetary_value"]]
    cl = gg_model.expected_customer_lifetime_value(
        transaction_model=bg_nbd_model, data=rep_df,
        future_t=future_t_months, time_unit="D", discount_rate=discount_rate,
    )
    out.loc[rep.index] = _per_customer(cl, rep.index, "clv").values

    one_timers = clv_df[clv_df["frequency"] == 0]
    if len(one_timers):
        horizon_days = round(future_t_months * 365 / 12)
        pop_mean = float(gg_model.expected_new_customer_spend().mean())
        purchases = predict_purchases(bg_nbd_model, one_timers, future_t=horizon_days)
        out.loc[one_timers.index] = (purchases * pop_mean).values
    return out
