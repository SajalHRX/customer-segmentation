"""Segment profiling, validation, and actions — Phase 6 (docs 12, 13, 14).

Turns the K=3 clustering labels (+ the one-timer group = 4 segments) into NAMED, VALIDATED personas
with predicted CLV and recommended actions. This module holds the reusable logic; the notebooks
(06a profiling+validation, 06b grid+recommendations) call it.

Profiling discipline (doc 13): profile on RAW units (the scaled centroids are uninterpretable) and
report MEDIANS (robust to residual skew). Names are derived from the behavioural signature AFTER
inspecting the profile — never imposed from a textbook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import utils

CLUSTER_FEATURES = ["Recency", "Frequency", "Monetary", "Tenure"]

# Cluster labels in a sensible reading order (value-descending), and the data-derived persona names
# (doc 13: named after inspecting the raw-unit profile, not before).
SEGMENT_ORDER = ["R0", "R1", "R2", "one-timer"]
SEGMENT_NAMES = {
    "R0": "Champions",      # recent · very frequent · high-spend · long-tenured · ~80% of revenue
    "R1": "Rising",         # recent · newest tenure · CLV share double its revenue share (growing)
    "R2": "At-Risk",        # established but lapsed ~1yr · P(alive) fallen · CLV share < revenue share
    "one-timer": "One-Timers",  # single purchase · low value · ~28% of customers, ~3% of revenue
}


def load_segment_panel() -> pd.DataFrame:
    """Join the per-customer artifacts into one analysis table, keyed by Customer ID.

    Combines the segment label, RAW core features (R/F/M/Tenure), CLV predictions (clv, p_alive,
    expected_purchases) and supporting variables (Country, ReturnRate, wholesaler flag). The CLV and
    supporting columns are EXTERNAL to clustering — used for the non-circular validation (doc 13 C).
    """
    p = utils.DATA_PROCESSED
    seg = pd.read_parquet(p / "segment_assignment.parquet")
    core = pd.read_parquet(p / "core_features.parquet")
    clvp = pd.read_parquet(p / "clv_predictions.parquet")[["clv", "p_alive", "expected_purchases"]]
    sup = pd.read_parquet(p / "supporting_features.parquet")
    df = core.join(seg).join(clvp).join(sup)
    df["segment"] = pd.Categorical(df["segment"], categories=SEGMENT_ORDER, ordered=True)
    df["persona"] = df["segment"].map(SEGMENT_NAMES).astype("category")
    return df


def profile_table(df: pd.DataFrame) -> pd.DataFrame:
    """The segment profile table (doc 13 §1): raw-unit medians + headcount/revenue/CLV shares.

    One row per segment, in SEGMENT_ORDER. Columns: headcount n and %, median Recency/Frequency/
    Monetary/Tenure, median P(alive) and CLV, and the strategic pair revenue-share (value NOW) vs
    CLV-share (value FUTURE) — the gap between them is the most decision-relevant signal.
    """
    n_total = len(df)
    rev_total = df["Monetary"].sum()
    clv_total = df["clv"].sum()

    g = df.groupby("segment", observed=True)
    prof = g.agg(
        n=("segment", "size"),
        Recency=("Recency", "median"),
        Frequency=("Frequency", "median"),
        Monetary=("Monetary", "median"),
        Tenure=("Tenure", "median"),
        p_alive=("p_alive", "median"),
        CLV=("clv", "median"),
        ReturnRate=("ReturnRate", "median"),
    )
    prof.insert(0, "persona", [SEGMENT_NAMES[s] for s in prof.index])
    prof["pct"] = (prof["n"] / n_total * 100).round(1)
    prof["rev_share_pct"] = (g["Monetary"].sum() / rev_total * 100).round(1)
    prof["clv_share_pct"] = (g["clv"].sum() / clv_total * 100).round(1)
    return prof


# --------------------------------------------------------------------------- Stage 2: validation (doc 13)
def kruskal_effect_sizes(df: pd.DataFrame, features, group: str = "segment") -> pd.DataFrame:
    """Tier A — Kruskal-Wallis omnibus + effect size per feature (doc 13).

    KW is a rank-based non-parametric ANOVA (right for skewed RFM); at n≈5,900 the p-value is ~0 for
    everything and uninformative, so we report the EFFECT SIZE η²_H = (H − k + 1)/(N − k) ∈ [0,1] and
    RANK features by it — the discriminative-feature ranking that justifies the persona names.
    """
    from scipy.stats import kruskal

    k = df[group].nunique()
    N = len(df)
    rows = []
    for f in features:
        samples = [v[f].dropna().to_numpy() for _, v in df.groupby(group, observed=True)]
        H = float(kruskal(*samples).statistic)
        rows.append({"feature": f, "H": H, "eta2_H": max((H - k + 1) / (N - k), 0.0)})
    return pd.DataFrame(rows).set_index("feature").sort_values("eta2_H", ascending=False)


def cliffs_delta(x, y) -> float:
    """Cliff's delta — a non-parametric pairwise effect size in [−1, 1] (doc 13 post-hoc magnitude).

    δ = P(x > y) − P(x < y). |δ|: <0.147 negligible, <0.33 small, <0.474 medium, else large. Computed
    from the Mann-Whitney U so it is fast and ties-aware. Two segments small on ALL features = "secret
    twins" (a sign K is too high).
    """
    from scipy.stats import mannwhitneyu

    x, y = np.asarray(x), np.asarray(y)
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2.0 * u / (len(x) * len(y)) - 1.0


def classifier_separation(df: pd.DataFrame, features=CLUSTER_FEATURES, group: str = "segment",
                          seed: int = utils.RANDOM_SEED, cv: int = 5) -> dict:
    """Tier B — can a classifier recover the segments from the features? (doc 13).

    Hide the labels, train a random forest to predict the segment, score by CROSS-VALIDATED accuracy
    (so we measure real separability, not memorisation). High accuracy vs the majority-class baseline =
    the segments are separable in combination; the feature importances are the multivariate reading.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    X = df[features].to_numpy()
    y = df[group].astype(str).to_numpy()
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    clf.fit(X, y)
    return {
        "cv_accuracy": float(scores.mean()),
        "cv_std": float(scores.std()),
        "baseline": float(pd.Series(y).value_counts(normalize=True).max()),
        "importances": pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False),
    }


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramér's V — association between two categoricals in [0, 1] (doc 13, Tier C / RFM agreement)."""
    from scipy.stats import chi2_contingency

    ct = pd.crosstab(a, b)
    chi2 = chi2_contingency(ct, correction=False)[0]   # no Yates correction (convention for Cramér's V)
    n = ct.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def rfm_quintile_segments(df: pd.DataFrame) -> pd.Series:
    """Tier C (rule-based RFM) — the classic R/F/M quintile score, INDEPENDENT of the clustering.

    R is reverse-scored (recent = high); F and M score higher = better. Frequency is rank-broken
    before quantiling (many ties at 1/2/3). Returns the summed RFM score (3–15) per customer; high
    agreement with the clusters corroborates, structured divergence = the clustering refining the
    rules' arbitrary quintile cutoffs.
    """
    r = pd.qcut(df["Recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    f = pd.qcut(df["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    m = pd.qcut(df["Monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    return (r + f + m).rename("RFM_score")


# --------------------------------------------------------------------------- Stage 3: actions (docs 12, 14)
# One segment, one clear action — clarity is the deliverable (doc 14 §3). Spend follows VALUE AT STAKE
# (revenue + CLV share), not headcount; the segment × CLV grid refines intensity within a segment.
SEGMENT_ACTIONS = {
    "R0": "Protect",     # Champions — retain / VIP, do NOT discount (they would buy anyway)
    "R1": "Grow",        # Rising — nurture the newest high-potential into Champions
    "R2": "Win-back",    # At-Risk — reactivate, prioritising the high-CLV lapsers
    "one-timer": "Convert",  # One-Timers — low-cost second-purchase nudge; do not overspend
}


def assign_customer_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Attach persona, action, and a within-segment CLV tier to every customer (docs 12, 14).

    The CLV tier (Low/Mid/High terciles *within* each segment) is the grid's second axis: it sets the
    intensity of the segment's action — e.g. a High-CLV At-Risk customer is the best place to spend a
    retention pound, a Low-CLV one is not worth chasing.
    """
    out = df.copy()
    out["persona"] = out["segment"].map(SEGMENT_NAMES).astype("category")
    out["action"] = out["segment"].map(SEGMENT_ACTIONS).astype("category")
    out["clv_tier"] = out.groupby("segment", observed=True)["clv"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["Low", "Mid", "High"]))
    return out


def value_at_stake(df: pd.DataFrame) -> pd.DataFrame:
    """Per-segment headcount / revenue / CLV shares — the lens for budgeting (doc 14 §3).

    Spend should follow money on the line (revenue now + CLV future), NOT headcount: prioritising by
    headcount would pour budget into the ~28% one-timers who are barely worth anything.
    """
    n_total, rev_total, clv_total = len(df), df["Monetary"].sum(), df["clv"].sum()
    g = df.groupby("segment", observed=True)
    out = pd.DataFrame({
        "persona": [SEGMENT_NAMES[s] for s in g.size().index],
        "action": [SEGMENT_ACTIONS[s] for s in g.size().index],
        "headcount_pct": (g.size() / n_total * 100).round(1),
        "rev_share_pct": (g["Monetary"].sum() / rev_total * 100).round(1),
        "clv_share_pct": (g["clv"].sum() / clv_total * 100).round(1),
    })
    return out


def segment_clv_grid(df: pd.DataFrame, value: str = "clv", aggfunc: str = "sum") -> pd.DataFrame:
    """The segment × CLV-tier grid (doc 12) — value (or count) in each segment × Low/Mid/High cell."""
    d = df if "clv_tier" in df else assign_customer_actions(df)
    return d.pivot_table(index="segment", columns="clv_tier", values=value, aggfunc=aggfunc,
                         observed=True)
