"""Unit tests for src.segments (Phase 6 profiling + validation, doc 13)."""
import numpy as np
import pandas as pd

from src import segments


def _toy() -> pd.DataFrame:
    return pd.DataFrame({
        "segment": pd.Categorical(["R0", "R0", "R1", "one-timer"],
                                  categories=segments.SEGMENT_ORDER, ordered=True),
        "Recency": [10, 20, 50, 400],
        "Frequency": [10, 12, 3, 1],
        "Monetary": [4000, 4000, 800, 200],
        "Tenure": [700, 700, 300, 400],
        "p_alive": [0.97, 0.97, 0.90, 1.0],
        "clv": [1800, 1800, 800, 200],
        "ReturnRate": [0.10, 0.10, 0.0, 0.0],
    })


def test_segment_names_cover_order():
    assert set(segments.SEGMENT_NAMES) == set(segments.SEGMENT_ORDER)


def test_profile_table_medians_and_shares():
    prof = segments.profile_table(_toy())
    assert prof.loc["R0", "n"] == 2
    assert prof.loc["R0", "persona"] == "Champions"
    assert prof.loc["R0", "Monetary"] == 4000           # median of [4000, 4000]
    assert prof.loc["R1", "Frequency"] == 3
    # the three share columns each sum to ~100%
    for col in ["pct", "rev_share_pct", "clv_share_pct"]:
        assert abs(prof[col].sum() - 100) < 0.5
    # R0 dominates revenue: (4000+4000) of (4000+4000+800+200) = 8000/9000
    assert prof.loc["R0", "rev_share_pct"] == round(8000 / 9000 * 100, 1)


def test_cliffs_delta_extremes():
    assert segments.cliffs_delta([10, 11, 12], [1, 2, 3]) == 1.0      # x entirely above y
    assert segments.cliffs_delta([1, 2, 3], [10, 11, 12]) == -1.0     # x entirely below y
    assert abs(segments.cliffs_delta([1, 2, 3], [1, 2, 3])) < 1e-9    # identical -> 0


def test_kruskal_effect_sizes_ranks_discriminative_feature():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "segment": ["A"] * n + ["B"] * n,
        "signal": np.r_[rng.normal(0, 1, n), rng.normal(5, 1, n)],   # clearly separates A/B
        "noise": rng.normal(0, 1, 2 * n),                            # no separation
    })
    es = segments.kruskal_effect_sizes(df, ["signal", "noise"])
    assert es.index[0] == "signal"            # signal ranks first
    assert es.loc["signal", "eta2_H"] > 0.5
    assert es.loc["noise", "eta2_H"] < 0.1


def test_cramers_v_perfect_vs_independent():
    a = pd.Series(["x", "x", "y", "y"] * 25)
    assert segments.cramers_v(a, a) > 0.99                            # perfectly associated
    rng = np.random.default_rng(1)
    indep = pd.Series(rng.integers(0, 2, len(a)))
    assert segments.cramers_v(a, indep) < 0.3                         # ~independent


def test_classifier_separation_recovers_blobs():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)
    df = pd.DataFrame(X, columns=["Recency", "Frequency"])
    df["segment"] = y
    res = segments.classifier_separation(df, features=["Recency", "Frequency"])
    assert res["cv_accuracy"] > 0.9 and res["cv_accuracy"] > res["baseline"]


def test_value_at_stake_shares_and_actions():
    vs = segments.value_at_stake(_toy())
    assert vs.loc["R0", "action"] == "Protect"
    assert vs.loc["one-timer", "action"] == "Convert"
    assert abs(vs["rev_share_pct"].sum() - 100) < 0.5


def test_assign_customer_actions_adds_columns_and_tiers():
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "segment": pd.Categorical(["R0"] * n + ["R1"] * n, categories=segments.SEGMENT_ORDER, ordered=True),
        "Monetary": rng.uniform(100, 5000, 2 * n),
        "clv": rng.uniform(50, 3000, 2 * n),
    })
    out = segments.assign_customer_actions(df)
    assert {"persona", "action", "clv_tier"} <= set(out.columns)
    assert set(out["clv_tier"].cat.categories) == {"Low", "Mid", "High"}
    assert out.loc[out.segment == "R0", "action"].iloc[0] == "Protect"
