# Phase 4 — Teaching Figures Index

Mechanism-revealing figures for the clustering math (one observable plot per method in
`src/clustering.py`). Pair each with the matching section of
`planning/docs/Clustering_Mathematics_Reference.docx`. All are generated **data-driven** from the
real repeat-buyer matrix by the `notebooks/demo_*.py` scripts — nothing hardcoded; each demo reads
the chosen K from `data/processed/cluster_choice.json`.

## Two kinds of figure

- **All-K** — the figure already sweeps K internally, so one image covers every K. No variants.
- **K-specific** — the figure depends on a single deployed K, so there are `_k4` / `_k5` variants
  alongside the default (`= K=3`, the chosen K). Compare them side by side to see why K=3 won.

## The figures

| File | Shows | Kind | Demo script | Math § |
|---|---|---|---|---|
| `kmeans_lloyd.png` (+ `_k4`,`_k5`) | K-Means as EM: E-step assign + M-step move, inertia falling | **K-specific** | `demo_kmeans_lloyd.py` | A1 |
| `gmm_em.png` (+ `_k4`,`_k5`) | GMM EM: soft-membership colour blend + covariance ellipses; log-likelihood rising | **K-specific** | `demo_gmm_em.py` | A2 |
| `internal_indices.png` (+ `_k4`,`_k5`) | (A) silhouette anatomy *(K-specific)* · (B) elbow geometry · (C) CH · (D) DB *(B/C/D sweep all K)* | **mixed** | `demo_internal_indices.py` | B1–B4 |
| `gap_statistic.png` | null vs real, log Wₖ curves, Gap(k) ± sₖ | **all-K** | `demo_gap_bic.py` | B5 |
| `bic_anatomy.png` | BIC by covariance type; −2ℓ vs p·ln n trade-off | **all-K** | `demo_gap_bic.py` | B6 |
| `stability.png` (+ `_k4`,`_k5`) | (A) bootstrap Jaccard per cluster · (B) consensus at K · (C) consensus over-split (K+5) | **K-specific** | `demo_stability.py` | C1–C2 |
| `comparison_validation.png` (+ `_k4`,`_k5`) | (A) ARI 3×3 · (B) ARI-vs-RI under corruption *(A/B K-specific)* · (C) cophenetic · (D) PCA biplot *(C/D K-independent)* | **mixed** | `demo_comparison.py` | C3–C5 |
| `onetimer_degeneracy.png` | why one-timers are split off: Recency=Tenure, Frequency=1, covariance is rank-2 not 4 | **K-independent** | `demo_onetimer_degeneracy.py` | doc 17 / 04a step 4a |

> Pipeline (non-teaching) figures live in `reports/figures/04a_choosing_k/` and
> `reports/figures/04b_method_comparison/` — the actual decision artifacts from the notebooks.

## Regenerate

```bash
cd ~/customer-segmentation
# default = chosen K (3)
.venv/bin/python notebooks/demo_kmeans_lloyd.py
# a specific K (writes the _k<K> variant)
.venv/bin/python notebooks/demo_kmeans_lloyd.py 4
.venv/bin/python notebooks/demo_kmeans_lloyd.py 5
```

The K-specific demos are: `demo_kmeans_lloyd`, `demo_gmm_em`, `demo_internal_indices`,
`demo_stability`, `demo_comparison`. The K-independent demos (no K argument) are `demo_gap_bic` and `demo_onetimer_degeneracy`.

## Cross-K takeaway (K=3 vs 4 vs 5)

| Signal | K=3 | K=4 | K=5 |
|---|---|---|---|
| Silhouette | **0.367** (peak) | 0.336 | 0.305 |
| ARI K-Means↔Ward | 0.61 | **0.43** | 0.68 |
| Min bootstrap Jaccard | 0.787 | 0.773 | 0.783 |
| Consensus crispness | **0.973** | 0.954 | 0.965 |
| Customer groups (+ one-timer) | 4 | 5 | 6 |

K=3 is best on silhouette/CH and crispest; K=4 is *weakest* on cross-method agreement (0.43,
despite the elbow/DB liking it); K=5 adds groups without a silhouette gain. No K is dramatically
better — the data is a continuum — so K=3 wins on tightness + parsimony.
