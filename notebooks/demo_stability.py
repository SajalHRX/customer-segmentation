"""Teaching demo (not pipeline): OBSERVE cluster stability (doc 10).

Nothing hardcoded — reads the chosen K and real matrix, drives `src/clustering.py`. Stability asks
the question internal indices cannot: do the segments REPRODUCE under perturbation, or are they
artifacts? Two complementary views, plus a good-K-vs-over-split contrast:

  (A) Bootstrap Jaccard per cluster at the chosen K — with the doc-10 stability bands.
  (B) Consensus matrix at the chosen K — crisp blocks = stable.
  (C) Consensus matrix at an OVER-SPLIT K (chosen + a few) — watch the blocks smear.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import clustering, utils

SEED = utils.RANDOM_SEED
rng = np.random.default_rng(SEED)
K = json.load(open(utils.DATA_PROCESSED / "cluster_choice.json"))["chosen_k"]
K_OVER = K + 5                                   # an obviously over-split K for contrast
X = pd.read_parquet(utils.DATA_PROCESSED / "clustering_matrix_main.parquet")

# (A) bootstrap Jaccard per cluster at the chosen K (the real function).
jac = clustering.bootstrap_stability(X, K, method="kmeans", n_boot=100, frac=0.8)
print("bootstrap mean Jaccard per cluster (K={}):".format(K))
print(jac.round(3).to_string())

# (B)/(C) consensus on a subsample (memory/speed), at chosen K and at an over-split K.
sub = rng.choice(len(X), size=min(1000, len(X)), replace=False)
Xs = X.iloc[sub]
def consensus_ordered(k):
    M = clustering.consensus_matrix(Xs, k, method="kmeans", n_runs=30, frac=0.8)
    order = np.argsort(clustering.labels_of(clustering.fit_kmeans(Xs, k), Xs))
    return M[np.ix_(order, order)]
C_good = consensus_ordered(K)
C_over = consensus_ordered(K_OVER)
# Crispness = how close entries are to 0 or 1 (1 = perfectly block-diagonal).
crisp = lambda M: float(1 - 2 * np.mean(np.minimum(M, 1 - M)))
print(f"consensus crispness: K={K} -> {crisp(C_good):.3f} | K={K_OVER} -> {crisp(C_over):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

axes[0].bar([f"cluster {c}" for c in jac.index], jac.values,
            color=["#4f81a3", "#c0504d", "#9bbb59", "#8064a2", "#4bacc6"][:len(jac)])
for thr, lab, col in [(0.85, "highly stable", "#2c7a2c"), (0.75, "stable", "#b8860b"),
                      (0.60, "dissolves below", "#c0504d")]:
    axes[0].axhline(thr, ls="--", lw=1, color=col, label=f"{thr:g} {lab}")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("mean bootstrap Jaccard")
axes[0].set_title(f"(A) Bootstrap stability per cluster (K={K})\nJ = |A∩B| / |A∪B|, 100 resamples")
axes[0].legend(fontsize=7, loc="lower right")

sns.heatmap(C_good, cmap="rocket_r", vmin=0, vmax=1, ax=axes[1], cbar_kws={"label": "co-cluster"},
            xticklabels=False, yticklabels=False)
axes[1].set_title(f"(B) Consensus at chosen K={K}  (crispness {crisp(C_good):.2f})")
sns.heatmap(C_over, cmap="rocket_r", vmin=0, vmax=1, ax=axes[2], cbar_kws={"label": "co-cluster"},
            xticklabels=False, yticklabels=False)
axes[2].set_title(f"(C) Consensus over-split K={K_OVER}  (crispness {crisp(C_over):.2f})")

fig.suptitle("Cluster stability — does the partition reproduce? (observed on real repeat buyers)",
             fontsize=13)
fig.tight_layout()
out = utils.REPORTS_FIGURES / "teaching"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "stability.png", dpi=150, bbox_inches="tight")
print("saved:", out / "stability.png")
