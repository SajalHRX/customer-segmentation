"""Teaching demo (not pipeline): OBSERVE the comparison/validation methods (doc 11).

Nothing hardcoded — reads the chosen K and real matrix, drives `src/clustering.py`. Four panels:

  (A) ARI 3x3 — agreement among K-Means / GMM / Ward (the headline credibility result).
  (B) Why "Adjusted": corrupt labels step by step and watch ARI fall to 0 while the raw Rand Index
      plateaus high — the chance-correction, observed.
  (C) Cophenetic scatter — original distance vs dendrogram merge-height; the correlation IS
      cophenetic_corr. Ward vs single linkage, to see a faithful tree vs a poor one.
  (D) PCA biplot — how R/F/M/Tenure load onto the 2-D view every scatter uses.
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
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, rand_score

from src import clustering, utils

SEED = utils.RANDOM_SEED
rng = np.random.default_rng(SEED)
# Default to the chosen K; override with `python demo_comparison.py <K>` (K=3/4/5). Panels A (ARI) and
# B (corruption) are K-specific; C (cophenetic) and D (PCA biplot) are K-independent (unchanged).
CHOSEN = json.load(open(utils.DATA_PROCESSED / "cluster_choice.json"))["chosen_k"]
K = int(sys.argv[1]) if len(sys.argv) > 1 else CHOSEN
SUFFIX = "" if K == CHOSEN else f"_k{K}"
X = pd.read_parquet(utils.DATA_PROCESSED / "clustering_matrix_main.parquet")
Xnp = X.to_numpy()
n = len(X)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# ---- (A) ARI 3x3 among the three methods at the chosen K ------------------------------------
labels = {
    "K-Means": clustering.labels_of(clustering.fit_kmeans(X, K), X),
    "GMM": clustering.labels_of(clustering.fit_gmm(X, K, covariance_type="full"), X),
    "Ward": clustering.labels_of(clustering.fit_ward(X, K)),
}
ari = clustering.ari_matrix(labels).astype(float)
print("ARI matrix:\n", ari.round(3).to_string())
sns.heatmap(ari, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1, square=True,
            cbar_kws={"label": "ARI"}, ax=axes[0, 0])
axes[0, 0].set_title(f"(A) Cross-method agreement (ARI), K={K}")

# ---- (B) Adjusted vs raw Rand under progressive label corruption ----------------------------
base = labels["K-Means"]
fracs = np.linspace(0, 1, 11)
ari_curve, ri_curve = [], []
for f in fracs:
    corrupt = base.copy()
    m = rng.choice(n, int(f * n), replace=False)
    corrupt[m] = rng.integers(0, K, size=len(m))          # randomly relabel a fraction f
    ari_curve.append(adjusted_rand_score(base, corrupt))
    ri_curve.append(rand_score(base, corrupt))
axes[0, 1].plot(fracs, ri_curve, marker="s", color="#9aa7b4", label="Rand Index (raw)")
axes[0, 1].plot(fracs, ari_curve, marker="o", color="#1f4e79", label="Adjusted Rand Index")
axes[0, 1].axhline(0, color="black", lw=0.8)
axes[0, 1].set_title("(B) Why 'Adjusted': raw RI plateaus high; ARI → 0 at random")
axes[0, 1].set_xlabel("fraction of labels randomised")
axes[0, 1].set_ylabel("agreement vs original")
axes[0, 1].legend(fontsize=8)
print(f"\nat 100% corruption: RI={ri_curve[-1]:.3f} (still high) vs ARI={ari_curve[-1]:.3f} (~0)")

# ---- (C) Cophenetic scatter: original distance vs tree merge-height --------------------------
sub = rng.choice(n, size=400, replace=False)             # subsample for a readable pair scatter
Xs = Xnp[sub]
dorig = pdist(Xs)
coph_ward = cophenet(linkage(Xs, "ward"), dorig)[1]
coph_single = cophenet(linkage(Xs, "single"), dorig)[1]
r_ward = np.corrcoef(dorig, coph_ward)[0, 1]
r_single = np.corrcoef(dorig, coph_single)[0, 1]
print(f"\ncophenetic r: ward={r_ward:.3f}  single={r_single:.3f}")
axes[1, 0].scatter(dorig, coph_ward, s=3, alpha=0.15, color="#4f81a3", label=f"Ward (r={r_ward:.2f})")
axes[1, 0].scatter(dorig, coph_single, s=3, alpha=0.15, color="#c0504d",
                   label=f"single (r={r_single:.2f})")
lim = [0, dorig.max()]
axes[1, 0].plot(lim, lim, color="black", ls="--", lw=0.8)
axes[1, 0].set_title("(C) Cophenetic: original distance vs tree merge-height\n(correlation = cophenetic_corr)")
axes[1, 0].set_xlabel("original pairwise distance  d_ij")
axes[1, 0].set_ylabel("cophenetic distance  c_ij")
axes[1, 0].legend(fontsize=8)

# ---- (D) PCA biplot: feature loadings on the 2-D view ---------------------------------------
pca = PCA(n_components=2, random_state=SEED).fit(Xnp)
coords = pca.transform(Xnp)
ev = pca.explained_variance_ratio_
axes[1, 1].scatter(coords[:, 0], coords[:, 1], s=4, alpha=0.15, color="#9aa7b4")
scale = np.abs(coords).max() * 0.9
for j, feat in enumerate(X.columns):
    vx, vy = pca.components_[0, j] * scale, pca.components_[1, j] * scale
    axes[1, 1].arrow(0, 0, vx, vy, color="#c0504d", width=0.01, head_width=0.12, length_includes_head=True)
    axes[1, 1].text(vx * 1.1, vy * 1.1, feat, color="#1f4e79", fontsize=10, ha="center", va="center")
axes[1, 1].set_title("(D) PCA biplot — how features load on PC1/PC2")
axes[1, 1].set_xlabel(f"PC1 ({ev[0]:.0%} variance)")
axes[1, 1].set_ylabel(f"PC2 ({ev[1]:.0%} variance)")
print(f"\nPCA explained: PC1={ev[0]:.1%}, PC2={ev[1]:.1%}, together={ev.sum():.1%}")
print("loadings (rows=features, cols=PC1,PC2):")
print(pd.DataFrame(pca.components_.T, index=X.columns, columns=["PC1", "PC2"]).round(3).to_string())

fig.suptitle("Comparison & validation methods — observed on the real repeat-buyer matrix", fontsize=13)
fig.tight_layout()
out = utils.REPORTS_FIGURES / "teaching"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / f"comparison_validation{SUFFIX}.png", dpi=150, bbox_inches="tight")
print("\nsaved:", out / f"comparison_validation{SUFFIX}.png")
