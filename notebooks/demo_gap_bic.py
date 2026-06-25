"""Teaching demo (not pipeline): OBSERVE the gap statistic and GMM BIC (docs 10/11).

Nothing hardcoded — reads the real repeat-buyer matrix and drives `src/clustering.py`. Produces two
mechanism-revealing figures:

  gap_statistic.png  (A) what the NULL is: real data vs a uniform PCA-box sample;
                     (B) log W_k observed vs null — the gap IS the vertical distance;
                     (C) Gap(k) with ±s_k, and the Tibshirani choice.
  bic_anatomy.png    (D) BIC vs K per covariance type (watch `diag` collapse → singularity);
                     (E) BIC = (−2ℓ) + (p·ln n): the fit term falls, the penalty rises — observed.
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
from sklearn.decomposition import PCA

from src import clustering, utils

SEED = utils.RANDOM_SEED
rng = np.random.default_rng(SEED)
K_CHOSEN = json.load(open(utils.DATA_PROCESSED / "cluster_choice.json"))["chosen_k"]
X = pd.read_parquet(utils.DATA_PROCESSED / "clustering_matrix_main.parquet")
Xnp = X.to_numpy()
n, d = Xnp.shape
out = utils.REPORTS_FIGURES / "teaching"
out.mkdir(parents=True, exist_ok=True)

# =================================================================== GAP STATISTIC
gap_df = clustering.gap_statistic(X, range(1, 9), n_refs=10, reference="pca")
gap_k = clustering.gap_choice(gap_df)
gap_df["null_logW"] = gap_df["log_W"] + gap_df["gap"]   # mean_null = log_W + gap (by definition)
print(gap_df.round(3).to_string())
print(f"gap choice K = {gap_k}\n")

# One null reference sample (mirrors gap_statistic's PCA-box null) for the concept panel.
pca = PCA(random_state=SEED).fit(Xnp)
Xr = pca.transform(Xnp)
lo, hi = Xr.min(axis=0), Xr.max(axis=0)
null = pca.inverse_transform(rng.uniform(lo, hi, size=Xnp.shape))
real2, null2 = Xr[:, :2], pca.transform(null)[:, :2]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
axes[0].scatter(null2[:, 0], null2[:, 1], s=5, alpha=0.25, color="#9aa7b4", label="null (uniform PCA box)")
axes[0].scatter(real2[:, 0], real2[:, 1], s=5, alpha=0.35, color="#4f81a3", label="real customers")
axes[0].set_title("(A) The null: real data clumps; null fills the box")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].legend(fontsize=8)

axes[1].plot(gap_df.index, gap_df["log_W"], marker="o", color="#4f81a3", label="log Wₖ (observed)")
axes[1].plot(gap_df.index, gap_df["null_logW"], marker="s", ls="--", color="#9aa7b4",
             label="E[log Wₖ*] (null)")
axes[1].set_title("(B) Gap = vertical distance between null and observed")
axes[1].set_xlabel("K"); axes[1].set_ylabel("log within-cluster SS"); axes[1].legend(fontsize=8)

axes[2].errorbar(gap_df.index, gap_df["gap"], yerr=gap_df["s_k"], marker="o", color="#1f4e79",
                 capsize=3)
axes[2].axvline(gap_k, color="#c0504d", ls="--", label=f"Tibshirani choice K={gap_k}")
axes[2].axvline(K_CHOSEN, color="#9bbb59", ls=":", label=f"deployed K={K_CHOSEN}")
axes[2].set_title("(C) Gap(k) ± sₖ — rises monotonically (no interior peak = continuum)")
axes[2].set_xlabel("K"); axes[2].set_ylabel("Gap(k)"); axes[2].legend(fontsize=8)

fig.suptitle("Gap statistic — observed on the real repeat-buyer matrix", fontsize=13)
fig.tight_layout()
fig.savefig(out / "gap_statistic.png", dpi=150, bbox_inches="tight")
print("saved:", out / "gap_statistic.png")

# =================================================================== GMM BIC
cov_types = ("spherical", "diag", "tied", "full")
bic = clustering.gmm_bic(X, range(1, 9), covariance_types=cov_types)
print("\n" + bic.round(0).to_string())

# Decompose BIC = (-2ℓ) + (p·ln n) for the 'full' family, recomputing each term from scratch.
def n_params_full(k, d):
    # means: k·d ; covariances (full, symmetric): k·d(d+1)/2 ; weights: k−1
    return k * d + k * d * (d + 1) // 2 + (k - 1)

fit_term, penalty, bic_check = [], [], []
ks = list(range(1, 9))
for k in ks:
    gm = clustering.fit_gmm(X, k, covariance_type="full")
    ll = gm.score(Xnp) * n                 # total log-likelihood (score = per-sample mean)
    p = n_params_full(k, d)
    fit_term.append(-2 * ll)
    penalty.append(p * np.log(n))
    bic_check.append(-2 * ll + p * np.log(n))
# Observability: our hand-computed BIC must match sklearn's (proves the formula, nothing fudged).
print("max |hand BIC − sklearn BIC| (full):", float(np.max(np.abs(np.array(bic_check) - bic["full"].values))))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
for cov in cov_types:
    axes[0].plot(bic.index, bic[cov], marker="o", label=cov)
bic_k = int(bic.min(axis=1).idxmin())
axes[0].axvline(bic_k, color="#c0504d", ls="--", label=f"min BIC K={bic_k}")
axes[0].set_title("(D) BIC vs K by covariance type (lower better)\n`diag` diving = singularity (doc 11)")
axes[0].set_xlabel("K"); axes[0].set_ylabel("BIC"); axes[0].legend(fontsize=8)

axes[1].plot(ks, fit_term, marker="o", color="#c0504d", label="−2ℓ  (misfit ↓ with K)")
axes[1].plot(ks, penalty, marker="s", color="#4f81a3", label="p·ln n  (complexity ↑ with K)")
axes[1].plot(ks, bic_check, marker="^", color="#1f4e79", label="BIC = sum")
axes[1].set_title("(E) BIC anatomy (full): fit vs complexity trade-off")
axes[1].set_xlabel("K"); axes[1].set_ylabel("value"); axes[1].legend(fontsize=8)

fig.suptitle("GMM BIC — observed on the real repeat-buyer matrix", fontsize=13)
fig.tight_layout()
fig.savefig(out / "bic_anatomy.png", dpi=150, bbox_inches="tight")
print("saved:", out / "bic_anatomy.png")
