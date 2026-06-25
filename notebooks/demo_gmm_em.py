"""Teaching demo (not pipeline): visualise EM for a Gaussian Mixture, the soft sibling of K-Means.

Runs GMM EM BY HAND on the real repeat-buyer data (2-D PCA view), capturing every iteration so we
can watch: (1) points shaded by SOFT responsibility (a blend of the 3 component probabilities), and
(2) each component's COVARIANCE ELLIPSE stretching/tilting as the M-step re-estimates it. Contrast
with K-Means: there assignments are hard and 'ellipses' are forced circles. We also track the
log-likelihood, which EM increases every iteration (vs K-Means inertia, which decreases).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

from src import utils

SEED = utils.RANDOM_SEED
REG = 1e-6   # reg_covar: tiny ridge on Σ to prevent singular collapse (doc 11)
# No hardcoded K: read the chosen K from the Phase-4a decision artifact.
K = json.load(open(utils.DATA_PROCESSED / "cluster_choice.json"))["chosen_k"]

X4 = pd.read_parquet(utils.DATA_PROCESSED / "clustering_matrix_main.parquet").to_numpy()
P = PCA(n_components=2, random_state=SEED).fit_transform(X4)
n = len(P)

# base colours as RGB so we can BLEND them by responsibility (soft membership).
base_rgb = np.array([[0.31, 0.51, 0.64],   # blue
                     [0.75, 0.31, 0.30],   # red
                     [0.61, 0.73, 0.35]])  # green

# --- initialise: spread means, CIRCULAR covariances (so we can watch them become ovals) ------
rng = np.random.default_rng(SEED)
means = P[rng.choice(n, K, replace=False)].copy()
scale = P.var(axis=0).mean()
covs = np.array([np.eye(2) * scale for _ in range(K)])
weights = np.full(K, 1.0 / K)

# --- EM, by hand, capturing every iteration --------------------------------------------------
history = []   # per iter: means, covs, resp, loglik (the state that PRODUCED resp/loglik)
prev_ll = None
for _ in range(40):
    # E-step: weighted component densities -> responsibilities (posterior membership).
    dens = np.stack(
        [weights[k] * multivariate_normal(means[k], covs[k], allow_singular=True).pdf(P)
         for k in range(K)], axis=1)                      # (n, K)
    total = dens.sum(axis=1, keepdims=True)
    resp = dens / np.clip(total, 1e-300, None)            # γ_ik
    loglik = float(np.log(np.clip(total, 1e-300, None)).sum())
    history.append({"means": means.copy(), "covs": covs.copy(),
                    "resp": resp.copy(), "loglik": loglik})
    if prev_ll is not None and abs(loglik - prev_ll) < 1e-4 * abs(prev_ll):
        break
    prev_ll = loglik

    # M-step: responsibility-weighted MLE updates.
    Nk = resp.sum(axis=0)                                 # soft counts
    weights = Nk / n
    means = (resp.T @ P) / Nk[:, None]
    new_covs = []
    for k in range(K):
        diff = P - means[k]
        cov = (resp[:, k:k + 1] * diff).T @ diff / Nk[k] + REG * np.eye(2)
        new_covs.append(cov)
    covs = np.array(new_covs)

lls = [h["loglik"] for h in history]
final_resp = history[-1]["resp"]
fence = float((final_resp.max(axis=1) < 0.6).mean())
print(f"converged in {len(history)} iterations")
print("log-likelihood per iteration:", [round(v) for v in lls])
print(f"fence-sitters at convergence (max γ < 0.6): {fence:.1%}")


def draw_ellipse(ax, mean, cov, n_std, color):
    """Draw the n_std covariance ellipse: axes = 2·n_std·√eigenvalue, angle from eigenvector."""
    vals, vecs = np.linalg.eigh(cov)              # symmetric -> real eig/ orthonormal vecs
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(mean, w, h, angle=angle, fill=False, edgecolor=color, lw=2))


# --- Plot: snapshots shaded by soft responsibility + ellipses, then the log-likelihood curve --
snaps = sorted(set([0, 1, 2, min(5, len(history) - 1), len(history) - 1]))
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
axes = axes.flat
for ax, it in zip(axes, snaps):
    h = history[it]
    point_rgb = np.clip(h["resp"] @ base_rgb, 0, 1)       # blend = soft membership
    ax.scatter(P[:, 0], P[:, 1], c=point_rgb, s=6, alpha=0.6)
    for k in range(K):
        c = base_rgb[k]
        draw_ellipse(ax, h["means"][k], h["covs"][k], 1.0, c)
        draw_ellipse(ax, h["means"][k], h["covs"][k], 2.0, c)
        ax.scatter(*h["means"][k], color="black", s=120, marker="X", edgecolor="white", lw=1.2)
    tag = "init" if it == 0 else ("converged" if it == len(history) - 1 else f"iter {it}")
    ax.set_title(f"{tag}  ·  ℓ = {h['loglik']:,.0f}")
    ax.set_xticks([]); ax.set_yticks([])

axll = axes[len(snaps)]
axll.plot(range(len(lls)), lls, marker="o", color="#1f4e79")
axll.set_title("Log-likelihood ℓ RISES every iteration (EM guarantee)")
axll.set_xlabel("iteration"); axll.set_ylabel("ℓ = Σ log Σ π·N")
for ax in axes[len(snaps) + 1:]:
    ax.axis("off")

fig.suptitle("EM for a Gaussian Mixture: soft membership (colour blend) + covariance ellipses, "
             "K=3, repeat buyers (PCA 2-D)", fontsize=12.5)
fig.tight_layout()
out = utils.REPORTS_FIGURES / "teaching"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "gmm_em.png", dpi=150, bbox_inches="tight")
print("saved:", out / "gmm_em.png")
