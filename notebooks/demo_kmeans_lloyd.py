"""Teaching demo (not pipeline): visualise Lloyd's algorithm for K-Means as an EM-style loop.

Runs K-Means BY HAND on the real repeat-buyer data, projected to 2-D (PCA) so we can watch it.
Each iteration = assignment step (E: points -> nearest centroid) + update step (M: centroid ->
members' mean). We capture every iteration to plot the centroids moving and the inertia J falling.
Deliberately uses a *bad* random init (not k-means++) so the motion is visible.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from sklearn.decomposition import PCA

from src import utils

SEED = utils.RANDOM_SEED
# No hardcoded K: default to the chosen K from the Phase-4a decision; allow an override via
# `python demo_kmeans_lloyd.py <K>` to compare K=3/4/5. Output filename gets a _k<K> suffix when overridden.
CHOSEN = json.load(open(utils.DATA_PROCESSED / "cluster_choice.json"))["chosen_k"]
K = int(sys.argv[1]) if len(sys.argv) > 1 else CHOSEN
SUFFIX = "" if K == CHOSEN else f"_k{K}"

# Real repeat-buyer matrix (scaled R/F/M/Tenure), projected to 2-D for the picture.
X4 = pd.read_parquet(utils.DATA_PROCESSED / "clustering_matrix_main.parquet").to_numpy()
P = PCA(n_components=2, random_state=SEED).fit_transform(X4)

# --- Lloyd's algorithm, by hand, capturing every iteration ---------------------------------
rng = np.random.default_rng(SEED)
C = P[rng.choice(len(P), K, replace=False)].copy()   # bad random init -> visible movement

history = []   # (centroids_before, labels, inertia) per iteration
for _ in range(12):
    # E-step: squared distance to each centroid, assign to nearest.
    d2 = ((P[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)   # (n, K)
    labels = d2.argmin(axis=1)
    inertia = d2[np.arange(len(P)), labels].sum()             # J = WCSS
    history.append((C.copy(), labels.copy(), float(inertia)))
    # M-step: each centroid -> mean of its members.
    newC = np.array([P[labels == k].mean(axis=0) if np.any(labels == k) else C[k]
                     for k in range(K)])
    if np.allclose(newC, C):
        break
    C = newC

inertias = [h[2] for h in history]
print(f"converged in {len(history)} iterations")
print("inertia per iteration:", [round(j) for j in inertias])

# --- Plot: 5 iteration snapshots + the inertia curve ---------------------------------------
# K-flexible palette (first 3 match the original 3-cluster colours, so K=3 is unchanged).
_PALETTE = ["#4f81a3", "#c0504d", "#9bbb59", "#8064a2", "#4bacc6", "#f79646", "#7f7f7f"]
colors = np.array([to_rgba(c) for c in _PALETTE[:K]])
snaps = sorted(set([0, 1, 2, min(4, len(history) - 1), len(history) - 1]))
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
axes = axes.flat

for ax, it in zip(axes, snaps):
    Cit, lab, j = history[it]
    ax.scatter(P[:, 0], P[:, 1], c=colors[lab], s=5, alpha=0.35)
    ax.scatter(Cit[:, 0], Cit[:, 1], c="black", s=240, marker="X", edgecolor="white", linewidth=1.5)
    # arrow from this iteration's centroid to where it moves next (the M-step shift).
    if it + 1 < len(history):
        Cnext = history[it + 1][0]
        for k in range(K):
            ax.annotate("", xy=Cnext[k], xytext=Cit[k],
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    tag = "init" if it == 0 else ("converged" if it == len(history) - 1 else f"iter {it}")
    ax.set_title(f"{tag}  ·  J = {j:,.0f}")
    ax.set_xticks([]); ax.set_yticks([])

# Last panel: inertia (WCSS) monotonically decreasing -> convergence.
axj = axes[len(snaps)]
axj.plot(range(len(inertias)), inertias, marker="o", color="#1f4e79")
axj.set_title("Inertia J falls every iteration (never increases)")
axj.set_xlabel("iteration"); axj.set_ylabel("J = within-cluster SS")
for ax in axes[len(snaps) + 1:]:
    ax.axis("off")

fig.suptitle(f"K-Means as EM: E-step (assign) + M-step (move centroid), K={K}, repeat buyers (PCA 2-D)",
             fontsize=13)
fig.tight_layout()
out = utils.REPORTS_FIGURES / "teaching"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / f"kmeans_lloyd{SUFFIX}.png", dpi=150, bbox_inches="tight")
print("saved:", out / f"kmeans_lloyd{SUFFIX}.png")
