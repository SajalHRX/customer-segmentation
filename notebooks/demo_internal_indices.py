"""Teaching demo (not pipeline): OBSERVE the internal K-selection indices (doc 10).

Nothing hardcoded — reads the chosen K and the real repeat-buyer matrix, then drives the actual
`src/clustering.py` functions to render four mechanism-revealing panels:

  (A) Silhouette ANATOMY  — per-point silhouette widths, grouped by cluster (which clusters are
      tight vs mushy), not just the average.
  (B) Elbow GEOMETRY      — the normalised inertia curve, the chord, and the vertical gaps whose
      maximum IS `knee_point` (so you can see why it picks the K it does).
  (C) Calinski-Harabasz   — the between/within variance ratio vs K (higher better).
  (D) Davies-Bouldin      — average worst-case cluster overlap vs K (lower better).
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
from sklearn.metrics import silhouette_samples

from src import clustering, utils

SEED = utils.RANDOM_SEED
K = json.load(open(utils.DATA_PROCESSED / "cluster_choice.json"))["chosen_k"]
K_RANGE = range(2, 9)            # doc 10's sane business search range (2–8), not an answer
X = pd.read_parquet(utils.DATA_PROCESSED / "clustering_matrix_main.parquet")

# Drive the REAL module — no re-implementation, no hardcoded metric values.
panel = clustering.sweep_k(X, K_RANGE, method="kmeans")
knee = clustering.knee_point(panel.index, panel["inertia"])
sil_k = int(panel["silhouette"].idxmax())
ch_k = int(panel["calinski_harabasz"].idxmax())
db_k = int(panel["davies_bouldin"].idxmin())
print(panel.round(3).to_string())
print(f"\nknee(elbow) K={knee} | silhouette K={sil_k} | CH K={ch_k} | DB K={db_k} | deployed K={K}")

# Per-point silhouette at the chosen K (for the anatomy panel).
labels = clustering.labels_of(clustering.fit_kmeans(X, K), X)
sample_sil = silhouette_samples(X.to_numpy(), labels)
mean_sil = sample_sil.mean()

colors = ["#4f81a3", "#c0504d", "#9bbb59", "#8064a2", "#4bacc6", "#f79646"]
fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))

# ---- (A) Silhouette anatomy at the chosen K -------------------------------------------------
axA = axes[0, 0]
y = 0
for c in range(K):
    vals = np.sort(sample_sil[labels == c])
    axA.barh(np.arange(y, y + len(vals)), vals, height=1.0, color=colors[c % len(colors)],
             edgecolor="none")
    axA.text(-0.05, y + len(vals) / 2, f"cluster {c}\n(n={len(vals)})", va="center", ha="right",
             fontsize=8)
    y += len(vals) + 40
axA.axvline(mean_sil, color="black", ls="--", lw=1.2, label=f"mean silhouette = {mean_sil:.3f}")
axA.set_title(f"(A) Silhouette anatomy at deployed K={K}\n"
              f"width = s(i) = (b−a)/max(a,b); negatives = likely mis-assigned")
axA.set_xlabel("silhouette s(i)")
axA.set_yticks([])
axA.legend(loc="lower right", fontsize=8)

# ---- (B) Elbow geometry: the chord and the gaps that knee_point maximises --------------------
axB = axes[0, 1]
ks = np.array(panel.index, dtype=float)
yv = panel["inertia"].to_numpy()
kn = (ks - ks.min()) / (ks.max() - ks.min())               # exactly what knee_point does
yn = (yv - yv.min()) / (yv.max() - yv.min())
chord = yn[0] + (yn[-1] - yn[0]) * kn
gap = chord - yn
axB.plot(kn, yn, marker="o", color="#1f4e79", label="normalised inertia")
axB.plot([kn[0], kn[-1]], [yn[0], yn[-1]], color="#9aa7b4", ls="-", label="chord (endpoints)")
for i in range(len(kn)):
    axB.plot([kn[i], kn[i]], [yn[i], chord[i]], color="#c0504d", lw=1, alpha=0.7)
ki = int(np.argmax(gap))
axB.scatter([kn[ki]], [yn[ki]], s=160, facecolor="none", edgecolor="#c0504d", lw=2,
            label=f"max gap → knee K={int(ks[ki])}")
axB.set_xticks(kn)
axB.set_xticklabels([int(k) for k in ks])
axB.set_title("(B) Elbow geometry — knee = K of MAX vertical gap below the chord")
axB.set_xlabel("K"); axB.set_ylabel("inertia (normalised)")
axB.legend(fontsize=8)

# ---- (C) Calinski-Harabasz vs K -------------------------------------------------------------
axC = axes[1, 0]
axC.plot(panel.index, panel["calinski_harabasz"], marker="o", color="#1f4e79")
axC.axvline(ch_k, color="#c0504d", ls="--", label=f"peak K={ch_k}")
axC.set_title("(C) Calinski-Harabasz = [tr(B)/(K−1)] / [tr(W)/(n−K)]  (higher = better)")
axC.set_xlabel("K"); axC.set_ylabel("CH score"); axC.legend(fontsize=8)

# ---- (D) Davies-Bouldin vs K ----------------------------------------------------------------
axD = axes[1, 1]
axD.plot(panel.index, panel["davies_bouldin"], marker="o", color="#1f4e79")
axD.axvline(db_k, color="#c0504d", ls="--", label=f"min K={db_k}")
axD.set_title("(D) Davies-Bouldin = mean_i max_j (sᵢ+sⱼ)/d(cᵢ,cⱼ)  (lower = better)")
axD.set_xlabel("K"); axD.set_ylabel("DB score"); axD.legend(fontsize=8)

fig.suptitle("Internal K-selection indices — observed on the real repeat-buyer matrix", fontsize=13)
fig.tight_layout()
out = utils.REPORTS_FIGURES / "teaching"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "internal_indices.png", dpi=150, bbox_inches="tight")
print("saved:", out / "internal_indices.png")
