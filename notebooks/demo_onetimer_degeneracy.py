"""Teaching demo (not pipeline): SHOW why one-timers can't be clustered (the degeneracy).

Nothing hardcoded — reads the raw core features and proves numerically that one-time buyers collapse
onto a lower-dimensional sliver of the 4-D feature space:
  - Frequency is constant (= 1)            -> one dead dimension
  - Recency == Tenure (first = last)       -> two features perfectly collinear
So their 4x4 feature covariance is RANK 2, not 4: there is nothing for clustering to discover.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import utils

FEATS = ["Recency", "Frequency", "Monetary", "Tenure"]
core = pd.read_parquet(utils.DATA_PROCESSED / "core_features.parquet")
ot = core["Frequency"] == 1                       # the one-timer rule
rb = ~ot

print(f"one-timers: {ot.sum():,} ({ot.mean():.1%}) | repeat: {rb.sum():,}\n")

# ---- numerical proofs --------------------------------------------------------------------
print("[1] Frequency for one-timers — unique values:", sorted(core.loc[ot, "Frequency"].unique()),
      "| std =", round(core.loc[ot, "Frequency"].std(), 6))
rt_gap = (core.loc[ot, "Recency"] - core.loc[ot, "Tenure"]).abs().max()
print(f"[2] Recency == Tenure for one-timers?  max|R − T| = {rt_gap}  ->  identical")
print(f"    corr(Recency, Tenure):  one-timers = {core.loc[ot, 'Recency'].corr(core.loc[ot, 'Tenure']):.3f}"
      f"  |  repeat = {core.loc[rb, 'Recency'].corr(core.loc[rb, 'Tenure']):.3f}")
ab_gap = (core.loc[ot, "AvgBasket"] - core.loc[ot, "Monetary"]).abs().max()
print(f"[3] AvgBasket == Monetary for one-timers?  max|AB − M| = {ab_gap}  ->  identical")

# Rank of the feature covariance per group (standardise by OVERALL std so it's scale-fair).
z = (core[FEATS] - core[FEATS].mean()) / core[FEATS].std()
cov_ot = np.cov(z[ot].T)
cov_rb = np.cov(z[rb].T)
eig_ot = np.sort(np.linalg.eigvalsh(cov_ot))[::-1]
eig_rb = np.sort(np.linalg.eigvalsh(cov_rb))[::-1]
print(f"\n[4] feature covariance eigenvalues (standardised):")
print(f"    one-timers: {np.round(eig_ot, 3)}  -> rank {np.linalg.matrix_rank(cov_ot)} of 4 (2 dead dims)")
print(f"    repeat:     {np.round(eig_rb, 3)}  -> rank {np.linalg.matrix_rank(cov_rb)} of 4")

# ---- figure ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (A) Recency vs Tenure — one-timers sit exactly on the y=x line.
axes[0].scatter(core.loc[rb, "Recency"], core.loc[rb, "Tenure"], s=5, alpha=0.25,
                color="#4f81a3", label="repeat buyers")
axes[0].scatter(core.loc[ot, "Recency"], core.loc[ot, "Tenure"], s=5, alpha=0.35,
                color="#c0504d", label="one-timers")
lim = [0, core["Tenure"].max()]
axes[0].plot(lim, lim, color="black", ls="--", lw=1, label="Recency = Tenure")
axes[0].set_xlabel("Recency (days)"); axes[0].set_ylabel("Tenure (days)")
axes[0].set_title("(A) One-timers lie EXACTLY on Recency = Tenure")
axes[0].legend(fontsize=8, markerscale=2)

# (B) Frequency: one-timers are a single spike at 1; repeat buyers spread.
axes[1].hist(np.clip(core.loc[rb, "Frequency"], 0, 30), bins=30, color="#4f81a3",
             alpha=0.8, label="repeat buyers")
axes[1].hist(core.loc[ot, "Frequency"], bins=30, color="#c0504d", alpha=0.8, label="one-timers (all = 1)")
axes[1].set_yscale("log")
axes[1].set_xlabel("Frequency (distinct invoices, clipped at 30)")
axes[1].set_ylabel("customers (log)")
axes[1].set_title("(B) Frequency is constant (=1) for one-timers → a dead dimension")
axes[1].legend(fontsize=8)

# (C) Eigenvalue spectrum: one-timers have 2 near-zero eigenvalues (rank 2).
x = np.arange(4)
w = 0.38
axes[2].bar(x - w / 2, eig_ot, w, color="#c0504d", label="one-timers")
axes[2].bar(x + w / 2, eig_rb, w, color="#4f81a3", label="repeat buyers")
axes[2].axhline(0, color="black", lw=0.6)
axes[2].set_xticks(x); axes[2].set_xticklabels([f"PC{i+1}" for i in range(4)])
axes[2].set_ylabel("eigenvalue (variance)")
axes[2].set_title("(C) Covariance spectrum: one-timers are RANK 2, not 4")
axes[2].legend(fontsize=8)

fig.suptitle("Why one-timers are split off: they collapse onto a 2-D sliver of the 4-D feature space",
             fontsize=13)
fig.tight_layout()
out = utils.REPORTS_FIGURES / "teaching"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "onetimer_degeneracy.png", dpi=150, bbox_inches="tight")
print("\nsaved:", out / "onetimer_degeneracy.png")
