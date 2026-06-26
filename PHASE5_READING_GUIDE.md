# Phase 5 (CLV) — Reading Guide

How to read what we built in Phase 5, in three passes: **concepts → code → analysis**. The math
lives in the design docs + code docstrings + `planning/docs/CLV_Mathematics_Reference.docx`.

## Pass 1 — Concepts (design docs)

1. **`planning/docs/12_CLV_and_Segments_Integration.docx`** — *first*. Why CLV is a SEPARATE track
   from clustering (no circularity) and how they meet in Phase 6. Frames everything.
2. **`planning/docs/15_BGNBD_GammaGamma_CLV_Engine.docx`** — the heart. The naming trap, BG/NBD
   (Poisson buying + Beta-Geometric dropout → P(alive)), Gamma-Gamma (spend + shrinkage), CLV.
3. **`planning/docs/08_Time_Window_and_CLV_Validation.docx`** — the anchor date + temporal holdout,
   incl. the trap (don't rank by raw MAE).
4. **`planning/docs/CLV_Mathematics_Reference.docx`** — formula + explanation + worked example +
   project number for every piece (parallel to the clustering math reference).

## Pass 2 — Code (`src/clv.py`, in pipeline order)

5. Read the functions top-to-bottom (they ARE the pipeline): `gamma_gamma_gate` →
   `fit_bg_nbd`/`predict_alive`/`predict_purchases` → `fit_gamma_gamma`/`predict_spend` →
   `predict_clv` → `evaluate_cutoff`/`calibration_by_frequency`. Each docstring cites a design doc.
6. **`tests/test_clv.py`** — the known-answer checks (gate passes/fails; data-shaping is correct).

## Pass 3 — Analysis (figures + notebook)

7. Walk `reports/figures/05_clv/` in this order:
   - `gamma_gamma_gate.png` — the honesty gate (passes, mild Spearman caveat)
   - `bgnbd_brain.png` — **the key one**: why recency drives churn
   - `bgnbd_outputs.png` — population model (Gamma λ, Beta p) + P(alive)
   - `gamma_gamma_shrinkage.png` — Bayesian shrinkage (thin evidence → mean)
   - `clv_components.png` — CLV = purchases × spend, and its distribution
8. **`notebooks/05a_clv_validation.ipynb`** — markdown cells: the 3/6/9-month backtest, calibration
   plots, and the honest **seasonality finding**. Figures in `reports/figures/05a_clv_validation/`.

## Cross-reference map

| Concept | Design doc | Figure | Code |
|---|---|---|---|
| Two tracks, no circularity | 12 | — | module docstring |
| Validity gate | 15 | `gamma_gamma_gate.png` | `gamma_gamma_gate` |
| BG/NBD (count + P(alive)) | 15 | `bgnbd_brain.png`, `bgnbd_outputs.png` | `fit_bg_nbd`, `predict_alive`, `predict_purchases` |
| Gamma-Gamma (spend, shrinkage) | 15 | `gamma_gamma_shrinkage.png` | `fit_gamma_gamma`, `predict_spend` |
| CLV = purchases × spend | 15, 12 | `clv_components.png` | `predict_clv` |
| Temporal-holdout validation | 08 | `calibration_small_multiples.png`, `parameter_stability.png` | `evaluate_cutoff`; `05a` |
| MAP vs MCMC / uncertainty | 15 | *(05b, upcoming)* | `fit_*(method=...)` |

## The 4 findings (the story of Phase 5)

1. **Gate is a qualified pass** — Pearson 0.02 (pass) but a mild monotonic creep (Spearman 0.25), flagged.
2. **The "brain"** — frequent + recent = alive; frequent + silent = churned. Recency drives churn.
3. **Shrinkage** — the less we know about a customer, the more we lean on the population mean.
4. **Seasonality limit** — BG/NBD assumes none; short holdouts ending at Christmas are biased, but the
   12-month horizon averages it out (same honesty stance as Phase 4's continuum).

## 15-minute version

Doc 15 §1–6 → `bgnbd_brain.png` + `gamma_gamma_shrinkage.png` + `clv_components.png` → the markdown of
`05a`. That's the whole CLV story.

## Status

`src/clv.py` engine + validation ✅ · `05a` validation ✅ · **`05b` production (MCMC) — next**.
One-line summary: `CLAUDE.md` → STATUS.
