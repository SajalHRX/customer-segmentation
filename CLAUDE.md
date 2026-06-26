# Customer Segmentation Project — Context & Resume File

> **Purpose of this file:** persistent project memory. If a session exits, reading this
> file (plus the `docs/` folder) restores full context to continue work. Keep the
> **STATUS** section below up to date at the end of each working session.

---

## 🔖 STATUS (update this every session)

- **Current phase:** CODING. Phase 1 (CLEANING) DONE & pushed: `src/data_prep.py` = load_raw +
  drop_invalid_records + resolve_non_purchases + build_clean_table (all unit-tested, doc 16);
  `notebooks/01_cleaning.ipynb` executed (raw-quality EDA + reconciliation figures);
  `data/processed/clean_transactions.parquet` (790,704 rows, 5,852 customers) + non_purchases.parquet saved.
  Reconciliation balances losslessly. **Phase 2 (EDA) DONE & pushed:** `02_eda.ipynb` — Pareto
  (top 20% = 77% of revenue), 27.6% one-time buyers, spend skew 25.6→0.27 via log1p, UK 83.8% of
  revenue; 5 figures in `reports/figures/02_eda/`. Every EDA finding empirically confirmed a design
  decision. **NEXT = Phase 3 features:** `src/features.py` (+ tests) + `notebooks/03_features.ipynb` —
  per-customer rollup anchored to 2012-01-01; the 3 lanes (core R/F/M/Tenure → log1p+RobustScaler;
  CLV inputs raw freq=inv−1/recency=t_x/T/monetary; supporting: wholesaler flag + return-rate +
  country) per docs 04/09/15/16/17. Watch: Tenure=first→anchor; AvgBasket profiling-only (=M/F).
  **Phase 3 (FEATURES) DONE & pushed:** `src/features.py` = build_core_features / build_clv_inputs
  (via PyMC-Marketing `rfm_summary`, day-based) / build_supporting_features / scale_features — 4 funcs,
  8 fast tests. `03_features.ipynb` Feature-EDA confirmed: LOG_COLS = F/M only (Recency raw skew 0.89;
  Tenure NOT logged — logging WORSENS it −0.62→−1.65); F–M corr 0.85 (H3 ✓); AvgBasket excluded
  (definitional M/F redundancy — correlation doesn't show it under log1p, was an overclaim, fixed);
  54 wholesalers (0.9%, >1169 units/invoice, IQR cutoff). Saved core_features / clustering_matrix /
  clv_inputs / supporting_features (.parquet) + `scaler_robust.joblib` to data/processed (gitignored).
  **Phase 4 (CLUSTERING) DONE (2026-06-24):** `src/clustering.py` = shared methods (fit_kmeans/gmm/ward,
  labels_of, sweep_k, knee_point, gap_statistic+gap_choice, gmm_bic, bootstrap_stability,
  consensus_matrix, ari_matrix, cophenetic_corr, pca_2d) + 12 tests (synthetic blobs, known K=3).
  TWO notebooks mirroring docs 10/11: `04a_choosing_k` + `04b_method_comparison` (NOT one 04_).
  **4a:** split off one-timers (1,618 = 27.6%) → cluster 4,234 repeat buyers (doc 17 ii); the
  full-pop silhouette was HIGHER but only via the degenerate one-timer blob — diagnosed, not followed.
  **4b/4c:** K=3 chosen by triangulation (silhouette+CH peak at 3, most stable: mean Jaccard 0.79,
  0 dissolving; K=4 close runner-up via elbow+DB). Gap→8 and GMM-BIC→8 diagnosed as continuum /
  GMM-singularity artifacts (diag BIC went negative), NOT followed. **04b:** at K=3 cross-method ARI
  is MODERATE not unanimous (K-Means↔Ward 0.61; GMM diverges 0.38/0.27; cophenetic 0.54) → honest
  "continuum, useful discretisation not natural kinds"; DEPLOY K-Means (silhouette 0.37, balanced
  sizes); GMM 4.3% fence-sitters. Saved: clustering_matrix_main, onetimer_mask, cluster_choice.json,
  cluster_labels (3 methods), segment_assignment (4 groups: R0/R1/R2 + one-timer). Deferred 4f
  robustness experiments (scaler/weighting, doc 09). NEXT = Phase 5 CLV: BG/NBD + Gamma-Gamma
  (PyMC-Marketing) on clv_inputs (docs 15, 12), then Phase 6 profile + name + cross-validate vs CLV.
- **Phase 4 fix + teaching (2026-06-25):** (1) `04a` gap now runs from K=1 (was 2) so the gap stat can
  return its "no clusters" verdict — gap→K=1 (continuum); CHOSEN_K unchanged (3). (2) Slow line-by-line
  walkthrough of `src/clustering.py` with the MATH for every method, the user's rule: NOTHING hardcoded
  + an OBSERVABLE figure per method ([[feedback-experiment-observability]]). Built 7 teaching demos in
  `notebooks/demo_*.py` → `reports/figures/teaching/` (kmeans_lloyd, gmm_em, internal_indices,
  gap_statistic, bic_anatomy, stability, comparison_validation) — all read K from cluster_choice.json.
  (3) `planning/docs/Clustering_Mathematics_Reference.docx` (+ `generate_math_reference.py`, run with
  SYSTEM python3): formula + explanation + worked example + project number for all 16 methods.
- **Phase 5 (CLV) IN PROGRESS — engine built (2026-06-26):** `src/clv.py` = the value track (docs 15/12/08),
  SEPARATE from clustering (no circularity). Functions: repeat_buyers, one_timer_share, gamma_gamma_gate,
  prepare_clv_data, fit_bg_nbd, predict_alive, predict_purchases, fit_gamma_gamma, predict_spend,
  predict_clv. 6 tests. Decisions: **MAP while building → MCMC for final**; horizon **12 months**
  (HORIZON_DAYS=365); discounting deferred (d=0, param ready); R/CLVTools cross-check = end-stage stretch.
  **Gamma-Gamma gate:** Pearson 0.021 PASSES (<0.30) but Spearman 0.254 + binned medians £244→£418 =
  mild monotonic caveat (reported, not hidden). **One-timers:** 1,673 (28.6%, day-based freq vs
  clustering's 1,618) — BG/NBD includes them, Gamma-Gamma excludes → population-mean spend fallback.
  **MAP results:** r=0.64 α=62 a=0.15 b=2.76; mean P(alive)=0.89 (2.7% churned); CLV mean £1,396
  median £536 (one-timer £311 vs repeat £1,831). 5 observable figs in reports/figures/05_clv/
  (gamma_gamma_gate, bgnbd_outputs, bgnbd_brain, gamma_gamma_shrinkage, clv_components).
- **05a validation DONE (2026-06-26):** `evaluate_cutoff` + `calibration_by_frequency` in src/clv.py;
  `05a_clv_validation.ipynb` runs 3/6/9-mo temporal holdouts. **Honest finding:** calibration SHAPE
  excellent (model ranks customers well) but a vertical bias (3mo ×0.77 under, 6mo ×0.98, 9mo ×1.12
  over) = **seasonality** (gift-ware Christmas peak; BG/NBD assumes constant rate, doc 15) — averages
  out over the 12-mo horizon. r/α stable, a/b drift smoothly with calibration length. Figs in
  reports/figures/05a_clv_validation/. Also added `planning/docs/CLV_Mathematics_Reference.docx`
  (+generator, SYSTEM python3) and root `PHASE5_READING_GUIDE.md`.
- **05b production DONE — PHASE 5 COMPLETE (2026-06-26):** `05b_clv_production.ipynb` refits BG/NBD +
  Gamma-Gamma with **MCMC** (draws=800 tune=800 chains=4; ~8s+6s). Converged: R-hat=1.00, ESS
  1.0–2.2k. Added `posterior_summary()` to src/clv.py. **Key correction (honesty):** distinguished
  TWO uncertainties — ESTIMATION (credible interval on E[CLV], TIGHT ±~2% because 5,852 customers pin
  the 4 params → ranking is confident) vs PREDICTIVE (individual outcome, WIDE ±50–150%, shrinks with
  frequency). My earlier "£450±[120,980]" pitch was predictive, not estimation; `expected_customer_*`
  gives estimation, `distribution_customer_spend` gives predictive. Figs: parameter_forest,
  clv_uncertainty (panel B = predictive width 1.5→0.2 vs freq, the clean observable). Saved
  **clv_predictions.parquet** (5852×7: p_alive, expected_purchases, clv — each w/ estimation bounds).
  NEXT = Phase 6: join clv_predictions to segment_assignment on Customer ID → profile + name segments
  + segment×CLV grid + free cross-validation (docs 12, 13).
- **Coding conventions in play:** hybrid (logic in `src/` unit-tested via `pytest -m "not slow"`; notebooks
  thin, call src). Notebooks authored as jupytext `.py` (percent) → `jupytext --to notebook --execute` →
  `.ipynb` WITH outputs (commit both). Processed data → Parquet in data/processed (gitignored; ~4s reload
  vs ~33s xlsx). `src/utils.py` holds constants (ANCHOR_DATE, paths, NON_PRODUCT_STOCKCODES, RANDOM_SEED).
  Notebook import shim: add project root to sys.path. ALWAYS narrate code line-by-line
  ([[feedback-explain-code-line-by-line]]).
- **Tooling locked (2026-06-21, discussion #11 → doc 18):** Python **3.12** (Homebrew python3.12) — NOT
  system 3.9 (too old for PyMC-Marketing). `.venv` rebuilt on 3.12; full stack verified (pandas 3.0.3,
  sklearn 1.9, pymc-marketing 0.19.4, pymc 5.28). CLV = **PyMC-Marketing** (Bayesian BG/NBD + Gamma-Gamma,
  uncertainty + posterior predictive checks), replacing unmaintained `lifetimes`. **Dual-language:** Python
  primary + planned R cross-validation layer (clustering via mclust/fpc::clusterboot, CLV via CLVTools) as a
  cross-tool reproducibility check (end-stage, not parallel). Doc gen uses SYSTEM python3 (has python-docx);
  the .venv (3.12) does NOT have python-docx.
- **Last done (2026-06-19→20):** (1) Cleaning decisions → doc 16 (DROP returns + track return-rate;
  invalid-records a-e, postage excluded; wholesaler → feature-eng as supporting). (2) Feature-engineering
  decisions (discussion #10 → doc 17): Tenure = first→anchor (customer age; PATCHED doc 04 which said
  first→last); Tenure transform decided by Feature EDA (prior=no log); single-purchase customers KEPT
  as distinct group (degeneracy Recency=Tenure, AvgBasket=Monetary for one-timers; cluster-structure
  decided in EDA — lean split-off; CLV: BG/NBD incl, Gamma-Gamma excl → pop-mean fallback; report share).
  (3) Built ALL 6 core design diagrams: `project-architecture`, `data-cleaning`, `feature-engineering`
  (3-lane fork), `two-track-modeling` (clustering ‖ CLV, no-circularity note), `validation-flow` (2 lanes),
  `segments-to-actions` (profile→name→segment×CLV grid→recommendations). PNGs in reports/figures via mmdc
  (bold title + grey italic descriptor; `&middot;`/`&times;`/`&amp;` entities render, `&rarr;` does NOT).
- **Doc-sync decisions (2026-06-24, discussion #12):** (1) BACK-PATCHED doc 17's CLV line — it said
  `lifetimes.summary_data_from_transaction_data` but code uses PyMC-Marketing `rfm_summary` (day-based);
  now flagged SUPERSEDED-by-doc-18. (2) DROPPED "product mix" as a feature AND a validation dimension —
  no category column exists (4,620 StockCodes / 5,271 free-text Descriptions), so categories would be
  manufactured noise for a supporting-only var off the critical path (doc 04: every feature justified).
  Removed from doc 17, doc 13 external-validation battery, project-architecture lane, and
  `design/feature-engineering.md`; CLV + country + return-rate remain for external validation.
- **Key principle (new):** CORE vs SUPPORTING variables — core lane (R/F/M/Tenure) drives clustering;
  supporting lane (wholesaler flag, Country, return-rate, CLV) NEVER enters clustering,
  only feeds EDA/validation/profiling/recommendations. A supporting var explains & validates segments,
  never defines them.
- **Working pattern:** topic by topic, in depth; after each topic is decided, append a doc to
  `planning/docs/_generate_docs.py`, regenerate. Technical register by default (plain language only
  when asked). Standing rule: every experiment observable ([[feedback-experiment-observability]]).
  Preview Mermaid via `mmdc -i x.mmd -o reports/figures/design/x.png -p /tmp/puppeteer.json -b white -s 2`.
  Figures are GROUPED: design renders → `reports/figures/design/`; analysis figures →
  `reports/figures/<phase>/` via `utils.figure_path(phase, name)` (e.g. `01_cleaning/reconciliation.png`).
- **Current tree:** root `README.md` + `.gitignore` + `CLAUDE.md` + `requirements.txt`; `.venv/` (py3.12, gitignored);
  `data/{raw,processed}/`; `planning/docs/` (19 .docx: 00-18 + _generate_docs.py);
  `design/` (README + 7 diagrams incl. experiment-map); `notebooks/`, `src/`, `tests/`, `reports/figures/`
  (purpose-READMEs; PNGs of all 7 diagrams; no analysis code yet).
- **File structure (LOCKED):** hybrid = numbered notebooks (narrate + call src) + `src/`
  reusable tested modules, built LAZILY (no empty stubs). `planning/` = the "why" (decisions),
  `design/` = the "how it's wired" (architecture/pipeline diagrams as Mermaid markdown).
- **Decisions locked so far:** anchor date = 2012-01-01; CLV validated via temporal holdout at
  3/6/9-month cutoffs; log1p on F/M/AvgBasket; RobustScaler primary (+StandardScaler experiment);
  cluster on R/F/M/Tenure (AvgBasket profiling-only, = M/F redundancy); K by triangulation
  (silhouette+gap primary) × stability × business; compare K-Means/GMM/Ward via cross-method ARI;
  CLV = separate post-hoc layer joined on Customer ID (segment × CLV action grid); validate
  personas in 3 tiers (effect-size, classifier, external-variable gold standard); success =
  validated + interpretable + consistent + actionable + honest.
- **➡️ NEXT:** Design DONE — 7 diagrams (6 core + `experiment-map`). `data-eda` is NOT a diagram; it's
  real EDA in the coding phase (raw-quality pass atop `01_cleaning` + behavioural EDA in `02_eda`).
  START CODING: `notebooks/01_cleaning` + `src/data_prep.py` + `tests/test_data_prep.py` against doc 16 —
  first time hitting the real data (load xlsx, apply cleaning rules, row-count reconciliation).
  Design file naming convention still loosely TBD (only `project-architecture` formally agreed).
- **Reminder:** revoke leaked token ghp_5tk… (shared in chat 2026-06-17) still stands.
- **Not yet started:** new structure, any analysis code/notebooks.
- **GitHub auth note:** `gh` active account = SajalHRX (personal). Switch back for LinkedIn
  work with `gh auth switch -u sajsingh_LinkedIn`.

---

## Problem statement (LOCKED 2026-06-18 — discussion #1)

A UK-based online retailer has a finite marketing/CRM budget and currently treats all
customers alike. Using **only transactional history**, segment customers by purchasing
behaviour (RFM + tenure + basket value), validate the segments are real and stable, estimate
each customer's future value probabilistically (CLV), and recommend a differentiated action
per segment — so retention/marketing spend follows expected return.

**Scope (honest boundaries):** answers *who to target and how much effort* — NOT *what
product to recommend* (recommender systems) or *why they churned* (needs richer signals).
Framing chosen because it is the most ambitious framing the purely-transactional data
honestly supports.

## What this project is

A rigorous, explainable **customer-segmentation** project for a **CV / portfolio** piece
(owner: Sajal, MSc Statistics). It upgrades a weak synthetic-data notebook into a real,
statistically rigorous study on real e-commerce data.

**Interview one-liner:** clean 1M+ real transactions → RFM features → K-Means segmentation,
but *validated* (silhouette, gap statistic, bootstrap stability) and *compared* (K-Means vs
GMM vs hierarchical), plus a *probabilistic* BG/NBD + Gamma-Gamma CLV model, translated into
marketing actions.

## Locked-in decisions

- **Goal:** portfolio / CV piece.
- **Dataset:** real — UCI **Online Retail II** (NOT the original synthetic CSV).
- **Rigor:** FULL — cluster validation, method comparison, probabilistic CLV.
- **Style:** GitHub repo, numbered notebooks, reusable `src/` modules; work **phase by
  phase**; **every methodological choice must be explainable / justified**.

## The dataset (verified)

- **File:** `data/raw/online_retail_II.xlsx` (45 MB) — **never edit raw**.
- 1,067,371 rows; 2 sheets (Year 2009-2010 = 525,461, Year 2010-2011 = 541,910).
- 8 columns; 5,942 customers; 43 countries (92% UK); Dec 2009 → Dec 2011.
- **Cleaning to-do (quirks):** 243,007 missing Customer ID (22.8%) → drop for segmentation;
  19,494 cancellations (Invoice starts `C`); 22,950 negative-quantity returns.
- **vs synthetic data:** this real set has NO engagement columns (email/click/ad). We build
  RFM ourselves; clustering features = **RFM + tenure + average basket value**.

## Column importance map (the north star)

| Column | Importance | Drives |
|---|---|---|
| `Customer ID` | 5/5 | Unit of analysis |
| `InvoiceDate` | 5/5 | Recency, tenure, BG/NBD timing |
| `Invoice` | 4/5 | Frequency (+ `C` = cancellation) |
| `Quantity` × `Price` | 4/5 | Monetary, avg basket |
| `Country` | 2/5 | Geographic context (92% UK) |
| `StockCode` / `Description` | 1/5 | Optional product profiling |

**Rule:** every engineered feature must trace to a justified column. Never feed
low-signal/noise columns into clustering (a flaw in the original notebook).

## Planned repo structure

```
customer-segmentation/
├── CLAUDE.md            ← this file (context/resume)
├── README.md           ← (todo) storefront
├── requirements.txt    ← (todo)
├── .gitignore          ← (todo)
├── data/
│   ├── raw/            ← online_retail_II.xlsx (done)
│   └── processed/      ← (todo) cleaned + RFM tables
├── docs/               ← planning conversation as .docx (done) + _generate_docs.py
├── notebooks/          ← (todo) 01_data_cleaning ... 07_segments_and_recommendations
├── src/                ← (todo) data_prep.py, features.py, clustering.py, clv.py
├── reports/            ← (todo) figures/ + written report
└── tests/              ← (todo)
```

## The 7 planned notebooks

01 Cleaning → 02 EDA → 03 RFM → 04 Clustering → 05 Validation (silhouette/gap/bootstrap)
→ 06 CLV (BG/NBD + Gamma-Gamma) → 07 Segments & recommendations.

## Where the full context lives

The complete planning discussion is in `docs/` (open the `.docx` files):
`00_Project_Overview`, `01_Source_Notebook_Analysis`, `02_Project_Value`, `03_Dataset`,
`04_Column_Importance_Map`, `05_Problem_Formulation`, `06_Repo_Structure_and_Roadmap`.

---

## 🔁 How to resume (instructions for Claude)

1. Read this file's **STATUS** section for where we left off.
2. Skim `docs/00_Project_Overview.docx` (or the relevant topic doc) for detail.
3. Continue from **NEXT STEP**. Keep working phase by phase; explain every choice.
4. At session end, **update the STATUS section** above (Current phase / Last done / Next step).
