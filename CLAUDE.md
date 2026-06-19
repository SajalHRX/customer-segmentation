# Customer Segmentation Project — Context & Resume File

> **Purpose of this file:** persistent project memory. If a session exits, reading this
> file (plus the `docs/` folder) restores full context to continue work. Keep the
> **STATUS** section below up to date at the end of each working session.

---

## 🔖 STATUS (update this every session)

- **Current phase:** Design phase COMPLETE & pushed (8 commits, 2026-06-20). 18 planning docs +
  6 design diagrams. No analysis code yet. (Commit pause from 2026-06-19 is lifted — design batch committed.)
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
- **Key principle (new):** CORE vs SUPPORTING variables — core lane (R/F/M/Tenure) drives clustering;
  supporting lane (wholesaler flag, Country, product mix, return-rate, CLV) NEVER enters clustering,
  only feeds EDA/validation/profiling/recommendations. A supporting var explains & validates segments,
  never defines them.
- **Working pattern:** topic by topic, in depth; after each topic is decided, append a doc to
  `planning/docs/_generate_docs.py`, regenerate. Technical register by default (plain language only
  when asked). Standing rule: every experiment observable ([[feedback-experiment-observability]]).
  Preview Mermaid via `mmdc -i x.mmd -o reports/figures/x.png -p /tmp/puppeteer.json -b white -s 2`.
- **Current tree:** root `README.md` + `.gitignore` + `CLAUDE.md` + `requirements.txt`;
  `data/{raw,processed}/`; `planning/docs/` (18 .docx: 00-17 + _generate_docs.py);
  `design/` (README + 6 diagrams: project-architecture, data-cleaning, feature-engineering, two-track-modeling,
  validation-flow, segments-to-actions); `notebooks/`, `src/`, `tests/`, `reports/figures/` (purpose-READMEs;
  PNGs of all 6 diagrams; no code yet).
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
