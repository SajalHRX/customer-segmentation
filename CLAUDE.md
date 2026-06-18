# Customer Segmentation Project — Context & Resume File

> **Purpose of this file:** persistent project memory. If a session exits, reading this
> file (plus the `docs/` folder) restores full context to continue work. Keep the
> **STATUS** section below up to date at the end of each working session.

---

## 🔖 STATUS (update this every session)

- **Current phase:** Repo restructured to a clean slate. File structure for the actual
  analysis is intentionally NOT decided yet (the old notebooks/src/reports/tests scaffolding
  was rejected and deleted).
- **Last done (2026-06-18):** Cleaned the repo down to bare essentials. Root now holds only
  `.gitignore`, `CLAUDE.md`, `requirements.txt`, and `data/` (with the 45MB dataset).
  All planning material consolidated under `planning/docs/`. Deleted empty scaffolding
  dirs (notebooks/, src/, reports/, tests/). Pushed to private repo
  `SajalHRX/customer-segmentation` (branch `main`).
- **Current tree:** `.gitignore`, `CLAUDE.md`, `requirements.txt`, `data/raw/` (dataset),
  `planning/docs/` (7 .docx + _generate_docs.py).
- **➡️ NEXT STEP:** Decide the new analysis file structure with Sajal, then begin
  **data cleaning**: load both xlsx sheets, handle 22.8% missing Customer ID,
  `C`-cancellations, negative-qty returns → clean transaction table.
- **Not yet started:** new structure, any analysis code/notebooks.
- **GitHub auth note:** `gh` active account = SajalHRX (personal). Switch back for LinkedIn
  work with `gh auth switch -u sajsingh_LinkedIn`.

---

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
