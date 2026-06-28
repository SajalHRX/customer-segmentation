# Customer Segmentation & Lifetime Value — Online Retail II

A rigorous, **explainable** customer-segmentation study on ~1M real e-commerce transactions
(UCI *Online Retail II*). The goal isn't just to *produce* segments — it's to **validate** them,
**compare** methods, attach a **probabilistic** lifetime-value model, and turn the result into
prioritised marketing actions, surfacing every limitation honestly along the way.

> **One line:** clean 1M+ real transactions → engineer RFM features → segment customers (K-Means,
> *validated* with silhouette / gap / bootstrap stability, *compared* against GMM & hierarchical) →
> layer a *Bayesian* BG/NBD + Gamma-Gamma CLV model (PyMC-Marketing, with uncertainty) → a
> segment × value **action grid**.

> 📚 **New here, or want to teach this end-to-end?** Start with the
> [**Teaching Handbook**](TEACHING_HANDBOOK.md) — a beginner-friendly, chapter-by-chapter guide to how
> the whole project was built, with primers, figures to show, and exercises.

## The problem

A UK-based online retailer has a finite marketing budget and currently treats all customers the
same. **Which customers are worth what, and what should we do about each group?** Answering it means
separating customers by *behaviour*, forecasting each one's *future value*, and spending where the
money actually is.

## The answer — 4 validated personas, each with an action

| Persona | Action | Headcount | Revenue (now) | **Future value (CLV)** |
|---|---|---:|---:|---:|
| **Champions** | **Protect** — retain, VIP, don't discount | 26.7% | **80.7%** | **71.7%** |
| **Rising** | **Grow** — nurture into Champions | 21.0% | 7.9% | **16.4%** |
| **At-Risk** | **Win-back** — reactivate the high-CLV lapsers | 24.7% | 8.1% | 5.7% |
| **One-Timers** | **Convert** — cheap 2nd-purchase nudge | 27.6% | 3.2% | 6.2% |

Segments are **~equal in headcount but wildly unequal in value** — so budget follows *value at
stake*, not headcount. The richest cell is **High-CLV Champions (~£4.5M)**; the prime win-back target
is **High-CLV At-Risk (~£0.26M)**. Final per-customer table → `data/processed/customer_segments_actions.parquet`.

## How it was built (6 phases)

```
1M+ transactions → clean (790k rows) → features (R/F/M/Tenure)
   → cluster (K=3 + one-timers) → CLV (BG/NBD × Gamma-Gamma, MCMC) → profile + validate + act
```

| Phase | Notebooks | What |
|---|---|---|
| 1 Cleaning | `01_cleaning` | dedupe, returns, non-products → 790k clean rows, 5,852 customers |
| 2 EDA | `02_eda` | Pareto (top 20% = 77% revenue), 28% one-timers, skew, seasonality |
| 3 Features | `03_features` | per-customer R/F/M/Tenure (+ CLV inputs, supporting vars) |
| 4 Clustering | `04a_choosing_k`, `04b_method_comparison` | **K=3** by triangulation; K-Means vs GMM vs Ward |
| 5 CLV | `05a_clv_validation`, `05b_clv_production` | BG/NBD + Gamma-Gamma, MAP→**MCMC**, credible intervals |
| 6 Segments | `06a_profiling_validation`, `06b_recommendations` | name, **validate**, action grid |

## What makes it rigorous — the honesty

Every phase surfaced a real limitation instead of hiding it:

- **It's a continuum, not islands** — cross-method agreement is moderate (ARI 0.61), so we deploy a
  *useful discretisation*, not "natural kinds." K=3 chosen by triangulating silhouette + gap +
  stability, not one chart.
- **One-timers are degenerate** — their feature covariance is *rank-2, not 4* (proved numerically),
  so they're split off and handled separately rather than distorting the clustering.
- **CLV has a seasonality limit** — BG/NBD assumes a constant rate, but this is gift-ware with a
  Christmas peak; short backtests are biased, the 12-month horizon averages it out (validated 3/6/9-mo).
- **Estimation ≠ predictive uncertainty** — the credible interval on *expected* CLV is tight (±2%);
  an *individual's* outcome is wide (±50–150%). We separate them rather than overclaim.
- **The validation fires** — segments differ strongly on **predicted CLV (η²=0.62), a variable the
  clustering never saw** — two independent models agreeing the segments are real — while geography is
  an honest *null* (Cramér's V 0.09).

The methodology and reasoning behind every decision live in [`planning/docs/`](planning/docs) (21
documents), with two standalone **mathematics references**
([clustering](planning/docs/Clustering_Mathematics_Reference.docx),
[CLV](planning/docs/CLV_Mathematics_Reference.docx)) and per-phase reading guides.

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/` | `01_cleaning` … `06b_recommendations` — the phase-by-phase narrative (+ `demo_*` teaching figures) |
| `src/` | reusable, unit-tested modules: `data_prep`, `features`, `clustering`, `clv`, `segments`, `utils` |
| `tests/` | `pytest` suite (35 tests) over `src/` |
| `planning/docs/` | design decisions & the *why* (`_generate_docs.py` builds the `.docx`) + math references |
| `design/` | architecture & pipeline diagrams (the *how it's wired*) |
| `reports/figures/` | analysis figures per phase + a `teaching/` set explaining the math, observably |
| `data/raw,processed/` | the `.xlsx` and cleaned/feature/prediction tables (git-ignored) |
| `TEACHING_HANDBOOK.md` | beginner-friendly instructor's guide to the whole build (chapters + exercises) |
| `PHASE5_READING_GUIDE.md` | how to read the CLV phase (concepts → code → analysis) |

## Reproduce

Requires **Python ≥ 3.12** (PyMC-Marketing needs ≥ 3.10).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place online_retail_II.xlsx in data/raw/, then run the notebooks in order:
jupytext --to notebook --execute notebooks/01_cleaning.py   # … through 06b
pytest -q                                                   # 35 tests
```

Notebooks are authored as `jupytext` `.py` (percent) and executed to `.ipynb` with outputs; all
logic lives in `src/` so the notebooks stay thin.

## The bottom line

Behaviour is a continuum — but a *useful* 4-way discretisation that, independently, differs on future
value: **Champions to protect, Rising to grow, At-Risk to win back, One-Timers to convert cheaply.**
The grade is in the judgement on display — validate, compare, admit the continuum, report what failed —
not the cluster count or a pretty chart.

---

*Author: Sajal Singhal — MSc Statistics portfolio project.*
