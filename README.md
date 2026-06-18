# Customer Segmentation — Online Retail II

A rigorous, **explainable** customer-segmentation study on ~1M real e-commerce transactions
(UCI *Online Retail II*). The aim is not just to *produce* segments, but to *validate* them,
*compare* methods, attach a *probabilistic* lifetime-value model, and turn the result into
prioritised marketing actions.

> **One-liner:** clean 1M+ real transactions → engineer RFM features → segment customers
> (K-Means, **validated** with silhouette / gap / bootstrap stability, **compared** against
> GMM and hierarchical) → layer a **probabilistic** BG/NBD + Gamma-Gamma CLV model → translate
> into a segment × value action grid.

## Status

**Design complete; implementation starting.** The full methodology — every decision and its
reasoning — is recorded in [`planning/docs/`](planning/docs) (16 documents, 8 design
discussions). Architecture/pipeline diagrams live in [`design/`](design). No analysis code yet.

## Repository layout

| Path | Contents |
|---|---|
| `data/raw/` | original `online_retail_II.xlsx` (~45 MB, git-ignored — download separately) |
| `data/processed/` | cleaned + feature tables (git-ignored) |
| `planning/docs/` | design decisions & reasoning (the *"why"*) — `_generate_docs.py` builds the `.docx` |
| `design/` | architecture & pipeline diagrams (the *"how it's wired"*) |
| `notebooks/` | `01_cleaning` … `07_segments_recommendations` — phase-by-phase narrative |
| `src/` | reusable, unit-tested modules (data_prep, features, clustering, validation, clv, viz) |
| `reports/` | exported figures + written report |
| `tests/` | `pytest` unit tests on `src/` |

## Dataset

UCI *Online Retail II* (2009–2011): ~1.07M transactions, 5,942 customers, 43 countries.
The raw `.xlsx` is **not** committed; place it in `data/raw/` before running. See
`planning/docs/03_Dataset.docx` for the full profile.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place online_retail_II.xlsx in data/raw/, then open notebooks/ in order
```

---

*Author: Sajal Singhal — MSc Statistics portfolio project.*
