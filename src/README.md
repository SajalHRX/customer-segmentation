# src/

Reusable, importable, **unit-tested** logic — the engineering backbone the notebooks call.
Modules are added **only when a notebook first needs to reuse something** (no empty stubs).

Planned modules:

| Module | Responsibility |
|---|---|
| `data_prep.py` | cleaning: drop missing IDs, handle cancellations/returns, remove invalid prices/dupes |
| `features.py` | per-customer RFM + tenure + AvgBasket; `log1p`; anchor-date (2012-01-01) recency/tenure |
| `clustering.py` | scaling, K-Means / GMM / hierarchical fits, PCA for viz |
| `validation.py` | silhouette/gap/CH/DB, bootstrap stability (Jaccard), ARI, effect sizes |
| `clv.py` | BG/NBD + Gamma-Gamma summary table & predictions (note the frequency = invoices − 1 trap) |
| `viz.py` | shared plotting helpers (snake plot, calibration plot, comparison tables) |

Rule: when notebook logic is reused or grows past ~a screen, lift it here and import it back.
