# notebooks/

Numbered notebooks — one per project phase, meant to be read in order. They **narrate** the
analysis and **call** functions from `src/`; they should stay thin (no large logic blocks —
that lives in `src/` and is unit-tested).

| Notebook | Phase | Produces |
|---|---|---|
| `01_cleaning` | Data cleaning | clean transaction table (`data/processed/`) |
| `02_eda` | Exploratory analysis | figures + insights (revenue concentration, repeat rate, time/geography) |
| `03_features` | Feature engineering | per-customer RFM + tenure + AvgBasket table |
| `04_clustering` | Clustering | K-Means / GMM / hierarchical fits |
| `05_validation` | Cluster validation | silhouette / gap / stability + method comparison (ARI) |
| `06_clv` | Probabilistic CLV | BG/NBD + Gamma-Gamma predictions, holdout validation |
| `07_segments_recommendations` | Segments & actions | named personas + action / value-at-stake grid |

Each phase traces back to a recorded decision in `planning/docs/`.
