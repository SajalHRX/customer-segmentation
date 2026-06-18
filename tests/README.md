# tests/

`pytest` unit tests on `src/` — small **synthetic fixtures with hand-verifiable answers**.
We test the **deterministic, correctness-critical** logic (where a silent bug corrupts
everything downstream), not stochastic model *outputs* (cluster count, etc. — that's what the
validation notebooks are for).

| Test file | Checks |
|---|---|
| `test_data_prep.py` | missing-ID rows dropped; `C`-cancellations removed; negative-qty returns handled per decision; zero/negative prices removed; duplicates dropped; row-count reconciliation |
| `test_features.py` | RFM on a known fixture; `AvgBasket == Monetary / Frequency`; `log1p` handles zeros; recency/tenure vs the fixed anchor 2012-01-01 |
| `test_clv.py` | the naming trap: BG/NBD `frequency == distinct invoices − 1`; single-purchase customer → `frequency = 0`; `T` to anchor; recency = first→last span |
| `test_validation.py` | effect-size / Jaccard helpers return known values on toy inputs; clustering deterministic under a fixed seed |
