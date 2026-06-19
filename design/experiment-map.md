# Experiment Map (observability)

Every "try N variants and compare" point in the project, grouped by stage. The standing rule
([[feedback-experiment-observability]]): **each experiment ends in a comparison artifact** (a table
or plot) — never a silent pick. This map is the catalog so no experiment gets dropped.

> Rendered with `securityLevel: loose` + `htmlLabels: true` for the bold-title / italic-descriptor styling.

```mermaid
flowchart TD
    R["<b>Experiments</b><br/><span style='font-size:11px;color:#e5edf5;font-style:italic'>each ends in a comparison artifact (observability)</span>"]
    R --> FE["<b>Features</b>"]
    R --> CL["<b>Clustering</b>"]
    R --> CV["<b>CLV</b>"]
    FE --> FE1["<b>Scaling</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>RobustScaler vs StandardScaler</span>"]
    FE --> FE2["<b>Weighting</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>equal vs business vs PCA-whitened</span>"]
    CL --> CL1["<b>K selection</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>K = 2..8 &middot; silhouette/gap/CH/DB + stability</span>"]
    CL --> CL2["<b>Method</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>K-Means vs GMM vs Ward &middot; ARI, BIC</span>"]
    CL --> CL3["<b>Cluster structure</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>one-timers in vs split &middot; B2B/B2C together vs separate</span>"]
    CV --> CV1["<b>CLV holdout</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>3 / 6 / 9-month cutoffs &middot; calibration + param stability</span>"]

    classDef root fill:#1f4e79,stroke:#1f4e79,color:#ffffff,rx:6,ry:6;
    classDef group fill:#e8eef5,stroke:#1f4e79,stroke-width:2px,color:#1f4e79,rx:6,ry:6;
    classDef exp fill:#f5f7fa,stroke:#1f4e79,stroke-width:2px,color:#1f2937,rx:6,ry:6;
    class R root;
    class FE,CL,CV group;
    class FE1,FE2,CL1,CL2,CL3,CV1 exp;
```

## The catalog

| Experiment | Variants | Compared on | Stage | Doc |
|---|---|---|---|---|
| **Feature scaling** | RobustScaler vs StandardScaler | segment stability / ARI | Features | 09 |
| **Feature weighting** | equal vs business-weighted vs PCA-whitened | segments + metrics | Features | 09 |
| **K selection** | K = 2…8 | silhouette / gap / CH / DB + stability | Clustering | 10 |
| **Clustering method** | K-Means vs GMM vs Ward | cross-method ARI, BIC | Clustering | 11 |
| **Cluster structure** | one-timers in vs split-off; B2B/B2C together vs separate | EDA-driven | Clustering | 16, 17 |
| **CLV holdout** | 3 / 6 / 9-month cutoffs | calibration + parameter stability | Validation | 08 |

(Sub-experiments folded into the above: GMM `covariance_type` via BIC; hierarchical linkage = Ward.)
