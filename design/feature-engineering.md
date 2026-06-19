# Feature Engineering (detail)

Zoom-in on the **Feature Engineering** stage of `project-architecture.md`. Decisions recorded in
`planning/docs/17_Feature_Engineering.docx`. The clean transactions are rolled up to one row per
customer, inspected (Feature EDA), then split into **three lanes** — only **Core** drives the
clustering (the core-vs-supporting principle, doc 16).

> Rendered with `securityLevel: loose` + `htmlLabels: true` for the bold-title / italic-descriptor styling.

```mermaid
flowchart LR
    A["<b>Clean Transaction Table</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>completed purchases</span>"] --> B["<b>Aggregate to Customer Level</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>one row per customer</span>"] --> C["<b>Feature EDA</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>verify skew, correlations, wholesaler tail</span>"]
    C --> D["<b>Core Features</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>R / F / M / Tenure &middot; log + scale</span>"]
    C --> E["<b>CLV Inputs</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>frequency / recency / T / monetary &middot; raw</span>"]
    C --> F["<b>Supporting Variables</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>wholesaler flag &middot; return-rate &middot; country &middot; product mix</span>"]
    D --> G(["Clustering"])
    E --> H(["CLV model"])
    F --> I(["EDA / Validation / Profiling"])

    classDef stage fill:#f5f7fa,stroke:#1f4e79,stroke-width:2px,color:#1f2937,rx:6,ry:6;
    classDef target fill:#eef2f7,stroke:#9aa7b4,stroke-width:1px,color:#4b5563;
    class A,B,C,D,E,F stage;
    class G,H,I target;
```

## The three lanes

| Lane | Features | Goes to |
|---|---|---|
| **Core Features** | R / F / M / Tenure (AvgBasket profiling-only) → `log1p` + RobustScaler | **Clustering** |
| **CLV Inputs** | frequency=inv−1 / recency=t_x / T / monetary — **raw** | **CLV model** |
| **Supporting Variables** | wholesaler flag, return-rate, country, product mix | **EDA / Validation / Profiling** |

> **Feature EDA** is the checkpoint that *justifies* the transforms: it's where the RFM skew (→ `log1p`),
> the AvgBasket=M/F redundancy, and the wholesaler tail are confirmed before features are finalised.
