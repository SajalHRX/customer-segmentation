# Two-Track Modeling (detail)

Zoom-in on the **Modeling** stage of `project-architecture.md`. The key structural idea: clustering
and CLV are **two separate pipelines** on *different* representations of the same customers, meeting
only at the **join on Customer ID** — CLV **never** feeds back into clustering (no circularity, doc 12).
Decisions in `planning/docs/` 11 (methods), 12 (integration), 15 (CLV engine).

> Rendered with `securityLevel: loose` + `htmlLabels: true` for the bold-title / italic-descriptor styling.

```mermaid
flowchart LR
    A["<b>Core Features</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>scaled R / F / M / Tenure</span>"] --> B["<b>Clustering</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>K-Means / GMM / Ward</span>"] --> C["<b>Segment labels</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>one per customer</span>"]
    D["<b>CLV Inputs</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>raw frequency / recency / T / monetary</span>"] --> E["<b>BG/NBD</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>expected purchases + P(alive)</span>"] --> G["<b>Predicted CLV</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>+ P(alive) per customer</span>"]
    D --> F["<b>Gamma-Gamma</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>value per purchase</span>"] --> G
    C --> H{{"<b>Join on Customer ID</b>"}}
    G --> H
    H --> I["<b>Customer table</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>segment + CLV</span>"]
    N["<span style='font-size:11px;color:#9a3412;font-style:italic'>CLV is never an input to clustering<br/>(no circularity, doc 12)</span>"]
    B -.- N

    classDef stage fill:#f5f7fa,stroke:#1f4e79,stroke-width:2px,color:#1f2937,rx:6,ry:6;
    classDef join fill:#e8eef5,stroke:#1f4e79,stroke-width:2px,color:#1f2937;
    classDef note fill:#fff7ed,stroke:#fdba74,stroke-width:1px,color:#9a3412,rx:4,ry:4;
    class A,B,C,D,E,F,G,I stage;
    class H join;
    class N note;
```

## The two tracks

| Track | Input | Model | Output |
|---|---|---|---|
| **Clustering** | scaled Core Features | K-Means / GMM / Ward | segment label per customer |
| **CLV** | raw CLV Inputs | BG/NBD (purchases + P-alive) + Gamma-Gamma (value) | predicted CLV + P(alive) |

They run on **different representations** (scaled-log vs raw) and meet on **Customer ID** → the
final customer table carries **segment + CLV** together. Method *selection* and *validation* are the
next diagram (`validation-flow`).
