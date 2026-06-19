# Validation Flow (detail)

Zoom-in on the **Validation** stage of `project-architecture.md`. Validation proves *two* things, so
it has two lanes: **are the segments real?** (docs 10, 11, 13) and **does the CLV model predict?**
(doc 08). Both feed forward into Segmentation & Profiling. *(Note: "Choose K" is a bounded
sweep-score-decide step — fit + score clustering across K=2–8, then decide once; the downstream runs
once with the chosen K.)*

> Rendered with `securityLevel: loose` + `htmlLabels: true` for the bold-title / italic-descriptor styling.

```mermaid
flowchart LR
    subgraph SEG["Are the segments real?"]
      direction LR
      A1["<b>Choose K</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>statistical &times; stability &times; business</span>"] --> A2["<b>Compare methods</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>cross-method ARI (K-Means / GMM / Ward)</span>"] --> A3["<b>Validate personas</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>3 tiers: features &middot; external &middot; rule-based RFM</span>"] --> A4["<b>Validated segments</b>"]
    end
    subgraph CLVV["Does the CLV model predict?"]
      direction LR
      B1["<b>Temporal holdout</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>calibration vs holdout &middot; 3/6/9-mo</span>"] --> B2["<b>Calibration check</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>predicted vs actual &middot; parameter stability</span>"] --> B3["<b>Validated CLV</b>"]
    end
    A4 --> T(["Segmentation &amp; Profiling"])
    B3 --> T

    classDef stage fill:#f5f7fa,stroke:#1f4e79,stroke-width:2px,color:#1f2937,rx:6,ry:6;
    classDef target fill:#eef2f7,stroke:#9aa7b4,stroke-width:1px,color:#4b5563;
    class A1,A2,A3,A4,B1,B2,B3 stage;
    class T target;
    style SEG fill:#fbfcfd,stroke:#c7d2dc,color:#1f4e79;
    style CLVV fill:#fbfcfd,stroke:#c7d2dc,color:#1f4e79;
```

## The two lanes

| Lane | Steps | Source |
|---|---|---|
| **Are the segments real?** | Choose K (3-leg) → Compare methods (ARI) → Validate personas (3 tiers) | docs 10, 11, 13 |
| **Does the CLV model predict?** | Temporal holdout (3/6/9-mo) → Calibration check | doc 08 |

Method *selection* lives here, not in `two-track-modeling`. The fuller sub-structure (the legs and
tiers) is in docs 10 and 13.
