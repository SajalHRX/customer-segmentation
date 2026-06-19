# Segments to Actions (detail)

Zoom-in on the **Segmentation & Profiling → Recommendations** stages of `project-architecture.md`.
Consumes the validated customer table (segment + CLV) and turns it into the business deliverable —
closing the loop back to the problem statement (spend follows expected return). Docs 13, 14, 12.

> Rendered with `securityLevel: loose` + `htmlLabels: true` for the bold-title / italic-descriptor styling.

```mermaid
flowchart LR
    A["<b>Customer table</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>segment + predicted CLV</span>"] --> B["<b>Profile segments</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>raw-unit medians &middot; size &middot; revenue &amp; CLV share</span>"] --> C["<b>Name personas</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>from behavioural signature</span>"] --> D["<b>Segment &times; CLV grid</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>differentiated action &middot; prioritised by value-at-stake</span>"] --> E["<b>Recommendations</b><br/><span style='font-size:11px;color:#6b7280;font-style:italic'>table + grid + report</span>"]

    classDef stage fill:#f5f7fa,stroke:#1f4e79,stroke-width:2px,color:#1f2937,rx:6,ry:6;
    class A,B,C,D,E stage;
```

## The flow

| Node | What happens | Doc |
|---|---|---|
| **Customer table** | the joined output: segment + predicted CLV per customer | 12 |
| **Profile segments** | describe each segment in raw units — medians, size, revenue & CLV share | 13 |
| **Name personas** | name from the behavioural signature (not imposed textbook labels) | 13 |
| **Segment × CLV grid** | cross lifecycle × value → one action per cell, ranked by **value-at-stake** | 12, 14 |
| **Recommendations** | the deliverable: final customer table + action grid + written report | 14 |

> **Value-at-stake** = how much revenue + predicted CLV is *on the line* for a segment/cell — the
> number that prioritises spend by money-on-the-line, not headcount (the high-CLV at-risk cell wins).
