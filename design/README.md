# design/

Architecture and pipeline **diagrams** for the project — the *"how it's wired"*,
complementary to `planning/` (the *"why we chose it"*).

Diagrams are written as **Mermaid inside markdown** so they render natively on GitHub,
stay version-controlled, and diff cleanly. Exported PNGs (rendered via `mmdc`, higher
fidelity) live in `reports/figures/`.

## Diagrams

`project-architecture` is the top-level view; the rest are zoom-ins of its stages.

| File | Shows | Pipeline stage |
|---|---|---|
| `project-architecture.md` | top-level pipeline: Raw → Pre-processing → EDA → Features → Modeling → Validation → Segmentation → Recommendations | overview |
| `data-cleaning.md` | Load & Merge → Drop Invalid Records → Resolve Non-Purchases → Clean Dataset | Pre-processing |
| `feature-engineering.md` | aggregate → Feature EDA → 3-lane fork (Core / CLV inputs / Supporting) | Feature Engineering |
| `two-track-modeling.md` | clustering ‖ CLV, meeting on `Customer ID` (no circularity) | Modeling |
| `validation-flow.md` | two lanes: "are the segments real?" + "does the CLV model predict?" | Validation |
| `segments-to-actions.md` | profile → name personas → segment × CLV grid → recommendations | Segmentation → Recommendations |

> Styling: `securityLevel: loose` + `htmlLabels: true` for the bold-title / italic-descriptor look
> (the exported PNG honours it fully; GitHub's strict mode keeps the bold title but may drop the
> grey/italic descriptor styling).

Optional / not yet built: `experiment-map` (the tracked experiments — observability) and a light
`data-eda`.
