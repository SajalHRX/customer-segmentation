# design/

Architecture and pipeline **diagrams** for the project — the *"how it's wired"*,
complementary to `planning/` (the *"why we chose it"*).

Diagrams are written as **Mermaid inside markdown** so they render natively on GitHub,
stay version-controlled, and diff cleanly. Exported images (for the README) live in
`reports/figures/`.

Planned diagrams:

| File | Shows |
|---|---|
| `01_pipeline_dataflow.md` | End-to-end flow: raw → clean → features → (clustering ‖ CLV) → join → segments → recommendations |
| `02_two_track_architecture.md` | Clustering track and CLV track running in parallel, meeting on `Customer ID` (per planning doc 12) |
| `03_repo_architecture.md` | How `notebooks/ ↔ src/ ↔ data/ ↔ reports/` read and write each other |
| `04_validation_flow.md` | Three-leg K-selection framework + the 3-tier persona validation |
