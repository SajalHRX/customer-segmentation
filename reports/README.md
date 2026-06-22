# reports/

Project outputs for a reader who won't run the code.

- `figures/` — exported plots, **grouped into subfolders**:
  - `figures/design/` — renders of the `design/*.md` Mermaid diagrams (architecture/pipeline).
  - `figures/<phase>/` — analysis figures per pipeline phase (`01_cleaning/`, `02_eda/`, …),
    created on demand by `src.utils.figure_path(phase, name)`.
- the written statistical report (added at the end) — narrative tying methods → results →
  recommendations back to the problem statement.
