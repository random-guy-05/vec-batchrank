# Community Contribution submission text

## Title
VEC BatchRank — batch candidate selection and stability ranking for veckit

## Short description
VEC BatchRank turns the existing `veckit` local scorer into a multi-candidate model-selection workflow. It batch-scores directories/globs of `.h5ad` predictions across multiple scorer seeds, reports metric-level means and variability, identifies Pareto-optimal candidates, distinguishes highest-mean from stability-penalized selections, caches completed runs, and exports CSV/JSON/Markdown reports. When a published validation board is specified, it reproduces the current official aggregate using the Challenge's published anchors, hyperbolic skill transform, and metric weights; otherwise it uses an explicitly labeled candidate-relative consensus score rather than pretending an arbitrary pseudo-target has a leaderboard score.

## Contribution / community value
The canonical `veckit` tool answers how one prediction scores against a local target. During model development, competitors typically have many checkpoints, seeds, hyperparameter settings, or methods and need to decide which candidate to trust. VEC BatchRank supplies that missing selection layer without reimplementing VEC metrics or accessing hidden data. This is especially useful for the final phase, where official test attempts are scarce. The repository is open, installable, tested, and uses only user-supplied local files.

## Suggested tags
VEC; tooling; model selection; reproducibility; veckit; final-submission workflow
