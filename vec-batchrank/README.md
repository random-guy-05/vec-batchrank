# VEC BatchRank

**Batch model selection for the Virtual Embryo Challenge.** Score many candidate `.h5ad` predictions with [`veckit`](https://github.com/aristoteleo/veckit), measure seed sensitivity, reproduce the published validation-board aggregate when applicable, identify Pareto-optimal candidates, and produce a decision-ready report.

> Independent community tool. Not an official Virtual Embryo Challenge package and not affiliated with the organizers.

## Why

`veckit` answers: **how does this prediction score against a local target?**

VEC BatchRank answers the next development question: **I have 5–100 checkpoints / methods / hyperparameter settings; which one should I trust?**

This matters especially in the final test phase, where the Challenge rules currently allow only **two official submissions per board for the whole phase**. BatchRank does not touch hidden data; it only uses files you explicitly provide.

## Features

- Batch-discover `.h5ad` candidates from files, directories, or globs.
- Run the canonical `veckit.score(...)` Python API across multiple scorer seeds.
- **Official validation-score reproduction** for currently published boards using the organizers' hyperbolic skill transform, anchors, and metric weights.
- **Candidate-relative consensus ranking** when an official board/anchor set is not appropriate.
- Mean, SD, minimum, maximum, and a configurable **robust score** (`mean - λ·SD`).
- **Pareto frontier** across the official primary metrics.
- Preserves every `veckit` metric in `scores_long.csv`; only official primary metrics enter selection scores.
- Incremental cache so interrupted or expanded tournaments do not rescore unchanged candidate/seed combinations.
- Continues past individual failed candidates by default and records errors.
- CSV, JSON, Markdown, and terminal outputs.

## Install

```bash
pip install -e .
```

`vec-batchrank` depends on `veckit>=0.1.1`.

## Quick start

### T2, arbitrary pseudo-target

```bash
vec-batchrank \
  --task T2 \
  --setting heart \
  --predictions 'runs/*/prediction.h5ad' \
  --target pseudo_target.h5ad \
  --reference preceding_stage.h5ad \
  --seeds 0 1 2 3 4 \
  --out-dir batchrank_out
```

Because no official board was specified, the selection score is a **candidate-relative consensus score**, not a leaderboard score.

### Reproduce a published validation-board aggregate

When you are genuinely scoring against the corresponding validation target/reference:

```bash
vec-batchrank \
  --task T2 \
  --setting heart \
  --board T2:heart:val_interp \
  --predictions 'runs/*.h5ad' \
  --target validation_target.h5ad \
  --reference validation_reference.h5ad \
  --seeds 0 1 2 3 4
```

Current built-in board IDs (snapshot **2026-09-30**):

- `T1:val`
- `T2:embryo:val_interp`
- `T2:heart:val_extrap`
- `T2:heart:val_interp`
- `T3:gata4`

The organizers publish validation anchors at: https://virtualembryo.ai/challenge/evaluation

Hidden test-board anchors are not public and are intentionally not guessed.

## Outputs

`batchrank_out/` contains:

- `scores_long.csv` — one row per candidate × seed; full `veckit` metrics.
- `summary.csv` — metric means/SDs, mean selection score, stability, robust rank, Pareto flag.
- `report.md` — human-readable selection report.
- `run.json` — reproducibility manifest.
- `errors.csv` — any failed candidate/seed runs.
- `.cache/` — incremental scorer cache.

Example selection table:

```text
rank  candidate       mean     SD      robust   pareto
1     model_023.h5ad  68.18    0.19    67.99    yes
2     model_017.h5ad  68.41    0.72    67.69    yes
3     model_011.h5ad  68.15    2.61    65.54
```

The mean winner and robust winner can differ. That is intentional.

## How ranking works

### Official validation mode

With `--board`, BatchRank uses the Challenge's published validation anchors and current metric weights. For each metric, it reproduces the published hyperbolic skill transformation. A floor-model metric value maps to 0.5 skill, the attainable ceiling maps to 1.0, and worse-than-floor values smoothly approach zero. Zero-ideal signed metrics such as T2 `scale_log_ratio` and T3 `severity_slope` are scored on absolute value first.

The task score is then the organizer-specified weighted average × 100.

### Candidate-relative mode

Without `--board`, applying a validation board's anchors to an arbitrary pseudo-target would imply false precision. BatchRank therefore ranks only the supplied candidates. For each official primary metric it converts directionality into "higher is better", forms a 0–1 relative rank, and combines those ranks with the official task weights. The resulting 0–100 number is **only a tournament score among the candidates in that run**.

## Metric scope

The official Challenge scoring pages are authoritative. BatchRank's built-in snapshot uses the primary metrics currently documented there. `veckit` may expose extra constraints and diagnostics; BatchRank keeps those in the raw output but does not silently add them to the official aggregate.

## Important limitations

- Local pseudo-target performance is not a guarantee of hidden-test performance.
- Multi-seed stability measures scorer/subsampling sensitivity, not training-seed uncertainty unless your candidates themselves vary by training seed.
- Published anchors can change if the organizers amend the evaluation. This repository records the snapshot date; check the official evaluation page before a high-stakes final selection.
- Scoring is sequential by design. `veckit` dynamically loads task metric modules and large single-cell matrices can be memory intensive; naive threaded scoring is not a safe default.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check src tests
```

## Sources

- Challenge evaluation/scoring: https://virtualembryo.ai/challenge/evaluation
- Challenge rules: https://virtualembryo.ai/challenge/rules
- Community Contribution Award: https://virtualembryo.ai/challenge/community
- `veckit`: https://github.com/aristoteleo/veckit

## License

MIT.
