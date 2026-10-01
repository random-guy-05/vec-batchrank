# Official scoring snapshot used by VEC BatchRank 1.0.0

Snapshot date: **2026-09-30**.

Authoritative source: https://virtualembryo.ai/challenge/evaluation

BatchRank only uses the metrics currently documented by the organizers as ranking metrics. Extra values emitted by `veckit` are retained in raw results but are not silently inserted into an official aggregate.

## Task weights

### T1

- `de_score`: 25%, higher is better
- `de_direction`: 25%, higher is better
- `mmd_u`: 30%, lower is better
- `variogram`: 20%, lower is better

### T2

Four questions receive 25% each:

- Expression change: `de_score` 50%, `de_direction` 50%
- Cell-state distribution: `mmd_u` 60%, `variogram` 40%
- Tissue shape/growth: `d2_shape`, `occupancy_dice`, `scale_log_ratio` equally weighted
- Local spatial organization: `neighborhood_mmd` 100%

`scale_log_ratio` is zero-ideal and is scored on absolute value.

### T3

- `de_score`: 30%, higher is better
- `de_direction`: 25%, higher is better
- `severity_slope`: 25%, zero is ideal; scored on absolute value
- Cell-state distribution: 20%, split `mmd_u` 60% / `variogram` 40%

The official Task 3 page currently states that tissue shape is not part of the ranking aggregate even though `veckit` can report spatial diagnostics.

## Validation anchors

The actual constants are stored in `src/vec_batchrank/specs.py` and covered by tests asserting that every published floor maps to exactly 50 and every published ceiling maps to exactly 100.

Hidden-test anchors are intentionally absent because the organizers do not publish them.
