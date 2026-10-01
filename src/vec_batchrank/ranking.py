from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .specs import BOARD_SPECS, TASK_SPECS, BoardSpec, MetricSpec


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def objective_value(value: float, spec: MetricSpec) -> float:
    """Transform a metric so larger is always better for candidate-relative comparisons."""
    x = float(value)
    if spec.direction == "higher":
        return x
    if spec.direction == "lower":
        return -x
    return -abs(x)


def metric_skill(value, floor: float, ceiling: float, direction: str) -> float:
    """Official VEC hyperbolic per-metric skill in [0, 1].

    Missing/NaN metrics score 0. Two-sided zero-ideal metrics are scored on absolute
    value before applying the lower-is-better transform, matching the published rules.
    """
    if not _finite(value):
        return 0.0
    m = float(value)
    f = float(floor)
    c = float(ceiling)
    if direction == "zero":
        m, f, c = abs(m), abs(f), abs(c)
        direction = "lower"
    if direction == "higher":
        d = c - m
        d_floor = c - f
    elif direction == "lower":
        d = m - c
        d_floor = f - c
    else:
        raise ValueError(f"unknown direction: {direction}")
    if d_floor <= 0:
        raise ValueError(f"invalid anchors: floor={floor}, ceiling={ceiling}, direction={direction}")
    denom = d_floor + d
    if denom <= 0:
        return 1.0
    return float(min(d_floor / denom, 1.0))


def official_score(metrics: Mapping[str, object], board: str | BoardSpec) -> tuple[float, dict[str, float]]:
    """Return official-style validation score and per-metric skills for a published board."""
    b = BOARD_SPECS[board] if isinstance(board, str) else board
    specs = TASK_SPECS[b.task]
    skills: dict[str, float] = {}
    total = 0.0
    for name, spec in specs.items():
        if name not in b.anchors:
            raise KeyError(f"board anchors missing metric {name!r}")
        floor, ceiling = b.anchors[name]
        skill = metric_skill(metrics.get(name), floor, ceiling, spec.direction)
        skills[name] = skill
        total += spec.weight * skill
    return 100.0 * total, skills


def relative_scores(frame: pd.DataFrame, task: str) -> pd.Series:
    """Candidate-relative 0-100 consensus score using official metric weights.

    This is deliberately not presented as a leaderboard score. It ranks only the candidates
    supplied in `frame`, making it usable with arbitrary pseudo-targets when official anchors
    are not appropriate.
    """
    specs = TASK_SPECS[task]
    n = len(frame)
    if n == 0:
        return pd.Series(dtype=float)
    out = pd.Series(0.0, index=frame.index, dtype=float)
    for metric, spec in specs.items():
        if metric in frame.columns:
            vals = pd.to_numeric(frame[metric], errors="coerce")
        else:
            vals = pd.Series(np.nan, index=frame.index, dtype=float)
        obj = vals.map(lambda x: objective_value(x, spec) if _finite(x) else np.nan)
        if n == 1:
            rel = pd.Series(1.0 if obj.notna().iloc[0] else 0.0, index=frame.index)
        else:
            rank = obj.rank(method="average", ascending=False, na_option="bottom")
            rel = 1.0 - (rank - 1.0) / (n - 1.0)
            rel[obj.isna()] = 0.0
        out += spec.weight * rel
    return out * 100.0


def pareto_front(frame: pd.DataFrame, task: str) -> pd.Series:
    """Boolean Pareto frontier over official primary metrics (larger transformed = better)."""
    specs = TASK_SPECS[task]
    names = list(specs)
    mat = []
    for _, row in frame.iterrows():
        vals = []
        for name in names:
            v = row.get(name)
            vals.append(objective_value(v, specs[name]) if _finite(v) else -np.inf)
        mat.append(vals)
    arr = np.asarray(mat, dtype=float)
    frontier = np.ones(len(frame), dtype=bool)
    for i in range(len(frame)):
        for j in range(len(frame)):
            if i == j:
                continue
            if np.all(arr[j] >= arr[i]) and np.any(arr[j] > arr[i]):
                frontier[i] = False
                break
    return pd.Series(frontier, index=frame.index)


def aggregate_runs(long_df: pd.DataFrame, task: str, board: str | None, stability_penalty: float) -> pd.DataFrame:
    """Aggregate candidate×seed rows into candidate summaries and ranks."""
    metric_names = sorted({c for c in long_df.columns if c not in {"candidate", "path", "seed", "score"} and not c.startswith("_")})
    rows = []
    for candidate, g in long_df.groupby("candidate", sort=False):
        row: dict[str, object] = {
            "candidate": candidate,
            "path": g["path"].iloc[0],
            "n_seeds": int(g["seed"].nunique()),
        }
        for metric in metric_names:
            v = pd.to_numeric(g[metric], errors="coerce")
            row[metric] = float(v.mean()) if v.notna().any() else np.nan
            row[f"{metric}__sd"] = float(v.std(ddof=0)) if v.notna().any() else np.nan
        if board:
            scores = []
            for _, r in g.iterrows():
                s, _ = official_score(r.to_dict(), board)
                scores.append(s)
            row["selection_score_mean"] = float(np.mean(scores))
            row["selection_score_sd"] = float(np.std(scores))
            row["selection_score_min"] = float(np.min(scores))
            row["selection_score_max"] = float(np.max(scores))
            row["score_kind"] = "official_validation_reproduction"
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    if not board:
        means = summary.set_index("candidate")
        consensus = relative_scores(means, task)
        summary["selection_score_mean"] = summary["candidate"].map(consensus)
        # Seed-level relative scores are calculated jointly across candidates per seed.
        seed_scores = []
        for seed, sg in long_df.groupby("seed"):
            indexed = sg.set_index("candidate")
            rs = relative_scores(indexed, task)
            for cand, val in rs.items():
                seed_scores.append((cand, seed, float(val)))
        seed_df = pd.DataFrame(seed_scores, columns=["candidate", "seed", "relative_score"])
        stats = seed_df.groupby("candidate")["relative_score"].agg(["mean", "std", "min", "max"]).fillna(0.0)
        summary["selection_score_mean"] = summary["candidate"].map(stats["mean"])
        summary["selection_score_sd"] = summary["candidate"].map(stats["std"])
        summary["selection_score_min"] = summary["candidate"].map(stats["min"])
        summary["selection_score_max"] = summary["candidate"].map(stats["max"])
        summary["score_kind"] = "candidate_relative_consensus"
    summary["robust_score"] = summary["selection_score_mean"] - stability_penalty * summary["selection_score_sd"]
    mean_metrics = summary.set_index("candidate")
    summary["pareto"] = summary["candidate"].map(pareto_front(mean_metrics, task)).fillna(False)
    summary["rank_mean"] = summary["selection_score_mean"].rank(method="min", ascending=False).astype(int)
    summary["rank_robust"] = summary["robust_score"].rank(method="min", ascending=False).astype(int)
    return summary.sort_values(["rank_mean", "candidate"], kind="stable").reset_index(drop=True)
