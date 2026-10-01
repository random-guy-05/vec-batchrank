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
    """Transform a metric so larger is always better."""
    x = float(value)
    if spec.direction == "higher":
        return x
    if spec.direction == "lower":
        return -x
    return -abs(x)


def metric_skill(
    value,
    floor: float,
    ceiling: float,
    direction: str,
) -> float:
    """Official VEC hyperbolic per-metric skill in [0, 1]."""
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
        raise ValueError(
            f"invalid anchors: floor={floor}, ceiling={ceiling}, "
            f"direction={direction}"
        )

    denom = d_floor + d
    if denom <= 0:
        return 1.0
    return float(min(d_floor / denom, 1.0))


def official_score(
    metrics: Mapping[str, object],
    board: str | BoardSpec,
) -> tuple[float, dict[str, float]]:
    """Return official-style validation score and per-metric skills."""
    b = BOARD_SPECS[board] if isinstance(board, str) else board
    specs = TASK_SPECS[b.task]
    skills: dict[str, float] = {}
    total = 0.0

    for name, spec in specs.items():
        if name not in b.anchors:
            raise KeyError(f"board anchors missing metric {name!r}")
        floor, ceiling = b.anchors[name]
        skill = metric_skill(
            metrics.get(name),
            floor,
            ceiling,
            spec.direction,
        )
        skills[name] = skill
        total += spec.weight * skill

    return 100.0 * total, skills


def relative_scores(
    frame: pd.DataFrame,
    task: str,
) -> pd.Series:
    """Candidate-relative 0-100 consensus score using official weights."""
    specs = TASK_SPECS[task]
    n = len(frame)
    if n == 0:
        return pd.Series(dtype=float)

    out = pd.Series(0.0, index=frame.index, dtype=float)
    for metric, spec in specs.items():
        if metric in frame.columns:
            vals = pd.to_numeric(frame[metric], errors="coerce")
        else:
            vals = pd.Series(
                np.nan,
                index=frame.index,
                dtype=float,
            )

        obj = vals.map(
            lambda x, metric_spec=spec: (
                objective_value(x, metric_spec)
                if _finite(x)
                else np.nan
            )
        )

        if n == 1:
            rel = pd.Series(
                1.0 if obj.notna().iloc[0] else 0.0,
                index=frame.index,
            )
        else:
            rank = obj.rank(
                method="average",
                ascending=False,
                na_option="bottom",
            )
            rel = 1.0 - (rank - 1.0) / (n - 1.0)
            rel[obj.isna()] = 0.0

        out += spec.weight * rel

    return out * 100.0


def pareto_front(
    frame: pd.DataFrame,
    task: str,
) -> pd.Series:
    """Boolean Pareto frontier over official primary metrics."""
    specs = TASK_SPECS[task]
    names = list(specs)
    matrix = []

    for _, row in frame.iterrows():
        values = []
        for name in names:
            value = row.get(name)
            values.append(
                objective_value(value, specs[name])
                if _finite(value)
                else -np.inf
            )
        matrix.append(values)

    arr = np.asarray(matrix, dtype=float)
    frontier = np.ones(len(frame), dtype=bool)

    for i in range(len(frame)):
        for j in range(len(frame)):
            if i == j:
                continue
            if (
                np.all(arr[j] >= arr[i])
                and np.any(arr[j] > arr[i])
            ):
                frontier[i] = False
                break

    return pd.Series(frontier, index=frame.index)


def aggregate_runs(
    long_df: pd.DataFrame,
    task: str,
    board: str | None,
    stability_penalty: float,
) -> pd.DataFrame:
    """Aggregate candidate×seed rows into summaries and ranks."""
    excluded = {"candidate", "path", "seed", "score"}
    metric_names = sorted(
        {
            column
            for column in long_df.columns
            if column not in excluded
            and not column.startswith("_")
        }
    )

    rows = []
    for candidate, group in long_df.groupby(
        "candidate",
        sort=False,
    ):
        row: dict[str, object] = {
            "candidate": candidate,
            "path": group["path"].iloc[0],
            "n_seeds": int(group["seed"].nunique()),
        }

        for metric in metric_names:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )
            row[metric] = (
                float(values.mean())
                if values.notna().any()
                else np.nan
            )
            row[f"{metric}__sd"] = (
                float(values.std(ddof=0))
                if values.notna().any()
                else np.nan
            )

        if board:
            scores = []
            for _, result_row in group.iterrows():
                score, _ = official_score(
                    result_row.to_dict(),
                    board,
                )
                scores.append(score)

            row["selection_score_mean"] = float(
                np.mean(scores)
            )
            row["selection_score_sd"] = float(
                np.std(scores)
            )
            row["selection_score_min"] = float(
                np.min(scores)
            )
            row["selection_score_max"] = float(
                np.max(scores)
            )
            row["score_kind"] = (
                "official_validation_reproduction"
            )

        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    if not board:
        means = summary.set_index("candidate")
        consensus = relative_scores(means, task)
        summary["selection_score_mean"] = (
            summary["candidate"].map(consensus)
        )

        seed_scores = []
        for seed, seed_group in long_df.groupby("seed"):
            indexed = seed_group.set_index("candidate")
            relative = relative_scores(indexed, task)
            for candidate, value in relative.items():
                seed_scores.append(
                    (candidate, seed, float(value))
                )

        seed_df = pd.DataFrame(
            seed_scores,
            columns=[
                "candidate",
                "seed",
                "relative_score",
            ],
        )
        stats = (
            seed_df.groupby("candidate")["relative_score"]
            .agg(["mean", "std", "min", "max"])
            .fillna(0.0)
        )
        summary["selection_score_mean"] = (
            summary["candidate"].map(stats["mean"])
        )
        summary["selection_score_sd"] = (
            summary["candidate"].map(stats["std"])
        )
        summary["selection_score_min"] = (
            summary["candidate"].map(stats["min"])
        )
        summary["selection_score_max"] = (
            summary["candidate"].map(stats["max"])
        )
        summary["score_kind"] = (
            "candidate_relative_consensus"
        )

    summary["robust_score"] = (
        summary["selection_score_mean"]
        - stability_penalty
        * summary["selection_score_sd"]
    )

    mean_metrics = summary.set_index("candidate")
    summary["pareto"] = (
        summary["candidate"]
        .map(pareto_front(mean_metrics, task))
        .fillna(False)
    )
    summary["rank_mean"] = (
        summary["selection_score_mean"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    summary["rank_robust"] = (
        summary["robust_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return (
        summary.sort_values(
            ["rank_mean", "candidate"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
