from __future__ import annotations

import pandas as pd

from .specs import TASK_SPECS


def _fmt(x) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def markdown_report(
    summary: pd.DataFrame,
    task: str,
    board: str | None,
    seeds: list[int],
    errors: pd.DataFrame,
) -> str:
    lines = [
        "# VEC BatchRank report",
        "",
        f"- Task: **{task}**",
        f"- Seeds: `{', '.join(map(str, seeds))}`",
        (
            "- Ranking mode: **"
            + (
                "official validation-board reproduction"
                if board
                else "candidate-relative consensus"
            )
            + "**"
        ),
    ]

    if board:
        lines.append(f"- Published validation board anchors: **{board}**")
    else:
        lines.append(
            "- No board anchors supplied. Scores are relative only to the "
            "candidates in this run and are **not** leaderboard scores."
        )

    lines.extend(["", "## Selection summary", ""])
    cols = [
        "rank_mean",
        "candidate",
        "selection_score_mean",
        "selection_score_sd",
        "selection_score_min",
        "robust_score",
        "pareto",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    for _, row in summary.iterrows():
        values = [
            (
                str(row[col])
                if col in {"candidate", "pareto", "rank_mean"}
                else _fmt(row[col])
            )
            for col in cols
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Primary metric means", ""])
    metrics = list(TASK_SPECS[task])
    metric_cols = ["candidate"] + metrics
    lines.append("| " + " | ".join(metric_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(metric_cols)) + "|")

    for _, row in summary.iterrows():
        values = [str(row["candidate"])] + [
            _fmt(row.get(metric))
            for metric in metrics
        ]
        lines.append("| " + " | ".join(values) + " |")

    if not errors.empty:
        lines.extend(["", "## Errors", ""])
        for _, row in errors.iterrows():
            lines.append(
                f"- `{row['candidate']}` seed `{row['seed']}`: "
                f"{row['error']}"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "`rank_mean` selects the highest mean selection score. "
                "`robust_score` subtracts one standard deviation by default "
                "(configurable with `--stability-penalty`), so it favors "
                "candidates whose advantage persists across scorer seeds. "
                "`pareto=true` means no other candidate is at least as good "
                "on every official primary metric and strictly better on one."
            ),
            "",
            (
                "BatchRank never accesses hidden challenge data. It only "
                "scores against the target/reference files you explicitly "
                "provide through `veckit`."
            ),
        ]
    )
    return "\n".join(lines) + "\n"
