from __future__ import annotations

from pathlib import Path

import pandas as pd

from .specs import TASK_SPECS


def _fmt(x) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def markdown_report(summary: pd.DataFrame, task: str, board: str | None, seeds: list[int], errors: pd.DataFrame) -> str:
    lines = [
        "# VEC BatchRank report",
        "",
        f"- Task: **{task}**",
        f"- Seeds: `{', '.join(map(str, seeds))}`",
        f"- Ranking mode: **{'official validation-board reproduction' if board else 'candidate-relative consensus'}**",
    ]
    if board:
        lines.append(f"- Published validation board anchors: **{board}**")
    else:
        lines.append("- No board anchors supplied. Scores are relative only to the candidates in this run and are **not** leaderboard scores.")
    lines.extend(["", "## Selection summary", ""])
    cols = ["rank_mean", "candidate", "selection_score_mean", "selection_score_sd", "selection_score_min", "robust_score", "pareto"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in summary.iterrows():
        vals = [str(r[c]) if c in {"candidate", "pareto", "rank_mean"} else _fmt(r[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    lines.extend(["", "## Primary metric means", ""])
    metrics = list(TASK_SPECS[task])
    cols2 = ["candidate"] + metrics
    lines.append("| " + " | ".join(cols2) + " |")
    lines.append("|" + "|".join(["---"] * len(cols2)) + "|")
    for _, r in summary.iterrows():
        lines.append("| " + " | ".join([str(r["candidate"])] + [_fmt(r.get(m)) for m in metrics]) + " |")
    if not errors.empty:
        lines.extend(["", "## Errors", ""])
        for _, r in errors.iterrows():
            lines.append(f"- `{r['candidate']}` seed `{r['seed']}`: {r['error']}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`rank_mean` selects the highest mean selection score. `robust_score` subtracts one standard deviation by default (configurable with `--stability-penalty`), so it favors candidates whose advantage persists across scorer seeds. `pareto=true` means no other candidate is at least as good on every official primary metric and strictly better on one.",
            "",
            "BatchRank never accesses hidden challenge data. It only scores against the target/reference files you explicitly provide through `veckit`.",
        ]
    )
    return "\n".join(lines) + "\n"
