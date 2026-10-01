from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from . import __version__
from .io import candidate_labels, discover_predictions
from .ranking import BOARD_SPECS, aggregate_runs, official_score
from .report import markdown_report
from .runner import score_one, veckit_version


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vec-batchrank",
        description="Batch-score and rank Virtual Embryo Challenge candidate predictions with veckit.",
    )
    p.add_argument("--task", required=True, choices=["T1", "T2", "T3"])
    p.add_argument("--predictions", nargs="+", required=True, help="Files, directories, or quoted glob patterns.")
    p.add_argument("--target", required=True, type=Path, help="Local pseudo/validation target supplied to veckit.")
    p.add_argument("--reference", type=Path, help="T1/T2 preceding-stage reference. Strongly recommended for real models.")
    p.add_argument("--wt", type=Path, help="T3 matched wild-type reference.")
    p.add_argument("--setting", choices=["heart", "embryo"], help="T2 setting recorded by veckit; inferred from --board when possible, otherwise heart.")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Scorer seeds; default: 0 1 2 3 4")
    p.add_argument("--board", choices=sorted(BOARD_SPECS), help="Use current published validation anchors to reproduce the official aggregate score.")
    p.add_argument("--allow-reorder", action="store_true", help="Pass veckit's --allow-reorder behavior.")
    p.add_argument("--out-dir", type=Path, default=Path("batchrank_out"))
    p.add_argument("--stability-penalty", type=float, default=1.0, help="Robust score = mean - lambda*SD. Default 1.0.")
    p.add_argument("--no-cache", action="store_true", help="Disable incremental scoring cache.")
    p.add_argument("--fail-fast", action="store_true", help="Stop on first candidate/seed scoring error.")
    p.add_argument("--version", action="version", version=f"vec-batchrank {__version__}")
    return p


def _validate(args) -> None:
    if args.task == "T2" and args.setting is None:
        args.setting = "embryo" if args.board and ":embryo:" in args.board else "heart"
    elif args.setting is None:
        args.setting = "heart"
    if args.board and BOARD_SPECS[args.board].task != args.task:
        raise SystemExit(f"--board {args.board} belongs to {BOARD_SPECS[args.board].task}, not {args.task}")
    if args.task in {"T1", "T2"} and args.reference is None:
        raise SystemExit("T1/T2 model selection requires --reference so DE/change metrics are meaningful.")
    if args.task == "T3" and args.wt is None:
        raise SystemExit("T3 model selection requires --wt so perturbation-response metrics are meaningful.")
    for p in [args.target, args.reference, args.wt]:
        if p is not None and not p.exists():
            raise SystemExit(f"file not found: {p}")


def _console_table(summary: pd.DataFrame) -> Table:
    t = Table(title="VEC BatchRank")
    for col in ["rank_mean", "candidate", "selection_score_mean", "selection_score_sd", "robust_score", "pareto"]:
        t.add_column(col)
    for _, r in summary.iterrows():
        t.add_row(
            str(int(r["rank_mean"])),
            str(r["candidate"]),
            f"{r['selection_score_mean']:.3f}",
            f"{r['selection_score_sd']:.3f}",
            f"{r['robust_score']:.3f}",
            "yes" if bool(r["pareto"]) else "",
        )
    return t


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    _validate(args)
    predictions = discover_predictions(args.predictions)
    if not predictions:
        raise SystemExit("No .h5ad prediction files matched --predictions.")
    labels = candidate_labels(predictions)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = None if args.no_cache else args.out_dir / ".cache"
    rows: list[dict] = []
    errors: list[dict] = []
    console = Console()
    total = len(predictions) * len(args.seeds)
    done = 0
    for candidate in predictions:
        for seed in args.seeds:
            done += 1
            console.print(f"[{done}/{total}] {candidate.name} seed={seed}")
            try:
                payload = score_one(
                    candidate=candidate,
                    task=args.task,
                    target=args.target.resolve(),
                    reference=args.reference.resolve() if args.reference else None,
                    wt=args.wt.resolve() if args.wt else None,
                    setting=args.setting,
                    seed=seed,
                    allow_reorder=args.allow_reorder,
                    cache_dir=cache_dir,
                )
                result = payload["result"]
                row = {"candidate": labels[candidate], "path": str(candidate), "seed": seed}
                row.update(result.get("metrics", {}))
                if args.board:
                    row["score"], _ = official_score(row, args.board)
                rows.append(row)
            except Exception as exc:  # keep long batch jobs useful
                err = {"candidate": labels[candidate], "path": str(candidate), "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
                errors.append(err)
                console.print(f"[red]ERROR[/red] {err['error']}")
                if args.fail_fast:
                    raise
    long_df = pd.DataFrame(rows)
    error_df = pd.DataFrame(errors, columns=["candidate", "path", "seed", "error"])
    if long_df.empty:
        error_df.to_csv(args.out_dir / "errors.csv", index=False)
        raise SystemExit("Every scoring run failed. See errors.csv.")
    summary = aggregate_runs(long_df, args.task, args.board, args.stability_penalty)
    long_df.to_csv(args.out_dir / "scores_long.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    error_df.to_csv(args.out_dir / "errors.csv", index=False)
    report = markdown_report(summary, args.task, args.board, args.seeds, error_df)
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "vec_batchrank_version": __version__,
        "veckit_version": veckit_version(),
        "task": args.task,
        "board": args.board,
        "target": str(args.target.resolve()),
        "reference": str(args.reference.resolve()) if args.reference else None,
        "wt": str(args.wt.resolve()) if args.wt else None,
        "seeds": args.seeds,
        "stability_penalty": args.stability_penalty,
        "candidates": [str(p) for p in predictions],
    }
    (args.out_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    console.print(_console_table(summary))
    console.print(f"\nWrote {args.out_dir / 'report.md'}")
    if errors:
        console.print(f"[yellow]{len(errors)} scoring run(s) failed; see errors.csv.[/yellow]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
