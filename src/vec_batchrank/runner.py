from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from .io import cache_key, file_signature, safe_name


def veckit_version() -> str:
    try:
        return importlib.metadata.version("veckit")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _load_score():
    try:
        from veckit import score
    except ImportError as exc:
        raise RuntimeError(
            "veckit is required to score candidates. Install with `pip install veckit`."
        ) from exc
    return score


def score_one(
    *,
    candidate: Path,
    task: str,
    target: Path,
    reference: Path | None,
    wt: Path | None,
    setting: str,
    seed: int,
    allow_reorder: bool,
    cache_dir: Path | None,
) -> dict:
    cfg = {
        "candidate": file_signature(candidate),
        "target": file_signature(target),
        "reference": file_signature(reference) if reference else None,
        "wt": file_signature(wt) if wt else None,
        "task": task,
        "setting": setting,
        "seed": seed,
        "allow_reorder": allow_reorder,
        "veckit_version": veckit_version(),
    }
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{safe_name(candidate)}.{cache_key(cfg)}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            payload["_cache_hit"] = True
            return payload
    score = _load_score()
    kwargs = {"seed": seed, "allow_reorder": allow_reorder, "setting": setting}
    if task in {"T1", "T2"}:
        kwargs["reference"] = reference
    if task == "T3":
        kwargs["wt"] = wt
    result = score(task=task, input=candidate, target=target, **kwargs)
    payload = {"config": cfg, "result": result, "_cache_hit": False}
    if cache_path is not None:
        cache_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload
