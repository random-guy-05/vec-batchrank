from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path


def discover_predictions(items: list[str]) -> list[Path]:
    found: list[Path] = []
    for item in items:
        p = Path(item).expanduser()
        if p.is_dir():
            found.extend(sorted(p.glob("*.h5ad")))
            continue
        matches = [Path(x) for x in glob.glob(str(p), recursive=True)]
        if matches:
            found.extend(x for x in matches if x.is_file())
        elif p.is_file():
            found.append(p)
    # Deduplicate by resolved path while preserving order.
    seen = set()
    unique = []
    for p in found:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p.resolve())
    return unique


def file_signature(path: Path) -> dict[str, object]:
    st = path.stat()
    return {"path": str(path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def cache_key(config: dict[str, object]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:24]


def safe_name(path: Path) -> str:
    stem = path.stem
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)[:80]


def candidate_labels(paths: list[Path]) -> dict[Path, str]:
    """Return stable, human-readable labels, disambiguating repeated basenames."""
    base_counts = {}
    for p in paths:
        base_counts[p.name] = base_counts.get(p.name, 0) + 1
    labels = {}
    provisional = []
    for p in paths:
        label = p.name if base_counts[p.name] == 1 else f"{p.parent.name}/{p.name}"
        provisional.append((p, label))
    label_counts = {}
    for _, label in provisional:
        label_counts[label] = label_counts.get(label, 0) + 1
    for p, label in provisional:
        labels[p] = label if label_counts[label] == 1 else str(p)
    return labels
