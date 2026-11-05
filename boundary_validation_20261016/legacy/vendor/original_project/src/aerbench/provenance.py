from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_record_hash(record: Mapping[str, Any], excluded: Iterable[str] = ()) -> str:
    excluded_set = set(excluded)
    normalized: dict[str, Any] = {}
    for key in sorted(record):
        if key in excluded_set:
            continue
        value = record[key]
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = None if not np.isfinite(value) else float(value)
        normalized[str(key)] = value
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return sha256_bytes(payload)


def git_commit_or_unavailable(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable_not_a_git_worktree"


def package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not_installed"


def environment_snapshot(project_root: Path) -> dict[str, Any]:
    thread_names = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
    return {
        "timestamp_utc": utc_now_iso(),
        "operating_system": platform.platform(),
        "processor": platform.processor() or "unavailable",
        "python_version": sys.version,
        "python_executable": sys.executable,
        "packages": {
            name: package_version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "bsm2-python"]
        },
        "thread_settings": {name: os.environ.get(name) for name in thread_names},
        "power_mode": "TODO_record_manually_if_required",
        "git_commit": git_commit_or_unavailable(project_root),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=False, default=str)


def write_dataframe_with_hash(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {"path": str(path), "rows": int(len(frame)), "sha256": sha256_file(path)}


def save_run_provenance(
    *,
    project_root: Path,
    output_dir: Path,
    command: list[str],
    config_files: Iterable[Path],
    output_files: Iterable[Path] = (),
) -> Path:
    output_dir = Path(output_dir)
    config_records = []
    for path in config_files:
        path = Path(path)
        config_records.append({"path": str(path), "sha256": sha256_file(path)})
    outputs = []
    for path in output_files:
        path = Path(path)
        if path.exists():
            outputs.append({"path": str(path), "sha256": sha256_file(path)})
    manifest = {
        "command": command,
        "configuration": config_records,
        "outputs": outputs,
        "environment": environment_snapshot(project_root),
    }
    path = output_dir / "run_manifest.json"
    write_json(path, manifest)
    return path
