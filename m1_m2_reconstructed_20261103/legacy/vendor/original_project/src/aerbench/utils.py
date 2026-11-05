from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dirs(base: Path) -> None:
    for sub in ["figures", "tables", "timeseries", "manifests", "bsm1_validation"]:
        (base / sub).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def project_root_from_file(file: str) -> Path:
    return Path(file).resolve().parents[1]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
