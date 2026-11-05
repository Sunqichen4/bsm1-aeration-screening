"""Read-only provenance verification and a cache-free delivery archive."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
EXCLUDED_PARTS = {".runtime_cache", "__pycache__", ".pytest_cache", ".git"}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def verify_originals(workspace):
    workspace = Path(workspace).resolve()
    manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["files"]:
        name = entry.get("source", entry.get("protected_source"))
        path = (workspace / name).resolve()
        if not path.is_relative_to(workspace):
            raise ValueError("Source record escapes original workspace")
        actual = sha(path) if path.is_file() else None
        rows.append({"path": name, "expected_sha256": entry["sha256"], "actual_sha256": actual,
                     "unchanged": actual == entry["sha256"]})
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "workspace": str(workspace),
              "records_checked": len(rows), "unchanged_count": sum(x["unchanged"] for x in rows),
              "passed": all(x["unchanged"] for x in rows), "records": rows}
    write(ROOT / "originals_integrity.json", result)
    if not result["passed"]:
        raise RuntimeError("Protected original files differ; inspect originals_integrity.json")
    return {k: result[k] for k in ["records_checked", "unchanged_count", "passed"]}

def package(destination):
    destination = Path(destination).resolve()
    if destination.is_relative_to(ROOT) or destination.exists():
        raise ValueError("Archive must be a new file outside the study directory")
    files = sorted(p for p in ROOT.rglob("*") if p.is_file()
                   and not EXCLUDED_PARTS.intersection(p.relative_to(ROOT).parts)
                   and p.name not in {"package_manifest.json"}
                   and p.suffix not in {".pyc", ".nbc", ".nbi"})
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "excluded": sorted(EXCLUDED_PARTS),
                "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size}
                          for p in files]}
    write(ROOT / "package_manifest.json", manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files + [ROOT / "package_manifest.json"]:
            archive.write(path, arcname=ROOT.name + "/" + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed")
        for record in manifest["files"]:
            data = archive.read(ROOT.name + "/" + record["path"])
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise RuntimeError("Archive content digest failed")
    return {"path": str(destination), "files": len(files) + 1, "bytes": destination.stat().st_size,
            "sha256": sha(destination), "crc_and_content_sha256_verified": True}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-workspace", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    result = {}
    if args.original_workspace:
        result["originals"] = verify_originals(args.original_workspace)
    if args.archive:
        result["archive"] = package(args.archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
