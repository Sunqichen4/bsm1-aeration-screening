"""Regenerate publication-package inventory after an intentional packaging change."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.runtime_cache', 'runtime_cache', 'runs'}
EXCLUDED_FILES = {'RELEASE_MANIFEST.json'}

def included(path):
    rel = path.relative_to(ROOT)
    return (not any(part in EXCLUDED_DIRS for part in rel.parts)
            and path.name not in EXCLUDED_FILES
            and not path.name.startswith('~$')
            and path.suffix not in {'.pyc', '.pyo', '.tmp', '.bak'})

def main():
    # Inventory counts exclude the inventory and manifest themselves.
    files = sorted((p for p in ROOT.rglob('*') if p.is_file() and included(p)
                    and p.name != 'RELEASE_CONTENTS.csv'), key=lambda p: p.relative_to(ROOT).as_posix())
    groups = {}
    for path in files:
        rel = path.relative_to(ROOT)
        group = rel.parts[0] if len(rel.parts) > 1 else '(root files)'
        record = groups.setdefault(group, {'files': 0, 'bytes': 0})
        record['files'] += 1
        record['bytes'] += path.stat().st_size
    with (ROOT / 'RELEASE_CONTENTS.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'files', 'bytes'])
        writer.writeheader()
        for group, values in sorted(groups.items()):
            writer.writerow({'path': group, **values})
    files.append(ROOT / 'RELEASE_CONTENTS.csv')
    entries = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())]
    manifest = {'release_candidate': '2026.11.04-staging',
                'scope': 'Publication package inventory; internal scientific manifests retain their own scope.',
                'files': entries, 'total_files': len(entries), 'total_bytes': sum(e['bytes'] for e in entries)}
    (ROOT / 'RELEASE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'files': len(entries), 'bytes': manifest['total_bytes']}, indent=2))

if __name__ == '__main__':
    main()
