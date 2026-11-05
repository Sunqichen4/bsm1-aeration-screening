"""Verify the supplied release files using the standard library only."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / 'RELEASE_MANIFEST.json'
    if not path.is_file():
        print('FAIL: RELEASE_MANIFEST.json is missing', file=sys.stderr)
        return 1
    manifest = json.loads(path.read_text(encoding='utf-8'))
    failures = []
    for item in manifest['files']:
        target = (root / item['path']).resolve()
        if not target.is_relative_to(root):
            failures.append({'path': item['path'], 'reason': 'outside repository'})
        elif not target.is_file():
            failures.append({'path': item['path'], 'reason': 'missing'})
        elif target.stat().st_size != item['bytes'] or hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
            failures.append({'path': item['path'], 'reason': 'changed'})
    result = {'status': 'FAIL' if failures else 'PASS', 'files_checked': len(manifest['files']),
              'failures': failures, 'scope': 'Supplied-file integrity only; scientific checks are documented separately.'}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(bool(failures))

if __name__ == '__main__':
    raise SystemExit(main())
