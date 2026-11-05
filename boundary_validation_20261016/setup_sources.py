from pathlib import Path
import os, shutil, hashlib, json

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / 'model_revision_20261016'
if (ROOT/'source_M3_manifest.json').exists():
    raise FileExistsError('Historical bootstrap already completed; do not overwrite the frozen source snapshot.')
manifest = []
for folder, dirs, files in os.walk(SRC):
    dirs[:] = [d for d in dirs if d not in {'.runtime_cache', '__pycache__', '.pytest_cache'}]
    for name in sorted(files):
        p = Path(folder) / name
        manifest.append({'path': p.relative_to(SRC).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
print('Hashed', len(manifest), 'M3 files', flush=True)
(ROOT/'source_M3_manifest.json').write_text(json.dumps({'source':'../model_revision_20261016','files':manifest},ensure_ascii=False,indent=2),encoding='utf-8')
for name in ['process_model.py','statistical_models.py','run_study.py','fresh_reference.py','analyze_results.py','requirements-lock.txt']:
    shutil.copy2(SRC/name,ROOT/name)
shutil.copytree(SRC/'legacy',ROOT/'legacy',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(ROOT/'outputs/development').mkdir(parents=True)
for name in ['models.json','gates.json']:
    shutil.copy2(SRC/'outputs/development'/name,ROOT/'outputs/development'/name)
for name in ['fresh_reference.py','analyze_results.py']:
    p=ROOT/name
    import re
    text=re.sub(r'pd\.read_csv\(([^,\n\)]+)', r'pd.read_csv(\1, float_precision="round_trip"', p.read_text(encoding='utf-8'))
    p.write_text(text,encoding='utf-8',newline='\n')
(ROOT/'source_M3_freeze.json').write_bytes((SRC/'freeze.json').read_bytes())
(ROOT/'tests').mkdir(exist_ok=True)
print('Copied sources', flush=True)
