"""Protect the source M3 tree and package the complete cache-free M3B study."""
from __future__ import annotations
import argparse, hashlib, json, os, zipfile
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXCLUDED={'.runtime_cache','__pycache__','.pytest_cache','.git'}
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def verify_originals():
    manifest=json.loads((ROOT/'source_M3_manifest.json').read_text(encoding='utf-8'))
    source=(ROOT/manifest['source']).resolve()
    checks=[]
    for item in manifest['files']:
        path=(source/item['path']).resolve()
        if not path.is_relative_to(source):raise ValueError('Source path escapes M3')
        actual=sha(path) if path.is_file() else None
        checks.append(dict(item,actual_sha256=actual,unchanged=actual==item['sha256']))
    result={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'source':str(source),
            'records_checked':len(checks),'unchanged_count':sum(r['unchanged'] for r in checks),
            'passed':all(r['unchanged'] for r in checks),'records':checks}
    write(ROOT/'source_M3_integrity.json',result)
    if not result['passed']:raise ValueError('Original M3 changed')
    return {k:result[k] for k in ('records_checked','unchanged_count','passed')}
def package(destination):
    destination=Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(ROOT):raise ValueError('Use a new ZIP outside study')
    paths=[]
    for folder,dirs,names in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in EXCLUDED]
        for name in names:
            p=Path(folder)/name
            if name not in {'package_manifest.json','delivery_run.log'} and p.suffix not in {'.pyc','.nbc','.nbi','.tmp'}:paths.append(p)
    paths.sort()
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'excluded':sorted(EXCLUDED),
              'excluded_files':['delivery_run.log (active packaging stdout, receipt saved outside study)'],
              'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths]}
    write(ROOT/'package_manifest.json',manifest)
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in paths+[ROOT/'package_manifest.json']:
            archive.write(p,arcname=ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC failed')
        for record in manifest['files']:
            if hashlib.sha256(archive.read(ROOT.name+'/'+record['path'])).hexdigest()!=record['sha256']:raise ValueError('ZIP SHA failed')
    return {'path':str(destination),'files':len(paths)+1,'bytes':destination.stat().st_size,
            'sha256':sha(destination),'crc_and_content_sha256_verified':True}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path);args=p.parse_args()
    result={'source_M3':verify_originals()}
    if args.archive:result['archive']=package(args.archive)
    write(ROOT.parent/'M3B_交付核验_20261016.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
