"""Copy frozen artifacts to a Chinese/space path and audit/replay without refitting."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    target=ROOT.parent/'tmp'/'M3B 迁移复核 20261016'
    if target.exists():raise FileExistsError('Use a new portability directory')
    target.mkdir(parents=True)
    freeze=json.loads((ROOT/'freeze.json').read_text(encoding='utf-8'))
    names=[r['path'] for r in freeze['files']]+['freeze.json','reference_provenance.json','reference_execution.json','reference_summary.csv']
    for name in names:
        dst=target/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
    for folder in ('reference','reference_trajectories'):
        shutil.copytree(ROOT/folder,target/folder)
    env=os.environ.copy();env['PYTHONUTF8']='1';env['NUMBA_CACHE_DIR']=str(target/'.runtime_cache'/'production')
    checks=[]
    for stage in ('audit','replay'):
        command=[sys.executable,'-X','utf8','-B',str(target/'boundary_study.py'),stage]
        result=subprocess.run(command,cwd=target,env=env,capture_output=True,text=True,encoding='utf-8')
        (ROOT/f'portability_{stage}.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        checks.append({'stage':stage,'exit_code':result.returncode})
        if result.returncode:raise RuntimeError('Portable '+stage+' failed; inspect logs')
    audit=json.loads((target/'reference_audit.json').read_text(encoding='utf-8'))
    replay=json.loads((target/'prediction_replay.json').read_text(encoding='utf-8'))
    result={'passed':audit['passed'] and replay['passed'],'target_directory':str(target),
            'source_freeze_sha256':hashlib.sha256((ROOT/'freeze.json').read_bytes()).hexdigest(),
            'copied_freeze_sha256':hashlib.sha256((target/'freeze.json').read_bytes()).hexdigest(),
            'main_audit':audit,'prediction_replay':replay,'checks':checks,
            'native_ODE_executed':False,'half_step_scope':'Not included in this migration check; original half-step raw audit reported separately.'}
    (ROOT/'portability_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'maximum_peak_difference':replay['maximum_peak_difference'],'n_predictions':replay['n_predictions']},ensure_ascii=False))
if __name__=='__main__':main()
