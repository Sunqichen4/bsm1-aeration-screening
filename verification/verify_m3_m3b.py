"""Read-only M3/M3B replay. New reports/caches go only to --output.

Python 3.11 with the original requirements-lock.txt dependencies is required.
Example from the delivery root:
  python -X utf8 -B verification/verify_m3_m3b.py --output ../verification_new
This audits saved native trajectories; it does not run new native BSM1 ODEs
or refit scientific models. Run the original study commands in a separate
version directory for a complete scientific rerun.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time

NAMES=('model_revision_20261016','boundary_validation_20261016')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def worker(root,name,out):
    study=(root/name).resolve()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['NUMBA_CACHE_DIR']=str(out/'runtime_cache')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(study))
    import importlib.metadata
    import numpy as np
    import pandas as pd
    import fresh_reference as fr
    import run_study as s
    import analyze_results as analysis

    freeze=json.loads((study/'freeze.json').read_text(encoding='utf-8'))
    def hashes():
        return {entry['path']:hashlib.sha256((study/entry['path']).read_bytes()).hexdigest() for entry in freeze['files']}
    before=hashes()
    if before!={x['path']:x['sha256'] for x in freeze['files']}:
        raise ValueError('Frozen hash mismatch: '+name)
    # The original audit API writes only audit JSONs through this one helper.
    # Redirect those reports, preserving the source code and saved evidence.
    def redirect_audit(path,value,**kwargs):
        relative=Path(path).resolve().relative_to(study)
        if relative.as_posix() not in ('reference_audit.json','numerical/half_step_audit.json'):
            raise ValueError('Unexpected attempted audit output: '+str(relative))
        write(out/relative,value)
    fr._write_json=redirect_audit
    main=fr.audit_reference_outputs(study)
    half=fr.audit_half_step_outputs(study)
    if not main['passed'] or not half['passed']:
        raise ValueError('Raw saved reference audit failed; inspect output JSONs')

    read=lambda path:pd.read_csv(path,float_precision='round_trip')
    candidates=read(study/'new_candidates.csv')
    expected=read(study/'frozen_predictions.csv').sort_values(['candidate_id','model']).reset_index(drop=True)
    models=s.read(study/'outputs/development/models.json')
    gates=s.read(study/'outputs/development/gates.json')
    actual=s.all_predict(candidates,models,finite=True).merge(candidates,on=['candidate_id','scenario'],validate='many_to_one')
    actual['raw_compliant']=actual.predicted_peak<=4.
    actual['buffer_mgN_L']=actual.model.map({k:v['absolute_peak_buffer_mgN_L'] for k,v in gates.items()})
    actual['in_joint_domain']=(actual.load_coordinate.between(0,1)&(actual.oxygen_transfer_factor>=.7+.3*actual.load_coordinate-1e-12)&(actual.oxygen_transfer_factor<=1.))
    actual['in_feature_domain']=(actual.effective_kla>=28.+12.*actual.load_coordinate-1e-12)&(actual.effective_kla<=520.)
    actual['direct_release']=(actual.predicted_peak+actual.buffer_mgN_L<4.)&actual.in_joint_domain&actual.in_feature_domain
    actual['direct_reject']=(actual.predicted_peak-actual.buffer_mgN_L>4.)&actual.in_joint_domain&actual.in_feature_domain
    actual['recheck']=~(actual.direct_release|actual.direct_reject)
    actual=actual.sort_values(['candidate_id','model']).reset_index(drop=True)
    if not actual[['candidate_id','model']].equals(expected[['candidate_id','model']]):
        raise ValueError('Prediction identities differ')
    columns=('predicted_peak','raw_compliant','buffer_mgN_L','in_joint_domain','in_feature_domain','direct_release','direct_reject','recheck')
    equal={c:bool(np.array_equal(actual[c].to_numpy(),expected[c].to_numpy())) for c in columns}
    maxerr=float(np.max(np.abs(actual.predicted_peak-expected.predicted_peak)))
    replay={'passed':all(equal.values()),'prediction_count':len(actual),'maximum_peak_difference':maxerr,'exact_checks':equal}
    write(out/'prediction_replay.json',replay)
    if not replay['passed']:raise ValueError('Frozen prediction or gate replay mismatch')

    joined=analysis.join_evaluation(expected,read(study/'reference_summary.csv'),candidates)
    table_checks={}
    for filename,per_condition,keys in [('metrics_overall.csv',False,['stratum','model']),('metrics_by_condition.csv',True,['stratum','model','scenario'])]:
        calculated=analysis.aggregate_metrics(joined,by_condition=per_condition)
        recorded=read(study/filename)
        left=calculated.set_index(keys).sort_index().sort_index(axis=1)
        right=recorded.set_index(keys).sort_index().sort_index(axis=1)
        pd.testing.assert_frame_equal(left,right,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-10)
        calculated.to_csv(out/filename,index=False)
        table_checks[filename]={'passed':True,'rows':len(calculated),'numeric_tolerance':{'rtol':1e-12,'atol':1e-10}}
    after=hashes()
    report={'passed':before==after,'study':name,'scope':'frozen inputs, saved raw main/half trajectories, exact finite predictions and gate branches, independently reaggregated overall/per-condition metrics',
            'frozen_files':len(before),'frozen_inputs_unchanged':before==after,
            'main_cases':main['successful_count'],'half_cases':half['successful_count'],
            'prediction_replay':replay,'metric_checks':table_checks,
            'native_ODE_executed':False,'models_refit':False,
            'external_historical_worktree_protection':'not checked; package-internal frozen files and native source hashes checked',
            'python':sys.version,'executable':sys.executable,
            'packages':{n:importlib.metadata.version(n) for n in ('numpy','pandas','scipy','numba','bsm2-python')}}
    write(out/'verification_summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['passed'] else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    default=Path(__file__).resolve().parent
    if default.name=='verification':default=default.parent
    parser.add_argument('--package-root',type=Path,default=default)
    parser.add_argument('--output',type=Path,required=True,help='New output directory; must not exist')
    parser.add_argument('--worker',choices=NAMES,help=argparse.SUPPRESS)
    args=parser.parse_args()
    root=args.package_root.resolve()
    out=args.output.resolve()
    if any(out.is_relative_to(root/name) for name in NAMES):
        parser.error('Output must be outside the scientific study directories')
    if args.worker:return worker(root,args.worker,out)
    out.mkdir(parents=True,exist_ok=False)
    results=[]
    for name in NAMES:
        destination=out/name
        destination.mkdir()
        command=[sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--package-root',str(root),'--output',str(destination),'--worker',name]
        start=time.perf_counter()
        completed=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
        (destination/'stdout.log').write_text(completed.stdout,encoding='utf-8')
        (destination/'stderr.log').write_text(completed.stderr,encoding='utf-8')
        item={'study':name,'exit_code':completed.returncode,'elapsed_seconds':time.perf_counter()-start}
        results.append(item)
        print(json.dumps(item,ensure_ascii=False),flush=True)
    summary={'passed':all(x['exit_code']==0 for x in results),'scope':'saved-reference audit and frozen prediction/metric replay; no new native BSM1 runs or scientific refit','results':results}
    write(out/'verification_summary.json',summary)
    return 0 if summary['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
