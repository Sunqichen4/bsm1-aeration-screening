"""Reconstructed M1/M2 executable interface, 2026-11-03 v1.

The old portable release is unavailable. Unchanged scientific kernels and
frozen evidence are copied with source hashes. New outputs are never evidence
replacements. See README.md for the limits of reconstructed fitting commands.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse, hashlib, importlib.metadata, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import least_squares, minimize

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'legacy'))
import controlled_models as cm
VERSION='reconstructed-20261103-v1'

def read(name): return json.loads((ROOT/'inputs'/name).read_text(encoding='utf-8'))
def csv(name): return pd.read_csv(ROOT/'inputs'/name,float_precision='round_trip')
def save(out,name,value):
    (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def boolean(s):
    x=pd.Series(s).astype(str).str.lower().str.strip()
    if not x.isin(['true','false','0','1']).all():raise ValueError('Invalid Boolean data')
    return x.isin(['true','1']).to_numpy()
def err(raw,truth):
    raw,truth=np.asarray(raw,bool),np.asarray(truth,bool)
    return {'cases':len(raw),'false_acceptances':int((raw&~truth).sum()),'false_rejections':int((~raw&truth).sum())}
def write_frame(out,name,frame):frame.to_csv(out/name,index=False)
def compare_prediction(a,b):
    d=a.merge(b,on='candidate_id',suffixes=('_new','_frozen'),validate='one_to_one')
    if len(d)!=len(a) or len(d)!=len(b):raise AssertionError('Candidate sets differ')
    keys=['SNH_mean_eval','SNH_max_eval','SNO_mean_eval','DO_mean_eval']
    errors={k:float(np.max(np.abs(d[k+'_new']-d[k+'_frozen']))) for k in keys if k+'_frozen' in d}
    labels=int(np.count_nonzero(boolean(d.SNH_zero_violation_new)!=boolean(d.SNH_zero_violation_frozen)))
    ok=max(errors.values(),default=0.)<1e-9 and labels==0
    return {'cases':len(d),'max_absolute_differences':errors,'label_disagreements':labels,'tolerance':1e-9,'pass':ok}
def m1_logistic(frame,weights=None):
    w=read('M1_logistic_model.json')['weights'] if weights is None else weights
    x=np.column_stack((np.ones(len(frame)),frame.oxygen_transfer_factor*frame.kla/240.,frame.load_coordinate))
    return expit(x@np.asarray(w))
def predict(out,version,model,all_cases=True):
    if version=='M1':
        g=csv('M1_frozen_candidate_grid.csv');p=read('M1_model_parameters.json')
        r=cm.predict_cases(g,p,'full')
        comparison=compare_prediction(r,csv('M1_frozen_surrogate_predictions.csv'))
        probability=m1_logistic(g);f=csv('M1_frozen_logistic_predictions.csv')
        l=g[['candidate_id']].assign(probability=probability,raw_compliant=probability>=.5)
        j=l.merge(f,on='candidate_id',suffixes=('_new','_frozen'),validate='one_to_one')
        comparison['logistic_probability_max_abs_difference']=float(np.max(np.abs(j.probability_new-j.probability_frozen)))
        comparison['logistic_label_disagreements']=int((boolean(j.raw_compliant_new)!=boolean(j.raw_compliant_frozen)).sum())
        comparison['logistic_probability_tolerance']=1e-12
        comparison['pass']=bool(comparison['pass'] and comparison['logistic_probability_max_abs_difference']<comparison['logistic_probability_tolerance'] and comparison['logistic_label_disagreements']==0)
        write_frame(out,'M1_logistic_predictions.csv',l)
    else:
        g=csv('M2_frozen_candidate_grid.csv');p=read('M2_full_parameters.json' if model=='full' else 'M2_ablation_parameters.json')
        r=cm.predict_cases(g,p,model)
        f=csv('M2_frozen_predictions.csv');f=f[(f.model==model)&(f.limit_mgN_L==4.)]
        comparison=compare_prediction(r,f)
    write_frame(out,'predictions.csv',r)
    save(out,'prediction_comparison.json',{'version':version,'model':model,'scientific_integrations':len(r),'new_BSM1_integrations':0,**comparison})
    if not comparison['pass']:raise AssertionError('Prediction comparison failed')
    return comparison
def verify(out):
    manifest=json.loads((ROOT/'copied_source_manifest.json').read_text(encoding='utf-8'))
    failures=[]
    for name,record in manifest['files'].items():
        p=ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:failures.append(name)
    versions={}
    for name in ['numpy','pandas','scipy','numba','bsm2-python']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='not installed'
    result={'status':'PASS' if not failures else 'FAIL','checked_copied_files':len(manifest['files']),'failures':failures,'python':sys.version,'platform':platform.platform(),'packages':versions,'scope':'copied frozen evidence and retained scientific sources; no historical release recovery claimed'}
    save(out,'integrity.json',result)
    if failures:raise AssertionError('Input/source integrity failure')
    return result
def reproduce(out):
    """Regenerate primary screening and threshold-reuse counts from evidence."""
    f=csv('M2_frozen_predictions.csv');refs=csv('M2_reference_244.csv');rows=[]
    for (model,limit),p in f.groupby(['model','limit_mgN_L'],sort=True):
        j=p.merge(refs[['candidate_id','SNH_max_eval']],on='candidate_id',suffixes=('_pred','_ref'),validate='one_to_one')
        truth=j.SNH_max_eval_ref.to_numpy()<=limit
        raw=boolean(j.raw_compliant)
        rows.append({'model':model,'limit_mgN_L':float(limit),'subset':'all_available',**err(raw,truth)})
        c=j.grid_kind.eq('coarse') if 'grid_kind' in j else j.candidate_id.isin(csv('M2_frozen_candidate_grid.csv').query("grid_kind == 'coarse'").candidate_id)
        rows.append({'model':model,'limit_mgN_L':float(limit),'subset':'coarse',**err(raw[c],truth[c])})
    table=pd.DataFrame(rows);write_frame(out,'M2_screening_counts.csv',table)
    expected={'full':(0,0),'no_oxygen_demand_feedback':(10,1),'logistic_basic':(0,1),'logistic_shared_closure':(6,0)}
    checks=[]
    for model,pair in expected.items():
        r=table[(table.model==model)&(table.limit_mgN_L==4)&(table.subset=='all_available')].iloc[0]
        observed=[int(r.false_acceptances),int(r.false_rejections)]
        checks.append({'model':model,'observed':observed,'expected':list(pair),'pass':observed==list(pair)})
    audit=csv('M1_decision_audit.csv');gate=csv('gate_candidate_audit.csv')
    for name,data in [('M1_gate',audit),('new_Gate',gate)]:
        groups=['method']+(['buffer_policy','guard'] if 'buffer_policy' in data else [])
        rr=[]
        for key,g in data.groupby(groups):
            key=key if isinstance(key,tuple) else (key,)
            truth=boolean(g.SNH_zero_violation if 'SNH_zero_violation' in g else g.reference_compliant)
            rr.append({**dict(zip(groups,key)),**err(boolean(g.raw_compliant),truth),'reference_requests':int(boolean(g.recheck).sum()),**{'final_'+k:v for k,v in err(boolean(g.final_compliant),truth).items() if k!='cases'}})
        write_frame(out,name+'_counts.csv',pd.DataFrame(rr))
    result={'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL','checks':checks,'new_BSM1_integrations':0,'scope':'tables recomputed from frozen evidence; no new validation labels or historical timing runs'}
    save(out,'table_reproduction.json',result)
    if result['status']!='PASS':raise AssertionError('Primary count mismatch')
    return result
def route(frame,raw,method,policy='individual'):
    buffers=read('M1_frozen_buffers.json')[policy]
    b=buffers[method] if policy=='individual' else buffers
    result=[]
    for _,g in frame.assign(raw=raw).groupby('scenario',sort=False):
        g=g.sort_values('effective_kla').copy();r=boolean(g.raw)
        false=np.flatnonzero(~r)
        status='bracketed';lo=hi=np.nan
        if len(false)==0:status='no_noncompliant_bracket'
        else:
            last=int(false[-1]);tail=r[last+1:]
            if np.any(r[:last]):status='nonmonotone'
            elif len(tail)<3:status='insufficient_compliant_tail'
            else:lo=float(g.effective_kla.iloc[last]);hi=float(g.effective_kla.iloc[last+1])
        load=g.load_coordinate.to_numpy();alpha=g.oxygen_transfer_factor.to_numpy();k=g.effective_kla.to_numpy()
        domain=(load>=-1e-12)&(load<=1+1e-12)&(alpha>=.7+.3*load-1e-12)&(alpha<=1+1e-12)&(k>=28+12*load-1e-12)&(k<=520+1e-12)
        reject=(~r)&(k<lo-b['rejection_buffer'])&domain&(status=='bracketed')
        release=r&(k>hi+b['release_buffer'])&domain&(status=='bracketed')
        recheck=~(reject|release)
        g['recheck_new']=recheck;g['raw_new']=r
        g['rejection_boundary_new']=lo-b['rejection_buffer'];g['release_boundary_new']=hi+b['release_buffer'];g['bracket_status_new']=status
        result.append(g)
    return pd.concat(result,ignore_index=True)
def gate_audit(out):
    records=[]
    for name,grid,audit in [('M1',csv('M1_frozen_candidate_grid.csv'),csv('M1_decision_audit.csv')),('prospective_Gate',csv('gate_candidate_audit.csv').drop_duplicates('candidate_id'),csv('gate_candidate_audit.csv'))]:
        # Recompute both models; route uses only inputs/predictions/buffers.
        proc=cm.predict_cases(grid,read('M1_model_parameters.json'),'full')
        write_frame(out,name+'_fresh_process_predictions.csv',proc)
        raw_by={'six_state':dict(zip(proc.candidate_id,boolean(proc.SNH_zero_violation))),'logistic':dict(zip(grid.candidate_id,m1_logistic(grid)>=.5))}
        for method in ['six_state','logistic']:
            for policy in (['individual','shared'] if name=='M1' else ['individual']):
                f=audit[audit.method.eq(method)].copy()
                if 'buffer_policy' in f:f=f[f.buffer_policy.eq(policy)&f.guard.eq(20.)]
                pred=np.array([raw_by[method][cid] for cid in grid.candidate_id])
                routed=route(grid,pred,method,policy)
                j=routed[['candidate_id','raw_new','recheck_new','rejection_boundary_new','release_boundary_new','bracket_status_new']].merge(f,on='candidate_id',validate='one_to_one')
                truth=boolean(j.SNH_zero_violation if 'SNH_zero_violation' in j else j.reference_compliant)
                final=np.where(j.recheck_new,truth,j.raw_new)
                raw_diff=int(np.count_nonzero(j.raw_new!=boolean(j.raw_compliant)))
                route_diff=int(np.count_nonzero(j.recheck_new!=boolean(j.recheck)))
                final_diff=int(np.count_nonzero(final!=boolean(j.final_compliant)))
                rec={'study':name,'method':method,'buffer_policy':policy,'cases':len(j),'raw_label_disagreements':raw_diff,'route_disagreements':route_diff,'final_label_disagreements':final_diff,'reference_answer_uses':int(j.recheck_new.sum()),**err(j.raw_new,truth),'final_false_acceptances':int((final&~truth).sum()),'final_false_rejections':int((~final&truth).sum()),'pass':raw_diff==route_diff==final_diff==0}
                records.append(rec);write_frame(out,f'{name}_{method}_{policy}_decisions.csv',j.assign(final_new=final))
    result={'status':'PASS' if all(x['pass'] for x in records) else 'FAIL','new_BSM1_integrations':0,'fresh_process_integrations':228,'checks':records,'scope':'fresh predictions and independently reconstructed monotonicity/hull/buffer routes; frozen reference answers are consumed only for routed points; no raw-trajectory re-integration'}
    save(out,'gate_audit.json',result)
    if result['status']!='PASS':raise AssertionError('Gate routing comparison failed')
    return result
def fit_m2(out):
    d=csv('M1_development_750.csv');models=cm.fit_pair(d,out,read('M2_protocol.json'))
    rows=[]
    for name,m in models.items():
        old=read('M2_full_parameters.json' if name=='full' else 'M2_ablation_parameters.json')
        r={'model':name,'eligible_for_new_predictions':m['eligible_for_new_predictions'],'max_abs_parameter_difference':max(abs(m['parameters'][k]-old['parameters'][k]) for k in cm.FIT_NAMES)}
        fresh=pd.read_csv(out/name/'development_predictions.csv',float_precision='round_trip')
        old_pred=cm.steady_predictions(d,old,name);new_pred=cm.steady_predictions(d,m,name)
        r['development_classification_differences']=int(np.count_nonzero((old_pred<=4)!=(new_pred<=4)));r['max_abs_development_peak_difference']=float(np.max(np.abs(old_pred-new_pred)))
        rows.append(r)
    result={'scope':'retained original M2 kernel and prespecified 3-start fitting protocol; new execution against frozen parameters','comparisons':rows,'pass':all(x['eligible_for_new_predictions'] and x['development_classification_differences']==0 for x in rows)}
    save(out,'fit_comparison.json',result)
    return result
def fit_m1(out):
    # Exact old perturbation generator was not retained. Name this new protocol.
    d,target=cm._development(csv('M1_development_750.csv'));labels=target.astype(float)
    initial=read('historical_initialization.json')['parameters'];p=cm.default_parameters()
    p.update({k:float(v['value']) for k,v in initial.items() if k in p})
    indices=np.array([cm.PARAMETER_NAMES.index(n) for n in cm.FIT_NAMES]);b=np.array([cm.DEFAULT_BOUNDS[n] for n in cm.FIT_NAMES]);lower,span=b[:,0],b[:,1]-b[:,0]
    base=cm.parameter_array(p);z=(base[indices]-lower)/span
    rng=np.random.default_rng(20261013);starts=np.clip(np.vstack([z,z+rng.normal(0,.1,(2,len(z)))]),1e-8,1-1e-8)
    runs=[];fitted=[]
    for i,start in enumerate(starts):
        def residual(theta):
            x=base.copy();x[indices]=lower+span*theta
            return expit((4-cm.steady_predictions(d,x,'full'))/.5)-labels
        result=least_squares(residual,start,bounds=(0,1),method='trf',max_nfev=800,ftol=1e-7,xtol=1e-8,gtol=1e-7,diff_step=1e-6)
        x=base.copy();x[indices]=lower+span*result.x;fitted.append(dict(zip(cm.PARAMETER_NAMES,x)))
        runs.append({'start':i,'success':bool(result.success),'nfev':result.nfev,'loss':float(np.mean(result.fun**2)),'jacobian_rank':int(np.linalg.matrix_rank(result.jac))})
    selected=min(range(3),key=lambda i:runs[i]['loss']);parameters=fitted[selected]
    pred=cm.predict_cases(d,parameters,'full');steady=cm.steady_predictions(d,parameters,'full')
    audit={'max_abs_steady_finite_difference':float(np.max(np.abs(pred.SNH_max_eval-steady))),'label_disagreements':int(np.count_nonzero(boolean(pred.SNH_zero_violation)!=(steady<=4))),'lower_clips':int(pred.lower_clips_total.sum()),'upper_clips':int(pred.upper_clips_total.sum())}
    old=read('M1_model_parameters.json');frozen=cm.steady_predictions(d,old,'full')
    result={'protocol':'reconstructed_M1_historical_initialization_plus_two_normalized_gaussian_0.1_perturbations_v1','historical_perturbation_code_recovered':False,'seed':20261013,'parameters':parameters,'selected_start':selected,'runs':runs,'finite_audit':audit,'max_abs_development_peak_difference_from_frozen':float(np.max(np.abs(steady-frozen))),'development_classification_differences_from_frozen':int(np.count_nonzero((steady<=4)!=(frozen<=4))),'scope':'new explicitly reconstructed initialization protocol, not an exact historical optimizer replay; frozen publication parameters remain authoritative'}
    write_frame(out,'development_predictions.csv',pred);save(out,'parameters.json',result)
    return result
def fit_logistic(out):
    d=csv('M1_development_750.csv').sort_values(['scenario','kla']).reset_index(drop=True);y=boolean(d.target).astype(float)
    x=np.column_stack([np.ones(len(d)),d.effective_kla/240.,d.load_coordinate])
    def objective(w):
        z=x@w;return float(np.sum(np.logaddexp(0,z)-y*z)+.001*np.sum(w[1:]**2))
    def grad(w):
        g=x.T@(expit(x@w)-y);g[1:]+=.002*w[1:];return g
    m=minimize(objective,np.zeros(3),jac=grad,method='L-BFGS-B',bounds=[(None,None),(0,None),(None,0)],options={'maxiter':10000,'ftol':1e-14,'gtol':1e-10})
    records=[{'model':'M1_logistic','weights':m.x.tolist(),'success':bool(m.success),'objective':objective(m.x),'max_abs_probability_difference_from_frozen':float(np.max(np.abs(m1_logistic(d,m.x)-m1_logistic(d)))),'label_disagreements_from_frozen':int(np.count_nonzero((m1_logistic(d,m.x)>=.5)!=(m1_logistic(d)>=.5)))}]
    g=csv('M2_frozen_candidate_grid.csv');f=csv('M2_frozen_predictions.csv')
    for family in ['logistic_basic','logistic_shared_closure']:
        def features(frame):
            z=frame[['load_coordinate','effective_kla']].to_numpy(float)
            if family.endswith('shared_closure'):z=np.column_stack([z,cm.closure_features(frame)[['XH_eq','XA_eq']].to_numpy()])
            return z
        feat=features(d);mean,scale=feat.mean(0),feat.std(0);scale[scale==0]=1
        a=np.column_stack([np.ones(len(d)),(feat-mean)/scale]);ag=np.column_stack([np.ones(len(g)),(features(g)-mean)/scale])
        for limit in [2.,3.,4.,5.,6.]:
            target=(d.SNH_max_eval.to_numpy()<=limit).astype(float)
            rng=np.random.default_rng(20261014);starts=np.vstack([np.zeros(a.shape[1]),rng.uniform(-1,1,(2,a.shape[1]))]);runs=[]
            for start in starts:
                r=least_squares(lambda w:expit(a@w)-target,start,bounds=(-40,40),max_nfev=800,ftol=1e-7,xtol=1e-8,gtol=1e-7,diff_step=1e-6)
                runs.append(r)
            best=min(runs,key=lambda r:float(np.mean(r.fun**2)));prob=expit(ag@best.x)
            old=f[(f.model==family)&(f.limit_mgN_L==limit)];j=g[['candidate_id']].assign(new_probability=prob).merge(old[['candidate_id','probability','raw_compliant']],on='candidate_id')
            records.append({'model':family,'limit_mgN_L':limit,'weights':best.x.tolist(),'means':mean.tolist(),'standard_deviations':scale.tolist(),'success':bool(best.success),'loss':float(np.mean(best.fun**2)),'checked_frozen_cases':len(j),'max_abs_probability_difference_from_frozen':float(np.max(np.abs(j.new_probability-j.probability))),'label_disagreements_from_frozen':int(np.count_nonzero((j.new_probability>=.5)!=boolean(j.raw_compliant)))})
    result={'scope':'M1 documented objective; reconstructed M2 starts (zero plus two uniform [-1,1], seed 20261014) because original logistic source is unavailable; all fits use only 750 development cases, frozen weights never used as starts','models':records,'new_BSM1_integrations':0,'all_classifications_match_frozen':all(r['label_disagreements_from_frozen']==0 for r in records)}
    save(out,'logistic_fits.json',result);return result
def reference_smoke(out):
    from reference_runner import run_reference_case
    case=csv('M2_frozen_candidate_grid.csv').iloc[0].to_dict()
    result=run_reference_case(case,output_dir=out/'reference_cases',freeze_sha256=hashlib.sha256((ROOT/'inputs/M2_frozen_candidate_grid.csv').read_bytes()).hexdigest(),base=ROOT/'legacy/vendor/original_project')
    old=csv('M2_reference_244.csv');old=old[old.candidate_id.eq(case['candidate_id'])]
    ref=old.iloc[0]
    comparison={'candidate_id':case['candidate_id'],'new_BSM1_integrations':1,'SNH_peak_abs_difference':abs(result['SNH_max_eval']-float(ref.SNH_max_eval)),'SNH_mean_abs_difference':abs(result['SNH_mean_eval']-float(ref.SNH_mean_eval)),'classification_matches':result['SNH_zero_violation']==bool(ref.SNH_zero_violation),'scope':'one constant-condition native integration; not all historical raw trajectories'}
    save(out,'reference_comparison.json',comparison);return comparison
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['verify','predict','reproduce','gate-audit','fit-m1','fit-m2','fit-logistic','reference-smoke']);ap.add_argument('--output',required=True);ap.add_argument('--version',choices=['M1','M2'],default='M1');ap.add_argument('--model',choices=list(cm.MODELS),default='full');ap.add_argument('--all-cases',action='store_true');args=ap.parse_args()
    if args.command=='predict' and args.version=='M1' and args.model!='full':
        ap.error('M1 predictions support only --model full; no_oxygen_demand_feedback is an M2 model.')
    out=Path(args.output).resolve()
    if out==ROOT or ROOT/'inputs'==out or ROOT/'inputs' in out.parents or ROOT/'legacy'==out or ROOT/'legacy' in out.parents:raise ValueError('Outputs cannot overwrite package inputs or scientific sources')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output directory must be new or empty')
    out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    dispatch={'verify':verify,'reproduce':reproduce,'gate-audit':gate_audit,'fit-m1':fit_m1,'fit-m2':fit_m2,'fit-logistic':fit_logistic,'reference-smoke':reference_smoke}
    if args.command=='predict':result=predict(out,args.version,args.model,args.all_cases)
    else:result=dispatch[args.command](out)
    save(out,'execution.json',{'implementation':VERSION,'command':sys.argv[1:],'elapsed_seconds':time.perf_counter()-started,'new_outputs_only':True,'success':True});print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
