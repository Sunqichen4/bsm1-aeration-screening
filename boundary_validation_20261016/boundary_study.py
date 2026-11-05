"""M3B: frozen-model, prospective, fixed-grid boundary validation; no fitting."""
from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.runtime_cache'/'production'))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, hashlib, json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import run_study as prediction_library
import process_model as pm

LOADS = [.13,.31,.43,.61,.73,.91]
QS = [.27,.63,.38,.77,.34,.68]
FIXED_Z = [112.] + [float(z) for z in range(136,197,4)] + [220.]
M3_FREEZE_SHA = 'c5a5aa2e6ea73d2c0b143d1d011529d4d1b4b89dec29fba293dcbf0283f43965'
SOURCE_M3_MANIFEST_SHA = '300aacad5d3833e44af39ffde9105609e3f662cae6123fa2062d6ecd7419dfc3'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n',encoding='utf-8')
def read_csv(path): return pd.read_csv(path, float_precision='round_trip')

def verify_model_origin(root=ROOT):
    if sha(root/'source_M3_manifest.json') != SOURCE_M3_MANIFEST_SHA:
        raise ValueError('Changed M3 origin manifest')
    origin = load(root/'source_M3_manifest.json')
    manifest = {r['path']:r['sha256'] for r in origin['files']}
    for name in ('outputs/development/models.json','outputs/development/gates.json',
                 'process_model.py','statistical_models.py','run_study.py',
                 'legacy/controlled_models.py','legacy/M2_full_parameters.json'):
        if sha(root/name) != manifest[name]:
            raise ValueError('Changed frozen M3 model/source: '+name)
    if sha(root/'source_M3_freeze.json') != M3_FREEZE_SHA:
        raise ValueError('Unexpected M3 source freeze')
    return {'passed':True,'source_M3_freeze_sha256':M3_FREEZE_SHA,
            'model_sha256':sha(root/'outputs/development/models.json'),
            'gates_sha256':sha(root/'outputs/development/gates.json')}

def make_candidates(models):
    rows, selections = [], []
    for i,(L,q) in enumerate(zip(LOADS,QS),1):
        alpha_min=.7+.3*L
        alpha=alpha_min+q*(1.-alpha_min)
        scenario=f'm3b_n{i:02d}_l{L:.2f}_q{q:.2f}'
        def one(z, kind):
            return dict(scenario=scenario,load_coordinate=L,oxygen_transfer_factor=alpha,
                        kla=float(z/alpha),effective_kla=float(z),grid_kind=kind,
                        selection_reason='fixed_effective_kla_grid' if kind=='coarse' else 'legacy_steady_crossing_plus_minus_1')
        local=[one(z,'coarse') for z in FIXED_Z]
        def f(z):
            frame=pd.DataFrame([dict(one(z,'prediction_enriched'),candidate_id='selection_only')])
            return float(prediction_library.legacy_predict(frame,models['legacy_M2_binary'],finite=False)[0])-4.
        flo,fhi=f(112.),f(220.)
        record=dict(scenario=scenario,load_coordinate=L,q=q,oxygen_transfer_factor=alpha,
                    left_z=112.,right_z=220.,left_predicted_minus_4=flo,right_predicted_minus_4=fhi,
                    root_method='brentq_xtol_1e-10_rtol_1e-12',root_z=None,selection_status='no_endpoint_sign_change_skip_enrichment')
        root=None
        if flo==0.: root=112.
        elif fhi==0.: root=220.
        elif flo*fhi<0.: root=float(brentq(f,112.,220.,xtol=1e-10,rtol=1e-12))
        if root is not None:
            record.update(root_z=root,selection_status='one_legacy_steady_crossing_not_unique_or_minimal_claim')
            for z in (max(112.,root-1.),min(220.,root+1.)):
                duplicate=next((r for r in local if abs(r['effective_kla']-z)<=1e-9),None)
                if duplicate is not None:
                    duplicate['selection_reason']+=';also_legacy_enrichment'
                    duplicate['half_step_selected']=True
                else: local.append(dict(one(z,'prediction_enriched'),half_step_selected=True))
        else:
            # Pre-reference fallback: closest legacy prediction to threshold on
            # either predicted side, ties by z. This never uses new reference labels.
            legacy=[(r, f(r['effective_kla'])) for r in local]
            for side in (True,False):
                eligible=[(r,v) for r,v in legacy if (v<=0.)==side]
                if eligible: min(eligible,key=lambda t:(abs(t[1]),t[0]['effective_kla']))[0]['half_step_selected']=True
        for r in local:
            r.setdefault('half_step_selected',False)
            r['candidate_id']=f'm3b_{len(rows)+1:04d}'
            rows.append(r)
        selections.append(record)
    frame=pd.DataFrame(rows)
    return frame, pd.DataFrame(selections)

def apply_gates(fp, gates):
    fp=fp.copy()
    fp['raw_compliant']=fp.predicted_peak<=4.
    fp['buffer_mgN_L']=fp.model.map({k:v['absolute_peak_buffer_mgN_L'] for k,v in gates.items()})
    fp['in_joint_domain']=(fp.load_coordinate.between(0,1)&(fp.oxygen_transfer_factor>=.7+.3*fp.load_coordinate-1e-12)&(fp.oxygen_transfer_factor<=1.))
    fp['in_feature_domain']=(fp.effective_kla>=28.+12.*fp.load_coordinate-1e-12)&(fp.effective_kla<=520.)
    fp['direct_release']=(fp.predicted_peak+fp.buffer_mgN_L<4.)&fp.in_joint_domain&fp.in_feature_domain
    fp['direct_reject']=(fp.predicted_peak-fp.buffer_mgN_L>4.)&fp.in_joint_domain&fp.in_feature_domain
    fp['recheck']=~(fp.direct_release|fp.direct_reject)
    return fp

def initialize_protocol():
    protocol={
      'version':'M3B_boundary_supplement_20261016','created_at_utc':now(),
      'purpose':'Validate already frozen M3 models on a fixed effective-KLa grid near the actual reference threshold; no refitting.',
      'old_evidence_used':'M3 90 reference results informed the z range. They are design information, not unseen confirmation for M3B.',
      'model_policy':'Byte-identical M3 models, gates and prediction sources; six models evaluated on every frozen candidate.',
      'target':'Native BSM1 effluent NH4 maximum on 60 <= t < 67 d, mg N/L; peak <= 4 is compliant.',
      'temperature_C':15.,'tempmodel':False,'main_dt_min':1.,'half_dt_min':.5,
      'warmup_days':60.,'evaluation_days':7.,'main_workers':4,'half_workers':2,
      'conditions':[{'load':l,'q':q} for l,q in zip(LOADS,QS)],
      'alpha_formula':'.7+.3*L+q*(1-(.7+.3*L))','fixed_effective_kla':FIXED_Z,
      'primary_stratum':'108 fixed grid points (coarse); primary comparison independent of enrichment-model selection.',
      'secondary_stratum':'Up to 12 legacy M2 steady crossing c +/- 1 effective-KLa points, separately reported as prediction_enriched.',
      'crossing_rule':'brentq on [112,220], xtol=1e-10 rtol=1e-12 iff finite endpoint signs differ; endpoint zero accepted; no monotonicity/unique/minimal crossing claim.',
      'enrichment_bounds':'Clip c +/- 1 to [112,220]; deduplicate z within 1e-9 per condition. Coincident point retains coarse identity, noted in reason and half-step flag.',
      'no_crossing_rule':'Skip enrichment, preserve condition and endpoint predictions; no reference-guided extension.',
      'half_selection':'All unique c +/- 1 points selected before any new native labels; if no crossing, select each available predicted side nearest 4 on fixed grid, tie lower z.',
      'half_role':'Fresh integration at .5 min of frozen subset; report concentration differences and label flips, keep 1-min main labels.',
      'stopping':'Do not add cases, fit models, change parameters, buffers, threshold, or domains after observing M3B references, regardless of winner or missing sides.',
      'metrics':['RMSE','MAE','bias','raw false releases/rejections','Gate selected/corrected/missed errors','final false releases/rejections','per-condition actual dual-side coverage','nearest compliant/noncompliant margins','counts within 0.25,0.5,1.0 of 4','local observed sign changes'],
      'strata':['coarse','prediction_enriched','all'],'threshold_equality':'<=4 compliant; Gate strict < and > unchanged.',
      'inference_limits':['Six designed conditions are not independent identically distributed plant samples.','L and q are not factorial; no isolated load/alpha effects.','Do not pool M3 and M3B as an unbiased aggregate.','Legacy enrichment is model-informed; no general ranking based solely on this stratum.','Full_fixed_ratio vs no_feedback is matched five-parameter pair; full_calibrated has six.','Process historical biomass prior differs from pure-regressor features.','No new temperature, dynamics, physical identification, or measured energy validation.'],
      'input_precision':'All new scientific CSV reads use pandas float_precision=round_trip.',
      'provenance':'Actual candidates, all six finite predictions, routes, half-step identities and scientific sources frozen before new native integration.'}
    design={'question':'Do frozen M3 models resolve actual threshold sides and what reference cost does the fixed Gate require?',
            'claims_to_test':{'C1':'Each new condition has compliant and noncompliant references on the frozen fixed grid.','C2':'Model and Gate errors near 4 can be measured without post-reference tuning.','C3':'The preselected enriched references have stable labels under timestep halving.'},
            'failure_policy':'Report missing sides, negative model comparisons, gate cost and numerical flips without further tuning.',
            'author_input_needed':[]}
    save(ROOT/'protocol.json',protocol);save(ROOT/'design.json',design)

def freeze():
    if (ROOT/'freeze.json').exists(): raise FileExistsError('Already frozen; refusing overwrite')
    if (ROOT/'reference_provenance.json').exists() or (ROOT/'reference').exists(): raise ValueError('References already exist')
    origin=verify_model_origin()
    initialize_protocol()
    models=load(ROOT/'outputs/development/models.json');gates=load(ROOT/'outputs/development/gates.json')
    frame,selections=make_candidates(models)
    frame.to_csv(ROOT/'new_candidates.csv',index=False)
    frame=read_csv(ROOT/'new_candidates.csv')
    selections.to_csv(ROOT/'boundary_selection.csv',index=False)
    frame.loc[frame.half_step_selected].to_csv(ROOT/'half_step_candidates.csv',index=False)
    # Only historical INPUT coordinates are copied/read here, never their labels.
    historical_paths=[ROOT.parent/'model_revision_20261016/new_candidates.csv',
        ROOT.parent/'program_completion_20261015/portable_repro/data/M1_frozen_candidate_grid.csv',
        ROOT.parent/'program_completion_20261015/portable_repro/data/M2_frozen_candidate_grid.csv',
        ROOT.parent/'program_completion_20261015/portable_repro/extensions/gate_validation/frozen_candidates.csv']
    historical=[]
    for source in historical_paths:
        coords=pd.read_csv(source,float_precision='round_trip',usecols=['candidate_id','load_coordinate','oxygen_transfer_factor','kla','effective_kla'])
        coords['source_input_file']=source.relative_to(ROOT.parent).as_posix()
        historical.append(coords)
    old=pd.concat(historical,ignore_index=True)
    old.to_csv(ROOT/'source_historical_input_coordinates.csv',index=False)
    duplicates=[]
    for r in frame.itertuples():
        matches=old.loc[np.isclose(old.load_coordinate,r.load_coordinate,rtol=0,atol=1e-12)&np.isclose(old.effective_kla,r.effective_kla,rtol=0,atol=1e-9)]
        duplicates.extend(matches.candidate_id.tolist())
    if duplicates or frame.duplicated(['load_coordinate','effective_kla']).any(): raise ValueError('Repeated physical input')
    if not frame.kla.between(40.,520.).all(): raise ValueError('Out-of-domain commanded KLa')
    started=time.perf_counter()
    print('Computing frozen finite predictions for',len(frame),'candidates',flush=True)
    fp=prediction_library.all_predict(frame,models,finite=True)
    # Process finite output already carries raw_compliant; coordinates only here.
    fp=fp.merge(frame,on=['candidate_id','scenario'],validate='many_to_one')
    fp=apply_gates(fp,gates)
    process=fp.loc[fp.model.isin(pm.MODELS)]
    if not np.isfinite(fp.predicted_peak).all() or process.critical_NH4_DO_clips.sum()!=0 or process.steady_finite_abs_difference.max()>.001:
        raise ValueError('Finite predictions failed numerical gate')
    fp.to_csv(ROOT/'frozen_predictions.csv',index=False)
    save(ROOT/'pre_reference_checks.json',dict(origin,n_candidates=len(frame),n_conditions=int(frame.scenario.nunique()),
        primary_count=int(frame.grid_kind.eq('coarse').sum()),enriched_count=int(frame.grid_kind.eq('prediction_enriched').sum()),
        half_count=int(frame.half_step_selected.sum()),old_M3_input_duplicates=duplicates,
        physical_duplicates=0,critical_NH4_DO_clips=int(process.critical_NH4_DO_clips.sum()),
        steady_finite_max_abs_difference=float(process.steady_finite_abs_difference.max()),
        finite_prediction_wall_seconds=time.perf_counter()-started,new_reference_labels_read=False))
    paths=[ROOT/n for n in ('protocol.json','design.json','new_candidates.csv','half_step_candidates.csv','frozen_predictions.csv','boundary_selection.csv','pre_reference_checks.json','source_M3_manifest.json','source_M3_freeze.json','source_historical_input_coordinates.csv','requirements-lock.txt')]
    paths+=list((ROOT/'outputs/development').glob('*.json'))
    paths+=[ROOT/n for n in ('boundary_study.py','process_model.py','statistical_models.py','run_study.py','fresh_reference.py')]
    paths += [p for p in (ROOT/'legacy').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in sorted(set(paths))]
    save(ROOT/'freeze.json',{'version':'M3B','frozen_at_utc':now(),'source_M3_freeze_sha256':M3_FREEZE_SHA,
        'n_candidates':len(frame),'n_new_conditions':int(frame.scenario.nunique()),'new_reference_labels_read':False,
        'models_refit':False,'files':files,'source_hashes':{k:{'path':k+'.json','sha256':sha(ROOT/(k+'.json'))} for k in ('design','protocol')}})
    print('FROZEN',sha(ROOT/'freeze.json'),flush=True)

def replay():
    from fresh_reference import verify_freeze
    verify_freeze(ROOT);verify_model_origin()
    frame=read_csv(ROOT/'new_candidates.csv')
    expected=read_csv(ROOT/'frozen_predictions.csv').sort_values(['model','candidate_id']).reset_index(drop=True)
    actual=prediction_library.all_predict(frame,load(ROOT/'outputs/development/models.json'),finite=True)
    actual=apply_gates(actual.merge(frame,on=['candidate_id','scenario'],validate='many_to_one'),load(ROOT/'outputs/development/gates.json')).sort_values(['model','candidate_id']).reset_index(drop=True)
    checks={c:bool(np.array_equal(expected[c].to_numpy(),actual[c].to_numpy())) for c in ('predicted_peak','raw_compliant','buffer_mgN_L','in_joint_domain','in_feature_domain','direct_release','direct_reject','recheck')}
    result={'passed':all(checks.values()),'n_predictions':len(expected),'maximum_peak_difference':float(np.max(np.abs(expected.predicted_peak-actual.predicted_peak))),'checks':checks}
    save(ROOT/'prediction_replay.json',result)
    print(json.dumps(result),flush=True)
    if not result['passed']:raise ValueError('Prediction replay mismatch')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','main','half','audit','replay'])
    args=parser.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='replay':replay()
    else:
        import fresh_reference as fr
        if args.stage=='main':
            fr.run_frozen_references(ROOT,workers=4)
            result=fr.audit_reference_outputs(ROOT)
        elif args.stage=='half':
            fr.run_half_step_references(ROOT,workers=2)
            result=fr.audit_half_step_outputs(ROOT)
        else:result=fr.audit_reference_outputs(ROOT)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        if not result['passed']:raise RuntimeError('Raw reference audit failed')
if __name__=='__main__':main()
