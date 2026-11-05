"""Run the versioned development -> freeze -> reference -> audit pipeline."""
from __future__ import annotations
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ.setdefault("NUMBA_CACHE_DIR",str(ROOT/".runtime_cache"/"numba"))
for _k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[_k]="1"
import argparse,hashlib,json,time,importlib.metadata,importlib.util
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import process_model as pm
from statistical_models import fit_baselines,predict_baseline

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,"item") else str(x)),encoding="utf-8")

def scientific_sources():
    names=["process_model.py","statistical_models.py","run_study.py","design.json","protocol.json","data/development_750.csv","legacy/controlled_models.py","legacy/M2_full_parameters.json"]
    return {name:sha(ROOT/name) for name in names}

def verify_development_sources():
    record=read(ROOT/"outputs/development/run_started.json")
    if record.get("scientific_source_hashes")!=scientific_sources():
        raise RuntimeError("Development sources differ from the actual fit-start snapshot; new version/run required")

def legacy_module():
    name="m3_frozen_legacy_controlled"
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/"legacy/controlled_models.py")
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]

def legacy_predict(frame,model,finite=False):
    m=legacy_module()
    if not finite:return m.steady_predictions(frame,model["parameters"],model="full")
    g=m.predict_cases(frame,model["parameters"],model="full",dt_minutes=1.)
    g=g.rename(columns={"SNH_max_eval":"predicted_peak","SNH_mean_eval":"predicted_mean"}).assign(model="legacy_M2_binary")
    return g[["candidate_id","scenario","model","predicted_peak","predicted_mean","lower_clips_total","upper_clips_total"]]

def development():
    out=ROOT/"outputs"/"development"
    if out.exists():raise FileExistsError("Development run already exists; preserve results and use a new version.")
    protocol=read(ROOT/"protocol.json")
    d=pd.read_csv(ROOT/"data"/"development_750.csv")
    if len(d)!=750 or d.scenario.nunique()!=3 or not np.allclose(d.effective_kla,d.kla*d.oxygen_transfer_factor):
        raise ValueError("Unexpected development cases")
    y=d.SNH_max_eval.to_numpy(float)
    if not np.array_equal(y<=4.,d.target.astype(str).str.lower().isin(["true","1"])):raise ValueError("Inconsistent development labels")
    out.mkdir(parents=True)
    save(out/"run_started.json",{"at_utc":now(),"protocol_sha256":sha(ROOT/"protocol.json"),"data_sha256":sha(ROOT/"data/development_750.csv"),
        "scientific_source_hashes":scientific_sources(),
        "python":sys.version,"executable":sys.executable,"packages":{k:importlib.metadata.version(k) for k in ["numpy","pandas","scipy","numba","bsm2-python"]},
        "scope":"Development only. No old 244/96 cases or new reference labels are read."})
    models={};diagnostics={};predictions=[]
    for kind in pm.MODELS:
        print("Fitting",kind,flush=True)
        model=pm.fit_model(d,kind,out/kind,seed=protocol["seed"],max_nfev=protocol["max_nfev"],n_starts=protocol["process_starts"])
        models[kind]=model
        diagnostics[kind]=pm.jacobian_diagnostics(d,model)
        finite=pm.predict_finite(d,model)
        finite["steady_peak"]=pm.predict_steady(d,model)
        finite["reference_peak"]=y
        finite["steady_finite_abs_difference"]=np.abs(finite.predicted_peak-finite.steady_peak)
        finite.to_csv(out/kind/"development_finite.csv",index=False)
        predictions.append(finite)
        print(kind,"steady RMSE",model["training_rmse"],"finite RMSE",np.sqrt(np.mean((finite.predicted_peak-y)**2)),
              "critical clips",finite.critical_NH4_DO_clips.sum(),"equivalence",finite.steady_finite_abs_difference.max(),flush=True)
    stats=fit_baselines(d,out/"statistical",seed=protocol["seed"])
    models.update(stats)
    for name,model in stats.items():
        pred=predict_baseline(d,model)
        predictions.append(pd.DataFrame({"candidate_id":d.candidate_id,"scenario":d.scenario,"model":name,"predicted_peak":pred,"reference_peak":y}))
    legacy={"kind":"legacy_M2_binary","parameters":read(ROOT/"legacy/M2_full_parameters.json")["parameters"],
            "supervision":"historical_750_binary_labels_plus_historical_biomass_prior; contextual reference only"}
    models["legacy_M2_binary"]=legacy
    old=legacy_predict(d,legacy,finite=True)
    old["reference_peak"]=y
    predictions.append(old)
    combined=pd.concat(predictions,ignore_index=True)
    combined.to_csv(out/"development_predictions.csv",index=False)
    gates={}
    for name,g in combined.groupby("model"):
        gates[name]={"absolute_peak_buffer_mgN_L":float(np.max(np.abs(g.predicted_peak-g.reference_peak))),
                     "calibration":"maximum development absolute finite-peak residual; empirical only; no nominal coverage probability",
                     "n_calibration":len(g)}
    save(out/"models.json",models);save(out/"identifiability.json",diagnostics);save(out/"gates.json",gates)
    profile=[]
    for rho in protocol["rho_profile_grid"]:
        fit=pm.fit_model(d,"full_calibrated",out/"rho_profile"/f"rho_{rho:.9f}",seed=protocol["seed"],
                         max_nfev=protocol["max_nfev"],fixed_rho=rho,n_starts=protocol["process_starts"])
        profile.append({"fixed_rho":rho,"reoptimized_training_mse":fit["training_mse"],"reoptimized_training_rmse":fit["training_rmse"],
                        "effective_dimension":5,"fit_seconds":fit["fit_seconds"],"values":json.dumps(fit["values"]),
                        "selected_success":fit["runs"][fit["selected_start"]]["success"]})
        print("Profile rho",rho,"RMSE",fit["training_rmse"],flush=True)
    pd.DataFrame(profile).to_csv(out/"rho_profile.csv",index=False)
    quality=[]
    for name,g in combined.groupby("model"):
        raw=g.predicted_peak<=4.;truth=g.reference_peak<=4.
        quality.append({"model":name,"cases":len(g),"rmse":float(np.sqrt(np.mean((g.predicted_peak-g.reference_peak)**2))),
            "mae":float(np.mean(np.abs(g.predicted_peak-g.reference_peak))),"false_releases":int((raw&~truth).sum()),
            "conservative_rejections":int((~raw&truth).sum()),"gate_buffer":gates[name]["absolute_peak_buffer_mgN_L"]})
    pd.DataFrame(quality).to_csv(out/"development_summary.csv",index=False)
    # Gate numerical suitability before any prospective prediction freeze.
    process_rows=combined.loc[combined.model.isin(pm.MODELS)]
    checks={"no_nonfinite":bool(np.isfinite(combined.predicted_peak).all()),
            "critical_clips_zero":bool((process_rows.critical_NH4_DO_clips==0).all()),
            "steady_finite_max_difference":float(process_rows.steady_finite_abs_difference.max()),
            "selected_optimizer_success":all(m["runs"][m["selected_start"]]["success"] for m in models.values() if "runs" in m)}
    checks["pass"]=bool(checks["no_nonfinite"] and checks["critical_clips_zero"] and checks["selected_optimizer_success"] and checks["steady_finite_max_difference"]<=.001)
    save(out/"development_checks.json",checks)
    verify_development_sources()
    save(out/"completed.json",{"at_utc":now(),"checks":checks,"next":"Freeze predictions before obtaining fresh reference labels."})
    if not checks["pass"]:raise RuntimeError("Development numerical gate failed; records preserved.")

def all_predict(frame,models,finite=False):
    result=[]
    for name,model in models.items():
        if name=="legacy_M2_binary":
            if finite:g=legacy_predict(frame,model,finite=True)
            else:g=frame[["candidate_id","scenario"]].copy().assign(predicted_peak=legacy_predict(frame,model),model=name)
        elif name in pm.MODELS:
            if finite:
                g=pm.predict_finite(frame,model)
                g["steady_peak"]=pm.predict_steady(frame,model)
                g["steady_finite_abs_difference"]=np.abs(g.predicted_peak-g.steady_peak)
            else:g=frame[["candidate_id","scenario"]].copy().assign(predicted_peak=pm.predict_steady(frame,model),model=name)
        else:g=frame[["candidate_id","scenario"]].copy().assign(predicted_peak=predict_baseline(frame,model),model=name)
        result.append(g)
    return pd.concat(result,ignore_index=True)

def freeze():
    if (ROOT/"freeze.json").exists():raise FileExistsError("Prospective freeze already exists")
    p=read(ROOT/"protocol.json");out=ROOT/"outputs/development"
    verify_development_sources()
    if not read(out/"development_checks.json")["pass"]:raise RuntimeError("Development checks failed")
    models=read(out/"models.json");gates=read(out/"gates.json")
    rows=[];dense=[];selection=[]
    for i,spec in enumerate(p["new_conditions"],1):
        l,q=spec["load"],spec["q"];amin=.7+.3*l;a=amin+q*(1.-amin)
        scenario=f"m3_n{i:02d}_l{l:.2f}_q{q:.2f}"
        for k in range(40,521,2):dense.append({"candidate_id":f"p{i:02d}_{k:03d}","scenario":scenario,"load_coordinate":l,"oxygen_transfer_factor":a,"kla":float(k)})
    dense=pd.DataFrame(dense)
    pred=all_predict(dense,models)
    for scenario,g in dense.groupby("scenario",sort=True):
        extras=set();coarse=set(p["fixed_coarse_kla"])
        for name in models:
            if name=="legacy_M2_binary":continue
            z=pred.loc[(pred.scenario==scenario)&(pred.model==name)].merge(g[["candidate_id","kla"]],on="candidate_id")
            safe=z.loc[z.predicted_peak<=4.].sort_values("kla")
            k=float(safe.kla.iloc[0]) if len(safe) else None
            selection.append({"scenario":scenario,"model":name,"first_predicted_safe_dense":k})
            if k is not None:extras.update([max(40.,k-2.),min(520.,k+2.)])
        extras=sorted(extras-coarse)
        if len(extras)>8:extras=[extras[j] for j in np.rint(np.linspace(0,len(extras)-1,8)).astype(int)]
        for k in sorted(coarse|set(extras)):
            source=g.iloc[0]
            rows.append({"candidate_id":f"m3_{len(rows)+1:04d}","scenario":scenario,"load_coordinate":source.load_coordinate,
                         "oxygen_transfer_factor":source.oxygen_transfer_factor,"kla":float(k),"effective_kla":float(k*source.oxygen_transfer_factor),
                         "grid_kind":"coarse" if k in coarse else "prediction_enriched","selection_reason":"fixed_coarse" if k in coarse else "union_prediction_boundary_plus_minus_2"})
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/"new_candidates.csv",index=False)
    pd.DataFrame(selection).to_csv(ROOT/"boundary_selection.csv",index=False)
    fp=all_predict(frame,models,finite=True)
    fp=fp.merge(frame,on=["candidate_id","scenario"],validate="many_to_one")
    if not np.isfinite(fp.predicted_peak).all():raise FloatingPointError("Frozen prediction invalid")
    process=fp.loc[fp.model.isin(pm.MODELS)]
    if process.critical_NH4_DO_clips.sum()!=0 or process.steady_finite_abs_difference.max()>.001:raise RuntimeError("Frozen finite prediction audit failed")
    fp["raw_compliant"]=fp.predicted_peak<=4.
    fp["buffer_mgN_L"]=fp.model.map({k:v["absolute_peak_buffer_mgN_L"] for k,v in gates.items()})
    fp["in_joint_domain"]=(fp.load_coordinate.between(0,1)&(fp.oxygen_transfer_factor>=.7+.3*fp.load_coordinate-1e-12)&(fp.oxygen_transfer_factor<=1))
    fp["in_feature_domain"]=(fp.effective_kla>=28.+12.*fp.load_coordinate-1e-12)&(fp.effective_kla<=520.)
    fp["direct_release"]=(fp.predicted_peak+fp.buffer_mgN_L<4.)&fp.in_joint_domain&fp.in_feature_domain
    fp["direct_reject"]=(fp.predicted_peak-fp.buffer_mgN_L>4.)&fp.in_joint_domain&fp.in_feature_domain
    fp["recheck"]=~(fp.direct_release|fp.direct_reject)
    fp.to_csv(ROOT/"frozen_predictions.csv",index=False)
    fine=[]
    for scenario,g in fp.loc[fp.model.eq("full_calibrated")].groupby("scenario"):
        r=g.assign(distance=np.abs(g.predicted_peak-4.)).sort_values(["distance","kla"]).iloc[0]
        fine.append(frame.loc[frame.candidate_id.eq(r.candidate_id)].iloc[0].to_dict())
    pd.DataFrame(fine).to_csv(ROOT/"half_step_candidates.csv",index=False)
    # Include all scientific executables, source copies, and selected development artifacts.
    paths=[ROOT/"protocol.json",ROOT/"design.json",ROOT/"new_candidates.csv",ROOT/"frozen_predictions.csv",ROOT/"boundary_selection.csv",ROOT/"half_step_candidates.csv",
           ROOT/"data/development_750.csv",out/"models.json",out/"gates.json",out/"development_checks.json"]
    paths.extend(out/x for x in ["run_started.json","completed.json","development_summary.csv","identifiability.json","rho_profile.csv","development_predictions.csv"])
    paths+=sorted(ROOT.glob("*.py"))+sorted((ROOT/"tests").glob("*.py"))+sorted((ROOT/"legacy").rglob("*.py"))
    files=[{"path":x.relative_to(ROOT).as_posix(),"sha256":sha(x)} for x in sorted(set(paths))]
    save(ROOT/"freeze.json",{"version":"M3","frozen_at_utc":now(),"n_new_conditions":frame.scenario.nunique(),"n_candidates":len(frame),
         "new_reference_labels_read":False,"development_training_completed_before_freeze":True,"selection_uses":"model predictions only",
         "files":files,"source_hashes":{k:{"path":f"{k}.json","sha256":sha(ROOT/f"{k}.json")} for k in ["design","protocol"]}})
    print("Frozen",len(frame),"new candidates across",frame.scenario.nunique(),"conditions",flush=True)

def references(workers=4):
    from fresh_reference import run_frozen_references,audit_reference_outputs
    run_frozen_references(ROOT,workers=workers)
    print(audit_reference_outputs(ROOT),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["fit","freeze","references"]);ap.add_argument("--workers",type=int,default=4);args=ap.parse_args()
    if args.stage=="fit":development()
    elif args.stage=="freeze":freeze()
    else:references(args.workers)
if __name__=="__main__":main()
