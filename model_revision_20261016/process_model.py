"""M3: continuous-task calibration with explicit observable parameter groups.

Historical biomass closure is retained. NH4/DO gauge is fixed at kNA=4.8;
the parameters are empirical task quantities, not identified physiology.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import least_squares

NAMES=("tau_h","ammonia_activity","k_oa","oxygen_gain","ss_to_nh","oxygen_per_ammonia")
BOUNDS=np.array(((2.,24.),(.2,25.),(.2,8.),(.1,1.5),(0.,.1),(.1,3.)))
CANONICAL_KNA=4.8
REMOVAL_CAP=40.
FIXED_RATIO=2.5/CANONICAL_KNA
MODELS=("full_calibrated","full_fixed_ratio","no_feedback")
INITIAL=np.array((1200.,75.,110.,6.,8.,2.))
UPPER=np.array((5000.,500.,1200.,80.,80.,8.))

def inputs(frame):
    required=["load_coordinate","oxygen_transfer_factor","kla"]
    if not frame.columns.is_unique:raise ValueError("Duplicate input columns")
    a=frame[required].to_numpy(float)
    if len(a)==0 or not np.isfinite(a).all() or (a[:,0]<0).any() or (a[:,1]<=0).any() or (a[:,1]>1).any() or (a[:,2]<0).any():
        raise ValueError("Invalid operating inputs")
    return np.ascontiguousarray(a)

@njit(cache=True)
def closure(load,alpha,kla):
    cod,nh4,activity=260.*(1.+.2*load),30.*(1.+.35*load),1.+.05*load
    arg=max(-60.,min(60.,.08*(alpha*kla-123.7)))
    ref=.08*(160.-123.7)
    response=(1.+np.exp(-ref))/(1.+np.exp(-arg))
    xh=1200.*(cod/260.)**.5*(1.+.018*(response-1.))
    xa=534.*response*(nh4/30.)**.5*activity
    return cod,nh4,activity,xh,xa

@njit(cache=True)
def positive_root(b,c):
    disc=np.sqrt(b*b+4.*c)
    return 2.*c/(b+disc) if b>=0 else .5*(-b+disc)

@njit(cache=True)
def equilibrium_at_do(so,p,cod,nh4,activity,xh,xa):
    t,A,koa,g,f,rho=p
    ah=.75*so/(.35+so+1e-9)*xh/1000.
    ss=positive_root(30.+1e-9+t*18.*ah-cod,cod*(30.+1e-9))
    rh=ah*ss/(30.+ss+1e-9)
    aa=A*activity*so**2/(koa**2+so**2+1e-9)*xa/100.
    incoming=nh4+f*ss
    snh=positive_root(1.+1e-9+t*aa-incoming,incoming*(1.+1e-9))
    qn=aa*snh/(1.+snh+1e-9)
    if qn>REMOVAL_CAP:
        qn=REMOVAL_CAP
        snh=incoming-t*qn
    return ss,max(0.,snh),rh,qn

@njit(cache=True)
def steady_one(p,case,code):
    load,alpha,kla=case
    cod,nh4,activity,xh,xa=closure(load,alpha,kla)
    t,A,koa,g,f,rho=p
    gain=alpha*kla*g/240.
    if code==1:
        so=(.4/t+gain*8.)/(1./t+gain)
    else:
        left,right=0.,8.
        for _ in range(48):
            so=.5*(left+right)
            ss,snh,rh,qn=equilibrium_at_do(so,p,cod,nh4,activity,xh,xa)
            balance=(.4-so)/t+gain*(8.-so)-.4*rh-rho*qn
            if balance>0:left=so
            else:right=so
        so=.5*(left+right)
    ss,snh,rh,qn=equilibrium_at_do(so,p,cod,nh4,activity,xh,xa)
    sno=2.+t*(3.*qn/CANONICAL_KNA-.25*rh)
    return np.array((xh,ss,xa,sno,snh,so))

@njit(cache=True)
def steady_batch(p,a,code):
    out=np.empty(len(a))
    for i in range(len(a)):out[i]=steady_one(p,a[i],code)[4]
    return out

def parameters(model):
    p=np.asarray(model["values"],float)
    if p.shape!=(6,) or not np.isfinite(p).all() or (p<BOUNDS[:,0]-1e-10).any() or (p>BOUNDS[:,1]+1e-10).any():
        raise ValueError("Invalid M3 parameters")
    if model["kind"] not in MODELS:raise ValueError("Unknown M3 kind")
    if model["kind"]!="full_calibrated" and abs(p[5]-FIXED_RATIO)>1e-12:raise ValueError("Fixed-ratio model changed")
    return p

def predict_steady(frame,model):
    return steady_batch(parameters(model),inputs(frame),int(model["kind"]=="no_feedback"))

@njit(cache=True)
def rhs(x,p,cod,nh4,activity,xheq,xaeq,alpha,kla,code):
    xh,ss,xa,sno,snh,so=x
    t,A,koa,g,f,rho=p
    rh=.75*ss/(30.+ss+1e-9)*so/(.35+so+1e-9)*xh/1000.
    unlimited=A*activity*snh/(1.+snh+1e-9)*so**2/(koa**2+so**2+1e-9)*xa/100.
    qn=min(unlimited,REMOVAL_CAP)
    demand=.4*rh+rho*qn if code==0 else 0.
    return np.array(((xheq-xh)/120.,(cod-ss)/t-18.*rh,(xaeq-xa)/120.,
        (2.-sno)/t+3.*qn/CANONICAL_KNA-.25*rh,(nh4-snh)/t+f*max(ss,0.)/t-qn,
        (.4-so)/t+alpha*g*kla/240.*(8.-so)-demand))

@njit(cache=True)
def simulate(p,case,code,dt_minutes,warmup_days=60.,eval_days=7.):
    load,alpha,kla=case
    cod,nh4,activity,xheq,xaeq=closure(load,alpha,kla)
    dt=dt_minutes/60.
    nw,ne=int(round(warmup_days*1440./dt_minutes)),int(round(eval_days*1440./dt_minutes))
    x=INITIAL.copy()
    low,high=np.full(6,np.inf),np.full(6,-np.inf)
    lower,upper=np.zeros(6,np.int64),np.zeros(6,np.int64)
    raw_min=np.full(6,np.inf)
    sums=np.zeros(6)
    cap_count=0
    for step in range(nw+ne):
        if step>=nw:
            low=np.minimum(low,x);high=np.maximum(high,x);sums+=x
        # Count cap activation at each RK4 stage as a trajectory diagnostic.
        stage=x
        ks=np.empty((4,6))
        for j in range(4):
            if j==1:stage=x+dt*.5*ks[0]
            elif j==2:stage=x+dt*.5*ks[1]
            elif j==3:stage=x+dt*ks[2]
            snh,so=stage[4],stage[5]
            q=p[1]*activity*snh/(1.+snh+1e-9)*so**2/(p[2]**2+so**2+1e-9)*stage[2]/100.
            cap_count+=int(q>REMOVAL_CAP)
            ks[j]=rhs(stage,p,cod,nh4,activity,xheq,xaeq,alpha,kla,code)
        candidate=x+dt/6.*(ks[0]+2.*ks[1]+2.*ks[2]+ks[3])
        for k in range(6):
            raw_min[k]=min(raw_min[k],candidate[k])
            lower[k]+=int(candidate[k]<0.);upper[k]+=int(candidate[k]>UPPER[k])
        x=np.minimum(UPPER,np.maximum(0.,candidate))
    return low,high,sums/ne,lower,upper,raw_min,cap_count

def predict_finite(frame,model,dt_minutes=1.,warmup_days=60.,eval_days=7.):
    if not np.isfinite([dt_minutes,warmup_days,eval_days]).all() or dt_minutes<=0 or warmup_days<0 or eval_days<=0:
        raise ValueError("Invalid integration windows")
    counts=np.array([warmup_days,eval_days])*1440./dt_minutes
    if not np.allclose(counts,np.round(counts),atol=1e-7,rtol=0) or round(counts[1])<1:raise ValueError("Unaligned or empty windows")
    a,p=inputs(frame),parameters(model)
    code=int(model["kind"]=="no_feedback")
    rows=[]
    for i,(_,r) in enumerate(frame.iterrows()):
        started=time.perf_counter()
        lo,hi,avg,lower,upper,raw_min,cap=simulate(p,a[i],code,dt_minutes,warmup_days,eval_days)
        if not np.isfinite(np.r_[lo,hi,avg,raw_min]).all():raise FloatingPointError("Nonfinite M3 integration")
        row={"candidate_id":str(r.get("candidate_id",f"d{i:04d}")),"scenario":str(r.get("scenario","unknown")),
             "model":model["kind"],"predicted_peak":hi[4],"predicted_mean":avg[4],"predicted_do":avg[5],
             "predicted_sno_canonical":avg[3],"raw_compliant":bool(hi[4]<=4.),"dt_minutes":dt_minutes,
             "finite_runtime_seconds":time.perf_counter()-started,"cap_stage_activations":int(cap),
             "lower_clips_total":int(lower.sum()),"upper_clips_total":int(upper.sum()),
             "critical_NH4_DO_clips":int(lower[[4,5]].sum()+upper[[4,5]].sum())}
        for j,n in enumerate(("XH","SS","XA","SNO","SNH","SO")):
            row[f"{n}_lower_clips"]=int(lower[j]);row[f"{n}_upper_clips"]=int(upper[j]);row[f"{n}_raw_min"]=float(raw_min[j])
        rows.append(row)
    return pd.DataFrame(rows)

def fit_model(frame,kind,output_dir,seed=20261016,max_nfev=1000,fixed_rho=None,n_starts=3):
    if kind not in MODELS:raise ValueError(kind)
    if fixed_rho is not None:
        if not np.isfinite(fixed_rho) or not BOUNDS[5,0]<=fixed_rho<=BOUNDS[5,1]:raise ValueError("Invalid fixed rho")
        if kind!="full_calibrated" and abs(fixed_rho-FIXED_RATIO)>1e-12:raise ValueError("Only full_calibrated supports a profile rho")
    if n_starts<1 or max_nfev<1:raise ValueError("Positive starts and evaluation budget required")
    out=Path(output_dir)
    if (out/"model.json").exists():raise FileExistsError("Existing fit must not be overwritten")
    a=inputs(frame)
    y=frame["SNH_max_eval"].to_numpy(float)
    if not np.isfinite(y).all() or (y<0).any():raise ValueError("Invalid targets")
    k=6 if kind=="full_calibrated" and fixed_rho is None else 5
    rng=np.random.default_rng(seed)
    starts=np.vstack((np.full(k,.5),rng.uniform(.05,.95,(n_starts-1,k))))
    lower,span=BOUNDS[:k,0],np.ptp(BOUNDS[:k],axis=1)
    ratio=FIXED_RATIO if fixed_rho is None else float(fixed_rho)
    def expand(z):
        p=np.r_[lower+span*z,ratio] if k==5 else lower+span*z
        return p
    code=int(kind=="no_feedback")
    steady_batch(np.r_[np.mean(BOUNDS[:5],axis=1),ratio],a[:1],code)
    def residual(z):return (steady_batch(expand(z),a,code)-y)/4.
    runs=[];best=None;started=time.perf_counter()
    for i,start in enumerate(starts):
        initial=float(np.mean(residual(start)**2)*16.)
        t0=time.perf_counter()
        opt=least_squares(residual,start,bounds=(0.,1.),max_nfev=max_nfev,method="trf",
            ftol=1e-9,xtol=1e-9,gtol=1e-9,diff_step=1e-5)
        mse=float(np.mean(opt.fun**2)*16.)
        runs.append({"start":i,"initial_mse":initial,"mse":mse,"success":bool(opt.success),"status":int(opt.status),
                     "message":str(opt.message),"nfev":int(opt.nfev),"elapsed_seconds":time.perf_counter()-t0,
                     "normalized_solution":opt.x.tolist(),"values":expand(opt.x).tolist()})
        if best is None or mse<best[0]:best=(mse,expand(opt.x),i)
    model={"kind":kind,"values":best[1].tolist(),"parameters":dict(zip(NAMES,best[1].tolist())),
           "fit_names":list(NAMES[:k]),"effective_dimension":k,"selected_start":best[2],"training_mse":best[0],
           "training_rmse":float(np.sqrt(best[0])),"fit_seconds":time.perf_counter()-started,
           "supervision":"existing_development_continuous_NH4_peaks_only","n_training_cases":len(frame),"fixed_cap":REMOVAL_CAP,
           "canonical_kNA":CANONICAL_KNA,"historical_biomass_prior":True,"runs":runs}
    out.mkdir(parents=True,exist_ok=True)
    (out/"model.json").write_text(json.dumps(model,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    pd.DataFrame(runs).to_csv(out/"optimizer_runs.csv",index=False)
    return model

def jacobian_diagnostics(frame,model):
    a=inputs(frame);p=parameters(model);k=model["effective_dimension"]
    # A fixed-rho profile is still model kind full_calibrated, but k=5.
    output=[]
    for h in (1e-4,1e-5,1e-6):
        jac=np.empty((len(a),k))
        for j in range(k):
            delta=h*np.ptp(BOUNDS[j]);lo=p.copy();hi=p.copy()
            lo[j]=max(BOUNDS[j,0],p[j]-delta);hi[j]=min(BOUNDS[j,1],p[j]+delta)
            jac[:,j]=(steady_batch(hi,a,int(model["kind"]=="no_feedback"))-steady_batch(lo,a,int(model["kind"]=="no_feedback")))/((hi[j]-lo[j])/np.ptp(BOUNDS[j]))
        _,s,v=np.linalg.svd(jac,full_matrices=False)
        normalized=(p[:k]-BOUNDS[:k,0])/np.ptp(BOUNDS[:k],axis=1)
        output.append({"normalized_step":h,"singular_values":s.tolist(),"relative_rank_tolerance":1e-6,
            "relative_rank":int(np.count_nonzero(s>s[0]*1e-6)),"dimension":k,
            "column_norms":dict(zip(NAMES[:k],np.linalg.norm(jac,axis=0).tolist())),
            "weakest_direction":dict(zip(NAMES[:k],v[-1].tolist())),
            "condition_number":float(s[0]/max(s[-1],1e-300)),
            "active_bounds":[NAMES[j] for j in range(k) if normalized[j]<1e-4 or normalized[j]>1-1e-4]})
    return output
