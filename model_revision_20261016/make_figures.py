"""Reproduce scientific diagnostic figures from recorded results only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
ORDER=["full_calibrated","full_fixed_ratio","no_feedback","poly_ridge","rbf_kernel_ridge","legacy_M2_binary"]
LABELS=["Full calibrated (6)","Full fixed ratio (5)","No feedback (5)","Polynomial ridge","RBF kernel ridge","Historical M2 (binary)"]
COLORS=["#286494","#5B90B4","#CF6B49","#8066A4","#39978C","#C5A33E"]
def style():
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":10,
        "axes.spines.top":False,"axes.spines.right":False,"axes.titlelocation":"left","axes.titlesize":11,
        "axes.labelsize":10,"figure.facecolor":"white","svg.fonttype":"none","pdf.fonttype":42})

def export(fig,name):
    out=ROOT/"figures";out.mkdir(exist_ok=True)
    fig.savefig(out/(name+".png"),dpi=260,bbox_inches="tight")
    fig.savefig(out/(name+".tiff"),dpi=600,bbox_inches="tight",pil_kwargs={"compression":"tiff_lzw"})
    fig.savefig(out/(name+".svg"),bbox_inches="tight")
    fig.savefig(out/(name+".pdf"),bbox_inches="tight")
    plt.close(fig)

def make():
    style()
    dev=ROOT/"outputs/development"
    diag=json.loads((dev/"identifiability.json").read_text(encoding="utf-8"))
    models=json.loads((dev/"models.json").read_text(encoding="utf-8"))
    profile=pd.read_csv(dev/"rho_profile.csv")
    train=pd.read_csv(dev/"development_summary.csv").set_index("model").loc[ORDER]
    fig,axes=plt.subplots(2,2,figsize=(12,8.4),layout="constrained")
    ax=axes[0,0]
    for name,color in zip(ORDER[:3],COLORS[:3]):
        s=np.array(diag[name][1]["singular_values"])
        ax.semilogy(np.arange(1,len(s)+1),s/s[0],"o-",label=LABELS[ORDER.index(name)],color=color)
    ax.axhline(1e-6,color="0.5",ls="--",lw=1,label="Relative rank cutoff")
    ax.set(xlabel="Singular-value index",ylabel="Singular value / largest",title="a  Normalized sensitivity spectrum (development)",xticks=range(1,7))
    ax.legend(fontsize=8,frameon=False)
    ax=axes[0,1]
    bounds=np.array([[2,24],[.2,25],[.2,8],[.1,1.5],[0,.1],[.1,3]])
    labels=[r"$\tau_h$",r"$A$",r"$K_{O,A}$",r"$g$",r"$f_{SN}$",r"$\rho$"]
    for i,name in enumerate(ORDER[:3]):
        values=np.array(models[name]["values"])
        x=(values-bounds[:,0])/np.ptp(bounds,axis=1)
        ax.scatter(x,np.arange(6)+(i-1)*.17,s=36,c=COLORS[i],marker=["o","s","^"][i],label=LABELS[i])
    ax.set(xlim=(-.05,1.05),xlabel="Position within declared search interval",yticks=range(6),yticklabels=labels,title="b  Fitted positions and boundary constraints")
    ax.axvline(0,color="0.7",ls=":");ax.axvline(1,color="0.7",ls=":");ax.invert_yaxis()
    ax=axes[1,0]
    ax.plot(profile.fixed_rho,profile.reoptimized_training_rmse,"o-",c=COLORS[0])
    ax.axvline(2.5/4.8,color=COLORS[1],ls="--",label="Historical normalized ratio")
    ax.scatter(models["full_calibrated"]["values"][5],models["full_calibrated"]["training_rmse"],marker="*",s=130,c=COLORS[2],label="Free-ratio fit")
    ax.set(xlabel=r"Fixed empirical ratio $\rho$",ylabel=r"Reoptimized training RMSE (mg N L$^{-1}$)",title="c  Profile after refitting the other five quantities")
    ax.legend(frameon=False,fontsize=8)
    ax=axes[1,1]
    ax.barh(np.arange(6),train.rmse,color=COLORS,height=.65)
    for i,v in enumerate(train.rmse):ax.text(v+.08,i,f"{v:.3f}",va="center",fontsize=9)
    ax.set(yticks=range(6),yticklabels=LABELS,xlabel=r"Development peak RMSE (mg N L$^{-1}$)",title="d  Target fidelity on the same 750 cases")
    ax.set_xlim(0,float(train.rmse.max())*1.22);ax.invert_yaxis()
    export(fig,"M3_identifiability_and_development")
    (ROOT/"figures/development_plot_data.csv").write_text(train.reset_index().to_csv(index=False),encoding="utf-8")
    if not (ROOT/"metrics_overall.csv").exists():return
    metrics=pd.read_csv(ROOT/"metrics_overall.csv")
    coarse=metrics.loc[metrics.stratum.eq("coarse")].set_index("model").loc[ORDER]
    allm=metrics.loc[metrics.stratum.eq("all")].set_index("model").loc[ORDER]
    joined=pd.read_csv(ROOT/"evaluation_joined.csv")
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout="constrained")
    ax=axes[0,0]
    ax.barh(range(6),coarse.rmse,color=COLORS,height=.65)
    for i,v in enumerate(coarse.rmse):ax.text(v+.06,i,f"{v:.3f}",va="center",fontsize=9)
    ax.set(yticks=range(6),yticklabels=LABELS,xlabel=r"New coarse-grid peak RMSE (mg N L$^{-1}$)",title="a  Concentration error on 42 fixed coarse candidates")
    ax.set_xlim(0,float(coarse.rmse.max())*1.25);ax.invert_yaxis()
    ax=axes[0,1]
    fr=allm.raw_false_releases.to_numpy();fn=allm.raw_false_rejections.to_numpy()
    ax.barh(range(6),fr,color="#CB6148",label="False release")
    ax.barh(range(6),fn,left=fr,color="#8066A4",label="Conservative rejection")
    for i,(a,b) in enumerate(zip(fr,fn)):ax.text(a+b+.25,i,f"{a} / {b}",va="center",fontsize=9)
    ax.set(yticks=range(6),yticklabels=LABELS,xlabel="Raw errors (all frozen candidates)",title="b  Threshold decisions before reference correction")
    ax.set_xlim(0,max(1,float((fr+fn).max()))*1.4);ax.invert_yaxis();ax.legend(frameon=False,fontsize=8)
    ax=axes[1,0]
    chosen=["full_calibrated","rbf_kernel_ridge","legacy_M2_binary"]
    lim=float(max(joined.SNH_max_eval.max(),joined.predicted_peak.max()))*1.04
    for name,marker in zip(chosen,["o","^","s"]):
        g=joined.loc[joined.model.eq(name)]
        i=ORDER.index(name)
        ax.scatter(g.SNH_max_eval,g.predicted_peak,s=18,alpha=.6,marker=marker,c=COLORS[i],label=LABELS[i])
    ax.plot([0,lim],[0,lim],ls="--",color="0.5",lw=1)
    ax.set(xlim=(0,lim),ylim=(0,lim),xlabel=r"Native reference peak (mg N L$^{-1}$)",ylabel=r"Predicted peak (mg N L$^{-1}$)",title="c  Paired predictions; all observed candidates retained")
    ax.legend(frameon=False,fontsize=8)
    ax=axes[1,1]
    count=allm.gate_selected.to_numpy();n=allm.n_candidates.to_numpy()
    ax.barh(range(6),count,color=COLORS,height=.65)
    for i,(v,den,err) in enumerate(zip(count,n,allm.final_errors)):
        ax.text(v+.5,i,f"{v}/{den}; final errors {err}",va="center",fontsize=8)
    ax.set(yticks=range(6),yticklabels=LABELS,xlabel="References requested by empirical concentration gate",title="d  Review cost and remaining final errors")
    ax.set_xlim(0,float(n.max())*1.65);ax.invert_yaxis()
    export(fig,"M3_new_condition_validation")
    metrics.to_csv(ROOT/"figures/validation_plot_data.csv",index=False)

if __name__=="__main__":make()
