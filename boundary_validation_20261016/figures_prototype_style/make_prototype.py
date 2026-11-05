from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(r'D:\论文\1\boundary_validation_20261016')))
import make_boundary_figures as base

ROOT=Path(r'D:\论文\1\boundary_validation_20261016')
out=Path(r'D:\论文\1\boundary_validation_20261016\figures_prototype_style')
joined, truth, overall, near, summary=base.load_checked(ROOT)
MODELS=base.MODELS; LABELS=base.LABELS; COLORS=base.COLORS; STYLES=base.STYLES
# clean journal-like style with restrained panel spines and a safe-zone tint.
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':11,
    'axes.labelsize':11.5,'axes.titlesize':12,'xtick.labelsize':9.5,'ytick.labelsize':9.5,
    'legend.fontsize':9.5,'figure.facecolor':'#F7F6F2','axes.facecolor':'#FFFFFF',
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#4A4A4A','axes.linewidth':0.75,
    'grid.color':'#D4D8DB','grid.linewidth':0.55,'grid.alpha':0.7,'savefig.facecolor':'#F7F6F2',
    'pdf.fonttype':42,'svg.fonttype':'none'})

def panel_label(ax,letter):
    ax.text(-.115,1.055,letter,transform=ax.transAxes,fontsize=15,fontweight='bold',va='top',ha='left')
    ax.grid(axis='y',zorder=0); ax.set_axisbelow(True)

def response():
    fig,axes=plt.subplots(2,3,figsize=(13.3,8.15),sharex=True,sharey=True)
    ymax=max(5.,float(np.ceil(max(joined.predicted_peak.max(),truth.SNH_max_eval.max())/5.)*5.))
    for i,(scenario,ax) in enumerate(zip(sorted(truth.scenario.unique()),axes.flat)):
        local=joined.loc[joined.scenario.eq(scenario)]; true=truth.loc[truth.scenario.eq(scenario)].sort_values('effective_kla')
        panel_label(ax,'abcdef'[i])
        inset=ax.inset_axes([.53,.56,.43,.35]); inset.set_facecolor('#FBFBFA')
        for name in MODELS:
            group=local.loc[local.model.eq(name)].sort_values('effective_kla')
            ax.plot(group.effective_kla,group.predicted_peak,color=COLORS[name],linestyle=STYLES[name],lw=1.75,alpha=.96,zorder=2)
            inset.plot(group.effective_kla,group.predicted_peak,color=COLORS[name],linestyle=STYLES[name],lw=.85,alpha=.88,zorder=2)
        coarse=true.loc[true.grid_kind.eq('coarse')]; enrich=true.loc[true.grid_kind.eq('prediction_enriched')]
        # lightly tint non-compliant region; threshold remains explicit and dominant.
        ax.axhspan(4.,12.,color='#F4C8BC',alpha=.15,zorder=0)
        ax.axhline(4.,color='#A24C43',lw=1.0,linestyle=(0,(3,2)),zorder=1)
        inset.axhspan(4.,ymax*1.035,color='#F4C8BC',alpha=.11,zorder=0)
        inset.axhline(4.,color='#A24C43',lw=.65,linestyle=(0,(3,2)),zorder=1)
        ax.scatter(coarse.effective_kla,coarse.SNH_max_eval,marker='o',s=36,c='#20252B',edgecolors='white',linewidths=.45,zorder=5)
        ax.scatter(enrich.effective_kla,enrich.SNH_max_eval,marker='D',s=49,facecolors='white',edgecolors='#9A2F75',linewidths=1.45,zorder=6)
        inset.scatter(coarse.effective_kla,coarse.SNH_max_eval,marker='o',s=7,c='#20252B',zorder=4)
        inset.scatter(enrich.effective_kla,enrich.SNH_max_eval,marker='D',s=10,facecolors='white',edgecolors='#9A2F75',linewidths=.65,zorder=5)
        inset.set_xlim(float(truth.effective_kla.min())-3,float(truth.effective_kla.max())+3); inset.set_ylim(0,ymax*1.035)
        inset.set_xticks([112,220]); inset.set_yticks([0,ymax]); inset.tick_params(axis='both',labelsize=7.5,pad=1,length=2)
        inset.set_title('Full sampled range',fontsize=8.3,pad=1.5); inset.spines[:].set_color('#B4B9BD'); inset.spines[:].set_linewidth(.55)
        L=float(true.load_coordinate.iloc[0]); alpha=float(true.oxygen_transfer_factor.iloc[0])
        ax.set_title(f'L = {L:.2f}, α = {alpha:.4f}',pad=8)
        ax.set_xlim(136.,196.); ax.set_ylim(0.,12.); ax.set_xticks([140,160,180]);
        if i//3==1: ax.set_xlabel('Effective KLa, αKLa (d⁻¹)')
        if i%3==0: ax.set_ylabel('Effluent NH₄ peak (mg N/L)')
        safe=int((true.SNH_max_eval<=4.).sum()); total=len(true)
        ax.text(.025,.965,f'{safe}/{total} compliant',transform=ax.transAxes,ha='left',va='top',fontsize=9.2,
            bbox=dict(facecolor='white',edgecolor='#D9D9D9',alpha=.9,pad=2.0))
    handles=[Line2D([0],[0],color=COLORS[name],linestyle=STYLES[name],lw=2.,label=LABELS[name]) for name in MODELS]
    handles += [Line2D([0],[0],marker='o',linestyle='none',color='#20252B',markersize=5.5,label='BSM1: fixed grid'),
                Line2D([0],[0],marker='D',linestyle='none',markerfacecolor='white',color='#9A2F75',markersize=5.5,label='BSM1: enriched'),
                Line2D([0],[0],color='#A24C43',linestyle=(0,(3,2)),lw=1.,label='Compliance limit: 4 mg N/L')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.027),ncol=3,frameon=False,handlelength=2.6,columnspacing=2.2,fontsize=9.2)
    fig.text(.5,.007,'Main axes: αKLa 136–196 d⁻¹ and peak 0–12 mg N/L; inset: all frozen candidates. Lines connect frozen predictions only.',ha='center',fontsize=8.5)
    fig.subplots_adjust(left=.065,right=.99,top=.94,bottom=.19,wspace=.16,hspace=.33)
    fig.savefig(out/'M3B_threshold_response_prototype.png',dpi=260,bbox_inches='tight',pad_inches=.1)
    fig.savefig(out/'M3B_threshold_response_prototype.pdf',dpi=300,bbox_inches='tight',pad_inches=.1); plt.close(fig)

def performance():
    primary=overall.loc[overall.stratum.eq('coarse')].set_index('model').loc[list(MODELS)].reset_index()
    local=near.loc[near.stratum.eq('coarse') & near.window_mgN_L.eq(.5)].set_index('model').loc[list(MODELS)].reset_index()
    n=int(primary.n_candidates.iloc[0]); nnear=int(local.n_candidates.iloc[0]); x=np.arange(len(MODELS))
    fig,axes=plt.subplots(2,2,figsize=(11.8,8.15))
    for i,ax in enumerate(axes.flat): panel_label(ax,'abcd'[i]); ax.set_xticks(x,base.SHORT); ax.margins(x=.06)
    ax=axes[0,0]; bars=ax.bar(x,primary.rmse,color=[COLORS[m] for m in MODELS],edgecolor='white',linewidth=.7,width=.68); ax.bar_label(bars,labels=[f'{v:.2f}' for v in primary.rmse],padding=2.5,fontsize=9.5); ax.set_ylim(0,max(float(primary.rmse.max())*1.2,.5)); ax.set_ylabel('Peak RMSE (mg N/L)'); ax.set_title(f'Fixed-grid concentration error (n = {n})',pad=8)
    ax=axes[0,1]; a=ax.bar(x-.18,primary.raw_false_releases,width=.36,color='#BC4C48',label='False release',edgecolor='white'); b=ax.bar(x+.18,primary.raw_false_rejections,width=.36,color='#447EAB',label='False rejection',edgecolor='white'); ax.bar_label(a,labels=['0']*len(a),padding=2,fontsize=9); ax.bar_label(b,padding=2,fontsize=9); ax.scatter(x-.18,np.zeros_like(x),s=32,facecolors='none',edgecolors='#BC4C48',linewidths=1.1,zorder=4); ax.set_ylim(0,max(float(max(primary.raw_false_releases.max(),primary.raw_false_rejections.max()))*1.25,2.)); ax.set_ylabel('Raw decision error (count)'); ax.set_title('Fixed-grid threshold decisions',pad=8); ax.legend(frameon=False,loc='upper right',fontsize=9)
    ax=axes[1,0]; bars=ax.bar(x,100.*primary.gate_selected_fraction,color=[COLORS[m] for m in MODELS],edgecolor='white',width=.68); ax.bar_label(bars,labels=[f'{int(r.gate_selected)}/{n}' for r in primary.itertuples()],padding=2.5,fontsize=9.2); ax.set_ylim(0,114); ax.set_yticks([0,25,50,75,100]); ax.set_ylabel('Gate reference fraction (%)'); ax.set_title('Fixed-grid reference cost',pad=8); ax.text(.5,.975,'Final errors = 0 for all models',transform=ax.transAxes,ha='center',va='top',fontsize=8.5,color='#52565A')
    ax=axes[1,1]; finite=local.rmse.notna(); bars=ax.bar(x[finite],local.loc[finite,'rmse'],color=[COLORS[MODELS[i]] for i in np.flatnonzero(finite)],edgecolor='white',width=.68); ax.bar_label(bars,labels=[f'{v:.2f}' for v in local.loc[finite,'rmse']],padding=2.5,fontsize=9.5); ax.set_ylim(0,max(float(local.rmse.max())*1.2,.5) if finite.any() else 1.); ax.set_ylabel('Peak RMSE (mg N/L)'); ax.set_title(f'Near threshold: |BSM1 − 4| ≤ 0.5 (n = {nnear})',pad=8)
    fig.text(.5,.015,'Preselected fixed grid in all panels. Historical M2 used binary supervision; other models used continuous targets.',ha='center',fontsize=8.5)
    fig.subplots_adjust(left=.075,right=.99,top=.94,bottom=.11,wspace=.24,hspace=.42)
    fig.savefig(out/'M3B_fixed_grid_performance_prototype.png',dpi=260,bbox_inches='tight',pad_inches=.1); fig.savefig(out/'M3B_fixed_grid_performance_prototype.pdf',dpi=300,bbox_inches='tight',pad_inches=.1); plt.close(fig)

response(); performance(); print('wrote',out)

