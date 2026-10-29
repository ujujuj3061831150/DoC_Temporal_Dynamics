"""Rebuild the three manuscript data figures at their final IEEE widths."""
from pathlib import Path
import sys, json, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap, to_rgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.temporal_dynamics.adapters import load_mendeley_temporal_dataset
skill = Path.home()/'.codex/skills/scipilot-figure-skill/scripts'
sys.path.insert(0, str(skill))
try:
    from visual_qa import audit_layout, render_preview
except ImportError:
    def audit_layout(fig):
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        from matplotlib.text import Text
        return [('WARN', 'Text outside figure: '+t.get_text())
                for t in fig.findobj(Text) if t.get_visible() and t.get_text()
                and not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0])]
    def render_preview(fig,path,dpi=300):
        fig.savefig(path,dpi=dpi)

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=ROOT/'manuscript_temporal/figures')
parser.add_argument('--input',type=Path,default=ROOT/'results/temporal_dynamics_v1')
args=parser.parse_args()
OUT=args.output
OUT.mkdir(parents=True,exist_ok=True)
QA=ROOT/'audit_outputs/manuscript_palette_20261029/data_figures'
QA.mkdir(parents=True,exist_ok=True)
alignment_scripts=Path.home()/'.codex/skills/nature-figure/scripts'
sys.path.insert(0,str(alignment_scripts))
from audit_panel_alignment import require_matplotlib_panel_alignment
plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.labelsize':7.5,
 'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7.5,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.65,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'text.color':'#202020','axes.labelcolor':'#202020',
 'xtick.color':'#303030','ytick.color':'#303030','axes.edgecolor':'#707070',
 'lines.solid_capstyle':'round','axes.titlepad':9})
BLUE='#385A73'; ORANGE='#A3816C'; GRAY='#78838C'
INK='#202020'; PALE='#F4F6F8'; RULE='#DDE2E7'
HEATMAP=LinearSegmentedColormap.from_list('manuscript_gray_blue',
    ['#F5F6F7','#D4DEE5','#91A8B9','#4B6B82','#243F52'])

def relative_luminance(color):
    values=np.asarray(to_rgb(color))
    linear=np.where(values<=.04045,values/12.92,((values+.055)/1.055)**2.4)
    return float(np.dot(linear,[.2126,.7152,.0722]))

def cell_label_color(value,mappable):
    luminance=relative_luminance(mappable.cmap(mappable.norm(value))[:3])
    contrast_white=1.05/(luminance+.05)
    contrast_ink=(luminance+.05)/(relative_luminance(INK)+.05)
    return 'white' if contrast_white>contrast_ink else INK
records={}

def decorate(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y',color=RULE,lw=.45)
    ax.tick_params(pad=3)

def panel(ax,letter,title):
    # Axis labels carry the variable names; only panel letters are repeated.
    ax.text(-.085,1.055,letter,transform=ax.transAxes,fontsize=8,
            fontweight='bold',ha='left',va='bottom')
def save(fig,name,exemptions=()):
    fig.canvas.draw()
    axes=[ax for ax in fig.axes if ax.get_label()!='<colorbar>']
    row_groups=[{'id':'main-row','panels':[chr(97+i) for i in range(len(axes))]}] if len(axes)>1 else None
    require_matplotlib_panel_alignment(fig,json_out=QA/(name+'.alignment.json'),
        overlay_svg=QA/(name+'.alignment.svg'),row_groups=row_groups,axes=axes,
        exemptions=exemptions,strict=True)
    records[name]=audit_layout(fig)
    render_preview(fig,str(QA/(name+'.png')),dpi=300)
    Image.open(QA/(name+'.png')).convert('L').save(QA/(name+'_grayscale.png'))
    fig.savefig(OUT/(name+'.pdf'))
    fig.savefig(OUT/(name+'.svg'))
    fig.savefig(OUT/(name+'.png'),dpi=600)
    plt.close(fig)

data=load_mendeley_temporal_dataset(repo_root=ROOT)
ids=data['subject_ids']; labels=np.asarray(data['labels'])
frame=pd.DataFrame({'subject':ids,'group':np.where(labels==1,'MCS','VS/UWS'),
 'lag1':[data['temporal_features'][s]['lag1'] for s in ids],
 'neg_cv':[data['temporal_features'][s]['neg_cv'] for s in ids]})
frame.to_csv(QA/'figure1_source.csv',index=False)
fig,axs=plt.subplots(1,3,figsize=(7.1398,2.18))
fig.subplots_adjust(left=.075,right=.98,bottom=.245,top=.81,wspace=.42)
ax=axs[0]
trace_rows=[]
for sid,color,style,label in [('Nekrasova_P36',BLUE,'-','MCS (P36)'),('Nekrasova_P25',ORANGE,'--','VS/UWS (P25)')]:
    with np.load(ROOT/'results/target_profiles_v11/raw_channels'/f'{sid}.npz') as d:
        g=d['powers'][100:130,0,:].mean(axis=1); ok=d['peaks'][100:130]<500
    g=(g-g[ok].mean())/g[ok].std();g[~ok]=np.nan
    trace_rows.extend({'subject':sid,'block_index':100+i,'time_s':4*i,'accepted':bool(ok[i]),'standardized_theta':value} for i,value in enumerate(g))
    ax.plot(np.arange(30)*4,g,color=color,ls=style,lw=1.05,label=label)
pd.DataFrame(trace_rows).to_csv(QA/'figure1_trace_source.csv',index=False)
ax.set(xlabel='Elapsed time (s)',ylabel=r'Normalized $\theta$ power',xlim=(0,116))
ax.set_xticks([0,60,120]);ax.set_xlim(0,120)
decorate(ax)
panel(ax,'a','  Bedside dynamics')
handles,labs=ax.get_legend_handles_labels()
fig.legend(handles,['MCS','VS/UWS'],loc='upper center',bbox_to_anchor=(.55,1.015),ncol=2,frameon=False,handlelength=2.5,columnspacing=2.5)
for ax,col,letter,ylabel in zip(axs[1:],['lag1','neg_cv'],['b','c'],[r'Persistence, $\rho_1$','Theta CV']):
    for position,(group,color,marker) in enumerate([(1,BLUE,'o'),(0,ORANGE,'s')]):
        values=frame.loc[labels==group,col].to_numpy()
        if col=='neg_cv': values=-values
        assert np.isfinite(values).all()
        if col=='neg_cv': assert (values>0).all(), 'CV must be positive for a log axis'
        ax.boxplot([values],positions=[position],widths=.48,whis=1.5,showfliers=False,
            patch_artist=True,manage_ticks=False,
            boxprops=dict(facecolor=color,alpha=.16,edgecolor=color,linewidth=.8),
            whiskerprops=dict(color=GRAY,linewidth=.7),capprops=dict(color=GRAY,linewidth=.7),
            medianprops=dict(color=INK,linewidth=1))
        offsets=(((np.arange(len(values))*.61803398875)%1)-.5)*.30
        # Jitter is horizontal only: every patient's measured y value is retained.
        ax.scatter(position+offsets,values,s=12,marker=marker,color=color,
                   edgecolors='white',linewidths=.3,zorder=4)
    if col=='neg_cv':
        ax.set_yscale('log');ax.set_yticks([.5,1,2,5,10],['0.5','1','2','5','10']);ax.minorticks_off()
        ax.set_ylim(.38,12)
    else:
        ax.set_ylim(-.13,1.0);ax.set_yticks([0,.4,.8])
    ax.set(xlim=(-.5,1.5),ylabel=ylabel)
    ax.set_xticks([0,1],['MCS','VS/UWS'])
    decorate(ax);panel(ax,letter,'')
save(fig,'fig1_mesocircuit_memory')

# These are separate feature/model configurations, not additive increments.
fig=plt.figure(figsize=(3.4868,2.18))
axs=[fig.add_axes([.16,.32,.27,.49]),fig.add_axes([.70,.32,.27,.49])]
ax=axs[0]
factorial=[]
matrix=np.array([[.8200,.8911],[.7893,.8591]])
heat=ax.pcolormesh(np.arange(3)-.5,np.arange(3)-.5,matrix,
                  cmap=HEATMAP,vmin=.75,vmax=.95,shading='flat',
                  edgecolors='white',linewidth=.7,rasterized=False)
ax.set_ylim(1.5,-.5)
for y,(label,static,ordered) in enumerate([('Linear',.8200,.8911),('Rank',.7893,.8591)]):
    for x,value in enumerate([static,ordered]):
        ax.text(x,y,f'{value:.4f}',ha='center',va='center',fontsize=7,
                color=cell_label_color(value,heat))
    factorial.extend([{'fusion':label,'suite':'Static','auroc':static},
                      {'fusion':label,'suite':'Lag-1','auroc':ordered}])
pd.DataFrame(factorial).to_csv(QA/'figure2_factorial_source.csv',index=False)
ax.set_xticks([0,1],['Static',r'$+\rho_1$']);ax.set_yticks([0,1],['Linear','Rank'])
ax.tick_params(axis='both',length=0,pad=4,labelsize=7.5)
for spine in ax.spines.values():spine.set_visible(False)
cax=fig.add_axes([.16,.115,.27,.025],label='<colorbar>')
bar=fig.colorbar(heat,cax=cax,orientation='horizontal',ticks=[.75,.85,.95])
bar.solids.set_rasterized(False)
bar.outline.set_visible(False);bar.ax.tick_params(labelsize=6.5,length=2,pad=2)
bar.set_label('AUROC',fontsize=7,labelpad=1)
panel(ax,'a','  Factorial')
ax=axs[1]
summary_path=ROOT/'results/matched_model_comparison_v1/summary.json'
contrasts=json.loads(summary_path.read_text(encoding='utf8'))['evaluations']['mendeley/orthogonal_lag1']['paired_contrasts']
forest=[]
for y,(method,label) in enumerate([('lda_shrinkage','LDA'),('logistic','Logistic'),('ridge','Ridge')]):
    record=contrasts['equal_mean_minus_'+method];x=record['auroc_difference'];lo,hi=record['ci95']
    ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt='o',ms=4,capsize=2,lw=1.05,
                color=BLUE,markeredgecolor='white',markeredgewidth=.65)
    forest.append({'comparison':'Equal mean minus '+label,'difference':x,'ci_lower':lo,'ci_upper':hi})
pd.DataFrame(forest).to_csv(QA/'figure2_paired_source.csv',index=False)
ax.axvline(0,c=GRAY,ls=':',lw=.8)
ax.set_yticks([0,1,2],['LDA','Logistic','Ridge'])
ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=5)
ax.set(xlim=(-.008,.11),ylim=(-.5,2.5),xlabel=r'$\Delta$AUROC');ax.set_xticks([0,.05,.10],['0','.05','.10'])
panel(ax,'b','  Paired gain')
save(fig,'fig2_factorial_ablations')

null=pd.read_csv(args.input/'shuffle_summary_10000.csv')['plus_lag1__C'].to_numpy()
pd.DataFrame({'shuffled_auroc':null}).to_csv(QA/'figure3_source.csv',index=False)
fig,ax=plt.subplots(figsize=(3.4868,1.98));fig.subplots_adjust(left=.18,right=.96,bottom=.255,top=.83)
counts,edges,_=ax.hist(null,bins=30,color='#BEC8CF',edgecolor='white',linewidth=.3,label='Shuffled')
assert int(counts.sum())==len(null)==10000
pd.DataFrame({'left':edges[:-1],'right':edges[1:],'count':counts.astype(int)}).to_csv(QA/'figure3_histogram_bins.csv',index=False)
ax.axvline(.82,c=ORANGE,ls='--',lw=1,label='Static')
ax.axvline(.8911,c=BLUE,lw=1.5,label='Intact')
assert np.sum(null>=.8911)==0
ax.set(xlim=(.785,.92),ylim=(0,counts.max()*1.12),xlabel='AUROC',ylabel='Permutations');ax.set_xticks([.80,.84,.88,.92])
from matplotlib.ticker import MaxNLocator
ax.yaxis.set_major_locator(MaxNLocator(nbins=3,integer=True))
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,1.015),ncol=3,frameon=False,columnspacing=.8,handlelength=1.3)
decorate(ax)
save(fig,'fig3_permutation_collapse')
(QA/'layout_audit.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(records,ensure_ascii=False,indent=2))
