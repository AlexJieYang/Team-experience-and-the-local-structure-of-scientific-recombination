"""Publication plates assembled from existing exported statistics, without refitting.

PDF/SVG retain vector geometry; PNGs at 600 dpi are distinct from notebook previews.
No significance-driven model selection or cross-model covariance is invented.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import json, html, math, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
from PIL import Image, ImageOps, ImageDraw

BLUE='#2166AC'; ORANGE='#D55E00'; TEAL='#009E73'; PURPLE='#7851A9'; GREY='#7B8790'
REGIME_COLORS=[BLUE,TEAL,'#D99A00',PURPLE]
REGIME_LABELS=['Low M / low min','Low M / high min','High M / low min','High M / high min']
DIVERGE=colors.LinearSegmentedColormap.from_list('blue_white_orange',[BLUE,'#FAFAFA',ORANGE])
MEASURES=['any_novel_pct','novel_share_pct','z10_asinh','z50_asinh','semantic_distance']
SHORT_MEASURES=['New pair','Pair share','z10','z50','Semantic']

@dataclass
class FigureSettings:
    width_mm: float=178
    font_pt: float=7.0
    panel_font_pt: float=8.0
    preview_dpi: int=150
    png_dpi: int=600
    export_svg: bool=True
    export_tiff: bool=False
    supplementary: bool=True
    kde_max_points: int=4000
    show_points_per_group: int=120
    random_seed: int=20261028
    strict_layout: bool=True
    main1_height_mm: float=106
    main2_height_mm: float=114
    main3_height_mm: float=112
    main4_height_mm: float=116
    main_box_aspect: float | None=None


def load_settings(path=None):
    path=Path(path or Path(__file__).with_name('figure_settings.json'))
    values=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    return FigureSettings(**values)


def _style(s):
    names={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Arial','Helvetica','Liberation Sans','DejaVu Sans'] if f in names),'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'font.size':s.font_pt,'axes.labelsize':s.font_pt,
        'axes.titlesize':s.panel_font_pt,'xtick.labelsize':s.font_pt-.3,'ytick.labelsize':s.font_pt-.3,
        'legend.fontsize':s.font_pt-.4,'axes.linewidth':.55,'lines.linewidth':.95,
        'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':2,'ytick.major.size':2,
        'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white',
        'axes.facecolor':'white','savefig.facecolor':'white','figure.dpi':120,
        'mathtext.fontset':'dejavusans','axes.labelpad':2,'hatch.linewidth':.4})
    return font


def _bool(s):return s.astype(str).str.lower().isin(['true','1'])
def _number(v,decimals=2):return f'{v:+.{decimals}f}' if np.isfinite(v) else ''
def _compact_n(n):return f'{n/1e6:.2f}M' if n>=1e6 else f'{n/1000:.0f}k' if n>=1000 else str(int(n))
def _finite_limit(a,floor=.1):
    x=np.asarray(a,float);x=x[np.isfinite(x)]
    return max(floor,float(np.max(np.abs(x)))) if len(x) else floor

def _zero(ax,axis='x'):
    if axis=='x':ax.axvline(0,color=GREY,lw=.65,ls=(0,(3,2)),zorder=0)
    else:ax.axhline(0,color=GREY,lw=.65,ls=(0,(3,2)),zorder=0)

def _empty(ax,text='Not available'):
    ax.text(.5,.5,text,ha='center',va='center',transform=ax.transAxes,color=GREY,fontsize=7)
    ax.set_xticks([]);ax.set_yticks([])

def check_layout(fig,name,strict=True):
    # Fixed-aspect axes with colorbars need a second layout pass before bounds are final.
    fig.canvas.draw();fig.canvas.draw()
    renderer=fig.canvas.get_renderer();W,H=fig.bbox.width,fig.bbox.height
    bad=[]
    for ax in fig.axes:
        if not ax.get_visible():continue
        # These are the text objects whose placement is controlled by the layout engine.
        texts=[ax.title,ax._left_title,ax._right_title,ax.xaxis.label,ax.yaxis.label,
               ax.xaxis.get_offset_text(),ax.yaxis.get_offset_text()]
        if ax.name=='3d':
            texts.append(ax.zaxis.label)
        for axis,limits in ([] if ax.name=='3d' else [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]):
            lo,hi=sorted(limits)
            for loc,label in zip(axis.get_majorticklocs(),axis.get_majorticklabels()):
                if lo-1e-9<=loc<=hi+1e-9:texts.append(label)
        for t in texts:
            if not t.get_visible() or not t.get_text():continue
            box=t.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > W+1 or box.y1 > H+1:
                bad.append(t.get_text())
    if bad and strict:raise RuntimeError(f'{name}: text outside the fixed canvas: {bad}')


class JournalPlots:
    def __init__(self,results_root,output_dir=None,settings=None):
        self.root=Path(results_root).resolve()
        self.s=settings or load_settings()
        self.out=Path(output_dir).resolve() if output_dir else self.root/'pnas_figures'
        if self.out==self.root or self.out==self.root/'tables':raise ValueError('Use a dedicated output subfolder')
        self.font=_style(self.s);self.rng=np.random.default_rng(self.s.random_seed)
        for rel in ['main','supplement','previews','source_data']:(self.out/rel).mkdir(parents=True,exist_ok=True)
        required=['profile_contrasts','surfaces','model_samples','observed_cells','annual_sample']
        for key in required:
            if not(self.root/'tables'/f'{key}.csv').exists():raise FileNotFoundError(f'Missing exported table: {key}.csv')
        self.tables={}
        table_names=['profile_contrasts','surfaces','model_samples','observed_cells','annual_sample',
            'profile_support','common_measure_contrasts','common_measure_scales','cohort_composition',
            'alluvial_composition','coefficients','field_contrasts','field_coverage']
        for path in [self.root/'tables'/f'{name}.csv' for name in table_names]:
            if not path.exists():continue
            dtype={'field_id':'string','field_id_text':'string'} if 'field' in path.name else None
            self.tables[path.stem]=pd.read_csv(path,dtype=dtype)
        self.p=self.tables['profile_contrasts'];self.sur=self.tables['surfaces'];self.samples=self.tables['model_samples']
        parquet=self.root/'tables/plot_sample.parquet'
        self.plot=pd.read_parquet(parquet) if parquet.exists() else pd.DataFrame()
        labels=Path(__file__).with_name('field_labels.csv')
        self.fields=pd.read_csv(labels,dtype='string').set_index('field_id').field_name.to_dict() if labels.exists() else {}
        self.fields={k:v.replace(' Science',' sci.') for k,v in self.fields.items()}
        cfg=self.root/'config_used.json';self.cfg=json.loads(cfg.read_text(encoding='utf-8')) if cfg.exists() else {}
        self.records=[];self.captions=[];self.atlas=None

    def fig(self,height_mm):
        fig=plt.figure(figsize=(self.s.width_mm/25.4,height_mm/25.4),layout='constrained')
        fig.get_layout_engine().set(w_pad=1.5/72,h_pad=2/72,wspace=.035,hspace=.04)
        return fig

    def title(self,ax,letter,title):
        title=textwrap.fill(title,width=20,break_long_words=False,break_on_hyphens=False) if len(title)>20 else title
        ax.set_title(f'{letter}  {title}',loc='left',fontweight='bold',fontsize=self.s.panel_font_pt,pad=4)
        ax.tick_params(pad=2)

    def main_axes(self,height_mm):
        """Two rows and three columns; optional aspect setting for customization."""
        fig=self.fig(height_mm);axs=fig.subplots(2,3)
        if self.s.main_box_aspect is not None:
            for ax in axs.flat:ax.set_box_aspect(self.s.main_box_aspect)
        return fig,axs

    def source(self,name,frame):
        frame.to_csv(self.out/'source_data'/f'{name}.csv',index=False)

    def get(self,name,mean=None):
        z=self.p.loc[self.p.model.eq(name)]
        return z.loc[np.isclose(z.mean_pivot,mean)] if mean is not None else z

    def n(self,name):
        q=self.samples.loc[self.samples.model.eq(name)]
        return int(q.n_estimated.iloc[0]) if len(q) else 0

    def legend(self,fig,loc='outside upper center'):
        handles=[Line2D([],[],color=BLUE,marker='o',ms=3,lw=.9,label='Baseline'),
                 Line2D([],[],color=ORANGE,marker='s',ms=3,lw=.9,label='Reference-conditioned')]
        fig.legend(handles=handles,loc=loc,ncol=2,frameon=False,handlelength=1.4,columnspacing=1.4)

    def _layout_check(self,fig,name):
        check_layout(fig,name,self.s.strict_layout)

    def save(self,fig,name,caption,main=False):
        self._layout_check(fig,name)
        directory=self.out/('main' if main else 'supplement')
        extensions=['pdf','png']+(['svg'] if self.s.export_svg else [])+(['tiff'] if self.s.export_tiff else [])
        for ext in extensions:
            path=directory/f'{name}.{ext}';tmp=path.with_name(f'{name}.writing.{ext}')
            kwargs={'dpi':self.s.png_dpi} if ext in ['png','tiff'] else {}
            if ext=='tiff':kwargs['pil_kwargs']={'compression':'tiff_lzw'}
            fig.savefig(tmp,format=ext,**kwargs);tmp.replace(path)
        preview=self.out/'previews'/f'{name}.png'
        fig.savefig(preview,dpi=self.s.preview_dpi)
        if self.atlas is not None:self.atlas.savefig(fig)
        self.records.append({'name':name,'kind':'main' if main else 'supplement','width_mm':self.s.width_mm,
            'height_mm':fig.get_figheight()*25.4,'pdf':str((directory/f'{name}.pdf').relative_to(self.out)),
            'preview':str(preview.relative_to(self.out)),'font':self.font})
        panels=[ax for ax in fig.axes if ax._left_title.get_text()]
        self.records[-1]['n_panels']=len(panels)
        self.captions.append(f'## {name}\n\n{caption}\n')
        plt.close(fig)

    def mesh(self,fig,ax,z,value,label,cmap='viridis',vmin=None,vmax=None,supported=False,colorbar=True,ylim=None):
        if z.empty:_empty(ax);return None
        q=z.copy()
        if supported:q.loc[~_bool(q.supported),value]=np.nan
        t=q.pivot(index='grounding_gap',columns='mean_pivot',values=value).sort_index().sort_index(axis=1)
        im=ax.pcolormesh(t.columns.to_numpy(float),t.index.to_numpy(float),t.to_numpy(float),
                         shading='nearest',cmap=cmap,vmin=vmin,vmax=vmax,rasterized=False)
        ax.set_facecolor('#F0F1F2')
        if ylim is None:
            valid=q.loc[q[value].notna(),'grounding_gap'];ylim=min(1,float(valid.max())+.025) if len(valid) else .65
        ax.set(xlim=(0,1),ylim=(0,max(.35,ylim)))
        ax.plot([0,min(1,ylim)],[0,min(1,ylim)],lw=.5,color='#8B949B')
        ax.set_xticks([0,.5,1]);ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        if colorbar:
            cb=fig.colorbar(im,ax=ax,pad=.02,fraction=.052,shrink=.91)
            cb.ax.tick_params(labelsize=self.s.font_pt-.6,pad=1,width=.4,length=2)
            cb.set_label(label,fontsize=self.s.font_pt-.3,labelpad=2)
        return im

    def observed_mesh(self,fig,ax,outcome,scale=1,label='',log=False):
        z=self.tables['observed_cells'];z=z.loc[z.outcome.eq(outcome)].copy()
        z['mean_pivot']=(z.mbin+.5)/20;z['grounding_gap']=(z.gbin+.5)/20
        if log:
            z['display']=np.where(z['count']>0,np.log10(z['count'].clip(lower=1)),np.nan)
        else:
            z['display']=scale*z['mean'];z.loc[z['count']<self.cfg.get('surface_cell_min',100),'display']=np.nan
        self.mesh(fig,ax,z,'display',label)
        return z

    def profile_lines(self,ax,names=('formation_base','formation_full'),ylabel='Formation contrast (pp)'):
        for name,color,marker in zip(names,[BLUE,ORANGE],['o','s']):
            z=self.get(name).sort_values('mean_pivot')
            if len(z):ax.errorbar(z.mean_pivot,z.estimate,yerr=1.96*z.se,color=color,marker=marker,ms=3.2,capsize=1.4,lw=.9)
        _zero(ax,'y');ax.set(xticks=[.4,.6,.8],xlabel='Mean distance, M',ylabel=ylabel)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    def forest(self,ax,names,labels,mean=.6,color=ORANGE,show_labels=True):
        rows=[]
        for i,(name,label) in enumerate(zip(names,labels)):
            q=self.get(name,mean)
            if len(q):
                r=q.iloc[0];ax.errorbar(r.estimate,i,xerr=1.96*r.se,fmt='o',ms=3,color=color,capsize=1.4,lw=.8)
                rows.append(r)
            else:ax.text(.98,i,'NA',transform=ax.get_yaxis_transform(),ha='right',va='center',color=GREY,fontsize=6.4)
        _zero(ax);ax.set(yticks=range(len(names)),yticklabels=labels if show_labels else [],ylim=(len(names)-.5,-.5))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4));return pd.DataFrame(rows)

    def evidence(self,fig,ax,names,labels,mean_values=(.4,.6,.8),field=False,limit=None,colorbar=True):
        values=np.full((len(names),len(mean_values)),np.nan)
        for i,name in enumerate(names):
            for j,m in enumerate(mean_values):
                q=self.get(name,m)
                if len(q):values[i,j]=q.estimate.iloc[0]
        limit=limit or _finite_limit(values,.5)
        cmap=DIVERGE.copy();cmap.set_bad('#EBEDEF')
        im=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap=cmap,vmin=-limit,vmax=limit)
        for (i,j),v in np.ndenumerate(values):
            if np.isfinite(v):ax.text(j,i,_number(v),ha='center',va='center',fontsize=6.3,
                color='white' if abs(v)>.67*limit else '#17212B')
            else:ax.text(j,i,'-',ha='center',va='center',color='#88929B',fontsize=7)
        ax.set(xticks=range(len(mean_values)),xticklabels=[f'{m:.1f}' for m in mean_values],yticks=range(len(names)),yticklabels=labels,xlabel='Mean distance, M')
        ax.tick_params(length=0);ax.spines[['left','bottom']].set_visible(False)
        if colorbar:
            cb=fig.colorbar(im,ax=ax,pad=.025,fraction=.04,shrink=.85)
            cb.ax.tick_params(labelsize=6.4,pad=1);cb.set_label('Contrast (pp)',fontsize=7,labelpad=2)
        return values

    def main1(self):
        fig,axs=self.main_axes(self.s.main1_height_mm)
        ax=axs[0,0];self.title(ax,'A','Experience profiles')
        examples=[([.55,.60,.65],1,'G = 0.05'),([.35,.70,.75],0,'G = 0.25')]
        for distances,y,label in examples:
            ax.scatter(distances,[y-.07,y,y+.07],s=18,c=[TEAL,BLUE,BLUE],zorder=3,edgecolors='white',linewidth=.3)
            ax.plot([min(distances),.6],[y+.23,y+.23],color=ORANGE,lw=1.1)
            ax.text(.02,y+.18,label,fontsize=7)
        ax.axvline(.6,ls=(0,(2,2)),color=GREY,lw=.7);ax.text(.62,1.4,'M = 0.60',fontsize=7)
        ax.set(xlim=(0,1),ylim=(-.35,1.65),xticks=[0,.5,1],yticks=[0,1],yticklabels=['Profile 2','Profile 1'],xlabel='Member source distance')
        self.title(axs[0,1],'B','Configuration map')
        density=self.observed_mesh(fig,axs[0,1],'any_novel',label='log10 papers',log=True)
        axs[0,1].set(xlabel='Mean distance, M',ylabel='Gap, G')
        self.title(axs[0,2],'C','New-pair formation')
        observed=self.observed_mesh(fig,axs[0,2],'any_novel',100,'New-pair papers (%)')
        axs[0,2].set(xlabel='Mean distance, M',ylabel='Gap, G')
        full_structure=all(self.tables['observed_cells'].outcome.eq(key).any() for key in ['n_reference_sources','reference_source_hhi'])
        for ax,key,label,letter,title in [(axs[1,0],'n_reference_sources','Source count','D','Source breadth'),
                    (axs[1,1],'reference_source_hhi','Source HHI','E','Source concentration')]:
            self.title(ax,letter,title)
            if full_structure:
                tab=self.observed_mesh(fig,ax,key,label=label)
                ax.set(xlabel='Mean distance, M',ylabel='Gap, G')
                self.source(f'Fig1{letter}',tab)
                continue
            q=self.plot.dropna(subset=['mean_pivot','grounding_gap',key]).copy()
            q['mean_pivot']=(np.minimum(np.floor(q.mean_pivot*20),19)+.5)/20
            q['grounding_gap']=(np.minimum(np.floor(q.grounding_gap*20),19)+.5)/20
            tab=q.groupby(['mean_pivot','grounding_gap'])[key].agg(['mean','count']).reset_index()
            tab.loc[tab['count']<30,'mean']=np.nan
            self.mesh(fig,ax,tab,'mean',label);ax.set(xlabel='Mean distance, M',ylabel='Gap, G')
            self.source(f'Fig1{letter}',tab)
        ax=axs[1,2];self.title(ax,'F','Publication cohorts')
        q=self.tables['cohort_composition'];q=q.loc[q.outcome.eq('any_novel')].groupby('cohort')[['count','sum']].sum().reset_index()
        ax.plot(q.cohort,100*q['sum']/q['count'],marker='o',ms=3,color=BLUE)
        ax.set(xlabel='Cohort start year',ylabel='New-pair papers (%)');ax.set_xticks([1990,2005,2020])
        self.source('Fig1B',density);self.source('Fig1C',observed);self.source('Fig1F',q)
        self.save(fig,'Fig1_knowledge_landscape',
            'A, Illustrative three-member distance profiles, not empirical observations. Both have M=0.60; G differs because the closest-member distance differs. '
            'B-C, Full eligible-sample counts and unadjusted new-pair frequencies in 0.05 by 0.05 bins. C masks bins below the configured density threshold. '+
            ('D-E, Mean source count and source HHI in the full eligible sample; cells below the configured full-sample count threshold are masked. ' if full_structure else
             'D-E, Mean source count and source HHI in the uniform plotting sample; cells with fewer than 30 plotting observations are masked. ')+
            'F, Unadjusted formation rates by five-year publication cohort, aggregated over configuration groups. Cohort bins are not future citation windows.',main=True)



    def main2(self):
        fig,axs=self.main_axes(self.s.main2_height_mm);self.legend(fig)
        self.title(axs[0,0],'A','Profile contrasts');self.profile_lines(axs[0,0])
        names=['formation_base','formation_add_count','formation_add_concentration','formation_add_age',
               'formation_add_visibility','formation_add_count_concentration','formation_full']
        labels=['Baseline','+ count','+ HHI','+ age','+ citation base','+ count and HHI','+ all four']
        self.title(axs[0,1],'B','Reference adjustment')
        self.forest(axs[0,1],names,labels,color=BLUE);axs[0,1].set_xlabel('Contrast at M = 0.6 (pp)')
        ax=axs[0,2];self.title(ax,'C','Context coverage')
        q=self.tables.get('profile_support',pd.DataFrame())
        if len(q):
            q=q.loc[q.model.eq('formation_full')].sort_values('mean_pivot')
            for i,r in enumerate(q.itertuples()):
                ax.plot([r.context_mass_reference,r.context_mass_here],[i,i],lw=1.1,color='#A8B3BA')
                ax.scatter(r.context_mass_reference,i,s=15,color=TEAL,marker='o',zorder=3)
                ax.scatter(r.context_mass_here,i,s=16,color=PURPLE,marker='^',zorder=3)
            ax.set(yticks=range(len(q)),yticklabels=[f'M = {x:.1f}' for x in q.mean_pivot],xlim=(0,1),xticks=[0,.5,1],xlabel='Fraction in common contexts')
            ax.legend(handles=[Line2D([],[],color=TEAL,marker='o',lw=0,ms=3,label='G=0.05'),Line2D([],[],color=PURPLE,marker='^',lw=0,ms=3,label='G=0.25')],frameon=False,fontsize=6.1,loc='lower left',ncol=2,handletextpad=.3,columnspacing=.6)
            self.source('Fig2C',q)
        else:_empty(ax)
        panels=[self.sur.loc[self.sur.model.eq(x)].copy() for x in ['formation_base','formation_full']]
        vv=np.concatenate([z.loc[_bool(z.supported),'contrast'].to_numpy(float) for z in panels]);lim=_finite_limit(vv)
        for ax,z,letter,title in zip([axs[1,0],axs[1,1]],panels,['D','E'],['Baseline surface','Conditioned surface']):
            self.title(ax,letter,title);self.mesh(fig,ax,z,'contrast','Contrast (pp)',DIVERGE,-lim,lim,True)
            ax.set(xlabel='Mean distance, M',ylabel='Gap, G')
        self.title(axs[1,2],'F','Model forms')
        modelnames=['formation_full','formation_venue_full','formation_flexible_full','formation_structural_full','formation_spline_full']
        self.evidence(fig,axs[1,2],modelnames,['Linear','Venue FE','Flexible refs','Structural FE','Spline M,G'],colorbar=True)
        self.source('Fig2A',self.p.loc[self.p.model.isin(['formation_base','formation_full'])]);self.source('Fig2B',self.p.loc[self.p.model.isin(names)])
        self.source('Fig2F',self.p.loc[self.p.model.isin(modelnames)])
        self.source('Fig2DE',pd.concat(panels,ignore_index=True))
        self.save(fig,'Fig2_reference_conditioning',
            f'A, Fixed-M contrasts for G=0.05 to 0.25 in baseline and reference-conditioned models fitted to the same {self.n("formation_full"):,} papers. '
            'B, Single and joint reference-structure additions on the same actual sample. C, The fraction of papers at each endpoint represented in common team-size/source-count-bin/HHI-bin contexts. '
            'D-E, Model contrasts against the model-specific reference coordinates stored in surfaces.csv, using a common color scale and masking unsupported cells. '
            'F, Estimated contrasts across model forms; the models may use different actual samples, as documented in model_samples.csv. Gray cells lack a supported exported contrast. '
            'All error bars are pointwise 95% normal-approximation intervals; conditioning differences describe statistical associations.',main=True)



    def main3(self):
        fig,axs=self.main_axes(self.s.main3_height_mm);self.legend(fig)
        q=self.tables.get('common_measure_contrasts',pd.DataFrame());q=q.loc[np.isclose(q.mean_pivot,.6)] if len(q) else q
        ax=axs[0,0];self.title(ax,'A','Measure contrasts')
        for off,suffix,col in [(-.1,'base',BLUE),(.1,'full',ORANGE)]:
            for i,y in enumerate(MEASURES):
                z=q.loc[q.outcome.eq(y)&q.model.str.endswith('_'+suffix)]
                if len(z):r=z.iloc[0];ax.errorbar(r.estimate_sd,i+off,xerr=1.96*r.se_sd,fmt='o',ms=3,color=col,capsize=1.3)
        ax.set(yticks=range(5),yticklabels=SHORT_MEASURES,ylim=(4.5,-.5),xlabel='Contrast / outcome SD');_zero(ax)
        for ax,key,letter,title,ylabel in [(axs[1,0],'n_reference_sources','D','Source breadth','Source-count contrast'),
                        (axs[1,1],'source_hhi_pct','E','Source HHI','HHI contrast (pp)')]:
            self.title(ax,letter,title);z=self.get('reference_profile_'+key).sort_values('mean_pivot')
            if len(z):ax.errorbar(z.mean_pivot,z.estimate,yerr=1.96*z.se,fmt='o-',ms=3,color=TEAL,capsize=1.3)
            ax.set(xticks=[.4,.6,.8],xlabel='Mean distance, M',ylabel=ylabel);_zero(ax,'y')
            self.source('Fig3'+letter,z)
        ax=axs[0,1];self.title(ax,'B','Conditioned profiles')
        full=self.tables.get('common_measure_contrasts',pd.DataFrame())
        values=np.full((5,3),np.nan)
        for i,y in enumerate(MEASURES):
            for j,m in enumerate([.4,.6,.8]):
                z=full.loc[full.outcome.eq(y)&full.model.str.endswith('_full')&np.isclose(full.mean_pivot,m)]
                if len(z):values[i,j]=z.estimate_sd.iloc[0]
        limit=_finite_limit(values,.05);cmap=DIVERGE.copy();cmap.set_bad('#EBEDEF')
        im=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,vmin=-limit,vmax=limit,aspect='auto')
        for (i,j),v in np.ndenumerate(values):
            if np.isfinite(v):ax.text(j,i,_number(v),ha='center',va='center',fontsize=6.4,color='white' if abs(v)>.7*limit else '#17212B')
        ax.set(xticks=[0,1,2],xticklabels=['0.4','0.6','0.8'],yticks=range(5),yticklabels=SHORT_MEASURES,xlabel='Mean distance, M')
        cb=fig.colorbar(im,ax=ax,pad=.02,fraction=.05);cb.set_label('SD units',fontsize=7);cb.ax.tick_params(labelsize=6.4)
        ax=axs[0,2];self.title(ax,'C','Contrast plane')
        for m,col in zip([.4,.6,.8],[BLUE,TEAL,ORANGE]):
            a=full.loc[full.outcome.eq('any_novel_pct')&full.model.str.endswith('_full')&np.isclose(full.mean_pivot,m)]
            b=full.loc[full.outcome.eq('semantic_distance')&full.model.str.endswith('_full')&np.isclose(full.mean_pivot,m)]
            if len(a) and len(b):
                a,b=a.iloc[0],b.iloc[0];ax.errorbar(a.estimate_sd,b.estimate_sd,xerr=1.96*a.se_sd,yerr=1.96*b.se_sd,fmt='o',ms=3.5,color=col,capsize=1.3)
                ax.annotate(f'{m:.1f}',(a.estimate_sd,b.estimate_sd),xytext={.4:(4,8),.6:(4,-13),.8:(-13,8)}[m],textcoords='offset points',fontsize=6.8)
        _zero(ax);_zero(ax,'y');ax.set(xlabel='New-pair contrast (SD)',ylabel='Semantic contrast (SD)')
        ax=axs[1,2];self.title(ax,'F','Sample check')
        a=self.get('formation_full',.6);b=full.loc[full.outcome.eq('any_novel_pct')&full.model.str.endswith('_full')&np.isclose(full.mean_pivot,.6)]
        for i,(z,col) in enumerate([(a,ORANGE),(b,TEAL)]):
            if len(z):r=z.iloc[0];ax.errorbar(r.estimate,i,xerr=1.96*r.se,fmt='o',ms=3,color=col,capsize=1.4)
        ax.set(yticks=[0,1],yticklabels=['Main sample','Common sample'],ylim=(1.5,-.5),xlabel='Formation contrast (pp)');_zero(ax)
        self.source('Fig3ABCF',full)
        self.save(fig,'Fig3_measure_specificity',
            f'A,B,C, Results on the common five-measure sample ({self.n("common_any_novel_pct_full"):,} actual estimated papers). Standardization divides existing contrasts and standard errors by the outcome SD in this same sample. '
            'Larger z10/z50 means greater conventionality; larger semantic distance is a different construct. D-E, Baseline configuration associations with existing source count and source HHI. '
            'C, Crosshairs show separate pointwise intervals for two outcomes. Labels give M. '
            'F, Reference-conditioned formation estimates at M=0.6 in the outcome-specific and common samples.',main=True)



    def spec_models(self):
        return [
          ('formation_full','Linear','Form'),('formation_venue_full','Venue FE','Form'),
          ('formation_flexible_full','Flexible refs','Form'),('formation_structural_full','Structural FE','Form'),
          ('formation_spline_full','Spline M,G','Form'),('sensitivity_split0_full','Half 1','Sample'),
          ('sensitivity_split1_full','Half 2','Sample'),('sensitivity_exclude2024_full','Before 2024','Sample'),
          ('sensitivity_exclude2023_2024_full','Before 2023','Sample'),('sensitivity_small_teams_full','Small teams','Sample'),
          ('sensitivity_diversity_full','Diversity','History'),('sensitivity_collaboration_full','Collaboration','History'),
          ('sensitivity_interdisciplinarity_full','Interdisciplinarity','History'),('sensitivity_status_full','Status','History'),
          ('sensitivity_history_depth_full','History depth','History'),('sensitivity_h_reference','h reference','History')]

    def spec_curve(self,ax,mean=.6,labels=False,monochrome=False):
        rows=[];palette={'Form':BLUE,'Sample':TEAL,'History':PURPLE}
        for i,(name,label,group) in enumerate(self.spec_models()):
            q=self.get(name,mean)
            if len(q):
                r=q.iloc[0];ax.errorbar(i,r.estimate,yerr=1.96*r.se,fmt='o',ms=2.6,color=ORANGE if monochrome else palette[group],capsize=1,lw=.65)
                rows.append(dict(model=name,label=label,group=group,**{k:r[k] for k in ['estimate','se','lower','upper']}))
        _zero(ax,'y');ax.set(xlim=(-.6,len(self.spec_models())-.4),ylabel='Formation contrast (pp)')
        if labels:ax.set_xticks(range(len(self.spec_models())),[x[1] for x in self.spec_models()],rotation=65,ha='right',fontsize=6.3)
        else:ax.set(xticks=[0,4,9,15],xticklabels=['1','5','10','16'],xlabel='Specification index')
        return pd.DataFrame(rows)

    def field_evidence(self,compact=False):
        cover=self.tables.get('field_coverage',pd.DataFrame()).sort_values('n_available',ascending=False).copy()
        levels=np.array([.4,.6,.8]);values=np.full((len(cover),3),np.nan)
        fp=self.tables.get('field_contrasts',pd.DataFrame())
        for i,r in enumerate(cover.itertuples()):
            for j,m in enumerate(levels):
                z=fp.loc[fp.model.eq(r.prefix+'_full')&np.isclose(fp.mean_pivot,m)]
                if len(z):values[i,j]=z.estimate.iloc[0]
        if compact:
            rows=np.isfinite(values).any(axis=1);cols=np.isfinite(values).any(axis=0)
            cover=cover.loc[rows];levels=levels[cols];values=values[np.ix_(rows,cols)]
        return cover.reset_index(drop=True),levels,values

    def main4(self):
        fig=self.fig(self.s.main4_height_mm)
        gs=fig.add_gridspec(2,3,height_ratios=[1,1]);self.legend(fig)
        ax=fig.add_subplot(gs[0,0]);self.title(ax,'A','Supported fields')
        complete,all_levels,all_values=self.field_evidence()
        cover,levels,values=self.field_evidence(compact=True)
        rows=[]
        if values.size:
            lim=_finite_limit(values,.5);cm=DIVERGE.copy();cm.set_bad('#EAECEF')
            im=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap=cm,vmin=-lim,vmax=lim)
            for (i,j),v in np.ndenumerate(values):
                ax.text(j,i,_number(v,1) if np.isfinite(v) else '-',ha='center',va='center',fontsize=7,
                        color='white' if np.isfinite(v) and abs(v)>.7*lim else '#606B73')
                rows.append(dict(field_id=cover.field_id.iloc[i],mean_pivot=levels[j],estimate=v))
            ax.set(yticks=range(len(cover)),yticklabels=[self.fields.get(str(x),str(x)) for x in cover.field_id],
                   xticks=range(len(levels)),xticklabels=[f'{m:.1f}' for m in levels],xlabel='Mean distance, M')
            ax.tick_params(axis='y',labelsize=7,length=0);ax.tick_params(axis='x',length=0)
            cb=fig.colorbar(im,ax=ax,pad=.025,fraction=.05,shrink=.88)
            cb.set_label('Formation contrast (pp)',fontsize=7);cb.ax.tick_params(labelsize=6.5)
        else:_empty(ax,'No supported field contrasts\nSee field support atlas')
        ax=fig.add_subplot(gs[0,1]);self.title(ax,'B','Cohort contrasts')
        cohort_years=[]
        for suffix,col,mark in [('base',BLUE,'o'),('full',ORANGE,'s')]:
            q=self.p.loc[self.p.model.str.fullmatch(r'cohort_\d{4}_'+suffix,na=False)&np.isclose(self.p.mean_pivot,.6)].copy()
            q['year']=q.model.str.extract(r'cohort_(\d{4})_',expand=False).astype(int);q=q.sort_values('year')
            if len(q):
                ax.errorbar(q.year,q.estimate,yerr=1.96*q.se,fmt=mark+'-',ms=3,color=col,capsize=1.2)
                cohort_years.extend(q.year.tolist())
        _zero(ax,'y');ax.set(xlabel='Cohort start year',ylabel='Formation contrast (pp)')
        if cohort_years:
            years=sorted(set(cohort_years));ax.set_xlim(min(years)-1,max(years)+1)
            if len(years)<=4:ax.set_xticks(years)
            else:ax.xaxis.set_major_locator(ticker.MaxNLocator(4,integer=True))
        ax=fig.add_subplot(gs[1,:2]);self.title(ax,'D','Specification curve')
        spec=self.spec_curve(ax,labels=False,monochrome=True);ax.tick_params(axis='x',labelsize=6.4)
        self.source('Fig4D',spec)
        for ax,stem,letter,title,ylabel in [(fig.add_subplot(gs[0,2]),'citation','C','Citation recognition','Percentile-point contrast'),
            (fig.add_subplot(gs[1,2]),'recognition_novel','E','Recognition | N','Probability-point contrast')]:
            self.title(ax,letter,title);self.profile_lines(ax,(stem+'_base',stem+'_full'),ylabel)
        self.source('Fig4A',pd.DataFrame(rows));self.source('Fig4CE',self.p.loc[self.p.model.str.startswith(('citation_','recognition_novel_'))])
        self.save(fig,'Fig4_generality_and_recognition',
            f'A, Reference-conditioned field-specific estimates. The complete {len(complete)} by {len(all_levels)} field/M grid contains {np.isfinite(all_values).sum()} supported estimates. '
            f'The main panel retains {len(cover)} fields and {len(levels)} M levels with at least one supported estimate; rows and columns containing no supported estimate are moved to FigS10. '
            'Inclusion depends on common support, not significance or effect magnitude. Partial missing cells remain gray/hyphens. Field order follows available sample size. '
            'B, Supported cohort contrasts at M=0.6. C, Citation-percentile associations. D, Numbered specifications in fixed model-definition order; full labels appear in FigS5 and source data. '
            'E, High recognition conditional on new-pair formation. Each baseline/conditioned pair shares an actual estimation sample; samples differ across outcomes and specifications. '
            'Error bars are existing pointwise 95% intervals. Full field intervals appear in FigS6; all-field coverage, sample loss and support reasons appear in FigS10.',main=True)


    def field_support_table(self):
        cover,levels,_=self.field_evidence();checks=self.tables.get('profile_support',pd.DataFrame());rows=[]
        for field in cover.itertuples():
            q=checks.loc[checks.model.eq(field.prefix+'_full')]
            for m in levels:
                z=q.loc[np.isclose(q.mean_pivot,m)]
                item=dict(field_id=str(field.field_id),prefix=field.prefix,mean_pivot=m,
                          n_available=field.n_available,n_estimated=field.n_estimated,
                          model_status=field.status,n_min_endpoint=np.nan,min_context_mass=np.nan)
                if len(z):
                    r=z.iloc[0];reason=str(r.reason) if pd.notna(r.reason) else ''
                    ok=str(r.supported).lower() in ['true','1']
                    state=0 if ok else 1 if reason=='sparse M/G endpoints' else 2 if reason=='no common exact team size' else 3 if reason=='insufficient common reference contexts' else 5
                    item.update(n_min_endpoint=min(r.n_from,r.n_to),min_context_mass=min(r.context_mass_here,r.context_mass_reference))
                else:reason=str(field.reason);state=4 if field.status=='not_estimated' else 5
                item.update(state=state,reason=reason);rows.append(item)
        return pd.DataFrame(rows)

    def field_support_atlas(self):
        cover,levels,_=self.field_evidence();q=self.field_support_table()
        ids=cover.field_id.astype(str).tolist();pos=np.arange(len(ids));labels=[self.fields.get(x,x) for x in ids]
        fig=self.fig(max(118,37+5.4*len(ids)));axs=fig.subplots(1,3,sharey=True)
        state_labels=['Supported','Sparse endpoint','No common size','Weak context','Not estimated','No export']
        codes=['S','D','T','C','N','?'];palette=[TEAL,'#DAB96C','#B7A0C9','#B4C5D4','#E9ECEF','#F0B6A7']
        fig.legend(handles=[Rectangle((0,0),1,1,color=c,label=f'{code}: {label}') for code,label,c in zip(codes,state_labels,palette)],
                   loc='outside upper center',ncol=3,frameon=False,fontsize=6.2,columnspacing=1)
        ax=axs[0];self.title(ax,'A','Field samples')
        for i,r in enumerate(cover.itertuples()):
            if np.isfinite(r.n_estimated) and r.n_estimated>0:
                ax.plot([r.n_estimated,r.n_available],[i,i],color='#BAC4CB',lw=1)
                ax.scatter(r.n_estimated,i,s=10,color=ORANGE,zorder=3)
            ax.scatter(r.n_available,i,s=10,color=BLUE,zorder=3)
        ax.set_xscale('log');ax.set(yticks=pos,yticklabels=labels,ylim=(len(ids)-.5,-.5),xlabel='Papers (log scale)')
        counts_axis=np.r_[cover.n_available.to_numpy(float),cover.n_estimated.to_numpy(float)]
        counts_axis=counts_axis[np.isfinite(counts_axis)&(counts_axis>0)]
        if len(counts_axis):
            ticks=np.geomspace(counts_axis.min(),counts_axis.max(),3)
            ax.set_xticks(np.unique(ticks));ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_:_compact_n(x)))
            ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.legend(handles=[Line2D([],[],marker='o',color=BLUE,lw=0,ms=3,label='Available'),Line2D([],[],marker='o',color=ORANGE,lw=0,ms=3,label='Estimated')],frameon=False,fontsize=6,loc='lower right')
        states=q.pivot(index='field_id',columns='mean_pivot',values='state').reindex(index=ids,columns=levels)
        ax=axs[1];self.title(ax,'B','All-field support')
        ax.imshow(states.to_numpy(float),aspect='auto',cmap=colors.ListedColormap(palette),vmin=-.5,vmax=5.5,interpolation='nearest')
        for (i,j),v in np.ndenumerate(states.to_numpy(int)):ax.text(j,i,codes[v],ha='center',va='center',fontsize=6.2,color='#15232B')
        counts=q.pivot(index='field_id',columns='mean_pivot',values='n_min_endpoint').reindex(index=ids,columns=levels).to_numpy(float)
        mass=q.pivot(index='field_id',columns='mean_pivot',values='min_context_mass').reindex(index=ids,columns=levels).to_numpy(float)
        ax=axs[2];self.title(ax,'C','Endpoint / context')
        cm=plt.get_cmap('viridis').copy();cm.set_bad('#E9ECEF')
        im=ax.imshow(np.ma.masked_invalid(mass),aspect='auto',cmap=cm,vmin=0,vmax=1)
        for (i,j),v in np.ndenumerate(counts):
            if np.isfinite(v):
                text=_compact_n(v)+('\n'+f'{mass[i,j]:.2f}' if np.isfinite(mass[i,j]) else '\n-')
                ax.text(j,i,text,ha='center',va='center',fontsize=6.1,color='white' if mass[i,j]<.45 else '#15232B',linespacing=1.05)
            else:ax.text(j,i,'-',ha='center',va='center',fontsize=6.4,color=GREY)
        cb=fig.colorbar(im,ax=ax,pad=.02,fraction=.04,shrink=.9);cb.set_label('Common-context fraction',fontsize=6.5);cb.ax.tick_params(labelsize=6)
        for j,ax in enumerate(axs):
            ax.tick_params(axis='y',labelsize=6.3,length=0)
            if j:ax.set(xticks=range(len(levels)),xticklabels=[f'{m:.1f}' for m in levels],xlabel='Mean distance, M');ax.tick_params(labelleft=False)
        self.source('FigS10_all_field_support',q)
        self.save(fig,'FigS10_field_support_and_coverage',
            'All fields and all predefined M targets are retained. A shows available and actual estimation counts. B gives the first failed support criterion (codes in the legend). In C, color and the lower line give the minimum common-context fraction across endpoints; the upper line gives the minimum endpoint count, with k denoting thousands. Missing values are dashes. Exact counts and fractions are retained in source data.')


    def kde(self,values,grid):
        x=np.asarray(values,float);x=x[np.isfinite(x)]
        if len(x)<5 or np.std(x)<1e-10:return np.zeros(len(grid))
        if len(x)>self.s.kde_max_points:x=self.rng.choice(x,self.s.kde_max_points,replace=False)
        return gaussian_kde(x)(grid)

    def distributions(self):
        fig=self.fig(118);axs=fig.subplots(2,3);summaries=[]
        variables=[('citation_percentile','Citation percentile',False),('semantic_distance','Semantic distance',False),
                   ('novel_share_pct','New-pair share > 0',True),('n_reference_sources','Source count',True),
                   ('reference_source_hhi','Source HHI',False),('grounding_gap','Configuration gap',False)]
        for k,(ax,(key,title,log)) in enumerate(zip(axs.flat,variables)):
            self.title(ax,chr(65+k),title)
            allx=self.plot[key].dropna().to_numpy(float)
            if key=='novel_share_pct':allx=allx[allx>0]
            if log:allx=np.log1p(allx)
            if not len(allx):_empty(ax);continue
            lo,hi=np.quantile(allx,[.001,.999]);grid=np.linspace(lo,hi,220)
            for group,col in enumerate(REGIME_COLORS):
                raw=self.plot.loc[self.plot.regime.eq(group),key].dropna().to_numpy(float)
                zero=(raw==0).mean() if len(raw) else np.nan;available_n=len(raw)
                if key=='novel_share_pct':raw=raw[raw>0]
                x=np.log1p(raw) if log else raw
                if not len(x):continue
                base=3-group;h=self.kde(x,grid);h=.62*h/h.max() if h.max()>0 else h
                ax.fill_between(grid,base+.07,base+.07+h,color=col,alpha=.5,lw=0)
                a,b,c=np.quantile(x,[.25,.5,.75]);summaries.append(dict(variable=key,regime=group,n_available=available_n,n_display_distribution=len(x),q25=a,median=b,q75=c,zero_fraction=zero,transform='log1p' if log else 'identity'));ax.plot([a,c],[base-.03,base-.03],lw=2.2,color=col,solid_capstyle='butt');ax.scatter([b],[base-.03],s=5,color='#18222B',zorder=3)
                ix=self.rng.choice(len(x),min(self.s.show_points_per_group,len(x)),replace=False)
                ax.scatter(x[ix],base-.12-.12*self.rng.random(len(ix)),s=1.6,alpha=.2,color=col,rasterized=True,edgecolors='none')
                if key=='novel_share_pct':ax.text(.98,base+.2,f'zero {zero:.0%}',ha='right',transform=ax.get_yaxis_transform(),fontsize=6.1)
            ax.set(yticks=range(4),yticklabels=REGIME_LABELS[::-1] if k%3==0 else [],ylim=(-.4,3.85),
                   xlabel='log(1 + value)' if log else title)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        self.source('FigS1_quantiles',pd.DataFrame(summaries))
        self.save(fig,'FigS1_raincloud_ridgeline',
            'Hybrid ridgeline/raincloud displays of the uniform plotting sample. Filled half densities use at most the configured number of deterministic subsampled observations per group; medians and interquartile ranges use all available plotting observations. '
            'Thin point clouds show display subsamples. Descriptive profiles classify M at 0.5 and minimum member distance at 0.25. '
            'C shows only positive new-pair shares, with zero mass annotated separately. Log transforms are labeled. Density grids cover the pooled 0.1st to 99.9th percentiles; quantile marks and sampled points retain their actual coordinates.')

    def alluvial(self,ax):
        f=self.tables['alluvial_composition'].copy();f['profile']=f.regime.map(dict(enumerate(REGIME_LABELS)))
        stages=['profile','novelty_class','recognition_class']
        orders=[REGIME_LABELS,['No new source pair','New source pair'],['Other citation ranks','High recognition']]
        short=[dict(zip(REGIME_LABELS,['Low M / low min','Low M / high min','High M / low min','High M / high min'])),
               {'No new source pair':'No new pair','New source pair':'New pair'},
               {'Other citation ranks':'Other ranks','High recognition':'High recognition'}]
        total=f.papers.sum();scale=.76/total;nodes={};gap=.045;xs=[.06,.5,.94]
        for j,key in enumerate(stages):
            heights=f.groupby(key).papers.sum();start=(1-(.76+gap*(len(orders[j])-1)))/2
            for label in orders[j]:
                height=float(heights.get(label,0))*scale;nodes[j,label]=[start,height,0.];start+=height+gap
        for row in f.sort_values(['regime','novelty_class','recognition_class']).itertuples():
            labels=[row.profile,row.novelty_class,row.recognition_class];h=row.papers*scale;ys=[]
            for j,label in enumerate(labels):node=nodes[j,label];ys.append(node[0]+node[2]);node[2]+=h
            for j in range(2):
                x0,x1=xs[j]+.007,xs[j+1]-.007;y0,y1=ys[j],ys[j+1];dx=(x1-x0)*.45
                verts=[(x0,y0),(x0+dx,y0),(x1-dx,y1),(x1,y1),(x1,y1+h),(x1-dx,y1+h),(x0+dx,y0+h),(x0,y0+h),(x0,y0)]
                codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MPath(verts,codes),facecolor=REGIME_COLORS[int(row.regime)],edgecolor='none',alpha=.42))
        for (j,label),(y,h,_) in nodes.items():
            ax.add_patch(Rectangle((xs[j]-.007,y),.014,h,facecolor='#4F606D',edgecolor='none'))
            ax.text(xs[j]+(.012 if j<2 else -.012),y+h/2,short[j][label],ha='left' if j<2 else 'right',va='center',fontsize=6.8,
                    bbox={'facecolor':'white','alpha':.75,'edgecolor':'none','pad':.4})
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')

    def composition(self):
        fig=self.fig(106);gs=fig.add_gridspec(2,3,height_ratios=[1.05,1])
        ax=fig.add_subplot(gs[0,:]);self.title(ax,'A','Cross-sectional composition');self.alluvial(ax)
        ax=fig.add_subplot(gs[1,0]);self.title(ax,'B','Unadjusted cohort profiles')
        q=self.tables['cohort_composition'];q=q.loc[q.outcome.eq('any_novel')]
        for k,col in enumerate(REGIME_COLORS):
            z=q.loc[q.regime.eq(k)].sort_values('cohort');ax.plot(z.cohort,100*z['mean'],marker='o',ms=2.3,color=col,label=REGIME_LABELS[k])
        ax.set(xlabel='Cohort start year',ylabel='New-pair papers (%)',xticks=[1990,2005,2020])
        ax.legend(frameon=False,fontsize=6.2,ncol=2,loc='upper left',handlelength=1.1,columnspacing=.8)
        ax=fig.add_subplot(gs[1,1:]);self.title(ax,'C','Within-profile composition')
        f=self.tables['alluvial_composition'].copy();f['state']=f.novelty_class.eq('New source pair').astype(int)*2+f.recognition_class.eq('High recognition').astype(int)
        tab=f.pivot_table(index='regime',columns='state',values='papers',aggfunc='sum',fill_value=0).reindex(index=range(4),columns=range(4),fill_value=0)
        fractions=tab.div(tab.sum(axis=1),axis=0);left=np.zeros(4)
        labels=['No pair / other rank','No pair / high rank','New pair / other rank','New pair / high rank']
        cols=['#DCE2E6','#8FA6B6','#A4C8BD','#277C68']
        for j in range(4):ax.barh(range(4),fractions[j],left=left,color=cols[j],height=.62,label=labels[j]);left+=fractions[j].to_numpy()
        ax.set(yticks=range(4),yticklabels=['1','2','3','4'],xlabel='Fraction of papers',ylabel='Descriptive profile',xlim=(0,1))
        ax.legend(frameon=False,fontsize=6.1,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.25),handlelength=1,columnspacing=.6)
        self.source('FigS2C',fractions.reset_index());self.source('FigS2A',self.tables['alluvial_composition'])
        self.save(fig,'FigS2_alluvial_composition',
            'A, Alluvial composition of papers across descriptive configuration, new-source-pair status, and recognition categories. The ribbons show cross-sectional counts. '
            'B, Raw new-pair proportions in five-year publication cohorts by the same four profiles. C, Within-profile proportions of the four novelty/recognition combinations; profile numbering follows the order in A. '
            'All summaries use existing full-sample count tables and exclude unknown binary outcomes. High recognition is defined against the input core normalization, not against the selected plotting sample.')

    def surface_atlas(self):
        fig=self.fig(208);axs=fig.subplots(5,3)
        scales=self.tables['common_measure_scales'].set_index('outcome')['sd'].to_dict()
        for i,(y,label) in enumerate(zip(MEASURES,SHORT_MEASURES)):
            sd=scales.get(y,np.nan)
            if not np.isfinite(sd) or sd<=0:raise ValueError(f'Invalid common-sample SD: {y}')
            panels=[]
            for suffix in ['base','full']:
                q=self.sur.loc[self.sur.model.eq('common_'+y+'_'+suffix)].copy()
                q['std_contrast']=q.contrast/sd;panels.append(q)
            limit=_finite_limit(np.concatenate([q.loc[_bool(q.supported),'std_contrast'].to_numpy(float) for q in panels]),.02)
            images=[]
            for j,q in enumerate(panels):
                ax=axs[i,j];self.title(ax,chr(65+3*i+j),label+(' / baseline' if j==0 else ' / conditioned'))
                images.append(self.mesh(fig,ax,q,'std_contrast','',DIVERGE,-limit,limit,True,colorbar=False))
                ax.set(ylabel='Gap, G' if j==0 else '',xlabel='Mean distance, M' if i==4 else '')
                if i<4:ax.tick_params(labelbottom=False)
            if images[0] is not None:
                cb=fig.colorbar(images[0],ax=axs[i,:2],pad=.015,fraction=.025,shrink=.88)
                cb.set_label('Contrast / SD',fontsize=6.5);cb.ax.tick_params(labelsize=6)
            q=panels[1].copy();q['std_slope']=q.gap_slope/sd
            slope_limit=_finite_limit(q.loc[_bool(q.supported),'std_slope'],.01)
            ax=axs[i,2];self.title(ax,chr(67+3*i),label+' / local G slope')
            self.mesh(fig,ax,q,'std_slope','SD / +0.1 G',DIVERGE,-slope_limit,slope_limit,True)
            ax.set(xlabel='Mean distance, M' if i==4 else '')
            if i<4:ax.tick_params(labelbottom=False)
            self.source('FigS3_'+y,pd.concat([panels[0].assign(spec='baseline'),q.assign(spec='conditioned')],ignore_index=True))
        self.save(fig,'FigS3_common_sample_surface_atlas',
            'Rows show the five outcomes on the same actual estimation papers. The first two columns show baseline and reference-conditioned contrasts divided by the same outcome SD, with a shared color scale within each row. The third column shows the exported conditioned local slope per +0.1 G, divided by that outcome SD, on its own labeled scale. Unsupported locations remain masked; contrasts use the exported model-specific reference coordinate.')

    def uncertainty(self):
        fig=self.fig(158);axs=fig.subplots(3,3)
        full=self.sur.loc[self.sur.model.eq('formation_full')].copy();base=self.sur.loc[self.sur.model.eq('formation_base')].copy()
        valid=full.loc[_bool(full.supported),'contrast'];limit=_finite_limit(valid)
        self.title(axs[0,0],'A','Conditioned estimate');self.mesh(fig,axs[0,0],full,'contrast','Contrast (pp)',DIVERGE,-limit,limit,True)
        self.title(axs[0,1],'B','Pointwise uncertainty');self.mesh(fig,axs[0,1],full,'se','SE (pp)','magma',supported=True)
        full['resolved']=np.where((full.lower>0)|(full.upper<0),1.,0.)
        self.title(axs[0,2],'C','CI excludes zero');self.mesh(fig,axs[0,2],full,'resolved','Indicator','Greys',0,1,True)
        for ax,m,letter in zip(axs[1],[.4,.6,.8],['D','E','F']):
            self.title(ax,letter,f'Gap slope: M~ {m:.1f}')
            for z,col in [(base,BLUE),(full,ORANGE)]:
                mv=z.mean_pivot.iloc[np.argmin(abs(z.mean_pivot.to_numpy()-m))]
                q=z.loc[z.mean_pivot.eq(mv)].sort_values('grounding_gap').copy()
                ok=_bool(q.supported);effect=q.gap_slope.where(ok).to_numpy();se=q.gap_slope_se.to_numpy()
                ax.plot(q.grounding_gap,effect,color=col,lw=.95)
                ax.fill_between(q.grounding_gap,effect-1.96*se,effect+1.96*se,color=col,alpha=.16,lw=0)
            _zero(ax,'y');ax.set(xlabel='Gap, G',ylabel='Formation pp / +0.1 G' if letter=='D' else '',xlim=(0,.65))
        for ax,key,letter,title in [(axs[2,0],'context_mass_here','G','Context mass: cell'),(axs[2,1],'context_mass_reference','H','Context mass: ref')]:
            self.title(ax,letter,title);q=full.copy();q.loc[q.n_cell<self.cfg.get('surface_cell_min',100),key]=np.nan
            self.mesh(fig,ax,q,key,'Fraction','viridis',0,1);ax.set(xlabel='Mean distance, M',ylabel='Gap, G' if letter=='G' else '')
        ax=axs[2,2];self.title(ax,'I','Estimation domain')
        q=full.copy();q['domain']=np.where(_bool(q.supported),2,np.where(q.n_cell>=self.cfg.get('surface_cell_min',100),1,0))
        self.mesh(fig,ax,q,'domain','',colors.ListedColormap(['#F3F3F3','#C0CBD2',TEAL]),0,2,ylim=.7,colorbar=False)
        ax.legend(handles=[Line2D([],[],marker='s',ms=4,lw=0,color=c,label=t) for c,t in [('#E1E4E6','Sparse'),('#C0CBD2','Excluded'),(TEAL,'Supported')]],loc='upper left',frameon=False,fontsize=6.1,handletextpad=.3,labelspacing=.2)
        ax.set(xlabel='Mean distance, M')
        self.legend(fig)
        self.source('FigS4',full)
        self.save(fig,'FigS4_uncertainty_and_support',
            'A-C, Reference-conditioned formation contrast, its standard error, and whether its pointwise 95% interval excludes zero. C is not a simultaneous significance map. '
            'D-F, Pointwise local derivative intervals for a +0.1 G increment at the closest exported M grid values to 0.4, 0.6, and 0.8. Unsupported locations are left as gaps. '
            'G-H, Context coverage fractions, with sparse cells masked. I distinguishes sparse cells, density-eligible cells excluded by other support checks, and the final supported domain. '
            'Support and uncertainty come from the existing R exports; no new regression is fitted.')

    def specification_atlas(self):
        fig=self.fig(110);gs=fig.add_gridspec(2,1,height_ratios=[1.25,1])
        ax=fig.add_subplot(gs[0]);self.title(ax,'A','Reference-conditioned specification curve')
        rows=self.spec_curve(ax,labels=True);ax.tick_params(axis='x',labelbottom=False)
        handles=[Line2D([],[],color=c,marker='o',lw=0,ms=3,label=g) for g,c in [('Form',BLUE),('Sample',TEAL),('History',PURPLE)]]
        ax.legend(handles=handles,frameon=False,ncol=3,loc='upper right',fontsize=6.7)
        ax=fig.add_subplot(gs[1]);self.title(ax,'B','Specification components')
        specs=self.spec_models();cf=self.tables['coefficients'];flags=[]
        for name,_,_ in specs:
            terms=cf.loc[cf.model.eq(name),'term'].astype(str)
            flags.append([terms.str.contains('log_n_sources',regex=False).any() or name=='formation_structural_full',
                terms.str.contains('reference_source_hhi',regex=False).any() or name=='formation_structural_full',
                terms.str.contains('mean_ref_age',regex=False).any(),terms.str.contains('log_ref_citation_base',regex=False).any(),
                name=='formation_venue_full',name=='formation_structural_full',name=='formation_spline_full',
                name.startswith('sensitivity_')])
        a=np.array(flags,dtype=int).T
        ax.imshow(a,aspect='auto',cmap=colors.ListedColormap(['#EFF1F2','#4F687B']),vmin=0,vmax=1)
        ax.set(yticks=range(8),yticklabels=['Source count','Source HHI','Reference age','Citation base','Venue-year FE','Structural FE','Spline M,G','Sensitivity subset/control'],
               xticks=range(len(specs)),xticklabels=[x[1] for x in specs])
        ax.tick_params(axis='x',rotation=65,labelsize=6.4,length=0);ax.tick_params(axis='y',length=0)
        for label in ax.get_xticklabels():label.set_ha('right')
        self.source('FigS5A',rows)
        self.source('FigS5B',pd.DataFrame(a.T,columns=['source_count','hhi','age','visibility','venue_fe','structural_fe','spline_mg','sensitivity']).assign(model=[x[0] for x in specs]))
        self.save(fig,'FigS5_specification_curve',
            'A, Reference-conditioned profile contrasts at M=0.6, in a fixed model-definition order, with pointwise intervals across the related specifications. '
            'B, Presence of reference-structure components and model features. Count/HHI indicators include the corresponding structural fixed effects. Sensitivity denotes an additional sample restriction or historical-control variant. '
            'Indicators are obtained from the fitted coefficient terms and the documented named model forms. No estimate is selected or omitted because of significance; unsupported profiles remain absent.')

    def field_atlas(self):
        fp=self.tables['field_contrasts'];cover=self.tables['field_coverage'].sort_values('n_available',ascending=False)
        ids=cover.loc[cover.field_id.astype(str).isin(set(fp.field_id.astype(str))),'field_id'].astype(str).tolist()
        means=sorted(fp.mean_pivot.dropna().unique())
        if not ids or not means:return
        positions={v:i for i,v in enumerate(ids)}
        fig=self.fig(max(72,min(140,34+5.5*len(ids))))
        axs=fig.subplots(1,len(means),sharey=True,squeeze=False)[0];self.legend(fig)
        for j,m in enumerate(means):
            ax=axs[j];self.title(ax,chr(65+j),f'Mean distance M={m:.1f}')
            for suffix,color,off,marker in [('base',BLUE,-.13,'o'),('full',ORANGE,.13,'s')]:
                z=fp.loc[fp.model.str.endswith('_'+suffix)&np.isclose(fp.mean_pivot,m)].copy()
                z['pos']=z.field_id.astype(str).map(positions)+off
                ax.errorbar(z.estimate,z.pos,xerr=1.96*z.se,fmt=marker,ms=2.8,color=color,capsize=1.2,lw=.8)
            observed=set(fp.loc[np.isclose(fp.mean_pivot,m),'field_id'].astype(str))
            for field,i in positions.items():
                if field not in observed:ax.text(.97,i,'-',transform=ax.get_yaxis_transform(),ha='right',va='center',color=GREY,fontsize=6.6)
            _zero(ax);ax.set(yticks=range(len(ids)),yticklabels=[self.fields.get(x,x) for x in ids],ylim=(len(ids)-.5,-.5),xlabel='Formation contrast (pp)')
            ax.tick_params(axis='y',labelsize=6.8);ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        self.source('FigS6',fp)
        self.save(fig,'FigS6_field_interval_atlas',
            'Each panel is a supported mean-distance target. Baseline and reference-conditioned estimates are overlaid with small vertical offsets; intervals are pointwise 95%. Only fields and targets with an exported supported contrast are included. FigS10 retains the full field/target domain. Samples match within each field/model pair.')

    def cohort_atlas(self):
        fig=self.fig(122);axs=fig.subplots(2,3);allvalues=[];panels=[]
        for suffix in ['base','full']:
            for m in [.4,.6,.8]:
                q=self.sur.loc[self.sur.model.str.fullmatch(r'cohort_\d{4}_'+suffix,na=False)].copy()
                if q.empty:panels.append((pd.DataFrame(),m));continue
                mv=q.mean_pivot.iloc[np.argmin(abs(q.mean_pivot.to_numpy()-m))];q=q.loc[q.mean_pivot.eq(mv)].copy()
                q.loc[~_bool(q.supported),'gap_slope']=np.nan;q['cohort']=q.model.str.extract(r'cohort_(\d{4})_',expand=False).astype(int)
                panels.append((q,mv));allvalues.extend(q.gap_slope.dropna().tolist())
        lim=_finite_limit(allvalues,.1)
        valid_g=[q.loc[q.gap_slope.notna(),'grounding_gap'].max() for q,_ in panels if len(q) and q.gap_slope.notna().any()]
        xmax=max(valid_g)+.025 if valid_g else .5
        for i,(ax,(q,mv)) in enumerate(zip(axs.flat,panels)):
            self.title(ax,chr(65+i),f'{"Baseline" if i<3 else "Conditioned"}, M={mv:.3f}')
            if q.empty:_empty(ax);continue
            t=q.pivot(index='cohort',columns='grounding_gap',values='gap_slope').reindex(index=range(1990,2025,5))
            x=t.columns.to_numpy(float);y=t.index.to_numpy(float)
            im=ax.pcolormesh(x,y,t.to_numpy(float),cmap=DIVERGE,vmin=-lim,vmax=lim,shading='nearest')
            ax.set_facecolor('#EFF1F2');ax.set(xlim=(0,xmax),xticks=np.arange(0,xmax+.001,.1),yticks=[1990,2005,2020],xlabel='Gap, G',ylabel='Cohort start year' if i%3==0 else '')
            cb=fig.colorbar(im,ax=ax,pad=.02,fraction=.05,shrink=.9);cb.set_label('pp / +0.1 G',fontsize=6.8);cb.ax.tick_params(labelsize=6.3)
        self.source('FigS7',pd.concat([q.assign(target_m=m) for q,m in panels if len(q)],ignore_index=True))
        self.save(fig,'FigS7_cohort_response_atlas',
            'Local formation-probability slopes by publication cohort, configuration gap, and the closest exported mean-unfamiliarity grid values to the three targets. '
            'Color limits are shared across all six panels; gray cells are unsupported. Five-year rows group publication cohorts and impose no future observation window. '
            'These pointwise model summaries do not demonstrate cohort differences without an additional formal comparison.')

    def static3d(self):
        fig=self.fig(122);gs=fig.add_gridspec(2,2);names=['formation_base','formation_full'];panels=[]
        for name in names:
            z=self.sur.loc[self.sur.model.eq(name)].copy();z.loc[~_bool(z.supported),'contrast']=np.nan
            t=z.pivot(index='grounding_gap',columns='mean_pivot',values='contrast').sort_index().sort_index(axis=1);panels.append((z,t))
        lim=_finite_limit(np.concatenate([t.to_numpy(float).ravel() for _,t in panels]))
        for j,((z,t),title) in enumerate(zip(panels,['Baseline','Reference-conditioned'])):
            ax=fig.add_subplot(gs[0,j],projection='3d');self.title(ax,chr(65+j),title)
            X,Y=np.meshgrid(t.columns.to_numpy(float),t.index.to_numpy(float));Z=t.to_numpy(float)
            ax.plot_surface(X,Y,Z,cmap=DIVERGE,vmin=-lim,vmax=lim,linewidth=.15,edgecolor='#9CA8AF',antialiased=True)
            ax.set(xlim=(0,1),ylim=(0,.7),zlim=(-lim,lim),xlabel='M',ylabel='G',zlabel='Contrast (pp)')
            ax.set_proj_type('ortho');ax.view_init(elev=28,azim=-125);ax.tick_params(labelsize=6.3,pad=0)
            ax.set_xticks([0,.5,1]);ax.set_yticks([0,.3,.6]);ax.zaxis.set_major_locator(ticker.MaxNLocator(3))
            for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.set_facecolor((1,1,1,0))
            ax.xaxis.labelpad=0;ax.yaxis.labelpad=0;ax.zaxis.labelpad=1
            ax=fig.add_subplot(gs[1,j]);self.title(ax,chr(67+j),title+' contours')
            self.mesh(fig,ax,z,'contrast','Contrast (pp)',DIVERGE,-lim,lim,True)
            if np.isfinite(Z).sum()>5:
                levels=np.linspace(-lim,lim,7);ax.contour(X,Y,np.ma.masked_invalid(Z),levels=levels,colors='#45535D',linewidths=.35,alpha=.65)
            ax.set(xlabel='Mean distance, M',ylabel='Gap, G' if j==0 else '')
        self.save(fig,'FigS8_static3D_and_contours',
            'Orthographic three-dimensional and two-dimensional contour views of the same baseline and reference-conditioned formation contrasts. '
            'Both 3D panels use identical elevation, azimuth, z limits and color limits. Unsupported cells remain gaps; no surface is interpolated across them. '
            'The 2D panels provide quantitative contour views. Colors and heights encode fitted contrasts.')

    def correlation_atlas(self):
        fig=self.fig(126);axs=fig.subplots(2,3);ax=axs[0,0];self.title(ax,'A','Measure correlations')
        keys=['mean_pivot','grounding_gap','n_reference_sources','reference_source_hhi','novel_share_pct','semantic_distance','prior_h_gini','prior_output_log_sd','log_min_history_refs','log_mean_history_refs']
        short=['M','G','Sources','HHI','Pair share','Semantic','h Gini','Output SD','Min history','Mean history']
        values=self.plot.reindex(columns=keys)
        corr=values.corr(method='spearman',min_periods=100).to_numpy(float)
        pairs_table=[]
        for i,a in enumerate(keys):
            for j,b in enumerate(keys[i+1:],start=i+1):
                pairs_table.append(dict(variable1=a,variable2=b,n=int(values[[a,b]].notna().all(axis=1).sum()),spearman=corr[i,j]))
        correlations=pd.DataFrame(pairs_table)
        masked=np.ma.array(corr,mask=np.triu(np.ones_like(corr,dtype=bool),k=1))
        im=ax.imshow(masked,cmap=DIVERGE,vmin=-1,vmax=1,aspect='auto')
        for i in range(len(keys)):
            for j in range(i+1):
                if np.isfinite(corr[i,j]):ax.text(j,i,f'{corr[i,j]:.1f}'.replace('0.','.'),ha='center',va='center',fontsize=6.0,color='white' if abs(corr[i,j])>.7 else '#26343E')
        ax.set(xticks=range(10),xticklabels=range(1,11),yticks=range(10),yticklabels=[f'{i+1} {x}' for i,x in enumerate(short)])
        ax.tick_params(length=0,labelsize=6.4);cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.04,pad=.025);cb.set_label('Spearman correlation',fontsize=6.8);cb.ax.tick_params(labelsize=6.3)
        pairs=[('mean_pivot','n_reference_sources','B','Experience / breadth'),('grounding_gap','reference_source_hhi','C','Gap / concentration'),
               ('grounding_gap','log_min_history_refs','D','Gap / history depth'),('grounding_gap','prior_h_gini','E','Gap / status')]
        labels={'mean_pivot':'M','grounding_gap':'G','n_reference_sources':'Source count','reference_source_hhi':'HHI','log_min_history_refs':'log(1 + min history refs)','prior_h_gini':'Prior h Gini'}
        for (x,y,letter,title),ax in zip(pairs,[axs[0,1],axs[0,2],axs[1,0],axs[1,1]]):
            self.title(ax,letter,title);q=self.plot[[x,y]].dropna()
            im=ax.hexbin(q[x],q[y],gridsize=25,mincnt=5,bins='log',yscale='log' if y=='n_reference_sources' else 'linear',cmap='viridis',linewidths=0,rasterized=True)
            ax.set(xlabel=labels[x],ylabel=labels[y]+(' (log)' if y=='n_reference_sources' else ''));ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
            if y=='n_reference_sources':ax.yaxis.set_major_locator(ticker.LogLocator(base=10,numticks=3))
            else:ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            cb=fig.colorbar(im,ax=ax,pad=.02,fraction=.05,shrink=.85);cb.set_label('Papers (log scale)',fontsize=6.5);cb.ax.tick_params(labelsize=6.1)
        ax=axs[1,2];self.title(ax,'F','Gap correlations')
        indices=[2,3,4,5,6,8];vals=[corr[1,j] for j in indices]
        ax.hlines(range(len(indices)),0,vals,color='#AAB5BD',lw=.8);ax.scatter(vals,range(len(indices)),s=13,color=BLUE)
        ax.set(yticks=range(len(indices)),yticklabels=[short[j] for j in indices],ylim=(len(indices)-.5,-.5),xlim=(-1,1),xticks=[-1,0,1],xlabel='Spearman correlation with G');_zero(ax)
        self.source('FigS9A',correlations)
        self.save(fig,'FigS9_measure_relationships',
            'A,F, Pairwise Spearman correlations from the uniform plotting sample, using pair-specific available observations; per-pair sample sizes are retained in the source table. '
            'B-E, Hexagonal count displays; bins with fewer than five observations are omitted. F uses a fixed set of variables, not a significance-ranked selection. '
            'These unadjusted displays describe pairwise measurement relationships.')

    def contacts(self,kind,columns):
        items=[r for r in self.records if r['kind']==kind]
        if not items:return None
        thumb_w=480;label_h=28;gap=18
        thumb_h=math.ceil(thumb_w*max(r['height_mm']/r['width_mm'] for r in items)) if kind=='main' else 490
        canvas=Image.new('RGB',(columns*thumb_w+(columns+1)*gap,math.ceil(len(items)/columns)*(thumb_h+label_h+gap)+gap),'white')
        draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(items):
            im=Image.open(self.out/r['preview']).convert('RGB');im.thumbnail((thumb_w,thumb_h),Image.Resampling.LANCZOS)
            x=gap+(i%columns)*(thumb_w+gap);y=gap+(i//columns)*(thumb_h+label_h+gap)
            draw.text((x,y),r['name'],fill='#1C2A35');canvas.paste(im,(x+(thumb_w-im.width)//2,y+label_h))
        path=self.out/f'{kind}_contact_sheet.png';canvas.save(path);return path

    def gallery(self):
        cards=[]
        for r,caption in zip(self.records,self.captions):
            directory='main' if r['kind']=='main' else 'supplement'
            cards.append(f'<article data-kind="{r["kind"]}"><h2>{html.escape(r["name"])}</h2><a href="{r["pdf"]}"><img loading="lazy" src="{r["preview"]}" alt="{html.escape(r["name"])}"></a><p><a href="{r["pdf"]}">Vector PDF</a> | <a href="{directory}/{r["name"]}.png">600-dpi PNG</a>'
                         +(f' | <a href="{directory}/{r["name"]}.svg">Editable SVG</a>' if self.s.export_svg else '')+'</p><details><summary>Caption</summary><p>'+html.escape(caption.split('\n\n',1)[1])+'</p></details></article>')
        document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SciSciNet2 publication plates</title><style>body{font:15px system-ui,sans-serif;color:#182734;background:#f6f7f8;margin:0}header,main{max-width:1200px;margin:auto;padding:24px}header{position:sticky;top:0;background:#f6f7f8;z-index:2}h1{font-size:22px}button{border:1px solid #bcc7cf;padding:8px 15px;background:white;border-radius:5px;cursor:pointer;margin-right:8px}article{background:white;padding:20px;margin:0 0 24px;border:1px solid #dce2e6;border-radius:7px}article h2{font-size:16px}img{display:block;width:100%;height:auto;max-width:1100px;margin:auto}a{color:#2166ac}details p{line-height:1.5}p{line-height:1.5}</style><header><h1>Experience configuration and scientific recombination</h1><p>Fixed-width publication plates. Previews are separate from the 600-dpi originals. Statistical models are not re-estimated.</p><button onclick="choose('all')">All</button><button onclick="choose('main')">Main figures</button><button onclick="choose('supplement')">Supplement</button><a href="figure_atlas.pdf">Complete PDF atlas</a></header><main>'''
        document+=''.join(cards)+'''</main><script>function choose(kind){document.querySelectorAll('article').forEach(x=>x.hidden=kind!=='all'&&x.dataset.kind!==kind)}</script></html>'''
        (self.out/'gallery.html').write_text(document,encoding='utf-8')

    def render(self):
        methods=[self.main1,self.main2,self.main3,self.main4]
        if self.s.supplementary:methods += [self.distributions,self.composition,self.surface_atlas,self.uncertainty,
            self.specification_atlas,self.field_atlas,self.cohort_atlas,self.static3d,self.correlation_atlas,self.field_support_atlas]
        with PdfPages(self.out/'figure_atlas.pdf',metadata={'Title':'SciSciNet2 publication figure atlas','Author':'SciSciNet2 analysis'}) as atlas:
            self.atlas=atlas
            for method in methods:method()
        self.atlas=None
        pd.DataFrame(self.records).to_csv(self.out/'figure_index.csv',index=False)
        (self.out/'CAPTIONS.md').write_text('# Figure captions\n\n'+ '\n'.join(self.captions),encoding='utf-8')
        (self.out/'figure_settings_used.json').write_text(json.dumps(asdict(self.s),indent=2),encoding='utf-8')
        self.contacts('main',2);self.contacts('supplement',3);self.gallery()
        return {'output_dir':self.out,'main':[self.out/r['pdf'] for r in self.records if r['kind']=='main'],
            'supplement':[self.out/r['pdf'] for r in self.records if r['kind']=='supplement'],
            'atlas':self.out/'figure_atlas.pdf','gallery':self.out/'gallery.html'}


def render_all(results_root,output_dir=None,settings=None):
    """Read existing tables and plotting sample; never prepare raw data or fit models."""
    return JournalPlots(results_root,output_dir,settings).render()
