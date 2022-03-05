import os
import scanpy
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
import scipy
from scipy import stats
import re
import sklearn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from collections import Counter
import random
import seaborn
import sys
import shutil
import scvelo as scv
import tqdm
import bbknn
#Load my pipeline functions
import importlib
import importlib.util
spec = importlib.util.spec_from_file_location("ScanpyUtilsMT", os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../utils/ScanpyUtilsMT.py"))
sc_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc_utils)
sc.settings.figdir='/wynton/group/ye/mtschmitz/figures/mouseWbGeCB202002/'
scv.settings.figdir='/wynton/group/ye/mtschmitz/figures/mouseWbGeCB202002/'
sc.settings.file_format_figs='pdf'
sc.settings.autosave=True
sc.settings.autoshow=False

def most_frequent(List): 
    return max(set(List), key = List.count)

def variance(X,axis=1):
    return( (X.power(2)).mean(axis=axis)-(np.power(X.mean(axis=axis),2)) )

cortexregions=['head','forebrain','cortex','motor']
newfile='/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseWbGe.h5ad'
adata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseWbPresupervisionProcessed.h5ad')
adata.X=adata.raw.X[:,adata.raw.var.index.isin(adata.var.index)].todense()
adata.obs_names_make_unique()
adata.var_names_make_unique()


adata.obs['region']=[re.sub('srr','',x) for x in adata.obs['region'].astype(str)]
adata.obs['region']=[re.sub('ctx','cortex',x) for x in adata.obs['region'].astype(str)]
adata.obs['region']=[re.sub('ssctx','cortex',x) for x in adata.obs['region'].astype(str)]

supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MouseSupervisednames6PreserveCompoundname.txt')
supercell=supercell.loc[supercell['supervised_name']!='nan',:]
ind=adata.obs.index[adata.obs['full_cellname'].isin(supercell['full_cellname'])]
supercell.index=supercell['full_cellname']
supercell=supercell.loc[ind,:]
if 'old_leiden' in adata.obs.columns:
    adata.obs['old_leiden']=adata.obs['old_leiden'].astype(str)
if 'supervised_name' in adata.obs.columns:
    adata.obs['supervised_name']=adata.obs['supervised_name'].astype(str)
print(supercell,flush=True)
adata.obs.loc[ind,'old_leiden']=supercell['leiden']
adata.obs.loc[ind,'supervised_name']=supercell['supervised_name']
adata.obs['old_leiden']=[re.sub('\.0','',str(x)) for x in adata.obs['old_leiden']]
adata.obs['old_leiden']=adata.obs['old_leiden'].astype(str)
adata.obs['supervised_name']=adata.obs['supervised_name'].astype(str)
print(adata.obs['old_leiden'],flush=True)
print(adata.obs['old_leiden'].isin(['nan']),flush=True)
print(adata.obs['old_leiden'].str.contains('36|26|nan'),flush=True)
#adata=adata[~adata.obs['old_leiden'].str.contains('36|26|nan'),:]
#adata=adata[~adata.obs['old_leiden'].isin(['42','26','36','22','11','nan',np.nan]),:]
adata=adata[~adata.obs['old_leiden'].isin(['11','nan',np.nan]),:]
adata=adata[~adata.obs['supervised_name'].isin(['nan']),:]

adata.obs['timepoint']=adata.obs['timepoint'].astype(str)
adata.obs.loc[adata.obs['dataset_name']=='PRJNA498989_OB_mouse','timepoint']=84
adata.obs.loc[adata.obs['dataset_name']=='PRJNA515751_konopka_striatum','timepoint']=30
adata.obs.loc[adata.obs['batch_name'].str.contains('P5_',case=False),'timepoint']=26
adata.obs.loc[adata.obs['batch_name'].str.contains('p07_Cortex_SRR11947654',case=False),'timepoint']=28
adata.obs.loc[adata.obs['timepoint']=='nan','timepoint']=84
adata.obs.loc[adata.obs['timepoint'].astype(float)>100,'timepoint']=84

adata.obs.loc[adata.obs['dataset_name'].str.contains('dev_hypo'),'timepoint']=[sc_utils.tp_format_mouse(x) for x in adata.obs.loc[adata.obs['dataset_name'].str.contains('dev_hypo'),'batch_name']]
adata.obs['timepoint']=adata.obs['timepoint'].astype(float)

adata.obs['batch_name'].cat.remove_unused_categories(inplace=True)
sc.pp.normalize_total(adata,exclude_highly_expressed=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata,n_top_genes=12000,batch_key='dataset_name',subset=False)#'dataset_name'
sc.pp.scale(adata,max_value=10)
sc.pp.pca(adata,n_comps=100)
#sc.pp.neighbors(adata)
bbknn.bbknn(adata,batch_key='batch_name',n_pcs=100,neighbors_within_batch=9)
sc.tl.leiden(adata,resolution=7)
sc.tl.umap(adata,spread=3,min_dist=.2)
adata.obsm['X_umap'][:,0]=-adata.obsm['X_umap'][:,0]

adata.write('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseWbGeProcessed.h5ad')

####

paldict={'CGE_NR2F2/PROX1': 'slategray',
    'G1-phase_SLC1A3/ATP1A1': '#17344c',
    'G2-M_UBE2C/ASPM': '#19122b',
    'LGE_FOXP1/ISL1': 'cyan',
    'LGE_FOXP1/PENK': 'navy',
    'LGE_FOXP2/TSHZ1': 'goldenrod',
    'LGE_MEIS2/PAX6': 'orangered',
    'LGE_MEIS2/PAX6/SCGN': 'orange',
    'LGE-OB_MEIS2/PAX6': 'red',
    'MGE_CRABP1/MAF': 'indigo',
    'MGE_CRABP1/TAC3': 'fuchsia',
    'MGE_LHX6/MAF': 'darksalmon',
    'MGE_LHX6/NPY': 'maroon',
    'RMTW_ZIC1/RELN': 'yellow',
    'S-phase_MCM4/H43C': 'lawngreen',
    'Transition': '#3c7632',
    'VMF_ZIC1/ZIC2': 'teal',
    'VMF_CRABP1/LHX8':'skyblue',
    'VMF_NR2F2/LHX6':'lightseagreen',
    'VMF_LHX1/POU6F2':'seagreen',
    'VMF_TMEM163/OTP':'lightsteelblue',
    'VMF_PEG10/DLK1':'steelblue',
    'LGE_FOXP1/ISL1/NPY1R':'mediumpurple',
    'nan':'black'
}

sc.pl.umap(adata,color=['supervised_name'],save='supervised_name')
#adata=adata[adata.obs['supervised_name'].astype(str)!='nan',:]
#adata.obs['supervised_name'].cat.remove_unused_categories(inplace=True)
my_pal=[paldict[x] for x in adata.obs['supervised_name'].cat.categories]
        
#adata=sc.AnnData(adata.raw.X,var=adata.raw.var,obsm=adata.obsm,uns=adata.uns,obs=adata.obs)
#adata.raw=adata
#sc.pp.normalize_per_cell(adata)
#sc.pp.log1p(adata)
#sc.pp.scale(adata,max_value=10)
sc.pl.umap(adata,color=['leiden'],save='leiden')
sc.pl.umap(adata,color=['old_leiden'],save='old_leiden')
sc.pl.umap(adata,color=['batch_name'],save='batch_name')
sc.pl.umap(adata,color=['timepoint'],save='timepoint', color_map=matplotlib.cm.RdYlBu_r)
sc.pl.umap(adata,color=['supervised_name'],legend_loc='on data',legend_fontsize=4,palette=my_pal,save='supervised_name_onplot')
sc.pl.umap(adata,color=['supervised_name'],save='supervised_name')
sc.pl.umap(adata,color=['leiden'],legend_loc='on data',legend_fontsize=4,save='leiden_onplot')
sc.pl.umap(adata,color=['dataset_name'],save='dataset_name')

adata.obs['region']=adata.obs['region'].astype(str).str.lower()
adata.obs['region']=[re.sub('srr','',x) for x in adata.obs['region']]

cortex_order=['mop','cortex','ctx','fb','forebrain','forebraindorsal','hippocampus']
cortex_colors=seaborn.color_palette("YlOrBr",n_colors=len(cortex_order)+2).as_hex()[2:]
ventral_tel=['cge', 'lge', 'mge']
vt_colors=seaborn.blend_palette(('red','dodgerblue'),n_colors=len(ventral_tel)).as_hex()
med_tel=['forebrainventral','hypothalamus']
mt_colors=seaborn.blend_palette(('grey','black'),n_colors=len(med_tel)).as_hex()
basal_gang=['striatumventral','striatum','striatumdorsal','subcortex','forebrainventrolateral', 'amygdala']
bg_colors=seaborn.color_palette("PRGn",n_colors=len(basal_gang)).as_hex()
ob=['ob']
ob_colors=['#FF00FF']

all_regions=cortex_order+ventral_tel+med_tel+basal_gang+ob
all_regions_colors=cortex_colors+vt_colors+mt_colors+bg_colors+ob_colors
#region_color_dict={}
#for r,c in zip(all_regions,all_regions_colors):
#    if r in adata.obs.regions.cat.categories:
#        region_color_dict[r]=c
region_color_dict=dict(zip(all_regions,all_regions_colors))

adata.obs.region=adata.obs.region.astype('category')
print(all_regions)
print(adata.obs.region.cat.categories)
print(set(adata.obs.region.cat.categories)-set(all_regions))
print(set(all_regions)-set(adata.obs.region.cat.categories))
region_pal=[region_color_dict[x] for x in adata.obs['region'].cat.categories]
adata.uns['region_colors']=region_pal
sc.pl.umap(adata,color='region',palette=region_pal,save='region')


easygenes= ['MKI67','SOX2','AQP4','EDNRB','IL33','PDGFRA','HOPX','ERBB4','CALB1','CALB2','GAD1','GAD2','GADD45G','LHX6','LHX7','LHX8','RBP4','RXRA','NXPH1','NXPH2','SST','NKX2-1','MAF','SP8','SP9','PROX1','VIP','CCK','NPY','LAMP5','HTR1A','HTR3A','NR2F1','NR2F2','TOX3','ETV1','SCGN','FOXP1','FOXP2','FOXP4','TSHZ1','OPRM1','CASZ1','HMX3','NKX6-2','TH','DDC','SLC18A2','PAX6','MEIS2','SHTN1','ISL1','PENK','DRD1','ADORA2A','CHAT','ZNF503','STXBP6','CRABP1','ZIC1','ZIC2','FOXG1','TAC1','TAC2','TAC3','TACR1','TACR2','TACR3','EBF1','DLX5','GSX2','RBFOX3','PPP1R17','EOMES','DCX','TUBB3','BCL11B','TLE4','FEZF2','SATB2','TBR1','AIF1','RELN','PECAM1','HBZ','ROBO1','ROBO2','ROBO3','ROBO4']
easygenes=[x for x in easygenes if x in adata.var.index]
sc.pl.umap(adata,color=easygenes,use_raw=False,save='FaveGenes')

INgenes=['MKI67','DCX','DLX2','ZIC1','SP8','SCGN','CCK','VIP','MEIS2','PAX6','FOXP2','TSHZ1','FOXP1','OPRM1','ISL1','PENK','LHX6','SST','NPY','CHODL','CRABP1','ANGPT2','MAF','TAC2','GBX1','HMX3','TH','OTP']         
INgenes=[x for x in INgenes if x in adata.var.index]
sc.pl.umap(adata,color=INgenes,use_raw=False,save='SupplementGenes')

f = plt.figure()
df_plot = adata.obs.groupby(['supervised_name', 'batch_name']).size().reset_index().pivot(columns='batch_name', index='supervised_name', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVbatchBar.pdf'), bbox_inches="tight")

f = plt.figure()
df_plot = adata.obs.groupby(['supervised_name', 'timepoint']).size().reset_index().pivot(columns='timepoint', index='supervised_name', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVtimepointBar.pdf'), bbox_inches="tight")

f = plt.figure()
df_plot = adata.obs.groupby(['supervised_name', 'region']).size().reset_index().pivot(columns='region', index='supervised_name', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVregionBar.pdf'), bbox_inches="tight")

f = plt.figure()
df_plot = adata.obs.groupby(['supervised_name', 'leiden']).size().reset_index().pivot(columns='leiden', index='supervised_name', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVleidenBar.pdf'), bbox_inches="tight")

f = plt.figure()
df_plot = adata.obs.groupby(['leiden', 'region']).size().reset_index().pivot(columns='region', index='leiden', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'regionVleidenBar.pdf'), bbox_inches="tight")

f = plt.figure()
df_plot = adata[adata.obs['region'].isin(cortexregions),:].obs.groupby(['supervised_name', 'region']).size().reset_index().pivot(columns='region', index='supervised_name', values=0)#.apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVcorticalregionBar.pdf'), bbox_inches="tight")

df_plot = adata.obs.groupby(['supervised_name', 'region']).size().reset_index().pivot(columns='region', index='supervised_name', values=0)#.apply(lambda g: g / g.sum(),1)
df_plot=df_plot.loc[df_plot.sum(1)>50,:]
regiontotals=adata.obs.groupby([ 'region']).size()
ax = (df_plot/regiontotals).T.plot(kind='bar', legend=False,stacked=False)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion of cells in region')
ax.grid(False)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'regionMakeupSupervisedNameBarNorm.pdf'), bbox_inches="tight")

sc.pl.umap(adata,color=['supervised_name'],save='supervised_name',palette=my_pal)
sc.pl.umap(adata,color=['supervised_name'],legend_loc='on data',legend_fontsize=4,save='supervised_name_onplot')

sc_utils.cell_cycle_score(adata)
#sc_utils.marker_analysis(adata,variables=['leiden'],markerpath=os.path.expanduser('~/markers.txt')) 
#sc_utils.log_reg_diff_exp(adata)#,method='t-test_overestim_var')
sufficient_cells=adata.obs['leiden'].value_counts().index[adata.obs['leiden'].value_counts()>10]
adata=adata[adata.obs['leiden'].isin(sufficient_cells),:]
sufficient_cells=adata.obs['supervised_name'].value_counts().index[adata.obs['supervised_name'].value_counts()>10]
adata=adata[adata.obs['supervised_name'].isin(sufficient_cells),:]
sc_utils.log_reg_diff_exp(adata,obs_name='leiden')#,method='t-test_overestim_var')
sc_utils.log_reg_diff_exp(adata,obs_name='supervised_name')#,method='t-test_overestim_var')

print(adata.X)
print(adata.raw.X)
hierarchy_key='supervised_name'
rgs=sc.tl.rank_genes_groups(adata,groupby=hierarchy_key,method='logreg',use_raw=False,copy=True).uns['rank_genes_groups']#,penalty='elasticnet',solver='saga')#or penalty='l1'
result=rgs
groups = result['names'].dtype.names
df=pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'scores']})
df.to_csv(os.path.join(sc.settings.figdir,"LogReg"+hierarchy_key+"Norm.csv"))
topgenes=df.iloc[0:4,['_n' in x for x in df.columns]].T.values
cols=df.columns[['_n' in x for x in df.columns]]
cols=[re.sub('_n','',x) for x in cols]
topdict=dict(zip(cols,topgenes))
topdict['TAC3']=['TAC2','TACR3']
sc.tl.dendrogram(adata,groupby=hierarchy_key)
print(adata.uns)
var_dict=dict(zip(adata.uns["dendrogram_"+hierarchy_key]['categories_ordered'],[topdict[x] for x in adata.uns["dendrogram_"+hierarchy_key]['categories_ordered']]))
sc.pl.matrixplot(adata,groupby=hierarchy_key,var_names=var_dict,save='top_degenes',cmap='RdBu_r',use_raw=False,dendrogram=True)

s_genes=['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8']
g2m_genes=['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']
s_genes=[x for x in s_genes if x in adata.var.index]
g2m_genes=[x for x in g2m_genes if x in adata.var.index]

gdata=adata.copy().T

import numba
@numba.njit()
def abscorrelation(x, y):
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0

    for i in range(x.shape[0]):
        mu_x += x[i]
        mu_y += y[i]

    mu_x /= x.shape[0]
    mu_y /= x.shape[0]

    for i in range(x.shape[0]):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif dot_product == 0.0:
        return 1.0
    else:
        return 1.0 - np.absolute(dot_product / np.sqrt(norm_x * norm_y))
    
gdata.obsm['X_pca']=gdata.obsm['PCs']

#sc.pp.neighbors(gdata,n_neighbors=25)
sc.pp.neighbors(gdata,metric=abscorrelation,n_neighbors=25)

sc.tl.leiden(gdata,resolution=5)
sc.tl.umap(gdata,spread=5,min_dist=.2)

sc.pl.umap(gdata,color='leiden',legend_loc='on data',palette="Set2",save='GeneLeiden')
print(gdata[s_genes+g2m_genes,:].obs.leiden.value_counts())
vcs=gdata[s_genes+g2m_genes,:].obs.leiden.value_counts()
cc_mods=vcs.index[vcs>5]
cc_genes=gdata.obs.index[gdata.obs.leiden.isin(cc_mods)]
adata.var['leiden']=gdata.obs['leiden']
del gdata

adata=adata[(~adata.obs.region.isin(['ob','OB']) &  ~adata.obs.supervised_name.str.contains('"')) | adata.obs.supervised_name.str.contains('MEIS2'),:]

bbknn.bbknn(adata,batch_key='batch_name',n_pcs=100,neighbors_within_batch=3)

sc.pp.highly_variable_genes(adata,flavor='seurat_v3',layer='unspliced',n_top_genes=15000)
adata.var['highly_variable_rank_u']=adata.var['highly_variable_rank']
sc.pp.highly_variable_genes(adata,flavor='seurat_v3',layer='spliced',n_top_genes=15000)
adata.var['highly_variable_rank_s']=adata.var['highly_variable_rank']
print(adata,flush=True)
adata.var['velocity_genes']=adata.var.loc[:,['highly_variable_rank_s','highly_variable_rank_u']].mean(1,skipna=False).rank()<4000
print(adata.var['velocity_genes'].value_counts())

adata.uns['iroot'] = np.flatnonzero(adata.obs.index==sc_utils.get_median_cell(adata,'supervised_name','Transition'))[0]
ncells=20
mgenes=(adata.layers['unspliced']>0).sum(0).A1 < ncells
print(mgenes)
adata.var.loc[mgenes,'velocity_genes']=False

#mgenes=(adata.layers['unspliced']>0).sum(0) > ncells
#adata=adata[:,mgenes.A1]
#adata=adata[:,adata.var['highly_variable']]
#adata.var['velocity_genes']=mgenes.A1&adata.var['highly_variable']
print(list(adata.var['velocity_genes']))
scv.pp.normalize_per_cell(adata)
print(adata)
print(adata.layers['spliced'].shape)
#scv.pp.filter_genes_dispersion(adata)#,n_top_genes=min(6000,adata.layers['spliced'].shape[1]))
scv.pp.log1p(adata)
scv.pp.remove_duplicate_cells(adata)
scv.pp.moments(adata, n_neighbors=20)
print(adata,flush=True)
scv.tl.recover_dynamics(adata,var_names='velocity_genes',use_raw=False)
print('Recovered', flush=True)
scv.tl.velocity(adata,mode='dynamical',filter_genes=False)
print('Velocity done',flush=True)
adata.layers['linear_velocity']=adata.layers['velocity'].copy()
adata.layers['linear_velocity'][:,adata.var.index.isin(cc_genes)]=0
adata.layers['cc_velocity']=adata.layers['velocity'].copy()
adata.layers['cc_velocity'][:,~adata.var.index.isin(cc_genes)]=0

scv.tl.velocity_graph(adata,mode_neighbors='connectivities',vkey='cc_velocity',approx=False)
scv.tl.transition_matrix(adata,vkey='cc_velocity')
scv.tl.terminal_states(adata,vkey='cc_velocity')
scv.tl.recover_latent_time(adata,vkey='cc_velocity')
scv.tl.velocity_confidence(adata,vkey='cc_velocity')
scv.tl.velocity_embedding(adata, basis='umap',vkey='cc_velocity')
adata.obs['cc_latent_time']=adata.obs['latent_time']
scv.pl.velocity_embedding_grid(adata, basis='umap',vkey='cc_velocity',save='_cc_grid',color_map=matplotlib.cm.RdYlBu_r)
try:
    sc.pl.umap(adata,color=['cc_velocity_length', 'cc_velocity_confidence','latent_time','root_cells','end_points','cc_velocity_pseudotime'],save='cc_velocity_stats')
except:
    print('fail cc')

scv.tl.velocity_graph(adata,mode_neighbors='connectivities',vkey='linear_velocity',approx=False)
scv.tl.transition_matrix(adata,vkey='linear_velocity')
scv.tl.terminal_states(adata,vkey='linear_velocity')
scv.tl.recover_latent_time(adata,vkey='linear_velocity')
scv.tl.velocity_embedding(adata, basis='umap',vkey='linear_velocity')
scv.tl.velocity_confidence(adata,vkey='linear_velocity')
scv.pl.velocity_embedding_grid(adata,vkey='linear_velocity',basis='umap',color='latent_time',save='_linear_grid',color_map=matplotlib.cm.RdYlBu_r)
try:
    sc.pl.umap(adata,color=['linear_velocity_length', 'linear_velocity_confidence','latent_time','root_cells','end_points','linear_velocity_pseudotime'],save='linear_velocity_stats')
except:
    'fail linear'
scv.utils.cleanup(adata)
adata.write(re.sub('Velocity','VelocityDYNAMICAL',newfile))

sc.tl.paga(adata,groups='supervised_name')
sc.tl.paga(adata,use_rna_velocity=True,groups='supervised_name')
sc.pl.paga_compare(adata,legend_fontsize=4,arrowsize=10,edge_width_scale=.4,threshold=np.quantile(adata.uns['paga']['connectivities'].data,.9))
sc.pl.paga_compare(adata,legend_fontsize=4,arrowsize=10,edge_width_scale=.4,threshold=np.quantile(adata.uns['paga']['connectivities'].data,.9),save='connectivity')
sc.pl.paga_compare(adata,solid_edges='connectivities',transitions='transitions_confidence',legend_fontsize=4,arrowsize=10,threshold=np.quantile(adata.uns['paga']['transitions_confidence'].data,.9),save='DYNAMICALvelocity')
adata.write(re.sub('Velocity','VelocityDYNAMICAL',newfile))
