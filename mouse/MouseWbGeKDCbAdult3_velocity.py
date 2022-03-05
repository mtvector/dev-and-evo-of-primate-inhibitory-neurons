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
sc.settings.figdir='/wynton/group/ye/mtschmitz/figures/mouseWbGeAdult2/'
scv.settings.figdir='/wynton/group/ye/mtschmitz/figures/mouseWbGeAdult2/'
sc.settings.file_format_figs='pdf'
sc.settings.autosave=True
sc.settings.autoshow=False

def most_frequent(List): 
    return max(set(List), key = List.count)

def variance(X,axis=1):
    return( (X.power(2)).mean(axis=axis)-(np.power(X.mean(axis=axis),2)) )

cortexregions=['head','forebrain','cortex','motor']
newfile='/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseWbGeAdult.h5ad'
adata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseWbGeAdult2Processed.h5ad')
adata.obs_names_make_unique()
adata.var_names_make_unique()

adata.obs['agg_supervised_name']='nan'
adata.obs['old_leiden']='nan'
supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MouseAdultAggSupervised.txt')
adata=adata[~adata.obs.index.duplicated(),:]
supercell=supercell.loc[supercell['agg_supervised_name']!='nan',:]
ind=adata.obs.index[adata.obs.index.isin(supercell['full_cellname'])]
supercell.index=supercell['full_cellname']
supercell=supercell.loc[~supercell.index.duplicated(),:]
supercell=supercell.loc[ind,:]
adata.obs.loc[ind,'agg_supervised_name']=supercell['agg_supervised_name']
adata.obs.loc[ind,'old_leiden']=supercell['leiden']
adata.obs['agg_supervised_name']=adata.obs['agg_supervised_name'].astype(str)
adata.obs['old_leiden']=adata.obs['old_leiden'].astype(str)

####################
cortex_order=['mop','cortex','forebrain','forebraindorsal','hippocampus']
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
region_color_dict=dict(zip(all_regions,all_regions_colors))

adata.obs.region=adata.obs.region.astype('category')
print(all_regions)
print(adata.obs.region.cat.categories)
print(set(adata.obs.region.cat.categories)-set(all_regions))
print(set(all_regions)-set(adata.obs.region.cat.categories))
region_pal=[region_color_dict[x] for x in adata.obs['region'].cat.categories]
adata.uns['region_colors']=region_pal
sc.pl.umap(adata,color='region',palette=region_pal,save='region')

sc.tl.dendrogram(adata,groupby='agg_supervised_name')
sc.pl.dendrogram(adata,groupby='agg_supervised_name',save='agg_supervised_name')
df_plot = adata.obs.groupby(['agg_supervised_name', 'region']).size().reset_index().pivot(columns='region', index='agg_supervised_name', values=0).apply(lambda g: g / g.sum(),1)
df_plot=df_plot.loc[adata.uns["dendrogram_agg_supervised_name"]['categories_ordered'],:]
these_colors=[region_color_dict[x] for x in df_plot.columns]
ax = df_plot.plot(kind='bar', legend=False,stacked=True,color=these_colors,fontsize=7.5)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion of cells in region')
ax.grid(False)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedNameRegionsStackBar.pdf'), bbox_inches="tight")


sc.pl.umap(adata,color=['LYNX1','PRUNE2','SCN2B','SLC24A2','HES6','EZH2','TUBB2B','TUBB3'],use_raw=False,save='AdultMarkers')

#############################

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

sc.tl.leiden(gdata,resolution=3)
sc.tl.umap(gdata,spread=5,min_dist=.2)

sc.pl.umap(gdata,color='leiden',legend_loc='on data',palette="Set2",save='GeneLeiden')
print(gdata[s_genes+g2m_genes,:].obs.leiden.value_counts())
vcs=gdata[s_genes+g2m_genes,:].obs.leiden.value_counts()
cc_mods=vcs.index[vcs>5]
cc_genes=gdata.obs.index[gdata.obs.leiden.isin(cc_mods)]
adata.var['leiden']=gdata.obs['leiden']
del gdata

#####################
#Downsample dataset
adata=adata[np.random.choice(adata.obs.index,100000,replace=False),:]
#####################
#Downsample and Equalize Terminal state sizes
adata.obs['agg_supervised_name']='nan'
adata.obs['old_leiden']='nan'
supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MouseAdultAggSupervised.txt')
adata=adata[~adata.obs.index.duplicated(),:]
supercell=supercell.loc[supercell['agg_supervised_name']!='nan',:]
ind=adata.obs.index[adata.obs.index.isin(supercell['full_cellname'])]
supercell.index=supercell['full_cellname']
supercell=supercell.loc[~supercell.index.duplicated(),:]
supercell=supercell.loc[ind,:]
adata.obs.loc[ind,'agg_supervised_name']=supercell['agg_supervised_name']
adata.obs.loc[ind,'old_leiden']=supercell['leiden']
adata.obs['agg_supervised_name']=adata.obs['agg_supervised_name'].astype(str)
adata.obs['old_leiden']=adata.obs['old_leiden'].astype(str)

countmat=adata.obs.astype(str).groupby(['leiden', 'agg_supervised_name']).size().reset_index().pivot(columns='agg_supervised_name', index='leiden', values=0)
if 'nan' in countmat.columns:
    countmat=countmat.drop('nan',axis=1)
leidentosuper=dict(countmat.idxmax(1))
adata.obs.loc[adata.obs['agg_supervised_name']=='nan','agg_supervised_name']=adata.obs.loc[adata.obs['agg_supervised_name']=='nan','leiden'].replace(leidentosuper)


"""
d={}
for c in adata.obs.agg_supervised_name.unique():
    if not any([x in c.upper() for x in ['MGE','RMTW','CGE','LGE','VMF','G2-M','S-PHASE','GLIA','NAN','TRANSITION','EXCITATORY']]):
        print(c)
        select_cells=adata.obs.agg_supervised_name.isin([c])
        d[c]=[adata.obs.index[i] for i in range(adata.shape[0]) if select_cells[i]]
terminal_states=list(d.keys())#+['MGE_CRABP1/MAF']


devs=adata.obs.index[~adata.obs.agg_supervised_name.isin(terminal_states)]
terms=adata.obs.index[adata.obs.agg_supervised_name.isin(terminal_states)]
devs=np.random.choice(devs,60000,replace=False)

adata=adata[list(devs)+list(terms),:]

max_cells_in_category=500
vcs=adata.obs.agg_supervised_name.value_counts()
for t in terminal_states:
    print(t)
    print(vcs[t])
    print(max(vcs[t]-max_cells_in_category,0))
    destroy=np.random.choice(adata.obs.index[adata.obs.agg_supervised_name==t],size=max(vcs[t]-max_cells_in_category,0),replace=False)
    adata=adata[~adata.obs.index.isin(destroy),:]
print(adata)
"""
##################################
sufficient_cells=adata.obs['batch_name'].value_counts().index[adata.obs['batch_name'].value_counts()>10]
adata=adata[adata.obs['batch_name'].isin(sufficient_cells),:]
adata.obs['batch_name'].cat.remove_unused_categories(inplace=True)
bbknn.bbknn(adata,batch_key='dataset_name',n_pcs=100,neighbors_within_batch=12)

#################################

sc.pp.highly_variable_genes(adata,flavor='seurat_v3',layer='unspliced',n_top_genes=15000)
adata.var['highly_variable_rank_u']=adata.var['highly_variable_rank']
sc.pp.highly_variable_genes(adata,flavor='seurat_v3',layer='spliced',n_top_genes=15000)
adata.var['highly_variable_rank_s']=adata.var['highly_variable_rank']
print(adata,flush=True)
adata.var['velocity_genes']=adata.var.loc[:,['highly_variable_rank_s','highly_variable_rank_u']].mean(1,skipna=False).rank()<7000
print(adata.var['velocity_genes'].value_counts())
print('#velocity genes',np.sum(adata.var.velocity_genes), flush=True)

adata.uns['iroot'] = np.flatnonzero(adata.obs.index==sc_utils.get_median_cell(adata,'supervised_name','Transition'))[0]
ncells=10
mgenes=(adata.layers['unspliced']>0).sum(0).A1 < ncells
print(mgenes)
adata.var.loc[mgenes,'velocity_genes']=False


#mgenes=(adata.layers['unspliced']>0).sum(0) > ncells
#adata=adata[:,mgenes.A1]
#adata=adata[:,adata.var['highly_variable']]
#adata.var['velocity_genes']=mgenes.A1&adata.var['highly_variable']
print('#velocity genes',np.sum(adata.var.velocity_genes), flush=True)
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
#adata.write(re.sub('Velocity','VelocityDYNAMICALsmall',newfile))
