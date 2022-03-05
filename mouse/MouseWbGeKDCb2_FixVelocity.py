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
adata=sc.read(re.sub('Velocity','VelocityDYNAMICAL',newfile))

print(adata)
print(adata.layers)
print(adata.layers['spliced'])

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

adata=adata[(~adata.obs.region.isin(['ob','OB']) &  ~adata.obs.supervised_name.str.contains('"')) | adata.obs.supervised_name.str.contains('MEIS2'),:]

bbknn.bbknn(adata,batch_key='batch_name',n_pcs=100,neighbors_within_batch=3)

#sc.pp.highly_variable_genes(adata,flavor='seurat_v3',layer='unspliced',n_top_genes=15000)
#adata.var['highly_variable_rank_u']=adata.var['highly_variable_rank']
#sc.pp.highly_variable_genes(adata,flavor='seurat_v3',layer='spliced',n_top_genes=15000)
#adata.var['highly_variable_rank_s']=adata.var['highly_variable_rank']
#print(adata,flush=True)
adata.var['velocity_genes']=adata.var.loc[:,['highly_variable_rank_s','highly_variable_rank_u']].mean(1,skipna=False).rank()<5000
#print(adata.var['velocity_genes'].value_counts())

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
#scv.pp.normalize_per_cell(adata)
print(adata)
print(adata.layers['spliced'].shape)
#scv.pp.log1p(adata)
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
    sc.pl.umap(adata,color=['linear_velocity_length', 'linear_velocity_confidence','latent_time','root_cells','end_points','linear_velocity_pseudotime'],save='linear_velocity_stats',color_map=matplotlib.cm.RdYlBu_r)
except:
    'fail linear'
scv.utils.cleanup(adata)
adata.write(re.sub('Velocity','VelocityDYNAMICAL',newfile))

sc.tl.paga(adata,groups='supervised_name')
sc.tl.paga(adata,use_rna_velocity=True,groups='supervised_name')
sc.pl.paga_compare(adata,legend_fontsize=4,arrowsize=10,edge_width_scale=.4,threshold=np.quantile(adata.uns['paga']['connectivities'].data,.9))
sc.pl.paga_compare(adata,legend_fontsize=4,arrowsize=10,edge_width_scale=.4,threshold=np.quantile(adata.uns['paga']['connectivities'].data,.9),save='connectivity')
sc.pl.paga_compare(adata,solid_edges='connectivities',transitions='transitions_confidence',legend_fontsize=4,arrowsize=10,threshold=np.quantile(adata.uns['paga']['transitions_confidence'].data,.9),save='DYNAMICALvelocity')
#adata.write(re.sub('Velocity','VelocityDYNAMICAL',newfile))
