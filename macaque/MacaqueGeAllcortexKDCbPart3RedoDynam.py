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
import bbknn
import tqdm
#Load my pipeline functions
import importlib
import importlib.util
spec = importlib.util.spec_from_file_location("ScanpyUtilsMT", os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../utils/ScanpyUtilsMT.py"))
sc_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc_utils)
figdir='/wynton/group/ye/mtschmitz/figures/macaqueGeAllcortexHippoCB202002/'
sc.settings.figdir=figdir
scv.settings.figdir=figdir
sc.settings.file_format_figs='pdf'
sc.settings.autosave=True
sc.settings.autoshow=False
import matplotlib.font_manager
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Nimbus Sans','Arial']})
matplotlib.rc('text', usetex=False)
scanpy.set_figure_params(scanpy=True,dpi_save=400)

newfile='/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMacaqueGeAllcortexHippocampus.h5ad'

adata=sc.read(re.sub('Velocity','VelocityDYNAMICAL',newfile))

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

adata.var['cc_genes']=adata.var.index.isin(cc_genes)
print(adata.var['cc_genes'].value_counts())
adata.layers['cc_velocity']=adata.layers['velocity'].copy()
adata.layers['cc_velocity'][:,~adata.var['cc_genes']]=0

scv.tl.velocity_graph(adata,mode_neighbors='connectivities',vkey='cc_velocity',approx=False)
scv.tl.transition_matrix(adata,vkey='cc_velocity')
scv.tl.terminal_states(adata,vkey='cc_velocity')
scv.tl.recover_latent_time(adata,vkey='cc_velocity')
scv.tl.velocity_confidence(adata,vkey='cc_velocity')
scv.tl.velocity_embedding(adata, basis='umap',vkey='cc_velocity')
scv.pl.velocity_embedding_grid(adata, basis='umap',vkey='cc_velocity',save='_cc_grid',color_map=matplotlib.cm.RdYlBu_r,dpi=300)
try:
    sc.pl.umap(adata,color=['cc_velocity_length', 'cc_velocity_confidence','latent_time','root_cells','end_points','cc_velocity_pseudotime'],save='cc_velocity_stats',color_map=matplotlib.cm.RdYlBu_r)
except:
    print('fail cc')

del adata.layers['cc_velocity']

adata.layers['linear_velocity']=adata.layers['velocity'].copy()
adata.layers['linear_velocity'][:,adata.var['cc_genes']]=0
scv.tl.velocity_graph(adata,mode_neighbors='connectivities',vkey='linear_velocity',approx=False)
scv.tl.transition_matrix(adata,vkey='linear_velocity')
scv.tl.terminal_states(adata,vkey='linear_velocity')
adata=adata.copy()
scv.tl.recover_latent_time(adata,vkey='linear_velocity',weight_diffusion=0)
scv.tl.velocity_embedding(adata, basis='umap',vkey='linear_velocity')
scv.tl.velocity_confidence(adata,vkey='linear_velocity')
scv.pl.velocity_embedding_grid(adata,vkey='linear_velocity',basis='umap',color='latent_time',save='_linear_grid',color_map=matplotlib.cm.RdYlBu_r,dpi=300)
try:
    sc.pl.umap(adata,color=['linear_velocity_length', 'linear_velocity_confidence','latent_time','root_cells','end_points','linear_velocity_pseudotime'],save='linear_velocity_stats',color_map=matplotlib.cm.RdYlBu_r)
except:
    'fail linear'

adata.write(re.sub('Velocity','VelocityDYNAMICAL',newfile))
scv.tl.paga(adata,groups='supervised_name',vkey='linear_velocity')
scv.pl.paga_compare(adata,legend_fontsize=4,arrowsize=10,edge_width_scale=.4,threshold=np.quantile(adata.uns['paga']['connectivities'].data,.9))
sc.pl.paga_compare(adata,legend_fontsize=4,arrowsize=10,edge_width_scale=.4,threshold=np.quantile(adata.uns['paga']['connectivities'].data,.9),save='connectivity')
sc.pl.paga_compare(adata,solid_edges='connectivities',transitions='transitions_confidence',legend_fontsize=4,arrowsize=10,threshold=np.quantile(adata.uns['paga']['transitions_confidence'].data,.9),save='DYNAMICALvelocity')
adata.write(re.sub('Velocity','VelocityDYNAMICAL',newfile))
