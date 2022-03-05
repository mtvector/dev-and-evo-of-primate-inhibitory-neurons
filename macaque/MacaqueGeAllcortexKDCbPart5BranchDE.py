
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
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.use('Agg')
from collections import Counter
import random
import seaborn
import sys
import shutil
import scvelo as scv
import tqdm
#Load my pipeline functions
import importlib
import importlib.util
spec = importlib.util.spec_from_file_location("ScanpyUtilsMT", os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../utils/ScanpyUtilsMT.py"))
import statsmodels
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

import diffxpy
import diffxpy.api as de

newfile='/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMacaqueGeAllcortexHippocampusProcessed.h5ad'
adata=sc.read(newfile)
'''supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MacaqueGEsupervisednamesHippoPredictedEnd.txt')
ind=adata.obs.index[adata.obs.index.isin(supercell['full_cellname'])]
supercell.index=supercell['full_cellname']
supercell=supercell.loc[ind,:]
adata.obs['latent_time']=np.nan
adata.obs['predicted_end']=''
adata.obs.loc[ind,['noctx_supervised_name','predicted_end','latent_time']]=supercell.loc[:,['noctx_supervised_name','predicted_end','latent_time']]
adata.obs.loc[adata.obs['predicted_end'].isna(),'noctx_supervised_name']=adata.obs['noctx_supervised_name'][adata.obs['predicted_end'].isna()]
'''
adata.obs['supervised_name']=adata.obs['noctx_supervised_name']
adata=adata[~adata.obs.supervised_name.str.contains('phase|Transition|G2-M',regex=True),:].copy()

adata.X=adata.raw.X[:,adata.raw.var.index.isin(adata.var.index)].todense()
adata.X=scipy.sparse.csr_matrix(adata.X)
adata=adata[:,adata.var['highly_variable']]

sf=adata.X.mean(1) 
sf = sf/sf.mean()
adata.obs['size_factors']=sf
adata.obs['platform']='V2'
adata.obs.loc[adata.obs['batch_name'].str.contains('2019'),'platform']='V3'

d={}
for c in adata.obs['supervised_name'].unique():
    print(c)
    adata.obs['current']=adata.obs['supervised_name']==c
    test = de.test.wald(
        data=adata,
        formula_loc="~ 1 + current + size_factors",
        factor_loc_totest="current",#as_numeric=['latent_time'],
        size_factors='size_factors',
        as_numeric=["size_factors"]
    )
    resultsdiffx=test.summary()
    resultsdiffx['neg_log_q']=-test.log10_qval_clean()
    resultsdiffx['signif']=(resultsdiffx['neg_log_q']>2) & (np.absolute(resultsdiffx['log2fc'])>1.2)
    resultsdiffx=resultsdiffx.loc[resultsdiffx['log2fc'].argsort(),:]
    d[c]=resultsdiffx
    
namedcolumns=[item for sublist in [q +'__' + d[q].columns for q in d.keys()] for item in sublist]
newdf=pd.concat(list(d.values()),axis=1)
newdf.columns=namedcolumns
newdf.to_csv(os.path.join(sc.settings.figdir,"DiffxpyMarkers.csv"))


sc.pp.normalize_total(adata,exclude_highly_expressed=True)
sc.pp.log1p(adata)
sc.pp.scale(adata,max_value=10)

d={}
for c in adata.obs['supervised_name'].unique():
    bdata=adata[adata.obs.supervised_name==c,:]
    lt=bdata.obs['latent_time'].to_numpy()
    cors=[]
    for x in tqdm.tqdm(range(adata.X.shape[1])):
        cors.append(list(scipy.stats.spearmanr(lt,bdata.X[:,x].flatten(),nan_policy='omit')))
    corsDF=pd.DataFrame(cors,columns=['rho','pval'],index=adata.var.index)
    qvals=statsmodels.stats.multitest.multipletests(corsDF['pval'],alpha=.01,method='bonferroni')
    corsDF['signif']=qvals[0]
    corsDF['qval']=qvals[1]
    corsDF['log10qval']=np.clip(-np.log10(qvals[1]+1e-10),-10,10)
    d[c]=corsDF

namedcolumns=[item for sublist in [q +'__' + d[q].columns for q in d.keys()] for item in sublist]
newdf=pd.concat(list(d.values()),axis=1)
newdf.columns=namedcolumns
newdf.to_csv(os.path.join(sc.settings.figdir,"BranchCorrelations.csv"))
