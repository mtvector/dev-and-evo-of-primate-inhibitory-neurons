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
print(dir(sc_utils))

def most_frequent(List): 
    return max(set(List), key = List.count)

def variance(X,axis=1):
    return( (X.power(2)).mean(axis=axis)-(np.power(X.mean(axis=axis),2)) )

newfile='/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseWbGePresupervisionProcessed.h5ad'

adata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseWbPresupervision.h5ad')
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(adata.obs['dataset_name'])
print(adata.obs['batch_name'])

adata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseWbPresupervisionProcessed.h5ad')
sc.pl.umap(adata,color=['dataset_name'],save='test_dataset_name')
adata.obs['region']=[re.sub('srr','',x) for x in adata.obs['region'].astype(str)]
adata.obs['region']=[re.sub('ctx','cortex',x) for x in adata.obs['region'].astype(str)]
adata.obs['region']=[re.sub('ssctx','cortex',x) for x in adata.obs['region'].astype(str)]
adata.obs['region']=adata.obs['region'].astype(str)
adata=adata[~adata.obs.batch_name.str.contains('(?i)P10_'),:]
adata=adata[~adata.obs.batch_name.str.contains('(?i)thalamic'),:]
adata=adata[~adata.obs.batch_name.str.contains('(?i)hindbrain'),:]
adata=adata[~adata.obs.batch_name.str.contains('(?i)midbrain'),:]
adata=adata[~adata.obs.dataset_name.str.contains('(?i)dropseq'),:]

adata.obs['supervised_name']='nan'
adata.obs['old_leiden']='nan'

supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MouseSupervisednames5PreserveCompoundname.txt')
supercell=supercell.loc[supercell['supervised_name']!='nan',:]
ind=adata.obs.index[adata.obs['full_cellname'].isin(supercell['full_cellname'])]
supercell.index=supercell['full_cellname']
supercell=supercell.loc[ind,:]
if 'old_leiden' in adata.obs.columns:
    adata.obs['old_leiden']=adata.obs['old_leiden'].astype(str)
if 'supervised_name' in adata.obs.columns:
    adata.obs['supervised_name']=adata.obs['supervised_name'].astype(str)
print(supercell)
adata.obs.loc[ind,'old_leiden']=supercell['leiden']
adata.obs.loc[ind,'supervised_name']=supercell['supervised_name']
adata.obs['old_leiden']=[re.sub('\.0','',str(x)) for x in adata.obs['old_leiden']]

supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MouseSupervisednames4PreserveCompoundname.txt')
supercell=supercell.loc[supercell['supervised_name']!='nan',:]
ind=adata.obs.index[adata.obs['full_cellname'].isin(supercell['full_cellname'])]
supercell.index=supercell['full_cellname']
supercell=supercell.loc[ind,:]
if 'old_leiden' in adata.obs.columns:
    adata.obs['old_leiden']=adata.obs['old_leiden'].astype(str)
if 'supervised_name' in adata.obs.columns:
    adata.obs['supervised_name']=adata.obs['supervised_name'].astype(str)
print(supercell)
adata.obs.loc[ind,'old_leiden']=supercell['leiden']
adata.obs.loc[ind,'supervised_name']=supercell['supervised_name']
adata.obs['old_leiden']=[re.sub('\.0','',str(x)) for x in adata.obs['old_leiden']]

inhibmat = adata[:,['DLX1','DLX2','DLX5','DLX6','GAD1','GAD2']].to_df()
inhibmat['leiden']=adata.obs['leiden']
meanmat=inhibmat.groupby('leiden').mean()
print(meanmat)
print(meanmat>meanmat.mean(0))
boolmat=(meanmat>meanmat.mean(0)).sum(1)
print(boolmat.index[boolmat>=3])
adata.obs['supervised_name']=adata.obs['supervised_name'].astype(str)
adata=adata[adata.obs['leiden'].isin(boolmat.index[boolmat>=3])|~adata.obs['supervised_name'].str.contains('nan|MCM4|ASPM|Transition'),:]
adata=adata[~adata.obs['supervised_name'].str.contains('CTX_IPC'),:]

print(adata,flush=True)
adata.X=adata.raw.X[:,adata.raw.var.index.isin(adata.var.index)].todense()
sc.pp.filter_cells(adata,min_genes=800)


sufficient_cells=adata.obs['batch_name'].value_counts().index[adata.obs['batch_name'].value_counts()>20]
adata=adata[adata.obs['batch_name'].isin(sufficient_cells),:]
sc.pl.umap(adata,color='supervised_name',save='test_supervised_name')

adata.obs['batch_name'].cat.remove_unused_categories(inplace=True)
sc.pp.normalize_total(adata,exclude_highly_expressed=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata,n_top_genes=12000,batch_key='dataset_name',subset=False)#'dataset_name'
sc.pp.scale(adata,max_value=10)
sc.pp.pca(adata,n_comps=100)
#sc.pp.neighbors(adata)
bbknn.bbknn(adata,batch_key='batch_name',n_pcs=100,neighbors_within_batch=3)
sc.tl.leiden(adata,resolution=6)
sc.tl.umap(adata,spread=3,min_dist=.2)

sc.pl.umap(adata,color=['leiden'],save='precull_leiden')
adata.write('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseWbGePresupervisionPrecull.h5ad')

####
adata.obs['supervised_name']=adata.obs['supervised_name'].astype(str)
adata.obs['leiden']=adata.obs['leiden'].astype(str)
countmat=adata.obs.astype(str).groupby(['leiden', 'supervised_name']).size().reset_index().pivot(columns='supervised_name', index='leiden', values=0)
if 'nan' in countmat.columns:
    countmat=countmat.drop('nan',axis=1)
countmat=countmat.div(countmat.sum(axis=1),axis=0)
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(countmat,flush=True)
leidentosuper=pd.Series(dict(countmat.idxmax(1)))
leidentosuper=dict(leidentosuper)
adata.obs['supervised_name']=adata.obs['leiden'].replace(leidentosuper)
divcells=adata.obs['supervised_name'].isin(['G2-M_UBE2C/ASPM','S-phase_MCM4/H43C','Transition'])

cortexregions=['head','forebrain','cortex','motor']
adata.obs.loc[adata.obs['region'].str.contains('|'.join(cortexregions)),'supervised_name'].replace({'G2-M_UBE2C/ASPM':'nan','S-phase_MCM4/H43C':'nan','Transition':'nan'})
dlx_positive=(adata[:,['DLX1','DLX2','DLX5','DLX6']].raw.X.todense().sum(1)>1).A1#.toarray().flatten()
cortical_inhib_progens=(dlx_positive & (adata.obs['region'].str.contains('|'.join(cortexregions))))

adata=adata[cortical_inhib_progens | (adata.obs['supervised_name']!='nan') ,:]
adata.obs.region=adata.obs.region.astype(str)
print(adata.obs.region.isin(['ob','OB']))
print(adata.obs.supervised_name.str.contains('"'),flush=True)
print(adata)
adata=adata[(~adata.obs.region.str.contains('ob|OB').fillna(False) &  ~adata.obs.supervised_name.str.contains('"').fillna(False)) | adata.obs.supervised_name.str.contains('OB').fillna(False),:]
print(adata)

#Redo PCA using only the DLX+ progenitors
sc.pp.pca(adata,n_comps=50)
#sc.pp.neighbors(adata)
bbknn.bbknn(adata,batch_key='batch_name',n_pcs=50,neighbors_within_batch=3)
sc.tl.leiden(adata,resolution=6)
sc.tl.umap(adata,spread=3,min_dist=.2)


####

paldict={'CGE_NR2F2/PROX1': 'slategray',
    'G1-phase_SLC1A3/ATP1A1': '#17344c',
    'G2-M_UBE2C/ASPM': '#19122b',
    'LGE_FOXP1/ISL1': 'cyan',
    'LGE_FOXP1/PENK': 'navy',
    'LGE_FOXP2/TSHZ1': 'goldenrod',
    'LGE_MEIS2/PAX6': 'orangered',
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
    'nan':'black'
}

countmat=adata.obs.astype(str).groupby(['leiden', 'supervised_name']).size().reset_index().pivot(columns='supervised_name', index='leiden', values=0)
print(countmat)
if 'nan' in countmat.columns:
    countmat=countmat.drop('nan',axis=1)
leidentosuper=pd.Series(dict(countmat.idxmax(1)))
leidentosuper=leidentosuper.astype(str)
leidentosuper.index=[re.sub('\.0','',str(x)) for x in leidentosuper.index]
leidentosuper=dict(leidentosuper)
print(leidentosuper)
adata.obs['supervised_name']=adata.obs['leiden'].replace(leidentosuper)

adata.obs.loc[:,['full_cellname','supervised_name','leiden']].to_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MouseSupervisednames6Preserve.txt',index=None)

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
#sc.pl.umap(adata,color=['region'],save='region')
sc.pl.umap(adata,color=['dataset_name'],save='dataset_name')
easygenes= ['MKI67','SOX2','AQP4','EDNRB','IL33','PDGFRA','HOPX','ERBB4','CALB1','CALB2','GAD1','GAD2','GADD45G','LHX6','LHX7','LHX8','RBP4','RXRA','NXPH1','NXPH2','SST','NKX2-1','MAF','SP8','SP9','PROX1','VIP','CCK','NPY','LAMP5','HTR1A','HTR3A','NR2F1','NR2F2','TOX3','ETV1','SCGN','FOXP1','FOXP2','FOXP4','TSHZ1','OPRM1','CASZ1','HMX3','NKX6-2','TH','DDC','SLC18A2','PAX6','MEIS2','SHTN1','ISL1','PENK','DRD1','ADORA2A','CHAT','ZNF503','STXBP6','CRABP1','ZIC1','ZIC2','FOXG1','TAC1','TAC2','TAC3','TACR1','TACR2','TACR3','EBF1','DLX5','GSX2','RBFOX3','PPP1R17','EOMES','DCX','TUBB3','BCL11B','TLE4','FEZF2','SATB2','TBR1','AIF1','RELN','PECAM1','HBZ','ROBO1','ROBO2','ROBO3','ROBO4']
easygenes=[x for x in easygenes if x in adata.var.index]
sc.pl.umap(adata,color=easygenes,use_raw=False,save='FaveGenes')

INgenes=['MKI67','DCX','DLX2','ZIC1','SP8','SCGN','CCK','VIP','MEIS2','PAX6','FOXP2','TSHZ1','FOXP1','OPRM1','ISL1','PENK','LHX6','SST','NPY','CHODL','CRABP1','ANGPT2','MAF','TAC2']         
INgenes=[x for x in INgenes if x in adata.var.index]
sc.pl.umap(adata,color=INgenes,use_raw=False,save='SupplementGenes')

sc.pl.umap(adata,color=['supervised_name'],save='supervised_name',palette=my_pal)
sc.pl.umap(adata,color=['supervised_name'],legend_loc='on data',legend_fontsize=4,save='supervised_name_onplot')

sc_utils.cell_cycle_score(adata)
sc_utils.marker_analysis(adata,variables=['leiden'],markerpath=os.path.expanduser('~/markers.txt')) 
sc_utils.log_reg_diff_exp(adata)#,method='t-test_overestim_var')
sc_utils.log_reg_diff_exp(adata,obs_name='leiden')#,method='t-test_overestim_var')
sc_utils.log_reg_diff_exp(adata,obs_name='supervised_name')#,method='t-test_overestim_var')


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
var_dict=dict(zip(adata.uns["dendrogram_['"+hierarchy_key+"']"]['categories_ordered'],[topdict[x] for x in adata.uns["dendrogram_['"+hierarchy_key+"']"]['categories_ordered']]))
sc.pl.matrixplot(adata,groupby=hierarchy_key,var_names=var_dict,save='top_degenes',cmap='RdBu_r',use_raw=False,dendrogram=True)
