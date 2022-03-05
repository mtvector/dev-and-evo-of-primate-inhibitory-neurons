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
import cellrank as cr
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

adata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityDYNAMICALMouseWbGe.h5ad')

adata.obs['timepoint']=adata.obs['timepoint'].astype(str)
adata.obs.loc[adata.obs['dataset_name']=='PRJNA498989_OB_mouse','timepoint']=84
adata.obs.loc[adata.obs['dataset_name']=='PRJNA515751_konopka_striatum','timepoint']=30
adata.obs.loc[adata.obs['batch_name'].str.contains('P5_',case=False),'timepoint']=26
adata.obs.loc[adata.obs['batch_name'].str.contains('p07_Cortex_SRR11947654',case=False),'timepoint']=28
adata.obs.loc[adata.obs['timepoint']=='nan','timepoint']=84
adata.obs.loc[adata.obs['timepoint'].astype(float)>100,'timepoint']=84

INgenes=['MKI67','DCX','DLX2','ZIC1','SP8','SCGN','CCK','VIP','MEIS2','PAX6','FOXP2','TSHZ1','FOXP1','OPRM1','ISL1','PENK','LHX6','SST','NPY','CHODL','CRABP1','ANGPT2','MAF','TAC2','GBX1','HMX3','TH','OTP']
INgenes=[x for x in INgenes if x in adata.var.index]
sc.pl.umap(adata,color=INgenes,use_raw=False,save='SupplementGenes')

print(adata.X)
print(np.all(np.isfinite(adata.X)))
adata=adata[:,np.isfinite(adata.X.sum(0))]
print(adata)
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
sc.tl.dendrogram(adata,groupby=hierarchy_key)
var_dict=dict(zip(adata.uns["dendrogram_"+hierarchy_key]['categories_ordered'],[topdict[x] for x in adata.uns["dendrogram_"+hierarchy_key]['categories_ordered']]))
sc.pl.matrixplot(adata,groupby=hierarchy_key,var_names=var_dict,save='top_degenes',cmap='RdBu_r',use_raw=False,dendrogram=True,standard_scale='var')

adata.obs.loc[adata.obs['dataset_name'].str.contains('dev_hypo'),'timepoint']=[sc_utils.tp_format_mouse(x) for x in adata.obs.loc[adata.obs['dataset_name'].str.contains('dev_hypo'),'batch_name']]
adata.obs['timepoint']=adata.obs['timepoint'].astype(float)

adata.obs['fixed_timepoint']=adata.obs['timepoint']
adata.obs.loc[adata.obs['fixed_timepoint']>21,'fixed_timepoint']=25
sc.pl.umap(adata,color='fixed_timepoint',save='fixed_timepoint', color_map=matplotlib.cm.RdYlBu_r)

adata.obs['region']=adata.obs['region'].astype(str)
adata.obs['region']=[re.sub('srr','',x) for x in adata.obs['region']]

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

sc.pl.umap(adata,color='latent_time',save='_linear_latent_time',color_map=matplotlib.cm.RdYlBu_r)

transition_matrix=scv.utils.get_transition_matrix(adata,vkey='linear_velocity',self_transitions=True)
transition_matrix=transition_matrix.astype(np.float64)
#transition_matrix=(transition_matrix.T/transition_matrix.sum(1)).T

transition_matrix[np.arange(0,transition_matrix.shape[0]),transition_matrix.argmax(1).A1]+=(1.0-transition_matrix.sum(1).A1)

print(np.allclose(np.sum(transition_matrix, axis=1), 1.0))

vk = cr.tl.kernels.PrecomputedKernel(transition_matrix,adata=adata)
#vk = cr.tl.kernels.VelocityKernel(adata,vkey='linear_velocity')
ck = cr.tl.kernels.ConnectivityKernel(adata)
ck.compute_transition_matrix()
vk.compute_transition_matrix()

k=vk+ck
k.compute_transition_matrix()

d={}
for c in adata.obs.agg_supervised_name.unique():
    if not any([x in c.upper() for x in ['G2-M','S-PHASE','GLIA','NAN','TRANSITION','EXCITATORY']]):
        print(c)
        select_cells=adata.obs.agg_supervised_name.isin([c])
        d[c]=[adata.obs.index[i] for i in range(adata.shape[0]) if select_cells[i]]
terminal_states=list(d.keys())+['MGE_CRABP1/MAF']

g = cr.tl.estimators.GPCCA(ck)
#g.compute_eigendecomposition()
g.compute_schur(n_components=15)

g.compute_macrostates(n_states=4, n_cells=30,cluster_key="agg_supervised_name")
adata.write('/wynton/home/ye/mschmitz1/GPCCAmacrostatesmouseAdult2.h5ad')
print(adata)
print(g)
adata.uns['macrostates']=g.macrostates
g.set_terminal_states_from_macrostates([y for y in adata.uns['macrostates'].unique() if any([x in y for x in terminal_states])])
g.compute_absorption_probabilities()
probDF=pd.DataFrame(g.absorption_probabilities,columns=g.absorption_probabilities.names,index=adata.obs.index)
adata.obs['predicted_end']=probDF.idxmax(1)
sc.pl.umap(adata,color=['predicted_end'],use_raw=False,save='GPCCA_macrostates_predicted_end')
g_fwd.plot_absorption_probabilities(same_plot=False, size=50, basis='X_umap',save='GPCCA_macrostates_all_probabiliities')
cr.tl.initial_states(adata, cluster_key='agg_supervised_name')
cr.pl.initial_states(adata, discrete=True,save='GPCCA_macrostates_initial states')

adata.obs.drop('terminal_states_probs',axis=1,inplace=True)
adata.write('/wynton/home/ye/mschmitz1/GPCCAmacrostatesmouseAdult2.h5ad')

del adata.obsm['to_terminal_states']
del adata.uns['to_terminal_states_names']
del g

#g=cr.tl.estimators.CFLARE(ck)
g = cr.tl.estimators.GPCCA(ck)
#g.compute_eigendecomposition()
g.compute_schur(n_components=50)

g.set_terminal_states(d,cluster_key='agg_supervised_name')
g.compute_absorption_probabilities(show_progress_bar=True)
#g.write('/wynton/home/ye/mschmitz1/GPCCAmouseAdult2.pickle')

probDF=pd.DataFrame(g.absorption_probabilities,columns=g.absorption_probabilities.names,index=adata.obs.index)
adata.obs['Adult']=adata.obs.dataset_name.str.contains('BICCN|OB|konopkka|adult')
probDF.loc[adata.obs['Adult'],:]=np.nan

trim=0.05
trimmedProbDF=probDF[(probDF<1-trim)]#& (probDF>trim)
probDF=(probDF-trimmedProbDF.mean(0))/trimmedProbDF.std(0)
adata.obs['predicted_end']=probDF.idxmax(1)
sc.pl.umap(adata,color=['predicted_end'],use_raw=False,save='GPPCA_predicted_end')

adata.obs.drop('terminal_states_probs',axis=1,inplace=True)
adata.write('/wynton/home/ye/mschmitz1/GPCCAmouseAdult2.h5ad')

del adata.obsm['to_terminal_states']
del adata.uns['to_terminal_states_names']
del g

g=cr.tl.estimators.CFLARE(ck)
#g = cr.tl.estimators.GPCCA(ck)
g.compute_eigendecomposition()
#g.compute_schur(n_components=15)

g.set_terminal_states(d,cluster_key='agg_supervised_name')
g.compute_absorption_probabilities(show_progress_bar=True)
#g.write('/wynton/home/ye/mschmitz1/CFLAREmouseAdult2.pickle')

probDF=pd.DataFrame(g.absorption_probabilities,columns=g.absorption_probabilities.names,index=adata.obs.index)
adata.obs['Adult']=adata.obs.dataset_name.str.contains('BICCN|OB|konopkka|adult')
probDF.loc[adata.obs['Adult'],:]=np.nan

adata.obs['eSPN_prob']=probDF['eSPN']
adata.obs['Lamp5_prob']=probDF['Lamp5']
adata.obs['Pvalb_prob']=probDF['Pvalb']
adata.obs['Sst_prob']=probDF['Sst']
adata.obs['dSPN_prob']=probDF['dSPN']
adata.obs['iSPN_prob']=probDF['iSPN']

trim=0.05
trimmedProbDF=probDF[(probDF<1-trim)]#& (probDF>trim)
probDF=(probDF-trimmedProbDF.mean(0))/trimmedProbDF.std(0)
adata.obs['predicted_end']=probDF.idxmax(1)
sc.pl.umap(adata,color=['eSPN_prob','Lamp5_prob','Pvalb_prob','Sst_prob','iSPN_prob','dSPN_prob'],use_raw=False,save='CFLARE_probabilities')
sc.pl.umap(adata,color=['predicted_end'],use_raw=False,save='CFLARE_predicted_end')

adata.obs.drop('terminal_states_probs',axis=1,inplace=True)
adata.write('/wynton/home/ye/mschmitz1/CFLAREmouseAdult2.h5ad')

