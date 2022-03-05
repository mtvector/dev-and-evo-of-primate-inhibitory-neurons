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
adata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseWbGeAdultProcessed.h5ad')
adata.X=adata.raw.X[:,adata.raw.var.index.isin(adata.var.index)]

adata.obs['region']=[re.sub('srr|samn','',x) for x in adata.obs['region'].astype(str)]
adata.obs['region']=[re.sub('ctx','cortex',x) for x in adata.obs['region'].astype(str)]
adata.obs['region']=[re.sub('ssctx','cortex',x) for x in adata.obs['region'].astype(str)]
adata.obs['region']=adata.obs.region.astype(str)
adata.obs.loc[adata.obs.region.str.contains('ltx'),'region']='cortex'
adata.obs.loc[adata.obs.region.str.contains('(?i)sscortex'),'region']='cortex'
adata.obs.loc[adata.obs.region.str.contains('dentgyr|^ca$'),'region']='hippocampus'

adata.obs['timepoint']=adata.obs['timepoint'].astype(str)
adata.obs.loc[adata.obs['dataset_name']=='PRJNA498989_OB_mouse','timepoint']=84
adata.obs.loc[adata.obs['dataset_name']=='PRJNA515751_konopka_striatum','timepoint']=30
adata.obs.loc[adata.obs['batch_name'].str.contains('P5_',case=False),'timepoint']=26
adata.obs.loc[adata.obs['batch_name'].str.contains('p07_Cortex_SRR11947654',case=False),'timepoint']=28
adata.obs.loc[adata.obs['timepoint']=='nan','timepoint']=84
adata.obs.loc[adata.obs['timepoint'].astype(float)>100,'timepoint']=84

adata.obs.loc[adata.obs['dataset_name'].str.contains('dev_hypo'),'timepoint']=[sc_utils.tp_format_mouse(x) for x in adata.obs.loc[adata.obs['dataset_name'].str.contains('dev_hypo'),'batch_name']]
adata.obs['timepoint']=adata.obs['timepoint'].astype(float)

#Remove saunders dropseq
adata=adata[~adata.obs['dataset_name'].str.contains('PRJNA478603'),:]
adult_sets=['BICCN','PRJNA498989','PRJNA515751','PRJNA547712','SRP135960']#'PRJNA478603'
print('|'.join(adult_sets),flush=True)
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

dlx_positive=(adata[:,['DLX2','DLX5','GAD1','GAD2']].X.todense().sum(1)>1).A1#.toarray().flatten()
inhibs=(dlx_positive | ~(adata.obs['dataset_name'].str.contains('|'.join(adult_sets))))
print(adata[:,['DLX2','DLX5','GAD1','GAD2']].X.todense(),flush=True)
print(adata[:,['DLX2','DLX5','GAD1','GAD2']].X.todense().sum(1).A1,flush=True)
print(dlx_positive,flush=True)
print(inhibs,flush=True)
print(adata)
adata=adata[inhibs,:]
print(adata,flush=True)
adata.obs['batch_name'].cat.remove_unused_categories(inplace=True)
sc.pp.normalize_total(adata,exclude_highly_expressed=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata,n_top_genes=12000,batch_key='dataset_name',subset=False)#'dataset_name'
adata.var['highly_variable']=(adata.var['highly_variable']&(adata.var['highly_variable_nbatches']>7))
sc.pp.scale(adata,max_value=10)
sc.pp.pca(adata,n_comps=100)
#sc.pp.neighbors(adata)
bbknn.bbknn(adata,batch_key='dataset_name',n_pcs=100,neighbors_within_batch=12)
sc.tl.leiden(adata,resolution=10)
sc.tl.umap(adata,spread=3,min_dist=.2)

allencells=pd.read_csv('/wynton/group/ye/mtschmitz/mousefastqpool/BICCN_metadata.csv')
allennamedf=allencells['sample_name'].str.split('-|_',expand=True)
adatasplitnamedf=pd.DataFrame(adata.obs.index)[0].str.split('_',expand=True)
allenids=pd.DataFrame(allennamedf[0]+'_'+allennamedf[1]+'_'+allennamedf[2]+'_'+allennamedf[4])
allenids.index=allenids[0]
allenids['realind']=allencells.index
adataids=pd.DataFrame(adatasplitnamedf[0]+'_'+adatasplitnamedf[2]+'_'+adatasplitnamedf[3]+'_'+adatasplitnamedf[5])
adataids.index=adataids[0]
adataids['realind']=adata.obs.index
adata.obs['allen_cluster_label']='nan'
adata.obs['allen_class_label']='nan'
adata.obs.loc[adataids.loc[set(adataids[0])&set(allenids[0]),'realind'],'allen_cluster_label']=allencells.loc[allenids.loc[set(adataids[0])&set(allenids[0]),'realind'],:]['cluster_label'].tolist()
adata.obs.loc[adataids.loc[set(adataids[0])&set(allenids[0]),'realind'],'allen_class_label']=allencells.loc[allenids.loc[set(adataids[0])&set(allenids[0]),'realind'],:]['class_label'].tolist()
adata.obs['simplified_allen']=adata.obs['allen_cluster_label'].str.split('_',expand=True)[1]
adata.obs.loc[adata.obs['simplified_allen'].astype(str).str.contains('(?i)CTX|Car3'),'simplified_allen']='Excitatory'
sc.pl.umap(adata,color=['allen_class_label','allen_cluster_label','simplified_allen'],save='allenlabels')

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
    'nan':'white',
    'Str_LHX8/CHAT':'olivedrab',
    'Amy/Hypo_HAP1/PEG10':'darkslateblue',
    'GP_GBX1/GABRA1':'teal',
    'vSTR_HAP1/ZIC1':'darkslategray',
    'Excitatory':'whitesmoke',
    'Ctx_PROX1/LAMP5':'darkgoldenrod', 
    'Ctx_LHX6/LAMP5':'rosybrown', 
    'OB-GC_RPRM':'palegoldenrod', 
    'OB-GC_STXBP6/PENK':'olive', 
    'OB-PGC_FOXP2/CALB1':'aquamarine', 
    'OB-PGC_TH/SCGN':'orange', 
    'OB-PGC_ZIC':'saddlebrown', 
    'Ctx_LHX6/PVALB':'hotpink', 
    'Ctx_PROX1/SNCG':'lightslategray', 
    'Ctx_LHX6/SST':'salmon', 
    'Ctx/BN_SST/CHODL':'maroon',
    'Ctx_CCK/VIP':'tan',
    'Ctx_PROX1/PAX6':'sienna',
    'Ctx_PVALB/VIPR2':'lightcoral',
    'BN-eSPN_FOXP2/TSHZ1':'springgreen', 
    'Str-dSPN_FOXP1/ISL1':'cyan', 
    'Str-iSPN_FOXP1/PENK':'navy',
    'vStr_DRD1/NPY1R':'violet',
    'Str-IN_CRABP1/MAF':'black',
    'Glia':'lightgoldenrodyellow',
    'OB-GC NR2F2/PENK':'brown',
    'Ctx_LAMP5/NDNF':'khaki',
    'Ctx_SST/NDNF':'peachpuff',
    'Ctx_CCK/DPY19L1':'dimgray'}

sc.pl.umap(adata,color=['supervised_name'],save='supervised_name')
#adata=adata[adata.obs['supervised_name'].astype(str)!='nan',:]
#adata.obs['supervised_name'].cat.remove_unused_categories(inplace=True)
my_pal=[paldict[x] for x in adata.obs['supervised_name'].cat.categories]
        
adata.write('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseWbGeAdult2Processed.h5ad')

INgenes=['MKI67','DCX','DLX2','ZIC1','SP8','SCGN','CCK','VIP','MEIS2','PAX6','FOXP2','TSHZ1','FOXP1','OPRM1','ISL1','PENK','LHX6','SST','NPY','CHODL','CRABP1','ANGPT2','MAF','TAC2']         
INgenes=[x for x in INgenes if x in adata.var.index]
sc.pl.umap(adata,color=INgenes,use_raw=False,save='SupplementGenes')

sc.pl.umap(adata,color=['leiden'],save='leiden')
sc.pl.umap(adata,color=['old_leiden'],save='old_leiden')
sc.pl.umap(adata,color=['batch_name'],save='batch_name')
sc.pl.umap(adata,color=['timepoint'],save='timepoint', color_map=matplotlib.cm.RdYlBu_r)
sc.pl.umap(adata,color=['supervised_name'],legend_loc='on data',legend_fontsize=4,palette=my_pal,save='supervised_name_onplot')
sc.pl.umap(adata,color=['supervised_name'],save='supervised_name')
sc.pl.umap(adata,color=['leiden'],legend_loc='on data',legend_fontsize=4,save='leiden_onplot')
#sc.pl.umap(adata,color=['region'],save='region')
sc.pl.umap(adata,color=['dataset_name'],save='dataset_name')
easygenes= ['MKI67','SOX2','AQP4','EDNRB','IL33','PDGFRA','HOPX','ERBB4','CALB1','CALB2','GAD1','GAD2','GADD45G','LHX6','LHX7','LHX8','RBP4','RXRA','NXPH1','NXPH2','SST','NKX2-1','MAF','SP8','SP9','PROX1','VIP','CCK','NPY','LAMP5','HTR1A','HTR3A','NR2F1','NR2F2','TOX3','ETV1','SCGN','FOXP1','FOXP2','FOXP4','TSHZ1','OPRM1','CASZ1','HMX3','NKX6-2','TH','DDC','SLC18A2','PAX6','MEIS2','SHTN1','ISL1','PENK','DRD1','ADORA2A','CHAT','ZNF503','STXBP6','CRABP1','ZIC1','ZIC2','FOXG1','TAC1','TAC2','TAC3','TACR1','TACR2','TACR3','EBF1','DLX1','DLX2','DLX5','DLX6','GSX2','RBFOX3','PPP1R17','EOMES','DCX','TUBB3','BCL11B','TLE4','FEZF2','SATB2','TBR1','AIF1','RELN','PECAM1','HBZ','ROBO1','ROBO2','ROBO3','ROBO4']
easygenes=[x for x in easygenes if x in adata.var.index]
sc.pl.umap(adata,color=easygenes,use_raw=False,save='FaveGenes')

INgenes=['MKI67','DCX','DLX2','ZIC1','SP8','SCGN','CCK','VIP','MEIS2','PAX6','FOXP2','TSHZ1','FOXP1','OPRM1','ISL1','PENK','LHX6','SST','NPY','CHODL','CRABP1','ANGPT2','MAF','TAC2']         
INgenes=[x for x in INgenes if x in adata.var.index]
sc.pl.umap(adata,color=INgenes,use_raw=False,save='SupplementGenes')

sc.pl.umap(adata,color=['supervised_name'],save='supervised_name',palette=my_pal)
sc.pl.umap(adata,color=['supervised_name'],legend_loc='on data',palette=my_pal,legend_fontsize=4,save='supervised_name_onplot')

adata.obs_names_make_unique()
adata.var_names_make_unique()

sc_utils.cell_cycle_score(adata)
#sc_utils.marker_analysis(adata,variables=['leiden'],markerpath=os.path.expanduser('~/markers.txt')) 
#sc_utils.log_reg_diff_exp(adata,obs_name='leiden')#,method='t-test_overestim_var')
#sc_utils.log_reg_diff_exp(adata,obs_name='supervised_name')#,method='t-test_overestim_var'

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
sc.pl.matrixplot(adata,groupby=hierarchy_key,var_names=var_dict,save='top_degenes'+hierarchy_key,cmap='RdBu_r',use_raw=False,dendrogram=True)

sufficient_cells=adata.obs['leiden'].value_counts().index[adata.obs['leiden'].value_counts()>5]
adata=adata[adata.obs['leiden'].isin(sufficient_cells),:]
adata.obs['leiden'].cat.remove_unused_categories(inplace=True)

hierarchy_key='leiden'
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
sc.pl.matrixplot(adata,groupby=hierarchy_key,var_names=var_dict,save='top_degenes'+hierarchy_key,cmap='RdBu_r',use_raw=False,dendrogram=True)

