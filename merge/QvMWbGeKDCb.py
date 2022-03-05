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
import tqdm
import scvelo as scv
import bbknn
import tqdm
#Load my pipeline functions
import importlib
import importlib.util
spec = importlib.util.spec_from_file_location("ScanpyUtilsMT", os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../utils/ScanpyUtilsMT.py"))
sc_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc_utils)
sc.settings.figdir='/wynton/group/ye/mtschmitz/figures/qvmGE/'
scv.settings.figdir='/wynton/group/ye/mtschmitz/figures/qvmGE/'
sc.settings.file_format_figs='pdf'
sc.settings.autosave=True
sc.settings.autoshow=False

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

def most_frequent(List): 
    return max(set(List), key = List.count)

def variance(X,axis=1):
    return( (X.power(2)).mean(axis=axis)-(np.power(X.mean(axis=axis),2)) )

orthos=pd.read_csv('/wynton/home/ye/mschmitz1/utils/HOM_AllOrganism.rpt',sep='\t')
orthos=orthos.loc[orthos['NCBI Taxon ID'].isin([10090,9606]),:]
classcounts=orthos['DB Class Key'].value_counts()
one2one=classcounts.index[list(classcounts==2)]
orthos=orthos.loc[orthos['DB Class Key'].isin(one2one),:]

htab=orthos.loc[orthos['NCBI Taxon ID']==9606,:]
mtab=orthos.loc[orthos['NCBI Taxon ID']==10090,:]
genemapping=dict(zip([x.upper() for x in mtab['Symbol']],htab['Symbol']))
print(len(genemapping.keys()),flush=True)
print(htab)
print(mtab,flush=True)
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
    'Cholinergic':'olivedrab',
    'Amy/Hypo_HAP1':'darkslateblue',
    'GP_GBX1':'teal',
    'NAc_HAP1':'darkslategray',
    'Excitatory':'whitesmoke',
    'Lamp5':'darkgoldenrod', 
    'Lamp5 Lhx6':'rosybrown', 
    'OB-GC_RPRM':'palegoldenrod', 
    'OB-GC_STXBP6/PENK':'olive', 
    'OB-PGC_FOXP2/CALB1':'aquamarine', 
    'OB-PGC_TH/SCGN':'orange', 
    'OB-PGC_ZIC':'saddlebrown', 
    'Pvalb':'hotpink', 
    'Sncg':'lightslategray', 
    'Sst':'salmon', 
    'Sst Chodl':'maroon',
    'Vip':'tan',
    'PAX6':'sienna',
    'Pvalb Vipr2':'lightcoral',
    'eSPN':'springgreen', 
    'dSPN':'cyan', 
    'iSPN':'navy',
    'v-dSPN':'violet',
    'Str_IN':'black',
    'Glia':'lightgoldenrodyellow'}

if True:#not os.path.exists('/wynton/group/ye/mtschmitz/CAT202002_h5ad/KDCbProcessedQvMMotor.h5ad'):
    qdata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMacaqueGeAllcortexHippocampusProcessed.h5ad')
    #species norm
    #qdata.X=qdata.raw.X[:,qdata.raw.var.index.isin(qdata.var.index)]#.todense()
    qdata.obs['supervised_name']=qdata.obs['noctx_supervised_name']
    qdata.var['maq_highly_variable']=list(qdata.var['highly_variable'])
    
    #hdata=sc.read(os.path.join('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityHumanMotor.h5ad'))
    #mdata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/MouseBrain.h5ad')
    #mdata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseGEClean.h5ad')
    mdata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseWbGeProcessed.h5ad')
    #species norm
    #mdata.X=mdata.raw.X[:,mdata.raw.var.index.isin(mdata.var.index)]#.todense()
    mdata.var['mouse_highly_variable']=list(mdata.var['highly_variable'])
    mdata=mdata[:,mdata.var.index.isin(genemapping.keys())]
    
    mdata.var.index=[x.upper() for x in mdata.var.index]
    qdata.var.index=[x.upper() for x in qdata.var.index]
    mdata.var.index=[genemapping[x] for x in mdata.var.index]
    qdata.obs['species']='Macaque'
    mdata.obs['species']='Mouse'
    qdata.var_names_make_unique()
    mdata.var_names_make_unique()
    qdata.obs_names_make_unique()
    mdata.obs_names_make_unique()
    print(mdata.var.index)
    print(qdata.var.index)
    adata=sc.AnnData.concatenate(qdata,mdata,batch_key='species_batch',index_unique=None)
    print(adata.var)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    mito_genes = [name for name in adata.var_names if name in ['ND1','ND2','ND4L','ND4','ND5','ND6','ATP6','ATP8','CYTB','COX1','COX2','COX3'] or name.startswith('CHRM-') or name.startswith('MT-')]
    adata._inplace_subset_var(~adata.var.index.isin(mito_genes))

    #species norm
    #adata.X=adata.raw.X[:,adata.raw.var.index.isin(adata.var.index)]
    #sc.pp.filter_genes(adata,min_cells=10)
    #sc.pp.normalize_total(adata,exclude_highly_expressed=True)
    #sc.pp.log1p(adata)
    #sc.pp.highly_variable_genes(adata,n_top_genes=6000,batch_key='species')
    #adata.var['highly_variable']=(adata.var['highly_variable']&(adata.var['highly_variable_nbatches']>1))
    #sc.pp.scale(adata,max_value=10)
    adata.var['highly_variable']=(adata.var['maq_highly_variable-0']&adata.var['mouse_highly_variable-1'])
    sc.pp.pca(adata,n_comps=50)
    #adata.obsm['X_pca']=adata.obsm['X_pca'][:,1:]
    #sc.pp.neighbors(adata)
    scv.pp.remove_duplicate_cells(adata)
    bbknn.bbknn(adata,batch_key='species',n_pcs=50,neighbors_within_batch=25)
    sc.tl.leiden(adata,resolution=1.5)#.8 works for cross species when not marker subsetted
    sc.tl.umap(adata)
    #adata=sc.AnnData(adata.raw.X,var=adata.raw.var,obsm=adata.obsm,uns=adata.uns,obs=adata.obs)
    #adata.raw=adata
    #sc.pp.normalize_per_cell(adata)
    #sc.pp.log1p(adata)
    #sc.pp.scale(adata,max_value=10)
    sc.pl.umap(adata,color=['leiden'],save='leiden')
    sc.pl.umap(adata,color=['species'],save='species')
    sc.pl.umap(adata,color=['batch_name'],save='batch_name')
    sc.pl.umap(adata,color=['region'],save='region')
    sc.pl.umap(adata,color=['timepoint'],save='timepoint')
    easygenes= ['MKI67','SOX2','AQP4','EDNRB','IL33','PDGFRA','HOPX','ERBB4','CALB1','CALB2','GAD1','GAD2','GADD45G','LHX6','LHX7','SST','NKX2-1','MAF','SP8','SP9','PROX1','VIP','CCK','NPY','LAMP5','HTR3A','NR2F1','NR2F2','TOX3','ETV1','SCGN','FOXP1','FOXP2','FOXP4','TH','DDC','SLC18A2','PAX6','MEIS2','SHTN1','ISL1','PENK','CRABP1','ZIC1','ZIC2','EBF1','DLX5','GSX2','RBFOX3','PPP1R17','EOMES','DCX','TUBB3','BCL11B','TLE4','FEZF2','SATB2','TBR1','AIF1','RELN','PECAM1','HBZ','ROBO1','ROBO2','ROBO3','ROBO4']
    easygenes=[x for x in easygenes if x in adata.var.index]
    sc.pl.umap(adata,color=easygenes,use_raw=False,save='FaveGenes')
    sc.pl.umap(adata,color=['supervised_name'],save='supervised_name')
    my_pal=[paldict[x] for x in adata.obs['supervised_name'].cat.categories]
    sc.pl.umap(adata,color=['supervised_name'],palette=my_pal,save='supervised_name')
    #sc_utils.cell_cycle_score(adata)

    #sc_utils.marker_analysis(adata,variables=['leiden'],markerpath=os.path.expanduser('~/markers.txt'))
    #sc_utils.log_reg_diff_exp(adata)
    #sc_utils.log_reg_diff_exp(adata,obs_name='supervised_name')
    adata.write('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbProcessedQvMMotor.h5ad')

adata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbProcessedQvMMotor.h5ad')
#This needs to go in a separate file if we're going to do RNA velocity
#Specifically should load only a processed HMQ object, rename
#Then only load names here or the KNN will be fqd below

seaborn.set(font_scale=.7)

df=pd.DataFrame(adata.X)
df['noctx_supervised_name']=list(adata.obs.noctx_supervised_name)
gb=df.groupby(['noctx_supervised_name']).mean()
seaborn.clustermap(gb.T.corr(),xticklabels=True,yticklabels=True)
plt.savefig(os.path.join(sc.settings.figdir,'AllclustersCorr.pdf'), bbox_inches="tight")

df=pd.DataFrame(adata[:,adata.var['highly_variable']].X)
df['supervised_name']=list(adata.obs.supervised_name)
gb=df.groupby(['supervised_name']).mean()
seaborn.clustermap(gb.T.corr(),xticklabels=True,yticklabels=True)
plt.savefig(os.path.join(sc.settings.figdir,'AllclustersCorrMarkers.pdf'), bbox_inches="tight")


df1=df.loc[list(adata.obs.species.isin(['Mouse'])),:].groupby(['supervised_name']).mean()
corr = 1 - df1.T.corr() 
corr_condensed = scipy.cluster.hierarchy.distance.squareform(corr) # convert to condensed
df1_link = scipy.cluster.hierarchy.linkage(corr_condensed, method='average')
df2=df.loc[list(adata.obs['species'].isin(['Macaque'])),:].groupby(['supervised_name']).mean()
corr = 1 - df2.T.corr() 
corr_condensed = scipy.cluster.hierarchy.distance.squareform(corr) # convert to condensed
df2_link = scipy.cluster.hierarchy.linkage(corr_condensed, method='average')

corrmat=pd.DataFrame(corr2_coeff(np.array(df1),np.array(df2)),index=df1.index,columns=df2.index)
xind=list(corrmat.index)
yind=list(corrmat.columns)
xind=list(corrmat.index)
xind.remove('Transition')
xind.insert(0, 'Transition')
xind.remove('G2-M_UBE2C/ASPM')
xind.insert(0, 'G2-M_UBE2C/ASPM')
xind.remove('S-phase_MCM4/H43C')
xind.insert(0, 'S-phase_MCM4/H43C')
xind.remove('LGE-OB_MEIS2/PAX6')
xind.insert(8, 'LGE-OB_MEIS2/PAX6')
yind=list(corrmat.columns)
yind.remove('Transition')
yind.insert(0, 'Transition')
yind.remove('G2-M_UBE2C/ASPM')
yind.insert(0, 'G2-M_UBE2C/ASPM')
yind.remove('S-phase_MCM4/H43C')
yind.insert(0, 'S-phase_MCM4/H43C')
yind.remove('G1-phase_SLC1A3/ATP1A1')
yind.insert(0, 'G1-phase_SLC1A3/ATP1A1')
print(xind)
print(yind)
print(corrmat)
seaborn.heatmap(corrmat.loc[xind,yind],cmap='coolwarm')
seaborn.set(font_scale=.4)
plt.savefig(os.path.join(sc.settings.figdir,'MouseVsMacaqueCorrNoclustermap.pdf'), bbox_inches="tight")

adata.obs['new_supervised_name']='nan'
adata.obs['new_supervised_name']=adata.obs['supervised_name']
del adata.uns['neighbors']
adata.obs['new_supervised_name']=adata.obs['new_supervised_name'].astype(str)
adata.obs.loc[adata.obs['species']!='Mouse','new_supervised_name']='nan'
adata.obs['nanlabel']=adata.obs['new_supervised_name']=='nan'
print('redo neighbors',flush=True)
bbknn.bbknn(adata,batch_key='nanlabel',n_pcs=50,neighbors_within_batch=10)
adata.uns['neighbors']['distances'].nonzero()
supervised=adata.obs['new_supervised_name']
spec=list(adata.obs['species'])
nz0=adata.uns['neighbors']['distances'].nonzero()[0]
nz1=adata.uns['neighbors']['distances'].nonzero()[1]
nz0i=adata.obs.index[nz0]
nz1i=adata.obs.index[nz1]
results=[nz0,
         nz1,
         adata.obs.loc[nz0i,'species'],
         adata.obs.loc[nz1i,'species'],
         adata.obs.loc[nz0i,'new_supervised_name'],
         adata.obs.loc[nz1i,'new_supervised_name']]
results=[list(x) for x in results]
df=pd.DataFrame(results,index=['ind1','ind2','species1','species2','classname1','classname2']).T
mat=df.groupby('ind1')['classname2'].value_counts(dropna=False).unstack('classname2').fillna(0)
mat=mat.loc[:,mat.columns!='nan']
#adata.obs.loc[adata.obs.index[mat.index],'new_supervised_name']=mat.idxmax(1)
adata.obs['new_supervised_name']=list(mat.idxmax(1))
adata.obs.loc[:,['full_cellname','new_supervised_name','leiden']].to_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MtoQSupervisedCellNames.txt',index=None)
sc.pl.umap(adata,color=['supervised_name'],palette=my_pal,save='MtoQ_supervised_name')

adata.obs['new_supervised_name']='nan'
adata.obs['new_supervised_name']=adata.obs['supervised_name']
del adata.uns['neighbors']
adata.obs['new_supervised_name']=adata.obs['new_supervised_name'].astype(str)
adata.obs.loc[adata.obs['species']!='Macaque','new_supervised_name']='nan'
adata.obs['nanlabel']=adata.obs['new_supervised_name']=='nan'
print('redo neighbors',flush=True)
bbknn.bbknn(adata,batch_key='nanlabel',n_pcs=50,neighbors_within_batch=10)
adata.uns['neighbors']['distances'].nonzero()
supervised=adata.obs['new_supervised_name']
spec=list(adata.obs['species'])
nz0=adata.uns['neighbors']['distances'].nonzero()[0]
nz1=adata.uns['neighbors']['distances'].nonzero()[1]
nz0i=adata.obs.index[nz0]
nz1i=adata.obs.index[nz1]
results=[nz0,
         nz1,
         adata.obs.loc[nz0i,'species'],
         adata.obs.loc[nz1i,'species'],
         adata.obs.loc[nz0i,'new_supervised_name'],
         adata.obs.loc[nz1i,'new_supervised_name']]
results=[list(x) for x in results]
df=pd.DataFrame(results,index=['ind1','ind2','species1','species2','classname1','classname2']).T
mat=df.groupby('ind1')['classname2'].value_counts(dropna=False).unstack('classname2').fillna(0)
mat=mat.loc[:,mat.columns!='nan']
#adata.obs.loc[adata.obs.index[mat.index],'new_supervised_name']=mat.idxmax(1)
adata.obs['new_supervised_name']=list(mat.idxmax(1))
adata.obs.loc[:,['full_cellname','new_supervised_name','leiden']].to_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/QtoMSupervisedCellNames.txt',index=None)
sc.pl.umap(adata,color=['supervised_name'],save='QtoM_supervised_name')
f = plt.figure()
df_plot = adata[adata.obs['species']=='Mouse',].obs.groupby(['supervised_name', 'new_supervised_name']).size().reset_index().pivot(columns='new_supervised_name', index='supervised_name', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
plt.xticks(fontsize=4)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'QtoMsupervisedtransferBar.pdf'), bbox_inches="tight")

f = plt.figure()
df_plot = adata[adata.obs['species']=='Macaque',].obs.groupby(['supervised_name', 'new_supervised_name']).size().reset_index().pivot(columns='new_supervised_name', index='supervised_name', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
plt.xticks(fontsize=4)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'MtoQsupervisedtransferBar.pdf'), bbox_inches="tight")


f = plt.figure()
df_plot = adata.obs.groupby(['supervised_name', 'species']).size().reset_index().pivot(columns='species', index='supervised_name', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVspeciesBar.pdf'), bbox_inches="tight")

adata=adata[~((adata.obs['species']=='Mouse') & (adata.obs['region']!='ob')),:]
adata.obs['new_supervised_name']='nan'
adata.obs['new_supervised_name']=adata.obs['supervised_name']
del adata.uns['neighbors']
adata.obs['new_supervised_name']=adata.obs['new_supervised_name'].astype(str)
adata.obs.loc[adata.obs['species']!='Macaque','new_supervised_name']='nan'
adata.obs['nanlabel']=adata.obs['new_supervised_name']=='nan'
print('redo neighbors',flush=True)
bbknn.bbknn(adata,batch_key='nanlabel',n_pcs=50,neighbors_within_batch=10)
adata.uns['neighbors']['distances'].nonzero()
supervised=adata.obs['new_supervised_name']
spec=list(adata.obs['species'])
nz0=adata.uns['neighbors']['distances'].nonzero()[0]
nz1=adata.uns['neighbors']['distances'].nonzero()[1]
nz0i=adata.obs.index[nz0]
nz1i=adata.obs.index[nz1]
results=[nz0,
         nz1,
         adata.obs.loc[nz0i,'species'],
         adata.obs.loc[nz1i,'species'],
         adata.obs.loc[nz0i,'new_supervised_name'],
         adata.obs.loc[nz1i,'new_supervised_name']]
results=[list(x) for x in results]
df=pd.DataFrame(results,index=['ind1','ind2','species1','species2','classname1','classname2']).T
mat=df.groupby('ind1')['classname2'].value_counts(dropna=False).unstack('classname2').fillna(0)
mat=mat.loc[:,mat.columns!='nan']
#adata.obs.loc[adata.obs.index[mat.index],'new_supervised_name']=mat.idxmax(1)
adata.obs['new_supervised_name']=list(mat.idxmax(1))
adata.obs.loc[:,['full_cellname','new_supervised_name','leiden']].to_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/QtoMSupervisedCellNames.txt',index=None)
sc.pl.umap(adata,color=['supervised_name'],save='QtoM_OBonly_supervised_name')
f = plt.figure()
df_plot = adata[adata.obs['species']=='Mouse',].obs.groupby(['supervised_name', 'new_supervised_name']).size().reset_index().pivot(columns='new_supervised_name', index='supervised_name', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True,color=my_pal)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
plt.xticks(fontsize=4)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'QtoM_OBonly_supervisedtransferBar.pdf'), bbox_inches="tight")


adata.obs['new_supervised_name']=adata.obs['supervised_name']
del adata.uns['neighbors']
adata.obs['new_supervised_name']=adata.obs['new_supervised_name'].astype(str)
adata.obs.loc[adata.obs['species']!='Mouse','new_supervised_name']='nan'
adata.obs['nanlabel']=adata.obs['new_supervised_name']=='nan'
print('redo neighbors',flush=True)
bbknn.bbknn(adata,batch_key='nanlabel',n_pcs=50,neighbors_within_batch=10)
adata.uns['neighbors']['distances'].nonzero()
supervised=adata.obs['new_supervised_name']
spec=list(adata.obs['species'])
nz0=adata.uns['neighbors']['distances'].nonzero()[0]
nz1=adata.uns['neighbors']['distances'].nonzero()[1]
nz0i=adata.obs.index[nz0]
nz1i=adata.obs.index[nz1]
results=[nz0,
         nz1,
         adata.obs.loc[nz0i,'species'],
         adata.obs.loc[nz1i,'species'],
         adata.obs.loc[nz0i,'new_supervised_name'],
         adata.obs.loc[nz1i,'new_supervised_name']]
results=[list(x) for x in results]
df=pd.DataFrame(results,index=['ind1','ind2','species1','species2','classname1','classname2']).T
mat=df.groupby('ind1')['classname2'].value_counts(dropna=False).unstack('classname2').fillna(0)
mat=mat.loc[:,mat.columns!='nan']
#adata.obs.loc[adata.obs.index[mat.index],'new_supervised_name']=mat.idxmax(1)
adata.obs['new_supervised_name']=list(mat.idxmax(1))
adata.obs.loc[:,['full_cellname','new_supervised_name','leiden']].to_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MtoQSupervisedCellNames.txt',index=None)
sc.pl.umap(adata,color=['supervised_name'],save='MtoQ_OBonly_supervised_name')

f = plt.figure()
df_plot = adata[adata.obs['species']=='Macaque',].obs.groupby(['supervised_name', 'new_supervised_name']).size().reset_index().pivot(columns='new_supervised_name', index='supervised_name', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
plt.xticks(fontsize=4)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'MtoQ_OBonly_supervisedtransferBar.pdf'), bbox_inches="tight")
