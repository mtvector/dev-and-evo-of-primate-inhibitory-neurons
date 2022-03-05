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
#Load my pipeline functions
import importlib
import importlib.util
spec = importlib.util.spec_from_file_location("ScanpyUtilsMT", os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../utils/ScanpyUtilsMT.py"))
sc_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc_utils)
figdir='/wynton/group/ye/mtschmitz/figures/macaqueGeAllcortexHippoPresuperviseCB202002/'
sc.settings.figdir=figdir
scv.settings.figdir=figdir
sc.settings.file_format_figs='pdf'
sc.settings.autosave=True
sc.settings.autoshow=False


def most_frequent(List): 
    return max(set(List), key = List.count)

def variance(X,axis=1):
    return( (X.power(2)).mean(axis=axis)-(np.power(X.mean(axis=axis),2)) )

filepath='/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_kallisto'
files=os.listdir(filepath)
files=[f for f in files if 'kOut' in f]
regionlist=['mix9','mix10','ge','mix8','amy','str','septum','otor','dfc','cing','pfc','parietal','v1','temp','somato','re-optic','poa','mix7','mix5','insula','hippo','hip','hippocampus']
fileList=[f for f in files if any([x in f.lower() for x in regionlist]) ]
tplist=['E40','E50','E65','E80','E90','E100','PEC_Yale','Mac2','Mix']
fileList=[f for f in fileList if any([x in f for x in tplist]) ]
print(fileList)
newfile='/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMacaqueGeAllcortexHippocampusPresupervise.h5ad'
min_genes=800
#fileList=[x for x in fileList if x != "E100insula_kOut"]

if True:#not os.path.exists(newfile):
    adatas=[]
    filename='a_cellbended_200_1000_300e_V0.2'
    for f in fileList:
        print(f,flush=True)
        try:
            if os.path.exists(os.path.join(filepath,f,filename,'a_cellbended_filtered.h5')):
                sadata = sc_utils.readCellbenderH5(os.path.join(filepath,f,filename,'a_cellbended_filtered.h5'))
                sadata =sadata[sadata.obs['latent_cell_probability']>.999,:]
                sc.pp.filter_cells(sadata,min_genes=min_genes)
            else:
                continue
                #sadata=sc_utils.loadPlainKallisto(os.path.join(filepath,f,'all'),min_genes=min_genes)
                
            sadata.obs.index=[re.sub("-1","",x) for x in sadata.obs.index]
            sadata.uns['name']=f
            sadata.obs['file_name']=f
            sadata.uns['name']=sc_utils.macaque_process_irregular_names(sadata.uns['name'])
            sadata.obs['batch_name']=str(sadata.uns['name'])
            sadata.obs['timepoint']=sc_utils.tp_format_macaque(sadata.uns['name'])
            regionstring=sc_utils.region_format_macaque(sadata.uns['name'])
            regionstring=regionstring.lower()
            sadata.obs['region']=regionstring
            if not os.path.exists(os.path.join(filepath,f,'cbdoublets.txt')):
                doublets=sc_utils.doublescrub(sadata)
                print(doublets,flush=True)
                pd.DataFrame(doublets).to_csv(os.path.join(filepath,f,'cbdoublets.txt'),index=False, header=False)
            try:
                ddf=pd.read_csv(os.path.join(filepath,f,'cbdoublets.txt'),index_col=False, header=None)
                doublets=list(ddf[0])
            except:
                doublets=[]
            sadata=sadata[~sadata.obs.index.isin(doublets),:]
            pd.DataFrame(sadata.obs.index).to_csv(os.path.join(filepath,f,'cellbendedcells.txt'),index=False, header=False)
            if sadata.shape[0]>10:
                adatas.append(sadata)
        except Exception as e:
            print(e)
            print('fail')

    #adatas=adatas+[multi]
    adata=sc.AnnData.concatenate(*adatas)
    adata.var.columns = adata.var.columns.astype(str)
    adata.obs.columns = adata.obs.columns.astype(str)
    adata.obs['clean_cellname']=[re.sub('-[0-9]+','',x) for x in  adata.obs.index]
    adata.obs['full_cellname']=adata.obs['clean_cellname'].astype(str)+'_'+adata.obs['batch_name'].astype(str)
    adata.obs.index=list(adata.obs['full_cellname'])
    adata.raw=adata
    adata.write(newfile)        

    
if True:#not os.path.exists(re.sub('\.h5ad','Processed.h5ad',newfile)):
    adata=sc.read(newfile)
    multiseq=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMultiseq.h5ad')
    multiseq=multiseq[multiseq.obs.index,:]
    multiseqregion=[any([x in f.lower() for x in regionlist]) for f in multiseq.obs['region']]
    multiseq=multiseq[multiseqregion,:]
    adata=sc.AnnData.concatenate(adata,multiseq,index_unique=None)
    print("1",adata.obs)
    supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MacaqueAllSupervisedCellNames.txt')
    ind=adata.obs.index[adata.obs['full_cellname'].isin(supercell['full_cellname'])]
    supercell.index=supercell['full_cellname']
    supercell=supercell.loc[ind,:]
    adata.obs.loc[ind,'supervised_name']=supercell['supervised_name']
    adata.obs['supervised_name']=adata.obs['supervised_name'].astype(str)
    
    supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MacaqueGEsupervisednames4.txt')
    ind=adata.obs.index[adata.obs['full_cellname'].isin(supercell['full_cellname'])]
    supercell.index=supercell['full_cellname']
    supercell=supercell.loc[ind,:]
    adata.obs.loc[ind,'supervised_name']=supercell['supervised_name']
    adata.obs['supervised_name']=adata.obs['supervised_name'].astype(str)
    
    print("2",adata)
    print(adata.obs['supervised_name'].unique())
    excitatorycells=[x for x in adata.obs['supervised_name'].unique() if 'excitatory' in x.lower()]
    #adata=adata[~adata.obs['supervised_name'].isin(excitatorycells),:]
    '''
    adata=adata[~adata.obs['supervised_name'].isin(['Endothelial Cell','Pericyte',
                                                        'Microglia','Blood',
                                                    'Choroid Cell']),:]
    print("3",adata)
    print(adata.obs['supervised_name'].unique())
    cortexregions=['motor','dfc','cingulate','insula','parietal','pfc','somato','somatosensory','temporal','v1']
    adata=adata[~(adata.obs['region'].isin(cortexregions) & adata.obs['supervised_name'].isin(['EDNRB+ Astrocyte','Radial Glia','Late Radial Glia','IL33+ Astrocyte','OPC','Oligodendrocyte','S-phase Dividing Cell','MGE IPC','CGE IPC','G2/M Dividing Cell'])),:]
    print("4",adata)
    print(adata.obs['supervised_name'].unique())
    adata.obs.loc[adata.obs['region'].isin(cortexregions),'supervised_name']="Cortical "+adata.obs.loc[adata.obs['region'].isin(cortexregions),'supervised_name']
    '''
    adata.raw=adata
    adata.var_names_make_unique()
    ribo_genes=[name for name in adata.var_names if name.startswith('RPS') or name.startswith('RPL') ]
    adata.obs['percent_ribo'] = np.sum(
    adata[:, ribo_genes].X, axis=1) / np.sum(adata.X, axis=1)
    mito_genes = [name for name in adata.var_names if name in ['ND1','ND2','ND4L','ND4','ND5','ND6','ATP6','ATP8','CYTB','COX1','COX2','COX3'] or name.startswith('chrM-') or name.startswith('MT-')]
    adata.obs['percent_mito'] = np.sum(
    adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
    print(adata)
    sc.pl.violin(adata,groupby='batch_name',keys=['percent_ribo','percent_mito'],rotation=45,save='ribomito')


    #adata=adata[adata.obs['percent_ribo']<.5,:]
    #adata=adata[adata.obs['percent_ribo']<.2,:]
    adata._inplace_subset_obs(adata.obs['percent_ribo']<.4)
    adata._inplace_subset_obs(adata.obs['percent_mito']<.15)
    adata._inplace_subset_var(~adata.var.index.isin(mito_genes))
    print(adata)
    sc.pp.filter_genes(adata,min_cells=10)
    #sc.pp.filter_cells(adata,min_counts=1000)
    sc.pp.filter_cells(adata,min_genes=min_genes)
    sc.pl.highest_expr_genes(adata, n_top=20, )
    print(adata)
    sc.pp.normalize_total(adata,exclude_highly_expressed=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata,n_top_genes=6000,batch_key='batch_name',subset=False)
    sc.pp.scale(adata,max_value=10)
    sc.pp.pca(adata,n_comps=100)
    #sc.pp.neighbors(adata)
    bbknn.bbknn(adata,batch_key='batch_name',n_pcs=100,neighbors_within_batch=3)
    sc.tl.leiden(adata,resolution=10)
    sc.tl.umap(adata,spread=2)
    adata.write(re.sub('\.h5ad','Processed.h5ad',newfile))
    #adata=sc.AnnData(adata.raw.X,var=adata.raw.var,obsm=adata.obsm,uns=adata.uns,obs=adata.obs)
    #adata.raw=adata
    #sc.pp.normalize_per_cell(adata)
    #sc.pp.log1p(adata)
    #sc.pp.scale(adata,max_value=10)
    sc.pl.umap(adata,color=['leiden'],save='leiden')
    sc.pl.umap(adata,color=['batch_name'],save='batch_name')
    sc.pl.umap(adata,color=['region'],save='region')
    sc.pl.umap(adata,color=['timepoint'],save='timepoint')
    easygenes= ['MKI67','SOX2','AQP4','EDNRB','IL33','PDGFRA','HOPX','ERBB4','CALB1','CALB2','GAD1','GAD2','GADD45G','LHX6','SST','NKX2-1','MAF','SP8','SP9','PROX1','VIP','CCK','NPY','LAMP5','HTR3A','NR2F1','NR2F2','TOX3','ETV1','SCGN','FOXP1','FOXP2','FOXP4','TH','PAX6','MEIS2','SHTN1','ISL1','PENK','CRABP1','ZIC1','ZIC2','EBF1','DLX5','GSX2','RBFOX3','PPP1R17','EOMES','DCX','TUBB3','BCL11B','TLE4','FEZF2','SATB2','TBR1','AIF1','RELN','PECAM1','HBZ','ROBO1','ROBO2','ROBO3','ROBO4']
    easygenes=[x for x in easygenes if x in adata.var.index]
    sc.pl.umap(adata,color=easygenes,use_raw=False,save='FaveGenes')

    sc.pl.umap(adata,color=['supervised_name'],save='supervised_name')
    countmat=adata.obs.astype(str).groupby(['leiden', 'supervised_name']).size().reset_index().pivot(columns='supervised_name', index='leiden', values=0)
    if 'nan' in countmat.columns:
        countmat=countmat.drop('nan',axis=1)
    leidentosuper=dict(countmat.idxmax(1))
    adata.obs['supervised_name']=adata.obs['supervised_name'].astype(str)
    adata.obs['supervised_name']='nan'
    for i in adata.obs.loc[adata.obs['supervised_name']=='nan',:].index:
        l=adata.obs['leiden'][i]
        adata.obs.loc[i,'supervised_name']=leidentosuper[l]

    adata.obs.loc[:,['full_cellname','supervised_name','leiden']].to_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MacaqueGEsupervisednamesHippo.txt',index=None)

    f = plt.figure()
    df_plot = adata.obs.groupby(['supervised_name', 'batch_name']).size().reset_index().pivot(columns='batch_name', index='supervised_name', values=0).apply(lambda g: g / g.sum(),1)
    ax = df_plot.plot(kind='bar', legend=False,stacked=True)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.ylabel('proportion')
    ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVbatchBar.pdf'), bbox_inches="tight")

    f = plt.figure()
    df_plot = adata.obs.groupby(['supervised_name', 'leiden']).size().reset_index().pivot(columns='leiden', index='supervised_name', values=0).apply(lambda g: g / g.sum(),1)
    ax = df_plot.plot(kind='bar', legend=False,stacked=True)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.ylabel('proportion')
    ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVleidenBar.pdf'), bbox_inches="tight")


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

    sc_utils.cell_cycle_score(adata)
    sc_utils.marker_analysis(adata,variables=['leiden'],markerpath=os.path.expanduser('~/markers.txt')) 
    sc_utils.log_reg_diff_exp(adata)


    adata.uns['iroot'] = np.flatnonzero(adata.obs.index==sc_utils.get_median_cell(adata,'supervised_name','S-phase Dividing Cell'))[0]
    sc.tl.diffmap(adata,n_comps=15)
    sc.tl.dpt(adata,n_branchings=0,n_dcs=15)
    #sc.pl.diffmap(adata,components='all',save='alldiffmap')

    sc.tl.paga(adata,groups='supervised_name')
    sc.pl.paga(adata,layout='eq_tree',save='normal_tree',legend_fontsize=4,edge_width_scale=.4,threshold=np.quantile(adata.uns['paga']['connectivities'].data,.9))
    sc.pl.paga_compare(adata,legend_fontsize=4,edge_width_scale=.4,threshold=np.quantile(adata.uns['paga']['connectivities'].data,.9),save='connectivity')

    adata.write(re.sub('\.h5ad','Processed.h5ad',newfile))

 


