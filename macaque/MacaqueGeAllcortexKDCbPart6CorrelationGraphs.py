
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
import copy

adata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMacaqueGeAllcortexHippocampusProcessed.h5ad')

def crazyshuffle(arr):
    x, y = arr.shape
    rows = np.indices((x,y))[0]
    cols = [np.random.permutation(y) for _ in range(x)]
    return arr[rows, cols]

supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MacaqueGEsupervisednamesHippoPredictedEnd.txt')
ind=adata.obs.index[adata.obs.index.isin(supercell['full_cellname'])]
supercell.index=supercell['full_cellname']
supercell=supercell.loc[ind,:]
adata.obs['latent_time']=np.nan
adata.obs['predicted_end']=''
adata.obs['noctx_supervised_name']=adata.obs['noctx_supervised_name'].astype(str)
adata.obs.loc[ind,['noctx_supervised_name','predicted_end','latent_time']]=supercell.loc[:,['noctx_supervised_name','predicted_end','latent_time']]
adata.obs['noctx_supervised_name']=adata.obs['supervised_name'].astype(str)

lineage_genes= ['LHX8','LHX6','NKX2-1','MAF','CRABP1','RBP4','TAC3','PROX1','NR2F1','NR2F2','SP8','ETV1','PAX6','TSHZ1','FOXP1','FOXP2','CASZ1','MEIS2','EBF1','ISL1','PENK','ADORA2A','ZIC1','ZIC4','NPY','CORT','CHODL']#GSX2,VAX1,'SALL3'
lineage_genes=[x for x in lineage_genes if x in adata.var.index]
import networkx as nx
import statsmodels
import statsmodels.stats
import statsmodels.stats.multitest
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
n_bootstraps=10000
def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))


adata=adata[adata.obs['latent_time'].to_numpy().argsort(),:]

d={}

for branch in ['S-phase|G2-M','G1-phase','MGE|LGE|CGE|VMF|Transition']:#'MGE','CGE','LGE'
    values=[]
    times=[]
    bdata=adata[adata.obs['noctx_supervised_name'].str.contains(branch),:].copy()
    bdata.obs_names_make_unique()
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    n=200       
    for x in tqdm.tqdm(chunks(bdata.obs.index[bdata.obs.latent_time.to_numpy().argsort()],n)):
        if len(x)>20:
            times.append(bdata[x,:].obs['latent_time'].mean())
            mat=pd.DataFrame(bdata[x,lineage_genes].X,index=x,columns=lineage_genes)
            values.append(mat.corr())

    values=np.array(values)
    times=np.array(times)
    
    
    df1=pd.DataFrame(bdata[:,lineage_genes].X)
    cors=df1.corr().to_numpy()
    bootstraps=[]
    df1_mat=df1.to_numpy()
    bootstrapcounts=np.zeros((len(lineage_genes),len(lineage_genes)))
    bootstrapcounts+=.1
    abscors=np.abs(cors)
    for q in tqdm.tqdm(range(n_bootstraps)):
        #bootstraps.append(df1.loc[np.random.choice(df1.index,size=df1.shape[0],replace=False),np.random.choice(df1.columns,size=df1.shape[1],replace=False)].corr().to_numpy())
        df2=pd.DataFrame(crazyshuffle(df1_mat))
        bootstrapcounts+=(abscors<np.abs(df2.corr().to_numpy()))
        #bootstraps.append(df2.corr().to_numpy())
    #bootstraps=np.array(bootstraps)
    #baseline_cor_pval=(((np.abs(cors.to_numpy())<np.abs(bootstraps)).sum(0)+1)/(n_bootstraps+1))
    baseline_cor_pval=bootstrapcounts/(n_bootstraps+.1)
    coefs=np.zeros((len(lineage_genes),len(lineage_genes)))
    intercepts=np.zeros((len(lineage_genes),len(lineage_genes)))
    pvals=np.zeros((len(lineage_genes),len(lineage_genes)))
    
    for i in range(len(lineage_genes)):
        for j in range(len(lineage_genes)):
            notnan=~np.isnan(values[:,i,j])
            if np.sum(notnan)>10:
                out=scipy.stats.linregress(times[notnan],values[notnan,i,j])
                coefs[i,j]=out[0]
                intercepts[i,j]=out[1]
                pvals[i,j]=out[3]
            else:
                coefs[i,j]=np.nan
                intercepts[i,j]=np.nan
                pvals[i,j]=np.nan
    d[branch]=dict(zip(['reg_coef','reg_intercept','reg_pval','branch_cor','branch_cor_pval'],[coefs,intercepts,pvals,cors,baseline_cor_pval]))
    np.fill_diagonal(d[branch]['branch_cor'],np.nan)
    #Multiple test correctuion using off-diagonal only
    s=d[branch]['branch_cor_pval'].shape
    flat_e=d[branch]['branch_cor_pval'][np.where(~np.eye(d[branch]['branch_cor_pval'].shape[0],dtype=bool))]
    qvals=statsmodels.stats.multitest.multipletests(flat_e,alpha=.05)
    d[branch]['branch_cor_qval']=copy.deepcopy(d[branch]['branch_cor_pval'])
    d[branch]['branch_cor_qval'][np.where(~np.eye(d[branch]['branch_cor_qval'].shape[0],dtype=bool))]=qvals[1]



for k in d.keys():
    for i in d[k].keys():
        sanitize_k="".join(x for x in k if x.isalnum())
        sanitize_i="".join(x for x in i if x.isalnum())
        d[k][i]=pd.DataFrame(d[k][i],index=lineage_genes,columns=lineage_genes)
        pd.DataFrame(d[k][i],index=lineage_genes,columns=lineage_genes).to_csv(os.path.join(sc.settings.figdir,sanitize_k+"_"+sanitize_i+"_CorrAnalysis.csv"))


k='MGE|LGE|CGE|VMF|Transition'
seaborn.set(font_scale=.6)
import networkx as nx
g=nx.networkx.convert_matrix.from_pandas_adjacency(d[k]['branch_cor'])
weights=1-(d[k]['branch_cor'])
for e in g.edges():
    g.add_edge(e[0],e[1],
        weight=weights.loc[e[0],e[1]],
        cor=d[k]['branch_cor'].loc[e[0],e[1]],
        qval=d[k]['branch_cor_qval'].loc[e[0],e[1]])
loc=nx.drawing.layout.kamada_kawai_layout(g,scale=2)

for k in d.keys():
    print(k)
    sanitize_k="".join(x for x in k if x.isalnum())
    seaborn.heatmap(d[k]['branch_cor'],cmap='RdBu_r', vmin=-.8, vmax=.8,xticklabels=True,yticklabels=True)
    plt.savefig(os.path.join(sc.settings.figdir,sanitize_k+'_BranchCor.pdf'), bbox_inches="tight")
    plt.clf()
    seaborn.heatmap(d[k]['branch_cor_pval'],xticklabels=True,yticklabels=True)
    plt.savefig(os.path.join(sc.settings.figdir,sanitize_k+'_BranchPval.pdf'), bbox_inches="tight")
    plt.clf()
    seaborn.heatmap(d[k]['branch_cor_qval'],xticklabels=True,yticklabels=True)
    plt.savefig(os.path.join(sc.settings.figdir,sanitize_k+'_BranchQval.pdf'), bbox_inches="tight")
    plt.clf()
    g=nx.networkx.convert_matrix.from_pandas_adjacency(d[k]['branch_cor'])
    weights=1-(d[k]['branch_cor'])
    for e in g.edges():
        g.add_edge(e[0],e[1],
            weight=weights.loc[e[0],e[1]],
            cor=d[k]['branch_cor'].loc[e[0],e[1]],
            qval=d[k]['branch_cor_qval'].loc[e[0],e[1]])
    l=[]
    for key in loc.keys():
        l.append(loc[key])
    maxes=np.array(l).max(0)
    mins=np.array(l).min(0)
    norm = matplotlib.colors.Normalize(vmin=-.8,vmax=.8)
    edges = g.edges()
    [g.remove_edge(u,v) for u,v in edges if g[u][v]['qval']>.05]
    colors = [plt.cm.RdBu_r(norm(g[u][v]['cor'])) for u,v in edges]
    weights = [np.abs(g[u][v]['cor'])*10 for u,v in edges]
    nx.draw_networkx_labels(g,pos=loc,font_weight="bold",font_size=9,font_color='black')
    nx.draw_networkx_edges(g,pos=loc,edge_color=colors,width=weights)
    axis = plt.gca()
    # maybe smaller factors work as well, but 1.1 works fine for this minimal example
    axis.set_xlim([mins[0]*1.2,maxes[0]*1.2])
    axis.set_ylim([mins[1]*1.2,maxes[1]*1.2])
    axis.set_facecolor('white')
    axis.grid(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.get_figure().savefig(os.path.join(sc.settings.figdir,sanitize_k+'Graph.pdf'), bbox_inches="tight")
    plt.clf()
