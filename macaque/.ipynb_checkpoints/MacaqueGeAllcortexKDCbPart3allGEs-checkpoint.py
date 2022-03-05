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
sc.settings.file_format_figs='pdf'
sc.settings.autosave=True
sc.settings.autoshow=False
import matplotlib.font_manager
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Nimbus Sans','Arial']})
matplotlib.rc('text', usetex=False)
scanpy.set_figure_params(scanpy=True,dpi_save=400)
import copy

newfile='/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMacaqueGeAllcortexHippocampus.h5ad'

adata=sc.read(re.sub('\.h5ad','Processed.h5ad',newfile))

[adata.obs.drop(x,axis=1,inplace=True) for x in adata.obs.columns if 'mkrscore' in x]

adata.obs['supervised_name']=adata.obs['supervised_name'].astype(str)
supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MacaqueGEsupervisednamesHippo3PreserveCompoundname.txt')
ind=adata.obs.index[adata.obs.index.isin(supercell['full_cellname'])]
supercell.index=supercell['full_cellname']
supercell=supercell.loc[ind,:]
adata.obs.loc[ind,'supervised_name']=supercell['supervised_name']
adata.obs['supervised_name']=adata.obs['supervised_name'].astype(str)
adata.obs['supervised_name']=[re.sub('Cortical ','',x) for x in adata.obs['supervised_name'] ]
adata.obs['region']=['hippocampus' if 'hip' in x else x for x in adata.obs.region]
adata.obs['general_region']=sc_utils.macaque_general_regions(adata.obs['region'])

supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MacaqueGEsupervisednamesHippoPredictedEnd.txt')
ind=adata.obs.index[adata.obs.index.isin(supercell['full_cellname'])]
supercell.index=supercell['full_cellname']
supercell=supercell.loc[ind,:]
adata.obs['latent_time']=np.nan
adata.obs['predicted_end']=''
adata.obs['noctx_supervised_name']=adata.obs['noctx_supervised_name'].astype(str)
adata.obs.loc[ind,['noctx_supervised_name','predicted_end','latent_time']]=supercell.loc[:,['noctx_supervised_name','predicted_end','latent_time']]
adata.obs.loc[adata.obs['predicted_end'].isna(),'noctx_supervised_name']=adata.obs['noctx_supervised_name'][adata.obs['predicted_end'].isna()]

adata.obs['supervised_name']=adata.obs['supervised_name'].astype('category')

paldict={'CGE_NR2F2/PROX1': 'yellow',
 'G1-phase_SLC1A3/ATP1A1': '#17344c',
 'G2-M_UBE2C/ASPM': '#19122b',
 'LGE_FOXP1/ISL1': 'cyan',
 'LGE_FOXP1/PENK': 'navy',
 'LGE_FOXP2/TSHZ1': 'goldenrod',
 'LGE_MEIS2/PAX6': 'orangered',
 'MGE_CRABP1/MAF': 'indigo',
 'MGE_CRABP1/TAC3': 'fuchsia',
 'MGE_LHX6/MAF': 'darksalmon',
 'MGE_LHX6/NPY': 'red',
 'RMTW_ZIC1/RELN': 'slategray',
 'S-phase_MCM4/H43C': 'lawngreen',
 'Transition': '#3c7632',
 'VMF_ZIC1/ZIC2': 'teal'}

GEs=['CRABP1','MGE','CGE','LGE','VMF']
GEmarkers=[['CRABP1','ANGPT2','LHX6','ETV1'],['NKX2-1','CRABP1','LHX6','ANGPT2','ETV1'],['NR2F1','NR2F2'],['MEIS2','FOXP1','FOXP2'],['NKX2-1','ZIC1','ZIC2','ZIC4']]
genelists={'CRABP1':['NKX2-1','ETV1','CRABP1','ANGPT2','CHRNA3','CHRNA5','CHRNA7','STXBP6','ETV5','ZIC1','TAC3','NXPH2','NXPH4','STXBP6','TRH','LHX8','NXPH1','RBP4','MAF','SHISA6','TACR3','COL19A1','LGI1','GRIA4'],
          'MGE':['NKX2-1','ETV1','CRABP1','ANGPT2','CHRNA3','CHRNA5','CHRNA7','TAC3','TRH','LHX8','LHX6','NXPH1','NXPH2','NXPH4','STXBP6','RBP4','MAF','ZIC1','NPY','CORT','PVALB'],
          'CGE':['NR2F1','NR2F2','SP8','PAX6','VIP','LAMP5','LHX6','CCK','SCNG','SCGN'],
          'LGE':['MEIS2','PAX6','SP8','SCGN','TSHZ1','CASZ1','OPRM1','FOXP2','FOXP1','FOXP4','PENK','ISL1','DRD1','ADORA2A'],
          'VMF':['ZIC1','ZIC2','ZIC4','LHX6','LHX8','NKX2-1','RELN','POMC','ISL1','SHH','NR2F2']}

for GE,markers in zip(GEs,GEmarkers):
    lineage_genes=genelists[GE]
    figdir='/wynton/group/ye/mtschmitz/figures/macaqueGeAllcortexHippoCB202002'+GE+'/'
    sc.settings.figdir=figdir
    scv.settings.figdir=figdir
    bdata=adata[(((adata[:,markers].X>1).sum(1)>1)&adata.obs['supervised_name'].str.contains('phase|G2-M'))|adata.obs.noctx_supervised_name.str.contains(GE),:].copy()
    keep_cat=bdata.obs['batch_name'].value_counts().index[bdata.obs['batch_name'].value_counts()>2]
    bdata=bdata[bdata.obs.batch_name.isin(keep_cat),:]
    bdata.obs['batch_name'] = bdata.obs.batch_name.cat.remove_unused_categories()
    bdata.X=bdata.raw.X[:,bdata.raw.var.index.isin(bdata.var.index)]
    bdata.obs['supervised_name'].cat.remove_unused_categories(inplace=True)
    my_pal=[paldict[x] for x in bdata.obs['supervised_name'].cat.categories]

    sc.pp.filter_genes(bdata,min_cells=20)
    lineage_genes=[x for x in lineage_genes if x in bdata.var.index]
    sc.pp.filter_cells(bdata,min_genes=500)
    sc.pp.normalize_total(bdata,exclude_highly_expressed=True)
    sc.pp.log1p(bdata)
    sc.pp.highly_variable_genes(bdata,n_top_genes=8000,batch_key='batch_name')
    bdata.var['highly_variable']=(bdata.var['highly_variable']&(bdata.var['highly_variable_nbatches']>1))
    sc.pp.scale(bdata,max_value=10)
    sc.pp.pca(bdata,n_comps=50)
    bbknn.bbknn(bdata,batch_key='batch_name',n_pcs=50,neighbors_within_batch=3)
    sc.tl.leiden(bdata,resolution=1.5)#.8 works for cross species when not marker subsetted
    sc.tl.umap(bdata,spread=1,min_dist=.1)

    genez=lineage_genes
    genez=[x for x in genez if x in bdata.var.index]
    sc.pl.umap(bdata,color=genez,use_raw=False,save=GE+'_fave_genes')
    sc.pl.umap(bdata,color='supervised_name',save='supervised_name',palette=my_pal)
    sc.pl.umap(bdata,color='region',use_raw=False,save='region')

    newfile='/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMacaqueGeAllcortexHippocampus'+GE+'only.h5ad'

    sc.pl.umap(bdata,color=['leiden'],save='leiden')
    hierarchy_key='leiden'
    rgs=sc.tl.rank_genes_groups(bdata,groupby=hierarchy_key,method='logreg',use_raw=False,copy=True).uns['rank_genes_groups']#,penalty='elasticnet',solver='saga')#or penalty='l1'
    result=rgs
    groups = result['names'].dtype.names
    df=pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'scores']})
    df.to_csv(os.path.join(sc.settings.figdir,"LogReg"+hierarchy_key+"Norm.csv"))
    topgenes=df.iloc[0:8,['_n' in x for x in df.columns]].T.values
    cols=df.columns[['_n' in x for x in df.columns]]
    cols=[re.sub('_n','',x) for x in cols]
    topdict=dict(zip(cols,topgenes))
    sc.tl.dendrogram(bdata,groupby=hierarchy_key)
    var_dict=dict(zip(bdata.uns["dendrogram_['"+hierarchy_key+"']"]['categories_ordered'],[topdict[x] for x in bdata.uns["dendrogram_['"+hierarchy_key+"']"]['categories_ordered']]))
    sc.pl.matrixplot(bdata,groupby=hierarchy_key,var_names=var_dict,cmap='RdBu_r',use_raw=False,dendrogram=True,save='leiden_marker')

    #bdata=bdata[np.random.choice(bdata.obs.index,size=20000,replace=False),:]
    continue
    sc.pp.highly_variable_genes(bdata,flavor='seurat_v3',layer='unspliced',n_top_genes=15000)
    bdata.var['highly_variable_rank_u']=bdata.var['highly_variable_rank']
    sc.pp.highly_variable_genes(bdata,flavor='seurat_v3',layer='spliced',n_top_genes=15000)
    bdata.var['highly_variable_rank_s']=bdata.var['highly_variable_rank']
    print(bdata,flush=True)
    bdata.var['velocity_genes']=bdata.var.loc[:,['highly_variable_rank_s','highly_variable_rank_u']].mean(1,skipna=False).rank()<2000
    print(bdata.var['velocity_genes'].value_counts())


    s_genes=['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8']
    g2m_genes=['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']
    s_genes=[x for x in s_genes if x in bdata.var.index]
    g2m_genes=[x for x in g2m_genes if x in bdata.var.index]

    gdata=bdata.copy().T
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
    cc_mods=vcs.index[vcs>3]
    cc_genes=gdata.obs.index[gdata.obs.leiden.isin(cc_mods)]
    bdata.var['leiden']=gdata.obs['leiden']
    del gdata

    ncells=50
    mgenes=(bdata.layers['unspliced']>0).sum(0).A1 < ncells
    bdata.var.loc[mgenes,'velocity_genes']=False
    #bdata.var['velocity_genes']=mgenes.A1&bdata.var['highly_variable']
    #bdata=bdata[:,mgenes.A1]
    #bdata=bdata[:,bdata.var['highly_variable']]
    scv.pp.normalize_per_cell(bdata)
    print(bdata)
    print(bdata.layers['spliced'].shape)
    #scv.pp.filter_genes_dispersion(bdata)#,n_top_genes=min(6000,bdata.layers['spliced'].shape[1]))
    scv.pp.log1p(bdata)
    scv.pp.remove_duplicate_cells(bdata)
    scv.pp.moments(bdata, n_neighbors=20)
    print(bdata,flush=True)
    scv.tl.recover_dynamics(bdata,var_names='velocity_genes',use_raw=False)
    print('Recovered', flush=True)
    scv.tl.velocity(bdata,mode='dynamical',filter_genes=False)
    print('Velocity done',flush=True)
    bdata.write(re.sub('Velocity','VelocityDYNAMICAL',newfile))

    bdata.layers['cc_velocity']=bdata.layers['velocity'].copy()
    bdata.layers['cc_velocity'][:,~bdata.var.index.isin(cc_genes)]=0

    scv.tl.velocity_graph(bdata,mode_neighbors='connectivities',vkey='cc_velocity',approx=False)
    scv.tl.transition_matrix(bdata,vkey='cc_velocity')
    scv.tl.terminal_states(bdata,vkey='cc_velocity')
    scv.tl.recover_latent_time(bdata,vkey='cc_velocity')
    scv.tl.velocity_confidence(bdata,vkey='cc_velocity')
    scv.tl.velocity_embedding(bdata, basis='umap',vkey='cc_velocity')
    scv.pl.velocity_embedding_grid(bdata, basis='umap',vkey='cc_velocity',save='_cc_grid',color_map=matplotlib.cm.RdYlBu_r,dpi=300)
    try:
        sc.pl.umap(bdata,color=['cc_velocity_length', 'cc_velocity_confidence','latent_time','root_cells','end_points','cc_velocity_pseudotime'],save='cc_velocity_stats',color_map=matplotlib.cm.RdYlBu_r)
    except:
        print('fail cc')

    del bdata.layers['cc_velocity']
    bdata.layers['linear_velocity']=bdata.layers['velocity'].copy()
    bdata.layers['linear_velocity'][:,bdata.var.index.isin(cc_genes)]=0
    scv.tl.velocity_graph(bdata,mode_neighbors='connectivities',vkey='linear_velocity',approx=False)
    scv.tl.transition_matrix(bdata,vkey='linear_velocity')
    scv.tl.terminal_states(bdata,vkey='linear_velocity')
    bdata=bdata.copy()
    scv.tl.recover_latent_time(bdata,vkey='linear_velocity')
    scv.tl.velocity_embedding(bdata, basis='umap',vkey='linear_velocity')
    scv.tl.velocity_confidence(bdata,vkey='linear_velocity')
    scv.pl.velocity_embedding_grid(bdata,vkey='linear_velocity',basis='umap',color='latent_time',save='_linear_grid',color_map=matplotlib.cm.RdYlBu_r,dpi=300)
    try:
        sc.pl.umap(bdata,color=['linear_velocity_length', 'linear_velocity_confidence','latent_time','root_cells','end_points','linear_velocity_pseudotime'],save='linear_velocity_stats',color_map=matplotlib.cm.RdYlBu_r)
    except:
        'fail linear'
    bdata.write(re.sub('Velocity','VelocityDYNAMICAL',newfile))
    
    def crazyshuffle(arr):
        x, y = arr.shape
        rows = np.indices((x,y))[0]
        cols = [np.random.permutation(y) for _ in range(x)]
        return arr[rows, cols]



    
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
        branchdata=adata[adata.obs['noctx_supervised_name'].str.contains(branch),:].copy()
        branchdata.obs_names_make_unique()
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        n=200       
        for x in tqdm.tqdm(chunks(branchdata.obs.index[branchdata.obs.latent_time.to_numpy().argsort()],n)):
            if len(x)>20:
                times.append(branchdata[x,:].obs['latent_time'].mean())
                mat=pd.DataFrame(branchdata[x,lineage_genes].X,index=x,columns=lineage_genes)
                values.append(mat.corr())

        values=np.array(values)
        times=np.array(times)


        df1=pd.DataFrame(branchdata[:,lineage_genes].X)
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
