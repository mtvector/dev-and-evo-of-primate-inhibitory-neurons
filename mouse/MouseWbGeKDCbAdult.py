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
sc.settings.figdir='/wynton/group/ye/mtschmitz/figures/mouseWbGeAdult/'
scv.settings.figdir='/wynton/group/ye/mtschmitz/figures/mouseWbGeAdult/'
sc.settings.file_format_figs='pdf'
sc.settings.autosave=True
sc.settings.autoshow=False

def most_frequent(List): 
    return max(set(List), key = List.count)

def variance(X,axis=1):
    return( (X.power(2)).mean(axis=axis)-(np.power(X.mean(axis=axis),2)) )

def read_mtx_market(p):
    from collections import Iterable
    def flatten(coll):
        for i in coll:
                if isinstance(i, Iterable) and not isinstance(i, str):
                    for subc in flatten(i):
                        yield subc
                else:
                    yield i

    var=[]
    obs=[]
    values=[]
    with open(p,'r') as f:
        for line in f:
            if line.startswith('%'):
                if 'GENE' in line:
                    var.append(line.split())
                if 'BARCODES' in line:
                    obs.append(line.split())
            else:
                values.append([int(x) for x in line.split()])

    var = list(flatten(var))
    obs = list(flatten(obs))
    values=np.array(values)
    shapes=values[0,:]
    values=values[1:,:]
    obs=[x for x in obs if '%' not in x]
    var=[x for x in var if '%' not in x]
    values[:,0:2]=values[:,0:2]-1
    return(anndata.AnnData(scipy.sparse.csr_matrix((values[:,2], (values[:,1], values[:,0])),shape=(shapes[1],shapes[0])),obs=pd.DataFrame(index=obs),var=pd.DataFrame(index=var)))


cortexregions=['head','forebrain','cortex','motor']


pth='/wynton/group/ye/mtschmitz/mousefastqpool'
dirs=['BICCN_zeng_MOp_v2','BICCN_zeng_MOp_v3','PRJNA515751_konopka_striatum','PRJNA498989_OB_mouse','PRJNA547712_dev_hypothalamus','SRP135960_linnarson_adultmouse']

dirdict={'SRP135960_linnarson_adultmouse':['ob','ctx','ca1','amygdala','striatum','dentgyr','cortex','hypothalamus'],
         'PRJNA547712_dev_hypothalamus':['e11','e12','e14','p14','p45']}

newfile='/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseWbGeAdult.h5ad'

min_genes=800

if True:#not os.path.exists(newfile):
    adatas=[]
    filename='a_cellbended_200_1000_300e_V0.2'
    for d in reversed(dirs):
        print(d,flush=True)
        filepath=os.path.join(pth,d)
        files=os.listdir(filepath)
        files=[f for f in files if 'kOut' in f]
        print(files)
        for f in files:
            print(f,flush=True)
            if d in dirdict.keys():
                regexp=re.compile('|'.join(dirdict[d]))
                print(regexp)
                if not regexp.search(f.lower()):
                    print('SKIPPIN '+f,flush=True)
                    continue
            try:
                if os.path.exists(os.path.join(filepath,f,filename,'a_cellbended_filtered.h5')):
                    sadata = sc_utils.readCellbenderH5(os.path.join(filepath,f,filename,'a_cellbended_filtered.h5'))
                    sadata =sadata[sadata.obs['latent_cell_probability']>.99,:]
                    sc.pp.filter_cells(sadata,min_genes=min_genes)
                else:
                    sadata=sc_utils.loadPlainKallisto(os.path.join(filepath,f,'all'),min_genes=min_genes)
                sadata.obs.index=[re.sub("-1","",x) for x in sadata.obs.index]
                sadata.uns['name']=f
                sadata.uns['name']=sc_utils.mouse_process_irregular_names(sadata.uns['name'])
                sadata.obs['batch_name']=str(sadata.uns['name'])
                sadata.obs['dataset_name']=d
                sadata.obs['timepoint']=sc_utils.tp_format_mouse(sadata.uns['name'])
                regionstring=sc_utils.region_format_mouse(sadata.uns['name'])
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
                sadata.var_names_make_unique()
                if 'PRJNA515751' in d:
                    sadata.obs['region']='striatum'
                    sadata.obs['tp']='p9'
                    sadata.obs['batch_name']=d
                pd.DataFrame(sadata.obs.index).to_csv(os.path.join(filepath,f,'cellbendedcells.txt'),index=False, header=False)
                if sadata.shape[0]>10:
                    adatas.append(sadata)
            except Exception as e:
                print(e)
                print('fail')

    """
    mtx_path='/wynton/group/ye/mtschmitz/mousefastqpool/PRJNA478603_saunders_mouse/'
    mats=[]
    filenames=[]
    for f in os.listdir(mtx_path):
        if 'raw.dge.txt' in f:
            print(f)
            filenames.append(f)
            xdata=read_mtx_market(os.path.join(mtx_path,f))
            xdata.obs['region']=re.sub('GSE116470_F_GRCm38\.81\.P60|\.raw.dge\.txt','',f)
            xdata.obs['timepoint']=81.0
            xdata.obs['batch_name']=f
            xdata.obs['dataset_name']='PRJNA478603_saunders'
            xdata.var_names_make_unique()
            mats.append(xdata)       
    adatas=adatas+mats    
    """
    xdata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseWbGeProcessed.h5ad')
    xdata.X=xdata.raw.X[:,xdata.raw.var.index.isin(xdata.var.index)]#.todense()
    xdata=xdata[~xdata.obs.region.str.contains('(?i)ob'),:]

    xdata.var_names_make_unique()
    #adatas=adatas+[multi]
    adatas.append(xdata)
    for q in adatas:
        q.var.index=[x.upper() for x in q.var.index]
        q.var_names_make_unique()
        q.obs_names_make_unique()
    print(adatas,flush=True)
    adata=sc.AnnData.concatenate(*adatas)
    adata.var.columns = adata.var.columns.astype(str)
    adata.obs.columns = adata.obs.columns.astype(str)
    adata.obs['clean_cellname']=[re.sub('-[0-9]+','',x) for x in  adata.obs.index]
    adata.obs['full_cellname']=adata.obs['clean_cellname'].astype(str)+'_'+adata.obs['batch_name'].astype(str)
    adata.obs.index=list(adata.obs['full_cellname'])
    adata.raw=adata
    adata.write(newfile)        
adata=sc.read(newfile)
adata.var.index=[x.upper() for x in adata.var.index]

print(adata.obs['supervised_name'].unique())

sc.pp.filter_cells(adata,min_genes=800)
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

adata.write('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMouseWbGeAdultProcessed.h5ad')

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
    'Glia':'lightgoldenrodyellow',
    'OB-GC NR2F2/PENK':'brown',
    'Ndnf Lamp5':'khaki',
    'Ndnf Sst':'peachpuff',
    'Cck Dpy1l1':'dimgray'}

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
sc.pl.umap(adata,color=['region'],save='region')
sc.pl.umap(adata,color=['dataset_name'],save='dataset_name')
easygenes= ['MKI67','SOX2','AQP4','EDNRB','IL33','PDGFRA','HOPX','ERBB4','CALB1','CALB2','GAD1','GAD2','GADD45G','LHX6','LHX7','LHX8','RBP4','RXRA','NXPH1','NXPH2','SST','NKX2-1','MAF','SP8','SP9','PROX1','VIP','CCK','NPY','LAMP5','HTR1A','HTR3A','NR2F1','NR2F2','TOX3','ETV1','SCGN','FOXP1','FOXP2','FOXP4','TSHZ1','OPRM1','CASZ1','HMX3','NKX6-2','TH','DDC','SLC18A2','PAX6','MEIS2','SHTN1','ISL1','PENK','DRD1','ADORA2A','CHAT','ZNF503','STXBP6','CRABP1','ZIC1','ZIC2','FOXG1','TAC1','TAC2','TAC3','TACR1','TACR2','TACR3','EBF1','DLX5','GSX2','RBFOX3','PPP1R17','EOMES','DCX','TUBB3','BCL11B','TLE4','FEZF2','SATB2','TBR1','AIF1','RELN','PECAM1','HBZ','ROBO1','ROBO2','ROBO3','ROBO4']
easygenes=[x for x in easygenes if x in adata.var.index]
sc.pl.umap(adata,color=easygenes,use_raw=False,save='FaveGenes')

INgenes=['MKI67','DCX','DLX2','ZIC1','SP8','SCGN','CCK','VIP','MEIS2','PAX6','FOXP2','TSHZ1','FOXP1','OPRM1','ISL1','PENK','LHX6','SST','NPY','CHODL','CRABP1','ANGPT2','MAF','TAC2']         
INgenes=[x for x in INgenes if x in adata.var.index]
sc.pl.umap(adata,color=INgenes,use_raw=False,save='SupplementGenes')

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
adata.obs.loc[adata.obs['simplified_allen'].astype(str).str.contains('(?i)CTX'),'simplified_allen']='Excitatory'
sc.pl.umap(adata,color=['allen_class_label','allen_cluster_label','simplified_allen'],save='allenlabels')


f = plt.figure()
df_plot = adata.obs.groupby(['supervised_name', 'batch_name']).size().reset_index().pivot(columns='batch_name', index='supervised_name', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVbatchBar.pdf'), bbox_inches="tight")

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

f = plt.figure()
df_plot = adata.obs.groupby(['supervised_name', 'leiden']).size().reset_index().pivot(columns='leiden', index='supervised_name', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVleidenBar.pdf'), bbox_inches="tight")

f = plt.figure()
df_plot = adata.obs.groupby(['leiden', 'region']).size().reset_index().pivot(columns='region', index='leiden', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'regionVleidenBar.pdf'), bbox_inches="tight")

f = plt.figure()
df_plot = adata[adata.obs['region'].isin(cortexregions),:].obs.groupby(['supervised_name', 'region']).size().reset_index().pivot(columns='region', index='supervised_name', values=0)#.apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVcorticalregionBar.pdf'), bbox_inches="tight")

df_plot = adata.obs.groupby(['supervised_name', 'region']).size().reset_index().pivot(columns='region', index='supervised_name', values=0)#.apply(lambda g: g / g.sum(),1)
df_plot=df_plot.loc[df_plot.sum(1)>50,:]
regiontotals=adata.obs.groupby([ 'region']).size()
ax = (df_plot/regiontotals).T.plot(kind='bar', legend=False,stacked=False)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion of cells in region')
ax.grid(False)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'regionMakeupSupervisedNameBarNorm.pdf'), bbox_inches="tight")

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
sc.tl.dendrogram(adata,groupby=hierarchy_key)
var_dict=dict(zip(adata.uns["dendrogram_['"+hierarchy_key+"']"]['categories_ordered'],[topdict[x] for x in adata.uns["dendrogram_['"+hierarchy_key+"']"]['categories_ordered']]))
sc.pl.matrixplot(adata,groupby=hierarchy_key,var_names=var_dict,save='top_degenes',cmap='RdBu_r',use_raw=False,dendrogram=True)
