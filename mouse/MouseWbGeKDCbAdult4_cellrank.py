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
sc.settings.figdir='/wynton/group/ye/mtschmitz/figures/mouseWbGeAdult2/'
scv.settings.figdir='/wynton/group/ye/mtschmitz/figures/mouseWbGeAdult2/'
sc.settings.file_format_figs='pdf'
sc.settings.autosave=True
sc.settings.autoshow=False
equalize=False

if equalize:
    sc.settings.figdir='/wynton/group/ye/mtschmitz/figures/mouseWbGeAdult2ET/'
    scv.settings.figdir='/wynton/group/ye/mtschmitz/figures/mouseWbGeAdult2ET/'

#adata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityDYNAMICALMouseWbGeAdult.h5ad')


if equalize:
    adata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityEqualizeTerminalsDYNAMICALMouseWbGeAdult.h5ad')
else:
    adata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityDYNAMICALMouseWbGeAdult.h5ad')
    #adata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityDYNAMICALsmallMouseWbGeAdult.h5ad')

adata=adata[:,np.isfinite(adata.X.mean(0))]
adata.obs['agg_supervised_name']='nan'
adata.obs['old_leiden']='nan'
supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MouseAdultAggSupervised.txt')
adata=adata[~adata.obs.index.duplicated(),:]
supercell=supercell.loc[supercell['agg_supervised_name']!='nan',:]
ind=adata.obs.index[adata.obs.index.isin(supercell['full_cellname'])]
supercell.index=supercell['full_cellname']
supercell=supercell.loc[~supercell.index.duplicated(),:]
supercell=supercell.loc[ind,:]
adata.obs.loc[ind,'agg_supervised_name']=supercell['agg_supervised_name']
adata.obs.loc[ind,'old_leiden']=supercell['leiden']
adata.obs['agg_supervised_name']=adata.obs['agg_supervised_name'].astype(str)
adata.obs['old_leiden']=adata.obs['old_leiden'].astype(str)
#adata.obs['agg_supervised_name']=adata.obs['agg_supervised_name']+"___"+adata.obs['old_leiden']

sc.pl.umap(adata,color=['agg_supervised_name'])

SPNgenes=['ISL1','TAC1','PDYN','NRXN1','EBF1','NPY1R','PCP4','DRD1','DRD2','PENK','ADORA2A','GPR6','FOXP1','FOXP2','FOXP4','CASZ1','OTOF','OPRM1','PBX3']
scanpy.pl.matrixplot(adata[adata.obs.agg_supervised_name.str.contains('FOXP|SPN'),:],groupby='agg_supervised_name',var_names=SPNgenes,cmap='coolwarm',dendrogram=True,use_raw=False,standard_scale='var',save='LGEonlyeSPNgenes')

sc.tl.dendrogram(adata,groupby='agg_supervised_name')
df_plot = adata.obs.groupby(['agg_supervised_name', 'region']).size().reset_index().pivot(columns='region', index='agg_supervised_name', values=0).apply(lambda g: g / g.sum(),1)
df_plot=df_plot.loc[adata.uns["dendrogram_agg_supervised_name"]['categories_ordered'],:]
these_colors=[region_color_dict[x] for x in df_plot.columns]
ax = df_plot.plot(kind='bar', legend=False,stacked=True,color=these_colors)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion of cells in region')
ax.grid(False)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedNameRegionsStackBar.pdf'), bbox_inches="tight")


"""
hierarchy_key='agg_supervised_name'
sufficient_cells=adata.obs[hierarchy_key].value_counts().index[adata.obs[hierarchy_key].value_counts()>10]
adata=adata[adata.obs[hierarchy_key].isin(sufficient_cells),:]
adata.obs[hierarchy_key].cat.remove_unused_categories(inplace=True)
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

hierarchy_key='old_leiden'
sufficient_cells=adata.obs[hierarchy_key].value_counts().index[adata.obs[hierarchy_key].value_counts()>10]
adata=adata[adata.obs[hierarchy_key].isin(sufficient_cells),:]
adata.obs[hierarchy_key].cat.remove_unused_categories(inplace=True)
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


sc.pl.violin(adata, ['n_genes'],groupby='dataset_name',
             jitter=0.4, multi_panel=True,rotation=90,stripplot=False,save='QC')
print('QC')

print(adata)

mouseMSNs=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/FOXTSHZ1mouseAdultStriatumMSNMarkers.txt',sep='\s')
mouseMSNs.index=mouseMSNs['Class']
mouseMSNs=mouseMSNs.iloc[:,3:]
mouseMSNs=mouseMSNs.div(mouseMSNs.max(0),axis=1)
mouseMSNs=mouseMSNs.loc[['Neuron_dSPN_Drd1','Neuron_iSPN_Adora2a','Neuron_SPN_BNST_Amygdala_Otof'],:]
mouseMSNs.drop('Syt6',inplace=True,axis=1)
scanpy.pl.matrixplot(adata[adata.obs.agg_supervised_name.str.contains('FOXP|SPN'),:],groupby='agg_supervised_name',var_names=[re.sub('\.1','',x.upper()) for x in mouseMSNs.columns],cmap='coolwarm',dendrogram=True,use_raw=False,standard_scale='var',save='LGEonlyeSPNgenes')
"""



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
    'Ctx_NR2F2/PAX6':'sienna',
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
sc.pl.umap(adata,color=['agg_supervised_name'])
my_pal=[paldict[x] for x in adata.obs['agg_supervised_name'].cat.categories]

sc.pl.umap(adata,color=['old_leiden'],legend_loc='on data',legend_fontsize=4,palette=my_pal,save='old_leiden')
sc.pl.umap(adata,color=['agg_supervised_name'],legend_loc='on data',legend_fontsize=4,palette=my_pal,save='agg_supervised_adult')
sc.pl.umap(adata,color='latent_time',save='_linear_latent_time',color_map=matplotlib.cm.RdYlBu_r)
ingenes=['VIP','VIPR2','SNCG','CCK','LAMP5','PROX1','NR2F2','SLC17A6','SLC17A7','MEIS2','TSHZ1','DLX2','DLX6','GAD1','GAD2','PAX6','ETV1','SCGN','TH','TRH','FOXP1','FOXP2','ISL1','PENK','ZIC1','RPRM','CRH','CALB1','CALB2','LHX6','LHX8','MAF','CRABP1','PVALB','SST','NPY','NPY1R','CHODL','CORT']
ingenes=[x for x in ingenes if x in adata.var.index]
sc.pl.umap(adata,color=ingenes,save='_INgenes2',use_raw=False)
ingenes=['VIP','VIPR2','SNCG','CCK','LAMP5','PROX1','NR2F2','SLC17A6','SLC17A7','MEIS2','TSHZ1','DLX2','DLX6','GAD1','GAD2','PAX6','ETV1','SCGN','TH','TRH','FOXP1','FOXP2','ISL1','PENK','ZIC1','RPRM','CRH','CALB1','CALB2','LHX6','LHX8','MAF','CRABP1','PVALB','SST','NPY','NPY1R','CHODL','CORT']
ingenes=[x for x in ingenes if x in adata.var.index]
sc.pl.umap(adata,color=ingenes,save='_INgenes',use_raw=False)


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

ckvk=vk+ck
ckvk.compute_transition_matrix()

d={}
for c in adata.obs.agg_supervised_name.unique():
    if not any([x in c.upper() for x in ['MGE','RMTW','CGE','LGE','VMF','G2-M','S-PHASE','GLIA','NAN','TRANSITION','EXCITATORY']]):
        print(c)
        select_cells=adata.obs.agg_supervised_name.isin([c])
        d[c]=[adata.obs.index[i] for i in range(adata.shape[0]) if select_cells[i]]
terminal_states=list(d.keys())#+['MGE_CRABP1/MAF']


g=cr.tl.estimators.CFLARE(ckvk)
g.compute_eigendecomposition()
g.set_terminal_states(d,cluster_key='agg_supervised_name')

#comment these two to go fast
g.compute_absorption_probabilities(show_progress_bar=True)

g.compute_lineage_drivers()
l=[]
for c in g.lineage_drivers.columns:
    print(c)
    col=g.lineage_drivers[c]
    srt=col.sort_values(ascending=False)
    l.append(srt.index)
    l.append(srt)
df=pd.DataFrame(l,index=np.repeat(g.lineage_drivers.columns,2)).T
df.to_csv(os.path.join(sc.settings.figdir,"lineage_drivers.csv"))


probDF=pd.DataFrame(g.absorption_probabilities,columns=g.absorption_probabilities.names,index=adata.obs.index)
probDF['supervised_name']=list(adata.obs['agg_supervised_name'])
probDF.index=adata.obs.index
adata.obs['Adult']=adata.obs.agg_supervised_name.isin(adata.uns['to_terminal_states_names']).fillna(value=False)
probDF.loc[list(adata.obs['Adult']),:]=np.nan
adata.obs['Adult']=adata.obs['Adult'].astype(str)

#adata.obs['eSPN_prob']=probDF['eSPN']
#adata.obs['Lamp5_prob']=probDF['Lamp5']
#adata.obs['Pvalb_prob']=probDF['Pvalb']
#adata.obs['Sst_prob']=probDF['Sst']
#adata.obs['dSPN_prob']=probDF['dSPN']
#adata.obs['iSPN_prob']=probDF['iSPN']

"""trim=0.05
trimmedProbDF=probDF[(probDF<1-trim)]#& (probDF>trim)
probDF=(probDF-trimmedProbDF.mean(0))/trimmedProbDF.std(0)
"""
#adata.obs['predicted_end']=probDF.idxmax(1)
#sc.pl.umap(adata,color=['eSPN_prob','Lamp5_prob','Pvalb_prob','Sst_prob','iSPN_prob','dSPN_prob'],use_raw=False,save='CFLARE_probabilities')
#g.plot_absorption_probabilities(same_plot=False, size=50, lineages=terminal_states[0:min(18,len(terminal_states))],basis='X_umap',save='CFLARE_all_probabiliities')
#sc.pl.umap(adata,color=['predicted_end'],use_raw=False,save='CFLARE_predicted_end')

adata.obs.drop('terminal_states_probs',axis=1,inplace=True)
adata.write('/wynton/home/ye/mschmitz1/CFLARE'+str(equalize)+'mouseAdult2.h5ad')

qvmdata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbProcessedQvMMotor.h5ad')


nz0=qvmdata.uns['neighbors']['distances'].nonzero()[0]
nz1=qvmdata.uns['neighbors']['distances'].nonzero()[1]
nz0i=qvmdata.obs.index[nz0]
nz1i=qvmdata.obs.index[nz1]
results=[nz0,
         nz1,
         qvmdata.obs.loc[nz0i,'species'],
         qvmdata.obs.loc[nz1i,'species'],
         qvmdata.obs.loc[nz0i,'supervised_name'],
         qvmdata.obs.loc[nz1i,'supervised_name']]
df=pd.DataFrame(results,index=['ind1','ind2','species1','species2','classname1','classname2']).T
mat=df.groupby('ind1')['classname2'].value_counts(dropna=False).unstack('classname2').fillna(0)
#mat=mat.loc[:,mat.columns!='nan']

crossdf=df.loc[(df['species1']=='Macaque')&(df['species2']=='Mouse'),:]

crossdf2=df.loc[(df['species2']=='Macaque')&(df['species1']=='Mouse'),:]

#crossdf=crossdf.loc[:,['ind1','ind2','classname1','classname2']]
#crossdf2=crossdf2.loc[:,['ind1','ind2','classname1','classname2']]
crossdf2.columns=['ind2','ind1','species1','species2','classname1','classname2']
mnndf=pd.concat([crossdf,crossdf2],)
mnndf['combo']=mnndf['ind1'].astype(str)+'_'+mnndf['ind2'].astype(str)

mnndf=mnndf.rename({'classname1':'classname2','classname2':'classname1'},axis=1)

mnndf=mnndf.loc[~mnndf['classname1'].str.contains('phase|G2-M|Transition'),:]
mnndf=mnndf.loc[~mnndf['classname2'].str.contains('phase|G2-M|Transition'),:]

cdfsums=mnndf.loc[mnndf['combo'].duplicated(),:].groupby('classname1')['classname2'].value_counts(normalize=True)

sankeydf=cdfsums.unstack().stack().reset_index()  

adata.obs['agg_supervised_name'].cat.remove_unused_categories(inplace=True)

##################Forwards
"""
devclusters=[c for c in adata.obs.agg_supervised_name.unique() if any([x in str(c).upper() for x in ['MGE','RMTW','CGE','LGE','VMF','G2-M','S-PHASE','GLIA','NAN','TRANSITION','Excitatory']])]
adata.obs['Adult']=adata.obs.dataset_name.str.contains('BICCN|OB|konopkka|adult')|~adata.obs.agg_supervised_name.isin(devclusters)
adultframe=adata.obs.loc[~adata.obs['Adult'],:]
adultframe['agg_supervised_name'].cat.remove_unused_categories(inplace=True)
adultframesums=adultframe.groupby(['agg_supervised_name'])['predicted_end'].value_counts(normalize=True)
adultsankeydf=adultframesums.unstack().stack().reset_index()"""
###################Backwards
#print(adata.obs)
#print(adata.obs.index)
#print(probDF.sort_values(c,ascending = False))
#print(adata.obs.index[probDF.sort_values(c,ascending = False).index[:500]])
##########Backwards top N mapping

max_cells_in_category=1000
vcs=probDF['supervised_name'].value_counts()
print(vcs)
for t in vcs.index:
    print(t)
    print(vcs[t])
    print(max(vcs[t]-max_cells_in_category,0))
    destroy=np.random.choice(probDF.index[probDF['supervised_name']==t],size=max(vcs[t]-max_cells_in_category,0),replace=False)
    probDF=probDF.loc[~probDF.index.isin(destroy),:]

topinds={}
for c in probDF.columns:
    topinds[c]=list(adata.obs.loc[probDF.sort_values(c,ascending = False).index[:100],'agg_supervised_name'])
adultsankeydf=pd.DataFrame(topinds).apply(lambda x: x.value_counts(normalize=True)).stack().reset_index()  
adultsankeydf.columns=['agg_supervised_name','predicted_end',0]

##########Backwards mean absorbtion normalized
"""
probDF['supervised_name']=list(adata.obs['supervised_name'])
meanDF=probDF.fillna(np.inf).groupby(['supervised_name']).mean().replace(np.inf, 0)
meanDF=meanDF/meanDF.sum(0)
#meanDF=meanDF.loc[meanDF.index.str.contains('VMF|LGE|MGE|CGE|RMTW'),:]
adultsankeydf=meanDF.stack().reset_index()
adultsankeydf.columns=['agg_supervised_name','predicted_end',0]
"""

adultsankeydf=adultsankeydf.loc[adultsankeydf['agg_supervised_name'].str.contains('MGE|CGE|VMF|LGE|RMTW'),:]
adultsankeydf=adultsankeydf.loc[~adultsankeydf['agg_supervised_name'].str.contains('supervised'),:]
adultsankeydf=adultsankeydf.loc[~adultsankeydf['predicted_end'].str.contains('supervised'),:]


cn1dict=dict(zip(sankeydf['classname1'].unique(),range(len(sankeydf['classname1'].unique()))))
cn1len=len(sankeydf['classname1'].unique())
cn2dict=dict(zip(sankeydf['classname2'].unique(),np.array(range(len(sankeydf['classname2'].unique())))+cn1len))
cn2len=len(sankeydf['classname2'].unique())
cn3dict=dict(zip(adultsankeydf['predicted_end'].unique(),np.array(range(len(adultsankeydf['predicted_end'].unique())))+cn1len+cn2len))
cn3len=len(adultsankeydf['predicted_end'].unique())
overalldict=dict(zip(list(cn1dict.keys())+list(cn2dict.keys())+list(cn3dict.keys()),list(cn1dict.values())+list(cn2dict.values())+list(cn3dict.values())))

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
    'Ctx_NR2F2/PAX6':'sienna',
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
#my_pal=[paldict[x] for x in supercell['agg_supervised_name'].unique()]
my_pal=[paldict[x] for x in adata.obs['agg_supervised_name'].cat.categories]


nodenames=list(cn1dict.keys())+list(cn2dict.keys())+list(cn3dict.keys())
nodecolors=[paldict[x] for x in nodenames]


alphacolors=['rgba'+str(matplotlib.colors.to_rgba(paldict[x],alpha=.59999999999999)) for x in paldict.keys()]
paldict=dict(zip(paldict.keys(),alphacolors))

toclass=list(sankeydf['classname1'])+list(adultsankeydf['predicted_end'])
fromclass=list(sankeydf['classname1'])+list(adultsankeydf['agg_supervised_name'])
values=list(sankeydf[0])+list(adultsankeydf[0])
color=[paldict[x] for x in toclass]
#linkcolor=[paldict[y] for x,y in zip(sankeydf[0]**2,sankeydf['classname1'])]
linkcolor=[re.sub('0.59999999999999','%.2f' %x ,paldict[y]) for x,y in zip(values,fromclass)]

import plotly
import plotly.graph_objects as go

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 5,
      thickness = 30,
      line = dict(color = "black", width = 0.5),
      label = nodenames,
      color = nodecolors
    ),
    link = dict(
      source = [cn1dict[x] for x in sankeydf['classname1']]+[cn2dict[x] for x in adultsankeydf['agg_supervised_name']],
      target = [cn2dict[x] for x in sankeydf['classname2']]+[cn3dict[x] for x in adultsankeydf['predicted_end']],
      value = values,
      color=linkcolor
      #color=[paldict[x] for x in sankeydf['classname1']]
  ))])

fig.update_layout(title_text="Cross Species MNN", font_size=20)
import plotly.io as pio
os.environ["PATH"] = os.environ["PATH"] +':/wynton/home/ye/mschmitz1/utils/miniconda3/envs/cellrank/bin/'
pio.write_image(fig, os.path.join(sc.settings.figdir,'CFLARE_Sankey.pdf'),width=1200,height=1200)

del adata.obsm['to_terminal_states']
del adata.uns['to_terminal_states_names']
del g

#g=cr.tl.estimators.CFLARE(ck)
g = cr.tl.estimators.GPCCA(ckvk)
#g.compute_eigendecomposition()
g.compute_schur(n_components=50)

g.set_terminal_states(d,cluster_key='agg_supervised_name')

g.compute_absorption_probabilities(show_progress_bar=True)
probDF=pd.DataFrame(g.absorption_probabilities,columns=g.absorption_probabilities.names,index=adata.obs.index)
g.plot_absorption_probabilities(same_plot=False, size=50, basis='X_umap',lineages=terminal_states[0:min(18,len(terminal_states))],save='GPCCA_all_probabiliities')

probDF['supervised_name']=list(adata.obs['agg_supervised_name'])
probDF.index=adata.obs.index
adata.obs['Adult']=adata.obs.agg_supervised_name.isin(adata.uns['to_terminal_states_names']).fillna(value=False)
probDF.loc[list(adata.obs['Adult']),:]=np.nan
adata.obs['Adult']=adata.obs['Adult'].astype(str)

#adata.obs['eSPN_prob']=probDF['eSPN']
#adata.obs['Lamp5_prob']=probDF['Lamp5']
#adata.obs['Pvalb_prob']=probDF['Pvalb']
#adata.obs['Sst_prob']=probDF['Sst']
#adata.obs['dSPN_prob']=probDF['dSPN']
#adata.obs['iSPN_prob']=probDF['iSPN']

"""trim=0.05
trimmedProbDF=probDF[(probDF<1-trim)]#& (probDF>trim)
probDF=(probDF-trimmedProbDF.mean(0))/trimmedProbDF.std(0)
"""
#adata.obs['predicted_end']=probDF.idxmax(1)
#sc.pl.umap(adata,color=['eSPN_prob','Lamp5_prob','Pvalb_prob','Sst_prob','iSPN_prob','dSPN_prob'],use_raw=False,save='GPPCA_probabilities')
#sc.pl.umap(adata,color=['predicted_end'],use_raw=False,save='GPPCA_predicted_end')

adata.obs.drop('terminal_states_probs',axis=1,inplace=True)
adata.write('/wynton/home/ye/mschmitz1/GPCCA'+str(equalize)+'mouseAdult2.h5ad')

nz0=qvmdata.uns['neighbors']['distances'].nonzero()[0]
nz1=qvmdata.uns['neighbors']['distances'].nonzero()[1]
nz0i=qvmdata.obs.index[nz0]
nz1i=qvmdata.obs.index[nz1]
results=[nz0,
         nz1,
         qvmdata.obs.loc[nz0i,'species'],
         qvmdata.obs.loc[nz1i,'species'],
         qvmdata.obs.loc[nz0i,'supervised_name'],
         qvmdata.obs.loc[nz1i,'supervised_name']]
df=pd.DataFrame(results,index=['ind1','ind2','species1','species2','classname1','classname2']).T
mat=df.groupby('ind1')['classname2'].value_counts(dropna=False).unstack('classname2').fillna(0)
#mat=mat.loc[:,mat.columns!='nan']

crossdf=df.loc[(df['species1']=='Macaque')&(df['species2']=='Mouse'),:]

crossdf2=df.loc[(df['species2']=='Macaque')&(df['species1']=='Mouse'),:]

#crossdf=crossdf.loc[:,['ind1','ind2','classname1','classname2']]
#crossdf2=crossdf2.loc[:,['ind1','ind2','classname1','classname2']]
crossdf2.columns=['ind2','ind1','species1','species2','classname1','classname2']
mnndf=pd.concat([crossdf,crossdf2],)
mnndf['combo']=mnndf['ind1'].astype(str)+'_'+mnndf['ind2'].astype(str)

mnndf=mnndf.rename({'classname1':'classname2','classname2':'classname1'},axis=1)

mnndf=mnndf.loc[~mnndf['classname1'].str.contains('phase|G2-M|Transition'),:]
mnndf=mnndf.loc[~mnndf['classname2'].str.contains('phase|G2-M|Transition'),:]

cdfsums=mnndf.loc[mnndf['combo'].duplicated(),:].groupby('classname1')['classname2'].value_counts(normalize=True)

sankeydf=cdfsums.unstack().stack().reset_index()  

adata.obs['agg_supervised_name'].cat.remove_unused_categories(inplace=True)

##################Forwards
"""
devclusters=[c for c in adata.obs.agg_supervised_name.unique() if any([x in str(c).upper() for x in ['MGE','RMTW','CGE','LGE','VMF','G2-M','S-PHASE','GLIA','NAN','TRANSITION','Excitatory']])]
adata.obs['Adult']=adata.obs.dataset_name.str.contains('BICCN|OB|konopkka|adult')|~adata.obs.agg_supervised_name.isin(devclusters)
adultframe=adata.obs.loc[~adata.obs['Adult'],:]
adultframe['agg_supervised_name'].cat.remove_unused_categories(inplace=True)
adultframesums=adultframe.groupby(['agg_supervised_name'])['predicted_end'].value_counts(normalize=True)
adultsankeydf=adultframesums.unstack().stack().reset_index()"""
###################Backwards
max_cells_in_category=1000
vcs=probDF['supervised_name'].value_counts()
for t in vcs.index:
    print(t)
    print(vcs[t])
    print(max(vcs[t]-max_cells_in_category,0))
    destroy=np.random.choice(probDF.index[probDF['supervised_name']==t],size=max(vcs[t]-max_cells_in_category,0),replace=False)
    probDF=probDF.loc[~probDF.index.isin(destroy),:]


topinds={}
for c in probDF.columns:
    topinds[c]=list(adata.obs.loc[probDF.sort_values(c,ascending = False).index[:100],'agg_supervised_name'])
adultsankeydf=pd.DataFrame(topinds).apply(lambda x: x.value_counts(normalize=True)).stack().reset_index()  
adultsankeydf.columns=['agg_supervised_name','predicted_end',0]
##########Backwards mean absorbtion normalized
"""
probDF['supervised_name']=list(adata.obs['supervised_name'])
meanDF=probDF.fillna(np.inf).groupby(['supervised_name']).mean().replace(np.inf, 0)
meanDF=meanDF/meanDF.sum(0)
#meanDF=meanDF.loc[meanDF.index.str.contains('VMF|LGE|MGE|CGE|RMTW'),:]
adultsankeydf=meanDF.stack().reset_index()
adultsankeydf.columns=['agg_supervised_name','predicted_end',0]
"""
adultsankeydf=adultsankeydf.loc[adultsankeydf['agg_supervised_name'].str.contains('MGE|CGE|VMF|LGE|RMTW'),:]
adultsankeydf=adultsankeydf.loc[~adultsankeydf['agg_supervised_name'].str.contains('supervised'),:]
adultsankeydf=adultsankeydf.loc[~adultsankeydf['predicted_end'].str.contains('supervised'),:]

cn1dict=dict(zip(sankeydf['classname1'].unique(),range(len(sankeydf['classname1'].unique()))))
cn1len=len(sankeydf['classname1'].unique())
cn2dict=dict(zip(sankeydf['classname2'].unique(),np.array(range(len(sankeydf['classname2'].unique())))+cn1len))
cn2len=len(sankeydf['classname2'].unique())
cn3dict=dict(zip(adultsankeydf['predicted_end'].unique(),np.array(range(len(adultsankeydf['predicted_end'].unique())))+cn1len+cn2len))
cn3len=len(adultsankeydf['predicted_end'].unique())
overalldict=dict(zip(list(cn1dict.keys())+list(cn2dict.keys())+list(cn3dict.keys()),list(cn1dict.values())+list(cn2dict.values())+list(cn3dict.values())))

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
    'Ctx_NR2F2/PAX6':'sienna',
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

#my_pal=[paldict[x] for x in supercell['agg_supervised_name'].unique()]
my_pal=[paldict[x] for x in adata.obs['agg_supervised_name'].cat.categories]


nodenames=list(cn1dict.keys())+list(cn2dict.keys())+list(cn3dict.keys())
nodecolors=[paldict[x] for x in nodenames]


alphacolors=['rgba'+str(matplotlib.colors.to_rgba(paldict[x],alpha=.59999999999999)) for x in paldict.keys()]
paldict=dict(zip(paldict.keys(),alphacolors))

toclass=list(sankeydf['classname1'])+list(adultsankeydf['predicted_end'])
fromclass=list(sankeydf['classname1'])+list(adultsankeydf['agg_supervised_name'])
values=list(sankeydf[0])+list(adultsankeydf[0])
color=[paldict[x] for x in toclass]
#linkcolor=[paldict[y] for x,y in zip(sankeydf[0]**2,sankeydf['classname1'])]
linkcolor=[re.sub('0.59999999999999','%.2f' %x ,paldict[y]) for x,y in zip(values,fromclass)]

import plotly
import plotly.graph_objects as go

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 5,
      thickness = 30,
      line = dict(color = "black", width = 0.5),
      label = nodenames,
      color = nodecolors
    ),
    link = dict(
      source = [cn1dict[x] for x in sankeydf['classname1']]+[cn2dict[x] for x in adultsankeydf['agg_supervised_name']],
      target = [cn2dict[x] for x in sankeydf['classname2']]+[cn3dict[x] for x in adultsankeydf['predicted_end']],
      value = values,
      color=linkcolor
      #color=[paldict[x] for x in sankeydf['classname1']]
  ))])

fig.update_layout(title_text="Cross Species MNN", font_size=20)
import plotly.io as pio
os.environ["PATH"] = os.environ["PATH"] +':/wynton/home/ye/mschmitz1/utils/miniconda3/envs/cellrank/bin/'
pio.write_image(fig, os.path.join(sc.settings.figdir,'GPCCA_Sankey.pdf'),width=1200,height=1200)

del adata.obsm['to_terminal_states']
del adata.uns['to_terminal_states_names']
del g

g = cr.tl.estimators.GPCCA(ckvk)
#g.compute_eigendecomposition()
g.compute_schur(n_components=100,method='krylov')
print('Schur complete',flush=True)
g.plot_spectrum(real_only=True,save='spectrum')
g.compute_macrostates(n_states=50, n_cells=30,cluster_key="agg_supervised_name")
print('Macrostates computed', flush=True)
#adata.write('/wynton/home/ye/mschmitz1/GPCCAmacrostatesmouseAdult2.h5ad')
adata.uns['macrostates']=list(g.macrostates.cat.categories)
g.set_terminal_states_from_macrostates([y for y in adata.uns['macrostates'].unique() if any([x in y for x in terminal_states])])
g.compute_absorption_probabilities()
probDF=pd.DataFrame(g.absorption_probabilities,columns=g.absorption_probabilities.names,index=adata.obs.index)
adata.obs['predicted_end']=probDF.idxmax(1)
#sc.pl.umap(adata,color=['eSPN_prob','Lamp5_prob','Pvalb_prob','Sst_prob','iSPN_prob','dSPN_prob'],use_raw=False,save='GPCCA_macrostates_selected_probabilities')
sc.pl.umap(adata,color=['predicted_end'],use_raw=False,save='GPCCA_macrostates_predicted_end')
g.plot_absorption_probabilities(same_plot=False, size=50,lineages=terminal_states[0:min(18,len(terminal_states))], basis='X_umap',save='GPCCA_macrostates_all_probabiliities')
cr.tl.initial_states(adata, cluster_key='agg_supervised_name')
cr.pl.initial_states(adata, discrete=True,save='GPCCA_macrostates_initial states')

adata.obs.drop('terminal_states_probs',axis=1,inplace=True)
#adata.write('/wynton/home/ye/mschmitz1/GPCCAmacrostatesmouseAdult2.h5ad')

del adata.obsm['to_terminal_states']
del adata.uns['to_terminal_states_names']
del g

