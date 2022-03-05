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
figdir='/wynton/group/ye/mtschmitz/figures/macaqueGeAllcortexHippoCB202002/'
sc.settings.figdir=figdir
scv.settings.figdir=figdir
sc.settings.file_format_figs='pdf'
sc.settings.autosave=True
sc.settings.autoshow=False
scanpy.set_figure_params(scanpy=True,dpi_save=400)
#my_pal= ['slategray','salmon','goldenrod','cyan','indigo','darkolivegreen','fuchsia']+seaborn.color_palette("nipy_spectral",n_colors=10).as_hex()+ seaborn.color_palette("cubehelix",n_colors=10).as_hex()+seaborn.color_palette("Set3_r",n_colors=10).as_hex()
paldict={'CGE_NR2F2/PROX1': 'slategray',
 'G1-phase_SLC1A3/ATP1A1': '#17344c',
 'G2-M_UBE2C/ASPM': '#19122b',
 'LGE_FOXP1/ISL1': 'cyan',
 'LGE_FOXP1/PENK': 'navy',
 'LGE_FOXP2/TSHZ1': 'goldenrod',
 'LGE_MEIS2/PAX6': 'orangered',
 'MGE_CRABP1/MAF': 'indigo',
 'MGE_CRABP1/TAC3': 'fuchsia',
 'MGE_LHX6/MAF': 'darksalmon',
 'MGE_LHX6/NPY': 'maroon',
 'RMTW_ZIC1/RELN': 'yellow',
 'S-phase_MCM4/H43C': 'lawngreen',
 'Transition': '#3c7632',
 'VMF_ZIC1/ZIC2': 'teal'}

cortex_order=['pfc','cingulate','motor','somato','temporal','insula','hippocampus','parietal','v1']
import matplotlib.font_manager
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Nimbus Sans','Arial']})
matplotlib.rc('text', usetex=False)

newfile='/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMacaqueGeAllcortexHippocampus.h5ad'
min_genes=800

#adata=sc.read(re.sub('\.h5ad','Clean.h5ad',newfile))
adata=sc.read('/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMacaqueGeAllcortexHippocampusPresupervise.h5ad')
#adata=adata[adata.obs['latent_cell_probability']>.9995,:]

adata.obs['region']=adata.obs.region.replace({'dfc':'pfc'})

print(adata.obs)
"""supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MacaqueGEsupervisednamesHippo3PreserveCompoundname.txt')
ind=adata.obs.index[adata.obs.index.isin(supercell['full_cellname'])]
supercell.index=supercell['full_cellname']
supercell=supercell.loc[ind,:]
adata.obs.loc[ind,'supervised_name']=supercell['supervised_name']
adata.obs.loc[ind,'leiden']=supercell['leiden']
adata.obs['supervised_name']=adata.obs['supervised_name'].astype(str)
adata.obs['supervised_name']=[re.sub('Cortical ','',x) for x in adata.obs['supervised_name'] ]
adata.obs['supervised_name']=adata.obs['supervised_name'].astype('category')
adata.obs['noctx_supervised_name']=adata.obs['supervised_name']"""

supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MacaqueGEsupervisednamesHippoPredictedEnd.txt')
ind=adata.obs.index[adata.obs.index.isin(supercell['full_cellname'])]
supercell.index=supercell['full_cellname']
supercell=supercell.loc[ind,:]
adata.obs['latent_time']=np.nan
adata.obs['predicted_end']='nan'
adata.obs['noctx_supervised_name']='nan'
#adata.obs['noctx_supervised_name']=adata.obs['noctx_supervised_name'].astype(str)
adata.obs.loc[ind,['noctx_supervised_name','predicted_end','latent_time']]=supercell.loc[:,['noctx_supervised_name','predicted_end','latent_time']]
adata.obs.loc[adata.obs['predicted_end'].isna(),'noctx_supervised_name']=adata.obs['noctx_supervised_name'][adata.obs['predicted_end'].isna()]
adata.obs['supervised_name']=adata.obs['noctx_supervised_name']


adata.obs['region']=['hippocampus' if 'hip' in x else x for x in adata.obs.region]
adata.obs['general_region']=sc_utils.macaque_general_regions(adata.obs['region'])

adata=adata[~adata.obs['supervised_name'].isin(['Cajal-Retzius Cell','Endothelial Cell','Pericyte',
                                                    'Microglia','Blood',
                                                    'Choroid Cell','nan',
                                                    'IL33+ Astrocyte','EDNRB+ Astrocyte','Ependymal Cell','Dividing OPC','OPC','Oligodendrocyte','Late Radial Glia','LateG1_EGFR/AQP4']),:]#'Cajal-Retzius Cell',

print("2",adata)
print(adata.obs['supervised_name'].unique())
excitatorycells=[x for x in adata.obs['supervised_name'].unique() if 'excitatory' in x.lower()]
adata=adata[~adata.obs['supervised_name'].isin(excitatorycells),:]
print("3",adata)
print(adata.obs['supervised_name'].unique())
dlx_positive=(adata[:,['DLX1','DLX2','DLX5','DLX6']].X.sum(1).A1>1)#.toarray().flatten()
progenregions=['septum','caudate','lge','cge','mge','ge','poa','optic']
progendomains=[f for f in adata.obs['region'].unique() if any([x in f for x in progenregions]) ]

dividing_cells=['S-phase Dividing Cell','MGE IPC','CGE IPC','G2/M Dividing Cell','Radial Glia','Late Progenitor','Late Radial Glia']
#div_bool=adata.obs['supervised_name'].isin(dividing_cells)
div_bool=adata.obs['supervised_name'].str.contains('G2-M|S-phase|G1-phase|Transition')
#Remove progens not in subcortical progen domains
adata=adata[~(~(adata.obs['region'].isin(progendomains) | dlx_positive) & div_bool),:]
#Remove noncortical glia
adata=adata[(adata.obs['region'].isin(progendomains) | ~adata.obs['supervised_name'].isin(['EDNRB+ Astrocyte','IL33+ Astrocyte','Dividing OPC','OPC','Oligodendrocyte','Ependymal Cell'])),:]
#adata=adata[~(adata.obs['region'].isin(cortex_order) & adata.obs['supervised_name'].isin(['EDNRB+ Astrocyte','Radial Glia','Late Progenitor','Late Radial Glia','IL33+ Astrocyte','Dividing OPC','OPC','Oligodendrocyte','S-phase Dividing Cell','Ependymal Cell','MGE IPC','CGE IPC','G2/M Dividing Cell'])),:]
print("4",adata)

print(adata.obs['supervised_name'].unique())
'''adata.obs.loc[adata.obs['region'].isin(cortex_order),'supervised_name']="Cortical "+adata.obs.loc[adata.obs['region'].isin(cortex_order),'supervised_name']

vcs=adata.obs.supervised_name.value_counts()
for c in vcs.index:
    if 'Cortical' in c:
        if vcs[c]/vcs[re.sub('Cortical ','',c)]<.3:
            adata.obs['supervised_name']=[re.sub('Cortical ','',x) if x==c else x for x in adata.obs['supervised_name'] ]

adata.obs['noctx_supervised_name']=[re.sub('Cortical ','',x) for x in adata.obs['supervised_name'] ]'''
adata.obs['noctx_supervised_name']=[re.sub('Cortical ','',x) for x in adata.obs['supervised_name'] ]

#adata=adata[np.random.choice(adata.obs.index,size=25000,replace=False),:]
adata.raw=adata

for gr in adata.obs['general_region'].unique():
    print(adata.obs.loc[adata.obs.general_region==gr,'region'].unique())

#cortex order above
cortex_colors=seaborn.color_palette("YlOrBr",n_colors=len(cortex_order)+2).as_hex()[2:]
ventral_tel=['ge','cge', 'cge_and_lge', 'lge','mge_and_cge_and_lge', 'mge']
vt_colors=seaborn.blend_palette(('fuchsia','dodgerblue'),n_colors=len(ventral_tel)).as_hex()
med_tel=['septum','pre-optic', 'hypothalamusandpoa','septumandnucleusaccumbens']
mt_colors=seaborn.blend_palette(('grey','black'),n_colors=len(med_tel)).as_hex()
basal_gang=['putamen_and_septum','str', 'putamenandclaustrumandbasalganglia', 'amy']
bg_colors=seaborn.color_palette("Greens",n_colors=len(basal_gang)).as_hex()

all_regions=cortex_order+ventral_tel+med_tel+basal_gang
all_regions_colors=cortex_colors+vt_colors+mt_colors+bg_colors
#region_color_dict=dict(zip(all_regions,all_regions_colors))


all_regions_dict={'pfc':'PFC',
 'cingulate':'Cingulate',
 'motor':'Motor',
 'somato':'Somato',
 'temporal':'Temporal',
 'insula':'Insula',
 'hippocampus':'Hippocampus',
 'parietal':'Parietal',
 'v1':'V1',
 'cge':'CGE',
 'cge_and_lge':'CGE+LGE',
 'lge':'LGE',
 'mge_and_cge_and_lge':'MGE+CGE+LGE',
 'mge':'MGE',
 'septum':'Septum',
 'ge':'dLGE',
 'xLGE':'dLGE',
 'pre-optic':'POA',
 'hypothalamusandpoa':'POH+POA',
 'septumandnucleusaccumbens':'Septum+NAc',
 'putamen_and_septum':'Septum+Putamen',
 'str':'Striatum',
 'putamenandclaustrumandbasalganglia':'Striatum+Claustrum',
 'amy':'Amygdala'}
all_regions=[all_regions_dict[x] for x in all_regions]
cortex_order=[all_regions_dict[x] for x in cortex_order]
region_color_dict=dict(zip(all_regions,all_regions_colors))

adata.obs['region']=adata.obs['region'].replace(all_regions_dict)
adata.obs.region=adata.obs.region.astype('category')
print(all_regions)
print(adata.obs.region.cat.categories)
adata.obs.region.cat.reorder_categories(all_regions,inplace=True,ordered=True)
adata.uns['region_colors']=all_regions_colors

adata.var_names_make_unique()
ribo_genes=[name for name in adata.var_names if name.startswith('RPS') or name.startswith('RPL') ]
adata.obs['percent_ribo'] = np.sum(
adata[:, ribo_genes].X, axis=1) / np.sum(adata.X, axis=1)
mito_genes = [name for name in adata.var_names if name in ['ND1','ND2','ND4L','ND4','ND5','ND6','ATP6','ATP8','CYTB','COX1','COX2','COX3'] or 'chrM-' in name or 'MT-' in name]
adata.obs['percent_mito'] = np.sum(
adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
print(adata,flush=True)
try:
    #sc.pl.violin(adata,groupby='batch_name',keys=['percent_ribo','percent_mito'],rotation=45,save='ribomito')
    print('violin bad')
except:
    print('noviolin')

adata.write(re.sub('\.h5ad','Clean.h5ad',newfile)) 

#adata=adata[adata.obs['percent_ribo']<.5,:]
#adata=adata[adata.obs['percent_ribo']<.2,:]
adata._inplace_subset_obs(adata.obs['percent_ribo']<.4)
adata._inplace_subset_obs(adata.obs['percent_mito']<.15)
adata._inplace_subset_var(~adata.var.index.isin(mito_genes))
print(adata)
sc.pp.filter_genes(adata,min_cells=20)
#sc.pp.filter_cells(adata,min_counts=1000)
sc.pp.filter_cells(adata,min_genes=min_genes)
print(adata)
sufficient_cells=adata.obs['batch_name'].value_counts().index[adata.obs['batch_name'].value_counts()>50]
adata=adata[adata.obs['batch_name'].isin(sufficient_cells),:]
adata.obs['batch_name'].cat.remove_unused_categories(inplace=True)


sc.pp.normalize_total(adata,exclude_highly_expressed=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata,n_top_genes=12000,batch_key='batch_name',subset=False)
sc.pp.scale(adata,max_value=10)
sc.pp.pca(adata,n_comps=100)
#sc.pp.neighbors(adata)
bbknn.bbknn(adata,batch_key='batch_name',n_pcs=100,neighbors_within_batch=3)
sc.tl.leiden(adata,resolution=5)
sc.tl.umap(adata,spread=4,min_dist=.2)

#Proper supervised_name colors
adata.obs['supervised_name'].cat.remove_unused_categories(inplace=True)
my_pal=[paldict[x] for x in adata.obs['supervised_name'].cat.categories]

sc.pl.umap(adata,color=['leiden'],save='leiden')
sc.pl.umap(adata,color=['batch_name'],save='batch_name')
sc.pl.umap(adata,color=['region'],save='region')
sc.pl.umap(adata,color=['timepoint'],save='timepoint',color_map=matplotlib.cm.RdYlBu_r)
sc.pl.umap(adata,color=['supervised_name'],legend_loc='on data',legend_fontsize=4,palette=my_pal,save='supervised_name_onplot')
sc.pl.umap(adata,color=['supervised_name'],save='supervised_name')
sc.pl.umap(adata,color=['leiden'],legend_loc='on data',legend_fontsize=4,save='leiden_onplot')
sc.pl.umap(adata, color=['general_region'],save='general_region')

adata.obs.loc[:,['full_cellname','supervised_name','leiden']].to_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MacaqueGEsupervisednamesHippo4.txt',index=None)

INgenes=['MKI67','SOX2','HOPX','DCX','FOXG1','GSX2','ERBB4','CALB1','CALB2','RSPO3','RELN','DLX2','DLX5','GAD1','GAD2','GADD45G','LHX6','LHX7','LHX8','SST','NKX2-1','MAF','CHODL','SP8','NR2F2','PROX1','VIP','CCK','NPY','PAX6','ETV1','SCGN','ROBO2','CASZ1','TSHZ1','OPRM1','FOXP1','FOXP2','FOXP4','VAX1','TH','DDC','SLC18A2','MEIS2','LAMP5','TLE4','ISL1','DRD1','PENK','ADORA2A','TAC1','TAC3','TACR1','TACR2','TACR3','CRABP1','ANGPT2','RBP4','STXBP6','ZIC1','ZIC4','EBF1','SATB2']
INgenes=[x for x in INgenes if x in adata.var.index]
sc.pl.umap(adata,color=INgenes,use_raw=False,save='INGenes')

INgenes=['MKI67','DCX','DLX2','ZIC1','SP8','SCGN','CCK','VIP','MEIS2','PAX6','FOXP2','TSHZ1','FOXP1','OPRM1','ISL1','PENK','LHX6','SST','NPY','CHODL','CRABP1','ANGPT2','MAF','TAC3']         
INgenes=[x for x in INgenes if x in adata.var.index]
sc.pl.umap(adata,color=INgenes,use_raw=False,save='SupplementGenes')


INgenes=['MEIS2','SCGN','FLRT2','PCDH9','KAT6B','ROBO2','TOX3','SLC10A4']
INgenes=[x for x in INgenes if x in adata.var.index]
sc.pl.umap(adata,color=INgenes,use_raw=False,save='LGESupplementGenes')

easygenes= ['MKI67','SOX2','AQP4','EDNRB','IL33','PDGFRA','HOPX','ERBB4','CALB1','CALB2','GAD1','GAD2','GADD45G','LHX6','LHX7','SST','NKX2-1','MAF','SP8','SP9','PROX1','VIP','CCK','NPY','LAMP5','HTR3A','NR2F1','NR2F2','TOX3','ETV1','SCGN','FOXG1','FOXP1','FOXP2','FOXP4','TH','DDC','SLC18A2','PAX6','MEIS2','SHTN1','ISL1','PENK','DRD1','ADORA2A','CHAT','CRABP1','ZIC1','EBF1','DLX2','DLX5','GSX2','RBFOX3','PPP1R17','EOMES','DCX','TUBB3','BCL11B','TLE4','FEZF2','SATB2','TBR1','AIF1','RELN','PECAM1','HBZ','ROBO1','ROBO2','ROBO3','ROBO4']
easygenes=[x for x in easygenes if x in adata.var.index]
sc.pl.umap(adata,color=easygenes,use_raw=False,save='FaveGenes')    
    
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
df_plot = adata[adata.obs['region'].isin(cortex_order),:].obs.groupby(['supervised_name', 'timepoint']).size().reset_index().pivot(columns='timepoint', index='supervised_name', values=0)#.apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVcorticaltimepointBar.pdf'), bbox_inches="tight")

f = plt.figure()
df_plot = adata.obs.groupby(['supervised_name', 'region']).size().reset_index().pivot(columns='region', index='supervised_name', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVregionBar.pdf'), bbox_inches="tight")

f = plt.figure()
df_plot = adata.obs.groupby(['timepoint', 'region']).size().reset_index().pivot(columns='region', index='timepoint', values=0)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'timepointVregionBar.pdf'), bbox_inches="tight")

f = plt.figure()
df_plot = adata.obs
these_regions=adata.obs.region.cat.categories
these_colors=[c for x,c in zip(all_regions,all_regions_colors) if x in df_plot['region'].unique()]
df_plot['region'].cat.reorder_categories(these_regions,ordered=True,inplace=True)
df_plot=df_plot.groupby(['timepoint', 'region']).size().reset_index().pivot(columns='region', index='timepoint', values=0)#.apply(lambda g: g / g.sum(),1)
ax = (df_plot).plot(kind='bar', legend=False,stacked=True,color=these_colors)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('number of cells')
plt.title('Distribution of regions sampled at each timepoint')
ax.grid(False)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'timepointVregionBar.pdf'), bbox_inches="tight")



f = plt.figure()
df_plot = adata[adata.obs['region'].isin(cortex_order),:].obs.groupby(['supervised_name', 'region']).size().reset_index().pivot(columns='region', index='supervised_name', values=0)#.apply(lambda g: g / g.sum(),1)
df_plot=df_plot.loc[df_plot.index.str.contains('LHX6|LGE_FOXP2|SCGN|PROX'),:]
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVcorticalregionBar.pdf'), bbox_inches="tight")

f = plt.figure()
df_plot = adata[adata.obs['region'].isin(cortex_order),:].obs
these_regions=[x for x in cortex_order if x in df_plot['region'].unique()]
these_colors=[c for x,c in zip(all_regions,all_regions_colors) if x in df_plot['region'].unique()]
df_plot['region'].cat.reorder_categories(these_regions,ordered=True,inplace=True)
regiontotals=df_plot.loc[df_plot['region'].isin(cortex_order),:].groupby([ 'region']).size()
df_plot=df_plot.groupby(['noctx_supervised_name', 'region']).size().reset_index().pivot(columns='region', index='noctx_supervised_name', values=0)#.apply(lambda g: g / g.sum(),1)
df_plot=df_plot.loc[df_plot.index.str.contains('LHX6|LGE_FOXP2|SCGN|PAX6|PROX'),:]
df_plot=df_plot.loc[df_plot.sum(1)>20,:]
ax = (df_plot/regiontotals).plot(kind='bar', legend=False,stacked=False,color=these_colors)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion of cells in region')
plt.title('Distribution of IN Classes across Cortical Regions')
ax.grid(False)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVcorticalregionBarNorm.pdf'), bbox_inches="tight")

sc.tl.dendrogram(adata,groupby='noctx_supervised_name')
df_plot = adata.obs.groupby(['noctx_supervised_name', 'region']).size().reset_index().pivot(columns='region', index='noctx_supervised_name', values=0).apply(lambda g: g / g.sum(),1)
df_plot=df_plot.loc[adata.uns["dendrogram_noctx_supervised_name"]['categories_ordered'],:]
these_colors=[region_color_dict[x] for x in df_plot.columns]
ax = df_plot.plot(kind='bar', legend=False,stacked=True,color=these_colors)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion of cells in region')
ax.grid(False)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedNameRegionsStackBar.pdf'), bbox_inches="tight")

df_plot = adata[adata.obs['region'].isin(cortex_order),:].obs.groupby(['supervised_name', 'timepoint']).size().reset_index().pivot(columns='timepoint', index='supervised_name', values=0)#.apply(lambda g: g / g.sum(),1)
df_plot=df_plot.loc[df_plot.sum(1)>50,:]
regiontotals=adata[adata.obs['region'].isin(cortex_order),:].obs.groupby([ 'timepoint']).size()
ax = (df_plot/regiontotals).plot(kind='bar', legend=False,stacked=False)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion of cells in timepoint')
ax.grid(False)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'supervisedVcorticalTpBarNorm.pdf'), bbox_inches="tight")

df_plot = adata.obs.groupby(['region', 'noctx_supervised_name']).size().reset_index().pivot(columns='noctx_supervised_name', index='region', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion of cells in region')
ax.grid(False)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'regionMakeupSupervisedNameStackBar.pdf'), bbox_inches="tight")

sc.pl.umap(adata,color=['noctx_supervised_name'],save='noctx_supervised_name',palette=my_pal)
sc.pl.umap(adata,color=['noctx_supervised_name'],legend_loc='on data',legend_fontsize=4,save='noctx_supervised_name_onplot')

super_color_dict=dict(zip(adata.obs['noctx_supervised_name'].cat.categories,adata.uns['noctx_supervised_name_colors']))
for gr in adata.obs['general_region'].unique():
    df_plot = adata.obs.loc[adata.obs.general_region==gr,:]
    df_plot['region'].cat.remove_unused_categories(inplace=True)
    df_plot['noctx_supervised_name'].cat.remove_unused_categories(inplace=True)
    these_regions=[x for x in all_regions if x in df_plot['region'].cat.categories]
    these_celltypes=[x for x in super_color_dict.keys() if x in df_plot['noctx_supervised_name'].cat.categories]
    these_colors=[super_color_dict[x] for x in these_celltypes]
    df_plot['region'].cat.reorder_categories(these_regions,ordered=True,inplace=True)
    df_plot['noctx_supervised_name'].cat.reorder_categories(these_celltypes,ordered=True,inplace=True)
    regiontotals=df_plot.groupby(['region']).size()
    df_plot=df_plot.groupby(['noctx_supervised_name', 'region']).size().reset_index().pivot(columns='region', index='noctx_supervised_name', values=0)#.apply(lambda g: g / g.sum(),1)
    #df_plot=df_plot.loc[df_plot.sum(1)>50,:]
    df_plot=(df_plot/regiontotals).T
    ax = df_plot.plot(kind='bar', legend=False,stacked=False,color=these_colors,width=.7)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.ylabel('proportion of cells in region')
    plt.title(gr)
    ax.grid(False)
    ax.get_figure().savefig(os.path.join(sc.settings.figdir,gr+'regionMakeupSupervisedNameBarNorm.pdf'), bbox_inches="tight")

    
df_plot = adata.obs.groupby(['supervised_name', 'region']).size().reset_index().pivot(columns='region', index='supervised_name', values=0)
print(df_plot)

adata.obs['general_region']=sc_utils.macaque_general_regions(adata.obs['region'])
sc.pl.umap(adata, color=['general_region'],save='general_region')

df_plot = adata.obs.groupby(['general_region', 'noctx_supervised_name']).size().reset_index().pivot(columns='noctx_supervised_name', index='general_region', values=0).apply(lambda g: g / g.sum(),1)
ax = df_plot.plot(kind='bar', legend=False,stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion of cells in region')
ax.grid(False)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'generalregionMakeupSupervisedNameStackedBar.pdf'), bbox_inches="tight")

sc.pl.umap(adata,color=['noctx_supervised_name'],save='noctx_supervised_name',palette=my_pal)
hierarchy_key='noctx_supervised_name'
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
sc.pl.matrixplot(adata,groupby=hierarchy_key,var_names=var_dict,save='top_degenes',cmap='RdBu_r',use_raw=False,dendrogram=True)

sc.pl.dotplot(adata[adata.obs.noctx_supervised_name.str.contains('^CGE|FOXP2|FOXP1|PAX6'),:],var_names=['NR2F2','PROX1','SP8','SCGN','PAX6','TH','ETV1','TSHZ1','CASZ1','FOXP2','FOXP4','MEIS2','FOXP1'],groupby='noctx_supervised_name',
                     cmap='coolwarm', use_raw=False,dendrogram=False,
                    var_group_labels=['CGE','CGE/dLGE','dLGE','d/iLGE','iLGE'],var_group_positions=[(0,1),(2,4),(5,6),(7,9),(10,10)],save='MIPdots',standard_scale='var')#var_group_labels=['CGE','dLGE','iLGE'],var_group_positions=[(0,3),(2,8),(6,9)]

sc.pl.tracksplot(adata[adata.obs.noctx_supervised_name.str.contains('^CGE|FOXP2|FOXP1|PAX6'),:],var_names=['NR2F2','PROX1','SP8','SCGN','PAX6','TH','ETV1','FOXP2','FOXP4','MEIS2','PCP4','FOXP1'],groupby='noctx_supervised_name',
                     cmap='coolwarm', use_raw=False,dendrogram=False,
                    var_group_labels=['CGE','CGE/dLGE','dLGE','d/iLGE','iLGE'],var_group_positions=[(0,1),(2,4),(5,6),(7,10),(11,11)],standard_scale=None,save='MIPtracks')#var_group_labels=['CGE','dLGE','iLGE'],var_group_positions=[(0,3),(2,8),(6,9)]

sc_utils.cell_cycle_score(adata)
sc_utils.marker_analysis(adata,variables=['leiden'],markerpath=os.path.expanduser('~/markers.txt')) 
adata.write(re.sub('\.h5ad','Processed.h5ad',newfile))

sc_utils.log_reg_diff_exp(adata)
#sc_utils.log_reg_diff_exp(adata,obs_name='supervised_name')
sc_utils.log_reg_diff_exp(adata,obs_name='noctx_supervised_name')

'''
#Go to part 3
adata.uns['iroot'] = np.flatnonzero(adata.obs.index==sc_utils.get_median_cell(adata,'supervised_name','Inhibitory IPC'))[0]
sc.tl.diffmap(adata,n_comps=15)
sc.tl.dpt(adata,n_branchings=0,n_dcs=15)
#sc.pl.diffmap(adata,components='all',save='alldiffmap')

sc.tl.paga(adata,groups='supervised_name')
sc.pl.paga(adata,layout='eq_tree',save='normal_tree',fontsize=4,edge_width_scale=.4)

adata.write(re.sub('\.h5ad','Processed.h5ad',newfile))
easygenes= ['MKI67','HMGB2','DCX','ERBB4','GAD2','GADD45G','LHX6','SST','NKX2-1','MAF','SP8','SP9','PROX1','VIP','CCK','LAMP5','NR2F1','NR2F2','TOX3','ETV1','SCGN','FOXP1','FOXP2','FOXP4','TAC1','TAC3','PCP4','UNC79','TH','PAX6','MEIS2','ISL1','PENK','CRABP1','ZIC1','DLX5','GSX2','RELN','ROBO1','ROBO2']
easygenes=[x for x in easygenes if x in adata.var.index]
mat=pd.DataFrame(adata[adata.obs['phase'].isin(['S','G2M']),easygenes].X,index=adata[adata.obs['phase'].isin(['S','G2M']),:].obs.index,columns=easygenes)
seaborn.set(font_scale=.6)
seaborn.clustermap(mat.corr(),yticklabels=True)
plt.savefig(os.path.join(sc.settings.figdir,'DividingCellCorrelationsClustermap.pdf'), bbox_inches="tight")

neurongroup=[x for x in adata.obs['supervised_name'].unique() if (('euron' in x) or ('IPC' in x))]
easygenes= ['ERBB4','GAD1','GAD2','GADD45G','LHX6','SST','NKX2-1','MAF','SP8','SP9','PROX1','VIP','CCK','LAMP5','NR2F1','NR2F2','TOX3','ETV1','SCGN','FOXP1','FOXP2','FOXP4','TAC1','TAC3','TH','PAX6','MEIS2','ISL1','PENK','CRABP1','ZIC1','ZIC2','DLX2','DLX5','GSX2','RELN','ROBO1','ROBO2','CALB1','CALB2']
easygenes=[x for x in easygenes if x in adata.var.index]
mat=pd.DataFrame(adata[adata.obs['supervised_name'].isin(neurongroup),easygenes].X,index=adata[adata.obs['supervised_name'].isin(neurongroup),:].obs.index,columns=easygenes)
seaborn.set(font_scale=.6)
seaborn.clustermap(mat.corr(),yticklabels=True)
plt.savefig(os.path.join(sc.settings.figdir,'InterneuronCorrelationsClustermap.pdf'), bbox_inches="tight")

sc.pl.dendrogram(adata,'supervised_name',save='super_dendro')
sc.pl.dendrogram(adata,'leiden',save='leiden_dendro')

s_genes=['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8']
g2m_genes=['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']
s_genes=[x for x in s_genes if x in adata.var.index]
g2m_genes=[x for x in g2m_genes if x in adata.var.index]

gdata=adata.copy().T
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
cc_mods=vcs.index[vcs>5]
cc_genes=gdata.obs.index[gdata.obs.leiden.isin(cc_mods)]
adata.var['leiden']=gdata.obs['leiden']
del gdata

ncells=50
mgenes=(adata.layers['unspliced']>0).sum(0) > ncells
adata.var['velocity_genes']=mgenes.A1&adata.var['highly_variable_velo']
adata=adata[:,mgenes.A1]
adata=adata[:,adata.var['highly_variable']]
scv.pp.normalize_per_cell(adata)
print(adata)
print(adata.layers['spliced'].shape)
#scv.pp.filter_genes_dispersion(adata)#,n_top_genes=min(6000,adata.layers['spliced'].shape[1]))
scv.pp.log1p(adata)
scv.pp.remove_duplicate_cells(adata)
scv.pp.moments(adata, n_neighbors=20)
print(adata,flush=True)
scv.tl.recover_dynamics(adata,var_names='velocity_genes',use_raw=False)
print('Recovered', flush=True)
scv.tl.velocity(adata,mode='dynamical',filter_genes=False)
print('Velocity done',flush=True)

adata.layers['linear_velocity']=adata.layers['velocity'].copy()
adata.layers['linear_velocity'][:,adata.var.index.isin(cc_genes)]=0
adata.layers['cc_velocity']=adata.layers['velocity'].copy()
adata.layers['cc_velocity'][:,~adata.var.index.isin(cc_genes)]=0

scv.tl.velocity_graph(adata,mode_neighbors='connectivities',vkey='cc_velocity',approx=False)
scv.tl.transition_matrix(adata,vkey='cc_velocity')
scv.tl.terminal_states(adata,vkey='cc_velocity')
scv.tl.recover_latent_time(adata,vkey='cc_velocity')
scv.tl.velocity_confidence(adata,vkey='cc_velocity')
scv.tl.velocity_embedding(adata, basis='umap',vkey='cc_velocity')
scv.pl.velocity_embedding_grid(adata, basis='umap',vkey='cc_velocity',save='_cc_grid')
try:
    sc.pl.umap(adata,color=['cc_velocity_length', 'cc_velocity_confidence','latent_time','root_cells','end_points','cc_velocity_pseudotime'],save='cc_velocity_stats')
except:
    print('fail cc')

scv.tl.velocity_graph(adata,mode_neighbors='connectivities',vkey='linear_velocity',approx=False)
scv.tl.transition_matrix(adata,vkey='linear_velocity')
scv.tl.terminal_states(adata,vkey='linear_velocity')
scv.tl.recover_latent_time(adata,vkey='linear_velocity')
scv.tl.velocity_embedding(adata, basis='umap',vkey='linear_velocity')
scv.tl.velocity_confidence(adata,vkey='linear_velocity')
scv.pl.velocity_embedding_grid(adata,vkey='linear_velocity',basis='umap',color='latent_time',save='_linear_grid')
try:
    sc.pl.umap(adata,color=['linear_velocity_length', 'linear_velocity_confidence','latent_time','root_cells','end_points','linear_velocity_pseudotime'],save='linear_velocity_stats')
except:
    'fail linear'

scv.utils.cleanup(adata)
adata.write(re.sub('Velocity','VelocityDYNAMICAL',newfile))
sc.tl.paga(adata,groups='supervised_name')
sc.tl.paga(adata,use_rna_velocity=True,groups='supervised_name')
sc.pl.paga_compare(adata,legend_fontsize=4,arrowsize=10,edge_width_scale=.4,threshold=np.quantile(adata.uns['paga']['connectivities'].data,.9))
sc.pl.paga_compare(adata,legend_fontsize=4,arrowsize=10,edge_width_scale=.4,threshold=np.quantile(adata.uns['paga']['connectivities'].data,.9),save='connectivity')
sc.pl.paga_compare(adata,solid_edges='connectivities',transitions='transitions_confidence',legend_fontsize=4,arrowsize=10,threshold=np.quantile(adata.uns['paga']['transitions_confidence'].data,.9),save='DYNAMICALvelocity')
adata.write(re.sub('Velocity','VelocityDYNAMICAL',newfile))




paths = [('mysterycells', ['Radial Glia','S-phase Dividing Cell','G2/M Dividing Cell','FOXP2+/ETV1+ Interneuron','Cortical FOXP2+/ETV1+ Interneuron','Cortical FOXP2/4+ Interneuron']),
         ('mysterycellsinsula', ['Radial Glia','S-phase Dividing Cell','G2/M Dividing Cell','MEIS2+ Insula Interneuron','FOXP2+/ETV1+ Interneuron','Cortical FOXP2+/ETV1+ Interneuron','Cortical FOXP2/4+ Interneuron']),
         ('cge', ['Radial Glia','S-phase Dividing Cell','G2/M Dividing Cell','CGE IPC','Maturing CGE Neuron']),
         ('mge', ['Radial Glia','S-phase Dividing Cell','G2/M Dividing Cell','MGE IPC','Maturing MGE Neuron']),
         ('CRABP1', ['Radial Glia','S-phase Dividing Cell','G2/M Dividing Cell','MGE IPC','CRABP1+ MGE Interneuron'])]

gene_names = ['HIST1H4C','HMGB2','GADD45G','GAD2','SP8','PAX6','ETV1','FOXP2','FOXP4','SCGN','TH','NR2F2','VIP','CCK']
gene_names = [x for x in gene_names if x in adata.var.index]
for ipath, (descr, path) in enumerate(paths):
    fig,ax=plt.subplots()
    sc.pl.paga_path(
        adata, path, gene_names,use_raw=False,
        show_node_names=True,
        ax=ax,
        ytick_fontsize=7,
        title_fontsize=7,
        left_margin=0.15,
        n_avg=40,
        annotations=['dpt_pseudotime'],#,'velocity_pseudotime'],
        groups_key='supervised_name',
        color_maps_annotations={'dpt_pseudotime': 'viridis'},
        show_yticks=True if ipath==0 else False,
#        show_colorbar=False,
#        color_map='Greys',
        title='{} path'.format(descr),
        show=False, return_data=True,
        save=descr+'paga_path_interneurons.pdf')
    plt.savefig(os.path.join(sc.settings.figdir,descr+'dpt_dynammical_pagapath_savefig.pdf'))
    plt.close(fig)
    plt.clf()


gene_names = ['HIST1H4C','HMGB2','GADD45G','GAD2','SP8','PAX6','ETV1','FOXP2','FOXP4','SCGN','TH','NR2F2','VIP','CCK']
gene_names = [x for x in gene_names if x in adata.var.index]
for ipath, (descr, path) in enumerate(paths):
    fig,ax=plt.subplots()
    sc.pl.paga_path(
        adata, path, gene_names,use_raw=False,
        show_node_names=True,
        ax=ax,
        ytick_fontsize=7,
        title_fontsize=7,
        left_margin=0.15,
        n_avg=40,
        annotations=['velocity_pseudotime'],#,'velocity_pseudotime'],
        groups_key='supervised_name',
        color_maps_annotations={'velocity_pseudotime': 'viridis'},
        show_yticks=True if ipath==0 else False,
#        show_colorbar=False,
#        color_map='Greys',
        title='{} path'.format(descr),
        show=False, return_data=True,
        save=descr+'paga_path_interneurons.pdf')
    plt.savefig(os.path.join(sc.settings.figdir,descr+'velo_pagapath_savefig.pdf'))
    plt.close(fig)
    plt.clf()

    
'''
