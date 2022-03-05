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

cflare=True
if cflare:
    adata=sc.read('/wynton/home/ye/mschmitz1/CFLAREFalsemouseAdult2.h5ad')
    model='CFLARE'
else:
    adata=sc.read('/wynton/home/ye/mschmitz1/GPCCAFalsemouseAdult2.h5ad')
    model='GPCCA'

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
adata=adata[~adata.obs['agg_supervised_name'].isnull(),:]
print(adata)

probDF=pd.DataFrame(adata.obsm['to_terminal_states'],columns=adata.uns['to_terminal_states_names'])
probDF['supervised_name']=list(adata.obs['agg_supervised_name'])
probDF.index=list(adata.obs.index)
adata.obs['Adult']=adata.obs.agg_supervised_name.isin(adata.uns['to_terminal_states_names']).fillna(value=False)
probDF.loc[list(adata.obs['Adult']),:]=np.nan
adata.obs['Adult']=adata.obs['Adult'].astype(str)
probDF=probDF.loc[~probDF['supervised_name'].isnull(),:]
print(probDF)

meanDF=probDF.fillna(np.inf).groupby(['supervised_name']).mean().replace(np.inf, 0)
meanDF=meanDF/meanDF.sum(0)
meanDF=meanDF.loc[meanDF.index.str.contains('VMF|LGE|MGE|CGE'),:]
seaborn.clustermap(meanDF)
plt.savefig(os.path.join(sc.settings.figdir,model+'_MeanAbsorbtion.pdf'), bbox_inches="tight")

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
adata.obs['agg_supervised_name']=adata.obs['agg_supervised_name'].astype('category')
adata.obs['agg_supervised_name'].cat.remove_unused_categories(inplace=True)
catnames=list(adata.obs['agg_supervised_name'].cat.categories)

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
    if 'RMTW' in t:
        destroy=probDF.index[probDF['supervised_name']==t]
    probDF=probDF.loc[~probDF.index.isin(destroy),:]
    

print(probDF)
probDF=probDF.loc[probDF['supervised_name'].str.contains('MGE|CGE|VMF|LGE|RMTW'),:]

print(probDF)
#adata.obs.drop('terminal_states_probs',axis=1,inplace=True)
topinds={}
for c in probDF.columns:
    topinds[c]=list(probDF.loc[probDF.sort_values(c,ascending = False).index[:100],'supervised_name'])
adultsankeydf=pd.DataFrame(topinds).apply(lambda x: x.value_counts(normalize=True)).stack().reset_index()  
adultsankeydf.columns=['agg_supervised_name','predicted_end',0]
adultsankeydf.loc[adultsankeydf.shape[0],:]=['RMTW_ZIC1/RELN','Ctx_PROX1/LAMP5',0.001]
del adata

##########Backwards mean absorbtion normalized
"""
probDF['supervised_name']=list(adata.obs['supervised_name'])
meanDF=probDF.fillna(np.inf).groupby(['supervised_name']).mean().replace(np.inf, 0)
meanDF=meanDF/meanDF.sum(0)
#meanDF=meanDF.loc[meanDF.index.str.contains('VMF|LGE|MGE|CGE|RMTW'),:]
adultsankeydf=meanDF.stack().reset_index()
adultsankeydf.columns=['agg_supervised_name','predicted_end',0]
"""

#adultsankeydf=adultsankeydf.loc[adultsankeydf['agg_supervised_name'].str.contains('MGE|CGE|VMF|LGE|RMTW'),:]
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
my_pal=[paldict[x] for x in catnames]


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
pio.write_image(fig, os.path.join(sc.settings.figdir,model+'_Sankey.pdf'),width=1200,height=1200)
