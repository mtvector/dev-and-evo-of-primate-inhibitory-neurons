
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
import cellrank as cr
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
import matplotlib.font_manager
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Nimbus Sans','Arial']})
matplotlib.rc('text', usetex=False)


newfile='/wynton/group/ye/mtschmitz/macaquedevbrain/CAT202002_h5ad/KDCbVelocityMacaqueGeAllcortexHippocampus.h5ad'
adata=sc.read(re.sub('Velocity','VelocityDYNAMICAL',newfile))
"""
lineage_genes= ['LHX6','NKX2-1','MAF','CRABP1','TAC3','PROX1','NR2F1','NR2F2','SP8','ETV1','PAX6','TSHZ1','FOXP1','FOXP2','CASZ1','MEIS2','EBF1','ISL1','PENK','ZIC1','ZIC4','HMX3','NPY','CORT','CHODL']
lineage_genes=[x for x in lineage_genes if x in adata.var.index]

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

adata=adata[adata.obs['latent_time'].argsort(),:]
d={}

for branch in ['phase|G2-M','MGE|LGE|CGE|VMF|Transition']:#'MGE','CGE','LGE',
    values=[]
    times=[]
    bdata=adata[adata.obs['noctx_supervised_name'].str.contains(branch),:].copy()
    bdata.obs_names_make_unique()
    def chunks(lst, n):
        #Yield successive n-sized chunks from lst.
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    n=1000       

    for x in tqdm.tqdm(chunks(bdata.obs.index,n)):
        if len(x)>20:
            times.append(bdata[x,:].obs['latent_time'].mean())
            mat=pd.DataFrame(bdata[x,lineage_genes].X,index=x,columns=lineage_genes)
            values.append(mat.corr())

    values=np.array(values)
    times=np.array(times)

    df1=pd.DataFrame(bdata[:,lineage_genes].X)
    cors=df1.corr() 
    bootstraps=[]
    for q in tqdm.tqdm(range(n_bootstraps)):
        #bootstraps.append(df1.loc[np.random.choice(df1.index,size=df1.shape[0],replace=False),np.random.choice(df1.columns,size=df1.shape[1],replace=False)].corr().to_numpy())
        df2=pd.DataFrame(adata[:,np.random.choice(adata.var.index[adata.var.highly_variable],size=len(lineage_genes),replace=False)].X)
        bootstraps.append(df2.corr().to_numpy())
    bootstraps=np.array(bootstraps)
    baseline_cor_pval=((np.abs(cors.to_numpy())>np.abs(bootstraps)).sum(0)+1)/(n_bootstraps+1)
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
    d[branch]=[coefs,intercepts,pvals,cors,baseline_cor_pval]    

    
titles=['regress_coef','intercept','regress_pval','corr_coef','corr_pval']
for i in range(5):
    print(i)
    cormat=np.triu(d['phase|G2-M'][i])+np.tril(d['MGE|LGE|CGE|VMF|Transition'][i])
    np.fill_diagonal(cormat,0)
    cdf=pd.DataFrame(cormat,index=lineage_genes,columns=lineage_genes)
    if (i == 2) | (i == 4):
        cdf=-np.log10(cdf)
    plt.figure(figsize=(14, 10))
    seaborn.heatmap(cdf,xticklabels=True,yticklabels=True,cmap='coolwarm')
    plt.show()
    plt.savefig(os.path.join(sc.settings.figdir, 'ProgensOverNeurons_'+titles[i]+'_CorrMat.pdf'), bbox_inches="tight")



adata
"""
# In[4]:


supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MacaqueGEsupervisednamesHippo3Preserve.txt')
pd.DataFrame(supercell['supervised_name'].unique())#.to_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/names.txt')


# In[32]:


adata.obs['supervised_name']=adata.obs['supervised_name'].astype(str)
adata.obs['noctx_supervised_name']=adata.obs['noctx_supervised_name'].astype(str)

supercell=pd.read_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MacaqueGEsupervisednamesHippo3PreserveCompoundname.txt')
ind=adata.obs.index[adata.obs.index.isin(supercell['full_cellname'])]
supercell.index=supercell['full_cellname']
supercell=supercell.loc[ind,:]
adata.obs.loc[ind,'supervised_name']=supercell['supervised_name']
adata.obs.loc[ind,'leiden_old']=supercell['leiden'].astype(str)
adata.obs['noctx_supervised_name']=adata.obs['supervised_name']

# In[44]:


#adata=adata[np.random.choice(adata.obs.index,size=15000,replace=False),:]




# In[12]:


sc.pl.umap(adata,color=['latent_time'], color_map=matplotlib.cm.RdYlBu_r,save='latent_time')


# In[19]:


cortex_order=['pfc','cingulate','motor','somato','temporal','insula','hippocampus','parietal','v1']
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
adata.obs['region']=adata.obs['region'].astype('category')
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


# In[22]:


sc.pl.umap(adata,color=['supervised_name'],legend_loc='on data',save='supervised_name_onplot')
sc.pl.umap(adata,color=['supervised_name'],save='supervised_name')
sc.pl.umap(adata,color=['region'],save='region')


# In[28]:

sc.pl.umap(adata,color=['leiden_old'],legend_loc='on data',save='old_leiden')
sc.set_figure_params(scanpy=True, fontsize=6)
_, ax = plt.subplots(figsize=(10, 5))
sc.pl.violin(adata, ['n_genes'],groupby='batch_name',ax=ax,
             jitter=0.4, multi_panel=True,rotation=90,stripplot=False,save='QC',font_scale=0.7)
print('QC')
print(adata)

hierarchy_key='leiden_old'
sufficient_cells=adata.obs[hierarchy_key].value_counts().index[adata.obs[hierarchy_key].value_counts()>20]
adata=adata[adata.obs[hierarchy_key].isin(sufficient_cells),:]
adata.obs[hierarchy_key].cat.remove_unused_categories(inplace=True)
adata=adata[:,np.isfinite(adata.X.sum(0))]
rgs=sc.tl.rank_genes_groups(adata,groupby=hierarchy_key,method='logreg',use_raw=False,copy=True).uns['rank_genes_groups']#,penalty='elasticnet',solver='saga')#or penalty='l1'
result=rgs
groups = result['names'].dtype.names
df=pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'scores']})
df.to_csv(os.path.join(sc.settings.figdir,"LogReg"+hierarchy_key+"Norm.csv"))
topgenes=df.iloc[0:3,['_n' in x for x in df.columns]].T.values
cols=df.columns[['_n' in x for x in df.columns]]
cols=[re.sub('_n','',x) for x in cols]
topdict=dict(zip(cols,topgenes))
sc.tl.dendrogram(adata,groupby=hierarchy_key)
var_dict=dict(zip(adata.uns["dendrogram_"+hierarchy_key]['categories_ordered'],[topdict[x] for x in adata.uns["dendrogram_"+hierarchy_key]['categories_ordered']]))
sc.pl.matrixplot(adata,groupby=hierarchy_key,var_names=var_dict,save='top_degenes',cmap='RdBu_r',use_raw=False,dendrogram=True)


sc.pl.umap(adata,color=['leiden'],legend_loc='on data',save='leiden_for_supervised')


# In[23]:


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


# In[24]:


ckvk=ck+vk
ckvk.compute_transition_matrix()


# In[26]:


d={}
for c in adata.obs.noctx_supervised_name.unique():
    if not any([x in c.lower() for x in ['g2-m','phase','trans','g1']]):
        print(c)
        select_cells=adata.obs.noctx_supervised_name.isin([c])
        lt_quantile=np.quantile(adata.obs.loc[select_cells,'latent_time'],.2)
        select_cells=select_cells&(adata.obs.loc[:,'latent_time']>lt_quantile)
        #lt_quantile=np.quantile(adata.obs.loc[select_cells,'latent_time'],.2)
        #select_cells=select_cells&(adata.obs.loc[:,'latent_time']<lt_quantile)
        d[c]=[adata.obs.index[i] for i in range(adata.shape[0]) if select_cells[i]]


# In[39]:


g=cr.tl.estimators.CFLARE(ckvk)
#g = cr.tl.estimators.GPCCA(ck)
g.compute_eigendecomposition()
#g.compute_schur(n_components=15)

g.set_terminal_states(d,cluster_key='supervised_name')


# In[40]:


g.plot_terminal_states()


# In[43]:


g.compute_absorption_probabilities(show_progress_bar=True)


# In[ ]:


g.plot_absorption_probabilities()


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


# In[ ]:


#for x in ['noctx_supervised_name','timepoint','region','latent_time']:
#    probDF[x]=adata.obs[x]


# In[ ]:


#renormalize so trimmed gaussian is standardized
#This is necessary because probabilities are affected by end cluster size
#Use dumb max since this will be the most likely call
trim=.05
trimmedProbDF=probDF[(probDF<1-trim) & (probDF>trim)]
probDF=trimmedProbDF
#probDF=(probDF-trimmedProbDF.mean(0))/trimmedProbDF.std(0)
adata.obs['predicted_end']=adata.obs['noctx_supervised_name']
adata.obs['predicted_end']=probDF.idxmax(1)
adata.obs.loc[adata.obs['supervised_name'].str.contains('hase|G2-M'),'predicted_end']=np.nan

sc.pl.umap(adata,color=['noctx_supervised_name','predicted_end'],use_raw=False,save='predicted_end_nonorm')

probDF=(probDF-trimmedProbDF.mean(0))/trimmedProbDF.std(0)
adata.obs['predicted_end']=probDF.idxmax(1)
adata.obs.loc[adata.obs['supervised_name'].str.contains('hase|G2-M'),'predicted_end']=np.nan
adata.obs.loc[:,['full_cellname','noctx_supervised_name','predicted_end','latent_time']].to_csv('/wynton/group/ye/mtschmitz/macaquedevbrain/MacaqueGEsupervisednamesHippoPredictedEnd.txt',index=None)

sc.pl.umap(adata,color=['noctx_supervised_name','predicted_end'],use_raw=False,save='predicted_end_renorm')


# In[ ]:


sc.pl.umap(adata,color='timepoint', color_map=matplotlib.cm.RdYlBu_r,save='timepoint')


# In[ ]:


p = seaborn.catplot(x="timepoint", y="latent_time",

                 col="predicted_end",#hue="region",

                data=adata.obs, kind="violin",

                height=4, aspect=.7)
p


# In[ ]:


#df=adata.obs.loc[adata.obs['supervised_name'].str.contains(['Transition']),['predicted_end','timepoint','region','latent_time']]


# In[ ]:


adata.obs['newborn_neuron']=False
for c in adata.obs['noctx_supervised_name'].unique():
    print(c)
    selected_cells=adata.obs['noctx_supervised_name']==c
    lt_quantile=np.quantile(adata.obs.loc[selected_cells,'latent_time'],.5)
    print(lt_quantile)
    adata.obs.loc[selected_cells & (adata.obs['latent_time']<lt_quantile),'newborn_neuron']=True
    
adata.obs['newborn_neuron']=adata.obs['newborn_neuron'].astype(bool)
adata.obs.loc[adata.obs.supervised_name.str.contains('ransit'),'newborn_neuron']=True
adata.obs.loc[adata.obs.supervised_name.str.contains('G2-M|G1-phase|S-phase'),'newborn_neuron']=False


# In[ ]:


sc.pl.umap(adata,color='newborn_neuron',save='newborn_neuron',color_map=matplotlib.cm.RdYlBu_r)


# In[ ]:


#cortex order above
cortex_order=['pfc','cingulate','motor','somato','temporal','insula','hippocampus','parietal','v1']
cortex_colors=seaborn.color_palette("YlOrBr",n_colors=len(cortex_order)+2).as_hex()[2:]
ventral_tel=['ge','cge', 'cge_and_lge', 'lge','mge_and_cge_and_lge', 'mge']
vt_colors=seaborn.blend_palette(('fuchsia','dodgerblue'),n_colors=len(ventral_tel)).as_hex()
med_tel=['septum','pre-optic', 'hypothalamusandpoa','septumandnucleusaccumbens']
mt_colors=seaborn.blend_palette(('grey','black'),n_colors=len(med_tel)).as_hex()
basal_gang=['putamen_and_septum','str', 'putamenandclaustrumandbasalganglia', 'amy']
bg_colors=seaborn.color_palette("Greens",n_colors=len(basal_gang)).as_hex()

all_regions=cortex_order+ventral_tel+med_tel+basal_gang
all_regions_colors=cortex_colors+vt_colors+mt_colors+bg_colors
region_color_dict=dict(zip(all_regions,all_regions_colors))


# In[ ]:


reg_key='region'
df_plot = adata.obs.loc[adata.obs.newborn_neuron,:]
df_plot=df_plot.loc[~df_plot['predicted_end'].isna(),:]
df_plot['predicted_end'].cat.remove_unused_categories(inplace=True)
df_plot=df_plot.groupby(['predicted_end',reg_key, 'timepoint']).size()

print(df_plot)
#df_plot=df_plot.reset_index().pivot(columns='timepoint', index='predicted_end', values=0)
#df_plot=df_plot.T.apply(lambda g: g / g.sum(),1)
#df_plot=df_plot.loc[df_plot.sum(1)>20,:]
#regiontotals=adata.obs.groupby([ 'timepoint']).size()
#df_plot=(df_plot/regiontotals).T

#Normalize each sample so you have the proportion of newborn cells in each region
df_plot=df_plot.div(df_plot.groupby([reg_key,'timepoint']).transform('sum'))
#normalize each timepoint so percents of regions sums to 1 for each timepoint
df_plot=df_plot.div(df_plot.groupby(['timepoint']).transform('sum'))

df_plot=pd.DataFrame(df_plot).reset_index()

df_plot[0]=df_plot[0].fillna(0)

df_plot.rename(columns={0:'prop'}, inplace=True)


print(df_plot)
df_plot=df_plot.loc[~df_plot['predicted_end'].isna(),:]
df_plot=df_plot.loc[df_plot['predicted_end']!='nan',:]

# In[ ]:


df_plot.groupby(['timepoint']).sum()


# In[ ]:


#https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
'''def draw_pie(dist, 
             xpos, 
             ypos, 
             size, 
             ax=None,
             colors=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))

    # for incremental pie slices
    cumsum = np.cumsum(dist)
    cumsum = cumsum/ cumsum[-1]
    pie = [0] + cumsum.tolist()
    for i,(r1, r2) in enumerate(zip(pie[:-1], pie[1:])):
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()

        xy = np.column_stack([x, y])
        if colors is not None:
            ax.scatter([xpos], [ypos], marker=xy, s=size,c=colors[i])
        else:
            ax.scatter([xpos], [ypos], marker=xy, s=size)

    return ax
'''

matplotlib.rcParams['figure.dpi'] = 500
def draw_pie(dist,xpos, ypos, size,ax=None, colors=None):
    dist_sum=np.sum(dist)
    if dist_sum>0:
        dist=dist/np.sum(dist)
    #assert sum(dist) <= 1, 'sum of dist needs to be < 1'

    markers = []
    previous = 0
    # calculate the points of the pie pieces
    for color, ratio in zip(colors, dist):
        this = 2 * np.pi * ratio + previous
        x  = [0] + np.cos(np.linspace(previous, this, 800)).tolist() + [0]
        y  = [0] + np.sin(np.linspace(previous, this, 800)).tolist() + [0]
        xy = np.column_stack([x, y])
        previous = this
        markers.append({'marker':xy, 's':np.abs(xy).max()**2*np.array(size), 'facecolor':color})

    # scatter each of the pie pieces to create pies
    for marker in markers:
        ax.scatter(xpos, ypos, **marker)

n_region=len(df_plot[reg_key].unique())
colors=all_regions_colors#seaborn.color_palette(n_colors=n_region).as_hex()
seaborn.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(10,10))
for i,end in enumerate(df_plot['predicted_end'].unique()[::-1]):
    for timepoint in df_plot['timepoint'].unique():
        ind=(df_plot['timepoint']==timepoint) & (df_plot['predicted_end']==end)
        draw_pie(df_plot.loc[ind,'prop'].to_numpy(), 
             timepoint, 
             end, 
             size=df_plot.loc[ind,'prop'].sum()*12000, 
             ax=ax,colors=colors)

plt.xlabel('Day')        
plt.ylabel('Terminal Class')  
plt.title('Distribution of Newborn INs During Development')
ax.xaxis.grid(False)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'CelltypeTimePies.pdf'), bbox_inches="tight")
plt.close()
fig = plt.figure(figsize=(5, 3))
patches=[matplotlib.patches.Patch(color=colors[i],label=df_plot[reg_key].unique()[i]) for i in range(n_region)]
ax = fig.add_subplot(111)
fig.legend(patches, df_plot[reg_key].unique(), loc='right', frameon=False)
plt.axis('off')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'CelltypeTimePiesLegend.pdf'), bbox_inches="tight")



n_region=len(df_plot['predicted_end'].unique())
colors=seaborn.color_palette(n_colors=n_region).as_hex()
colors=seaborn.color_palette().as_hex()
fig, ax = plt.subplots(figsize=(10,10))
for i,end in enumerate(df_plot[reg_key].unique()):
    for timepoint in df_plot['timepoint'].unique():
        ind=(df_plot['timepoint']==timepoint) & (df_plot[reg_key]==end)
        draw_pie(df_plot.loc[ind,'prop'].to_numpy(), 
             timepoint, 
             end, 
             size=df_plot.loc[ind,'prop'].sum()*9000, 
             ax=ax,colors=colors)

plt.xlabel('Day')        
plt.ylabel('Region')  
plt.title('Distribution of Newborn INs During Development')
ax.xaxis.grid(False)
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'RegionTimePies.pdf'), bbox_inches="tight")
plt.close()
fig = plt.figure(figsize=(5, 3))
patches=[matplotlib.patches.Patch(color=colors[i],label=df_plot['predicted_end'].unique()[i]) for i in range(n_region)]
ax = fig.add_subplot(111)
fig.legend(patches, df_plot['predicted_end'].unique(), loc='right', frameon=False)
plt.axis('off')
ax.get_figure().savefig(os.path.join(sc.settings.figdir,'RegionTimePiesLegend.pdf'), bbox_inches="tight")
plt.close()






reg_key='region'
df_plot = adata.obs.loc[adata.obs.newborn_neuron,:]
df_plot = df_plot.loc[~df_plot['predicted_end'].isin(['RMTW_ZIC1/RELN']),:]
df_plot=df_plot.loc[~df_plot['predicted_end'].isna(),:]
df_plot=df_plot.groupby(['predicted_end',reg_key,'timepoint']).size()

#Normalize each sample so you have the proportion of newborn cells in each region
df_plot=df_plot.div(df_plot.groupby([reg_key,'timepoint']).transform('sum'))

df_plot=pd.DataFrame(df_plot).reset_index()

df_plot[0]=df_plot[0].fillna(0)

df_plot.rename(columns={0:'prop'}, inplace=True)
df_plot=df_plot.loc[~df_plot['prop'].isna(),:]
df_plot['prop']=df_plot['prop'].fillna(0)
df_plot=df_plot.loc[(df_plot['prop'].astype(float)>0),:]

print(df_plot)

tps=sorted(df_plot.timepoint.value_counts().index.values)
pe=sorted(df_plot.predicted_end.value_counts().index.values)
df_plot['region']=df_plot['region'].astype(str)
fig, axes = plt.subplots(len(pe),len(tps), figsize=(10, 10),sharey=True)
matplotlib.rc_file_defaults()
seaborn.reset_orig()
for y, timepoint in enumerate(tps):
    for x, predicted_end in enumerate(pe):
        try:
            data=df_plot.loc[(df_plot.timepoint==timepoint),:]
            data['region']=data['region'].astype('category')
            cats=df_plot['region'][df_plot['region'].isin(data['region'])].unique()
            data['region'].cat.reorder_categories(cats,inplace=True)
            data=data.loc[(df_plot.predicted_end==predicted_end),:]
            seaborn.barplot(x='region',y='prop',data=data,palette=paldict,ax=axes[x,y])
            axes[x,y].tick_params(bottom=False)
            axes[x,y].set(xticklabels=[])
            axes[x,y].set(title=None)
            axes[x,y].set(xlabel=None)
            if x==(len(pe)-1):
                axes[x,y].set(xlabel=timepoint)
            if y>0:
                axes[x,y].set(yticklabels=[])
                axes[x,y].set(ylabel=None)
            else:
                axes[x,y].set(ylabel=predicted_end)
                #plt.ylabel(predicted_end,fontsize=2)
            
        except:
            pass
seaborn.axes_style("white")
fig.tight_layout(h_pad=0,w_pad=-0.3)
plt.savefig(os.path.join(sc.settings.figdir,'CelltypeRegionTimeBars.pdf'), bbox_inches="tight")




adata.obs['platform']='V2'
adata.obs.loc[adata.obs['batch_name'].str.contains('2019'),'platform']='V3'
sc.pl.violin(adata[adata.obs['end_points']>.7,:],groupby='noctx_supervised_name',keys='linear_velocity_length',rotation=90,save='VelocityLength')
crabdata=adata[adata.obs['supervised_name'].str.contains('CRABP1'),:].copy()

crabdata.X=crabdata.raw.X[:,crabdata.raw.var.index.isin(adata.var.index)]
sc.pp.filter_genes(adata,min_cells=20)
print(crabdata)
print(crabdata.X)
sc.pp.normalize_total(crabdata,exclude_highly_expressed=True)
sc.pp.log1p(crabdata)
sc.pp.scale(crabdata,max_value=10)
sc.pp.pca(crabdata,n_comps=50)
#sc.pp.neighbors(adata)

crabdata.obs['batch_name'].cat.remove_unused_categories(inplace=True)

import bbknn
bbknn.bbknn(crabdata,batch_key='batch_name',n_pcs=50,neighbors_within_batch=1)
#sc.pp.neighbors(crabdata)
sc.tl.leiden(crabdata,resolution=1)
sc.tl.umap(crabdata,spread=4,min_dist=.2)

sc.pl.umap(crabdata,color='noctx_supervised_name',save='CRABP1_supervised')
sc.pl.umap(crabdata,color='latent_time',save='CRABP1_latent_time', color_map=matplotlib.cm.RdYlBu_r)
sc.pl.umap(crabdata,color='region',save='CRABP1_region')
sc.pl.umap(crabdata,color=['CRABP1','ANGPT2','TAC3','MAF'],use_raw=False,save='CRABP1_Markers')
logFC=np.log(crabdata[crabdata.obs['supervised_name'].isin(['MAF+/CRABP1+ MGE Interneuron']),:].X.mean(0)+1)/(crabdata[~crabdata.obs['supervised_name'].isin(['MAF+/CRABP1+ MGE Interneuron']),:].X.mean(0)+1)
