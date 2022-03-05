import os
import scanpy
import scanpy as sc
import pandas as pd
import numpy as np
import sklearn
from sklearn import decomposition
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
matplotlib.use('Agg')
plt.switch_backend('agg')
import logging as logg
import seaborn
import pickle
import inspect
import re
import scipy

sc.set_figure_params(color_map="Purples")
outpath=sc.settings.figdir
#Init scanpyanalysis object by giving an outpath, then call functions statically
outpath = os.path.expanduser(outpath)
#sc.settings.autosave=True
#sc.settings.autoshow=False
if not os.path.exists(outpath):
    os.makedirs(outpath)

#Saves a scanpy object from this pipeline (includes name and list of operations run in uns) as a h5ad
def save_adata(adata,outpath=None):
    if outpath is None:
        outpath=sc.settings.figdir
    adata.write(os.path.join(outpath,"-".join([adata.uns['name']]+list(adata.uns['operations']))+".h5ad"))

#Template function for pipeline
def template(adata,save=False):
    #anndata.read_h5ad(file)
    adata.uns['operations']=np.append(adata.uns['operations'],inspect.stack()[0][3])
    if save:
        save_adata(adata,outpath)
    return(None)

#import a single 10x file, do basic
def import_file(filename,refname=None,outpath=None,filter_genes=30,save=False):
    if outpath is None:
        outpath=sc.settings.figdir
    adata=sc.read_10x_mtx(filename)
    adata.var_names_make_unique()
    #sc.pp.filter_cells(adata, min_genes=filter_genes)
    adata.obs['n_counts'] = adata.X.sum(axis=1)
    adata.uns['name']=filename.split(os.sep)[-3]
    adata.uns['operations']=['load']
    if save:
        save_adata(adata,outpath)
    return(adata)

#import a single 10x file, do basic
def import_dropseq(filename,outname='merged',outpath=None,filter_genes=500,save=False):
    def gsub(regex, sub, l):
        return([re.sub(regex, sub, x) for x in l])
    if outpath is None:
        outpath=sc.settings.figdir
    adata=sc.read(os.path.join(filename,'summary',outname+'_umi_expression_matrix.tsv')).transpose()
    adata.var.index=gsub('.*__','',adata.var.index)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=filter_genes)
    adata.obs['n_counts'] = adata.X.sum(axis=1)
    adata.uns['name']=filename.split(os.sep)[-1]
    adata.uns['operations']=['load']
    if save:
        save_adata(adata,outpath)
    return(adata)

def sanitize_types_for_h5(adata):
    adata.var=adata.var.convert_dtypes()
    adata.obs=adata.obs.convert_dtypes()

    for c,t in zip(adata.obs.columns,adata.obs.dtypes):
        adata.obs[c]=adata.obs[c].to_list()
        if t == object:
            print(c,t)
            adata.obs[c]=pd.to_numeric(adata.obs[c], errors='coerce')

    for c,t in zip(adata.var.columns,adata.var.dtypes):
        adata.var[c]=adata.var[c].to_list()
        if t == object:
            print(c,t)
            adata.var[c]=pd.to_numeric(adata.var[c], errors='coerce')
    
    if adata.raw is not None:
        adata.raw.var.columns = adata.raw.var.columns.astype(str)
        for c,t in zip(adata.raw.var.columns,adata.raw.var.dtypes):
            adata.raw.var[c]=adata.raw.var[c].to_list()
            if t == object:
                adata.var[c]=pd.to_numeric(adata.raw.var[c], errors='coerce')
    
    adata.obs=adata.obs.fillna(np.nan)
    adata.var=adata.var.fillna(np.nan)
    adata.var.columns = adata.var.columns.astype(str)
    adata.obs.columns = adata.obs.columns.astype(str)
    return(adata)

def sum_duplicate_var(adata):
    #Instead of removing duplicate vars, sum them
    #Make the first duplicate the sum of all duplicate
    for g in adata.var.index[adata.var.index.duplicated(keep='first')]:
        adata.X[:,np.where(adata.var.index.isin([g]))[0][0]]=adata.X[:,np.where(adata.var.index.isin([g]))[0]].sum(1)
    #remove all but first duplicate    
    return(adata[:,~adata.var.index.duplicated(keep='first')])

def loadPlainKallisto(all_path,min_genes=1,transcript_names=False):
    import scanpy as sc
    def sum_duplicate_var(adata):
        #Instead of removing duplicate vars, sum them
        #Make the first duplicate the sum of all duplicate
        for g in adata.var.index[adata.var.index.duplicated(keep='first')]:
            wherevect=np.where(adata.var.index.isin([g]))
            adata.X[:,wherevect[0][0]]=adata.X[:,wherevect[0]].sum(1)
        #remove all but first duplicate    
        return(adata[:,~adata.var.index.duplicated(keep='first')])        #remove all but first duplicate    
    if transcript_names:
        adata=sc.read_10x_mtx(all_path, var_names='gene_ids')
    else:
        adata=sc.read_10x_mtx(all_path, var_names='gene_symbols')
    print(adata,flush=True)
    #still needs preprocessing from tsv to 10X format (PreprocessKallisto shell script)
    introng=adata.var.index.str.contains('__I$|-I$')
    realcells=(adata.X>0).sum(1)>=min_genes
    adata._inplace_subset_obs(realcells)
    uadata=adata[:,introng].copy()
    sadata=adata[:,~introng].copy()
    sadata=sum_duplicate_var(sadata)
    realcells=(sadata.X>0).sum(1)>=min_genes
    #Transcript names is too slow to also process introns
    if transcript_names:
        sadata._inplace_subset_obs(realcells)
        sadata.obs.index=[re.sub('-[0-9]','',x) for x in sadata.obs.index]
        return(sadata)
    uadata._inplace_subset_obs(realcells)
    sadata._inplace_subset_obs(realcells)
    uadata=sum_duplicate_var(uadata)
    print(adata,flush=True)
    uadata.var.index=[re.sub('__I','',x) for x in uadata.var.index]
    sadata=sc.AnnData.concatenate(sadata,uadata,join='outer')
    uadata=sadata[sadata.obs['batch']=='1',:].copy()
    sadata=sadata[sadata.obs['batch']=='0',:].copy()
    uadata.obs.index=[re.sub('-[0-9]','',x) for x in uadata.obs.index]
    sadata.obs.index=[re.sub('-[0-9]','',x) for x in sadata.obs.index]
    sadata.obs=sadata.obs.drop('batch',axis=1)
    sadata.layers['unspliced']=uadata.X
    sadata.layers['spliced']=sadata.X
    sadata.layers["ambiguous"] = scipy.sparse.csr_matrix(np.zeros(sadata.X.shape))
    return(sadata)


def readCellbenderH5(filename,include_obs=True,transcript_names=False):
    import h5py
    import scanpy as sc
    import scipy
    f = h5py.File(filename, 'r')
    mat=f['matrix']
    if include_obs:
        cols=['latent_cell_probability','latent_RT_efficiency']
        obsdict={x:mat[x] for x in cols}
        obs=pd.DataFrame(obsdict,index=[x.decode('ascii') for x in mat['barcodes']])
    else:
        obs=pd.DataFrame(index=[x.decode('ascii') for x in mat['barcodes']])
        
    ad=sc.AnnData(X=scipy.sparse.csr_matrix((mat['data'][:], 
                                          mat['indices'][:], 
                                          mat['indptr'][:]),
                                        shape=(mat['shape'][1],mat['shape'][0])),
              var=pd.DataFrame(dict(mat['features'])),
              obs=obs,
              uns={'test_elbo':list(mat['test_elbo']),'test_epoch':list(mat['test_epoch'])})
    if transcript_names:
        ad.var.index=[x.decode('ascii') for x in ad.var['id']]
    else:
        ad.var.index=[x.decode('ascii') for x in ad.var['name']]
    introng=np.array(['__I' in x for x in ad.var.index])
    uadata=ad[:,introng].copy()
    sadata=ad[:,~introng].copy()

    uadata=sum_duplicate_var(uadata)
    sadata=sum_duplicate_var(sadata)

    uadata.var.index=[re.sub('__I','',x) for x in uadata.var.index]
    sadata=sc.AnnData.concatenate(sadata,uadata,join='outer')

    uadata=sadata[sadata.obs['batch']=='1',:].copy()
    sadata=sadata[sadata.obs['batch']=='0',:].copy()

    sadata.obs.index=[re.sub('-[0-9]','',x) for x in sadata.obs.index]
    uadata.obs.index=[re.sub('-[0-9]','',x) for x in uadata.obs.index]
    sadata.obs=sadata.obs.drop('batch',axis=1)

    sadata.layers['unspliced']=uadata.X
    sadata.layers['spliced']=sadata.X
    sadata.layers["ambiguous"] = scipy.sparse.csr_matrix(np.zeros(sadata.X.shape))
    return(sadata)


def get_median_cell(adata,groupname,groupval):
    #Returns the cell that's closest to the median PCA value
    dists=[]
    indexes=adata.obs.index[adata.obs[groupname]==groupval]
    medianvals=np.median(adata.obsm['X_pca'][adata.obs[groupname]==groupval,:],axis=0)
    pcamat=adata.obsm['X_pca'][adata.obs[groupname]==groupval,:]
    for i in range(pcamat.shape[0]):
    #Manhattan dist
        dists.append(np.sum(np.abs(pcamat[i,:]-medianvals)))
    return(indexes[np.argmin(dists)])

#Use these functions to modularize munging of tp, region names from name string
#These can be remade for other applications, or generalized so you can pass a list of functions
def tp_format_macaque(name):
    if isinstance(name, str):
        name=re.sub('PEC_YALE','E110',name)
        name=re.sub('PEC_Yale','E110',name)
        name=re.sub('Mac2','E65',name)
        searched=re.search("E[0-9]+",name)
        if searched is None:
            return('nan')
        tp=re.sub("^E","",searched.group(0))
        tp=float(tp)
        return(tp)
    else:
        return(np.nan)

def tp_format_human(name):
    if isinstance(name, str):
        if 'CS' in name:
            name=re.sub("DeeperCS","CS",name)
            searched=re.search("CS[0-9]+",name)
            if searched is None:
                return('nan')
            tp=re.sub("^CS","",searched.group(0))
            csdict={'10':28,'11':29,'12':30,'13':32,'14':33,'15':36,'16':40,'17':42,'18':44,'19':48,'20':52,'21':54,'22':55,'23':58}
            return(csdict[tp])
        name=re.sub("^GW-","GW",name)
        searched=re.search("GW[0-9]+",name)
        if searched is None:
            return('nan')
        tp=re.sub("^GW","",searched.group(0))
        tp=float(tp)
        tp=(tp-2)*7
        return(tp)
    else:
        return(np.nan)

    
def region_format_macaque(name):
    if isinstance(name, str):
        name=name.upper()
        name=re.sub('PEC_YALE_SINGLECELLRNASEQ_RMB[0-9]+','',name)
        name=re.sub('MAC2','',name)
        name=re.sub("E80MGE","E80_GE",name)
        name=re.sub('E[0-9]+',"",name)
        name=re.sub("-2019A_","",name)
        name=re.sub("-2019B_","",name)
        name=re.sub("-2019A_AND_E65-2019B_","",name)
        name=re.sub("-2019_","",name)
        name=re.sub("^_","",name)
        name=re.sub("CGE-LGE","CGE_AND_LGE",name)
        name=re.sub("LGE_AND_CGE","CGE_AND_LGE",name)
        name=re.sub("_1","",name)
        name=re.sub("_2","",name)
        name=re.sub("_OUT","",name)
        name=re.sub("_KOUT","",name)
        name=re.sub("SOMATOSENSORY","SOMATO",name)
        name=re.sub(".LOOM","",name)
        return(name)
    else:
        region='nan'
    return(region)

def region_format_human(name):
    if isinstance(name, str):
        name=re.sub("^GW-","GW",name)
        name=re.sub("^DeeperCS","CS",name)
        name=re.sub('GW[0-9]+',"",name)
        name=re.sub('CS[0-9]+',"",name)
        name=re.sub("_filtered","",name)
        name=re.sub("2_motor","",name)
        name=re.sub("2_motorVZ","",name)
        name=re.sub("T_","",name)
        name=re.sub("_nee-.+","",name)
        name=re.sub("M1_all","motor",name)
        name=re.sub("^_","",name)
        name=re.sub("_Out","",name)
        name=re.sub("_kOut","",name)
        return(name)
    else:
        region='nan'
    return(region)


def macaque_process_irregular_names(name):
    name=re.sub('_Out','',name)
    name=re.sub('_kOut','',name)
    mixSamples={'Mix1':"E100caudateANDextBGANDintBGANDputanum",
    'Mix2':"E100pulvinarANDdorsalThalamusANDventralThalamus",
    'Mix3':"E100lateralANDanteriorANDvermis",
    'Mix4':"E80lateralANDanteriorANDvermis",
    'Mix5':"E80putamenANDclaustrumANDbasalGanglia",
    'Mix6':"E80dorsalThalamusANDventralThalamus",
    'Mix7':"E100hypothalamusANDPoA",
    'Mix8':"E100septumANDnucleusAccumbens",
    'Mix9':"E100CGE-LGE",
    'Mix10':"E80CGE-LGE",
    'Mix11':"E80choroidANDE100choroid",
    'E10somato':'E100somatosensory',
    'E100insulaCP':'E100insula',
    'E40_hippocampus':'E40hippo',
    'E50_hippo-temp':'E50hippo',
    'E65_hippocampus':'E65hippo',
    'E65_somatosensory':'E65somatosensory',
    'E65_hypothalamus':'E65hypo'}
    if name in mixSamples.keys():
        name=mixSamples[name]
    return(name)

def macaque_correct_regions(regionlist):
    regionkey={'thal':'thalamus',
               'hippo':'hippocampus',
               'somato':'somatosensory',
               'hypo':'hypothalamus',
               'midbrain/pons':'midbrain',
               'motor-1':'motor',
               'motor-2':'motor',
               'temp':'temporal',
               'hip':'hippocampus',
               'cbc':'cerebellum',
               'md':'thalamus',
               'choroidandchoroid':'choroid',
               'hippocampus_choroid':'hippocampus_and_choroid',
               'mge_and_cge_and_lge':'ge',
                'pulvinaranddorsalthalamusandventralthalamus':'thalamus',
               'lateralandanteriorandvermis':'cerebellum',
               'cerebellum_pons':'cerebellum',
               'hypothalamusandpoa':'hypothalamus',
              'dorsalthalamusandventralthalamus':'thalamus'}
    newl=[]
    for l in regionlist:
        l=l.lower()
        if 'multi-seq' in l:
            l='multi-seq'
        if l in regionkey.keys():
            l=regionkey[l]
        newl.append(l)
    return(newl)

def mouse_process_irregular_names(name):
    name=re.sub('_Out','',name)
    name=re.sub('_kOut','',name)
    mixSamples={'680847414':"E63_motor_1",
    '675026503':"E63_motor_2",
    '680583717':"E63_motor_3",
    'neuron_1k_v3':"E18_cortex_1k_v3",
    'neurons_2000':"E18_cortex_2k_v2",
    'neuron_10k_v3':"E18_cortex_10k_v3"}
    if name in mixSamples.keys():
        name=mixSamples[name]
    return(name)

def tp_format_mouse(name):
    if isinstance(name, str):
        name=name.upper()
        name=re.sub('SRR[0-9+]_','',name)
        name=re.sub('SAMN[0-9+]_','',name)
        name=re.sub('[A-Za-z0-9]+_GSM34495','E63',name)
        searched=re.search("E[0-9]+",name, re.IGNORECASE)
        if 'L8TX' in name:
            return 84.0
        if searched is not None:
            tp=re.sub("^E","",searched.group(0))
            tp=float(tp)
            return(tp)
        searched=re.search("P[0-9]+",name, re.IGNORECASE)
        if searched is not None:
            tp=re.sub("^P","",searched.group(0))
            tp=float(tp)+21
            return(tp)
    else:
        return(np.nan)

def region_format_mouse(name):
    if isinstance(name, str):
        name=name.upper()
        if '171026' in name:
            return('MOp')
        name=re.sub('E[0-9]+',"",name)
        name=re.sub('P[0-9]+',"",name)
        name=re.sub('SRR[0-9]+_','',name)
        name=re.sub('SAMN[0-9]+_','',name)        
        name=re.sub("^_","",name)
        name=re.sub("[A-Za-z0-9]+_GSM34495",'OB',name)
        name=re.sub("WT[0-9]+","CORTEX",name)
        name=re.sub("WT[0-9]+-[0-9]+","CORTEX",name)
        name=re.sub("[0-9]+","",name)
        name=re.sub("-","",name)
        name=re.sub("\.","",name)
        name=re.sub("_[0-9]+","",name)
        name=re.sub("_","",name)
        name=re.sub("NEURONSSAMPLE","CORTEX",name)
        name=re.sub("CORTEXKV","CORTEX",name)
        name=re.sub("_OUT","",name)
        name=re.sub("_KOUT","",name)
        return(name)
    else:
        region='nan'
    return(region)


def macaque_general_regions(regionlist):
    regionkey={'pulvinar':'Diencephalon',
               'pituitary':'Diencephalon',
               'pre-optic':'MedialTelencephalon',
               'poa':'MedialTelencephalon',
               'midbrain':'Mesencephalon',
               'pons':'Rhombencephalon',
               'medulla':'Rhombencephalon',
               'cerebellum':'Rhombencephalon',
               'cbc':'Rhombencephalon',
               'drg':'Rhombencephalon',
               'amy':'Basal Ganglia',
               'putamen':'Basal Ganglia',
               'str':'Basal Ganglia',
               'cingulate':'Cortex',
               'striatum':'Basal Ganglia',
               'septum':'MedialTelencephalon',
               'v1':'Cortex',
               'dfc':'Cortex',
               'pfc':'Cortex',
               'motor':'Cortex',
               'temporal':'Cortex',
               'hippo':'Cortex',
               'hippocampus':'Cortex',
               'hypothalamus':'Diencephalon',
               'thalamus':'Diencephalon',
               'somatosensory':'Cortex',
               'somato':'Cortex',
               'parietal':'Cortex',
               'insula':'Cortex',
               'cge':'VentralTelencephalon',
               'mge':'VentralTelencephalon',
               'lge':'VentralTelencephalon',
               'ge':'VentralTelencephalon',
               'choroid':'Choroid',
               'md':'Mesencephalon'}
    newl=[]
    for l in regionlist:
        l=l.lower()
        matcharray=np.array(list(regionkey.keys()))[[x in l for x in regionkey.keys()]]
        if len(matcharray)>0:
            l=regionkey[matcharray[0]]
        else:
            l='nan'
        newl.append(l)
    return(newl)

#Imports a list of files, filters them with rough filtering, returns list of scanpy objects.
#For use with concat_files function
#Set filters functions to None if not processing developing macaque brain
def import_files(fileList,refname,groupname="group",n_counts=400,percent_mito=.6,percent_ribo=.6,filter_genes=10,tp_func=tp_format_macaque,region_func=region_format_macaque,fix_irreg=macaque_process_irregular_names,log=False,save=True):
    adatas=[]
    batch_categories=[]
    for filename in fileList:
        a = sc.read_10x_mtx(filename,refname)
        a.uns['operations']=['load_file']
        sc.pp.filter_cells(a, min_genes=filter_genes)
        a.obs['n_counts'] = a.X.sum(axis=1)
        a=quantify_ribo(a)
        a=quantify_mito(a)
        a=filter_custom(a,n_counts=n_counts, percent_mito=percent_mito,percent_ribo=percent_ribo)
        #a=knee_filter_cells(a)
        if log:
            sc.pp.log1p(a)
        a.uns['name']=filename.split(os.sep)[-3]
        a.obs['batch']=str(a.uns['name'])
        if fix_irreg is not None:
            a=fix_irreg(a)
        if tp_func is not None:
            print(a)
            print(a.uns['name'])
            a.obs['timepoint']=tp_func(a.uns['name'])
        if region_func is not None:
            a.obs['region']=region_func(a.uns['name'])
        batch_categories.append(a.uns['name'])
        adatas.append(a)
    return(adatas)

#merges a list of scanpy objects into a single object
#This is a starting point for the pipeline
def concat_files(fileList,refname,outpath=None,groupname="group",n_counts=300,percent_mito=.6,percent_ribo=.6,filter_genes=10,save=True):
    if outpath is None:
        outpath=sc.settings.figdir
    adatas=import_files(fileList=fileList,refname=refname,groupname=groupname,n_counts=n_counts,percent_mito=percent_mito,percent_ribo=percent_ribo,filter_genes=filter_genes)
    adata = sc.AnnData.concatenate(*adatas)
    adata.uns['name']=groupname
    adata.uns['operations']=['concat_files']
    sc.settings.autosave=True
    sc.settings.autoshow=False
    if save:
        save_adata(adata,outpath)
    return(adata)

#Not working yet
def concat_files_mnn(fileList,refname,outpath=None,groupname="group",n_top_genes=None,n_counts=400,percent_mito=.6,percent_ribo=.6,filter_genes=10,save=True):
    if outpath is None:
        outpath=sc.settings.figdir
    if n_top_genes is None:
        n_top_genes=np.sum(np.sum(adata.X,axis=1)>0)
    adata=concat_files(fileList=fileList,refname=refname,groupname="macaqueDevBrain",n_counts=500,percent_mito=.5,percent_ribo=.5,filter_genes=10,save=True)
    var_subset=sc.pp.filter_genes_dispersion(  # select highly-variable genes
        adata.X, flavor='seurat', n_top_genes=n_top_genes, log=True)
    var_subset=adata.var.index[var_subset.gene_subset]
    print(adata.X)
    print(var_subset)
    adatas=import_files(fileList=fileList,refname=refname,groupname=groupname,n_counts=n_counts,percent_mito=percent_mito,percent_ribo=percent_ribo,filter_genes=filter_genes,log=True)
    adata,mnnList,angleList = sc.pp.mnn_correct(*adatas,var_index=var_subset,do_concatenate=True)
    adata.uns['angles']=angleList
    adata.uns['mnn']=angleList
    adata.uns['name']=groupname+"_MNN"
    adata.uns['operations']=['concat_files_mnn']
    if save:
        save_adata(adata,outpath)
    return(adata)

#Basic filtering functions
def filter_custom(adata,percent_mito=.5,percent_ribo=.5,n_counts=400):
    #adata.write(os.path.expanduser("~/Desktop/tmp.h5ad"))
    #adata=adata[adata.obs.n_counts>n_counts,:]
    adata._inplace_subset_obs(adata.obs.n_counts>n_counts)
    #adata=adata[adata.obs.percent_mito<percent_mito,:]
    adata._inplace_subset_obs(adata.obs.percent_mito<percent_mito)
    #adata=adata[adata.obs.percent_ribo<percent_ribo,:]
    adata._inplace_subset_obs(adata.obs.percent_ribo<percent_ribo)
    adata.raw=adata
    return(adata)

#Count the percent of ribosomal genes per cell, add to obs
def quantify_ribo(adata,ribo_genes=None,save=False):
    if ribo_genes is None:
        ribo_genes=[name for name in adata.var_names if name.startswith('RPS') or name.startswith('RPL') ]
    adata.obs['percent_ribo'] = np.sum(
        adata[:, ribo_genes].X, axis=1) / np.sum(adata.X, axis=1)
    return(adata)

#Count the percent of mitochondrial genes per cell, add to obs
def quantify_mito(adata,mito_genes=None,save=False):
    if mito_genes is None:
        mito_genes = [name for name in adata.var_names if name in ['ND1','ND2','ND4L','ND4','ND5','ND6','ATP6','ATP8','CYTB','COX1','COX2','COX3'] or 'MT-' in name]
    adata.obs['percent_mito'] = np.sum(
        adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
    return(adata)

#Doesn't work properly on macaque data
def knee_filter_cells(adata,expect_cells=30000,save=False):
    knee_cells=list(getKneeEstimate(Counter(adata.obs.n_counts.to_dict()),expect_cells=expect_cells,plotfile_prefix=sc.settings.figdir))
    inthresh=np.array([x in knee_cells for x in adata.obs.index])
    if save:
        save_adata(adata,sc.settings.figdir)
    return(adata._inplace_subset_obs(inthresh))

#Standard normalizing and scaling, used before score_genes and PCA
def norm_and_scale(adata,n_top_genes=None,min_cells=5,save=False):
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if n_top_genes is None:
        n_top_genes=adata.shape[1]
    filter_result = sc.pp.filter_genes_dispersion( adata.X, flavor='cell_ranger', n_top_genes=n_top_genes, log=False)
    adata = adata[:, filter_result.gene_subset]     # subset the genes
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    adata.uns['operations']=np.append(adata.uns['operations'],inspect.stack()[0][3])
    if save:
        save_adata(adata,sc.settings.figdir)
    return(adata)

#Runs scanpy pca
def do_pca(adata,save=False):
    sc.tl.pca(adata)
    adata.obsm['X_pca'] *= -1  # multiply by -1 to match Seurat
    sc.pl.pca_scatter(adata)
    sc.pl.pca_variance_ratio(adata, log=True)
    sc.pl.pca_loadings(adata)
    adata.uns['operations']=np.append(adata.uns['operations'],inspect.stack()[0][3])
    if save:
        save_adata(adata,sc.settings.figdir)
    return(adata)

#Runs tsne, use after quantifying ribo and mito
def do_tsne(adata,save=False):
    sc.tl.tsne(adata, random_state=2, n_pcs=50)
    sc.pl.tsne(adata, color=['n_counts','percent_ribo','percent_mito'],  save="_percentRiboMito")
    adata.uns['operations']=np.append(adata.uns['operations'],inspect.stack()[0][3])
    if save:
        save_adata(adata,sc.settings.figdir)
    return(adata)

#Runs neighbors to build KNN graph from PCA, also runs UMAP
def neighbors_and_umap(adata,n_neighbors=100,save=False):
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata)
    adata.uns['operations']=np.append(adata.uns['operations'],inspect.stack()[0][3])
    if save:
        save_adata(adata,sc.settings.figdir)
    return(adata)

#Runs leiden clustering
def leiden(adata,save=False):
    sc.tl.leiden(adata)
    adata.uns['operations']=np.append(adata.uns['operations'],inspect.stack()[0][3])
    sc.pl.umap(adata,color="leiden",save="_leiden")
    if 'X_tsne' in adata.obsm.keys():
        sc.pl.tsne(adata,color="leiden",  save="_leiden")
    else:
        sc.pl.umap(adata,color="leiden",  save="_leiden")
    if save:
        save_adata(adata,sc.settings.figdir)
    return(adata)

#Function wraps together standard basic processing
def std_norm_transform(adata,n_top_genes=6000,log=True,tsne=True,save=False):
    if log:
        sc.pp.recipe_zheng17(adata,n_top_genes=n_top_genes)
    else:
         sc.pp.recipe_zheng17(adata,n_top_genes=n_top_genes,log=False)

    #adata=norm_and_scale(adata,n_top_genes)
    adata=do_pca(adata)
    if tsne:
        adata=do_tsne(adata)
    adata=neighbors_and_umap(adata)
    adata=leiden(adata)
    if save:
        save_adata(adata,sc.settings.figdir)
    return(adata)

#To Do, incomplete
def woublet(adata,save=False):
    print(adata.shape[0]/100000,"doublets")
    sc.tl.woublet(adata,expected_doublet_rate=adata.shape[0]/(100000))
    adata.uns['operations']=np.append(adata.uns['operations'],inspect.stack()[0][3])
    if save:
        save_adata(adata,sc.settings.figdir)
    return(adata)

#Strips genes from the expression matrix of a scanpy objects
def strip_genes(adata,genes,groupname='genegroup',save=False):
    genes=np.array([x for x in genes if x in adata.var.index])
    print(genes)
    print(np.array(adata[:,genes].X))
    adata.obsm[str(groupname)]=np.array(adata[:,genes].X.todense())
    adata.uns[groupname+'_genenames']=genes
    adata=adata[:,[x for x in adata.var.index if x not in genes]]
    adata.uns['operations']=np.append(adata.uns['operations'],inspect.stack()[0][3])
    if save:
        save_adata(adata,sc.settings.figdir)
    return(adata)

#Calculates the cell cycle score using Aviv Regevs cell cycle stage markers
def cell_cycle_score(adata,save=False):
    s_genes=['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8']
    g2m_genes=['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']
    sc.tl.score_genes_cell_cycle(adata,s_genes=s_genes,g2m_genes=g2m_genes)
    sc.pl.violin(adata, ['G2M_score','S_score'], groupby='leiden')
    if 'X_tsne' in adata.obsm.keys():
        sc.pl.tsne(adata, color=['G2M_score','S_score','phase','leiden'],save='_cc')
    sc.pl.umap(adata, color=['G2M_score','S_score','phase','leiden'],save='_cc')
    #adata.uns['operations']=np.append(adata.uns['operations'],inspect.stack()[0][3])
    if save:
        save_adata(adata,sc.settings.figdir)
    return(adata)

#Calculates logistic regression enrichment of leiden clusters
def log_reg_diff_exp(adata,obs_name='leiden',ncells=None,prefix='',min_in_group_fraction=.01,max_out_group_fraction=.5,method='logreg'):
    if ncells is None:
        cells=adata.obs.index
    else:
        cells=np.random.choice(adata.obs.index,ncells,replace=False)
    adata=sc.tl.rank_genes_groups(adata[cells,:], groupby=obs_name, method=method,copy=True,use_raw=False)#,min_in_group_fraction=min_in_group_fraction,max_out_group_fraction=max_out_group_fraction)
    #sc.tl.filter_rank_genes_groups(adata,groupby=obs_name,min_in_group_fraction=min_in_group_fraction,max_out_group_fraction=max_out_group_fraction,key='rank_genes_groups')
    #result = adata.uns['rank_genes_groups_filtered']
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    df=pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'scores']})
    adata.uns['log_reg_de']=df
    df.to_csv(os.path.join(sc.settings.figdir,prefix+obs_name+"LogRegMarkers.csv"))
    #adata.uns['operations']=np.append(adata.uns['operations'],inspect.stack()[0][3])
    return(adata)

#Input names of obsm for silhouettes, and obs for clustermethod and rands
#Provides basic metrics of clustering quality 
def get_cluster_metrics(adata,rands=[],clustermethod='leiden',silhouettes=['X_pca','X_umap']):
    import sklearn
    mets={}
    for i in silhouettes:
        mets[i+'_silhouette']=sklearn.metrics.silhouette_score( adata.obsm[i],adata.obs[clustermethod],metric='manhattan')
    for i in rands:
        mets[i+'_rand']=sklearn.metrics.adjusted_rand_score(adata.obs[i],adata.obs[clustermethod])
    mets['X_silhouette']=sklearn.metrics.silhouette_score( adata.X,adata.obs[clustermethod],metric='manhattan')
    return(mets)

def doublescrub(adata):
    fail=0
    cells=adata.obs.index
    try:
        import scrublet
        scrub = scrublet.Scrublet(adata.X)
        doublet_scores, predicted_doublets = scrub.scrub_doublets()
        scrubletdubs=cells[predicted_doublets]
        if len(scrubletdubs)==0 or len(scrubletdubs.shape)>1:
            predicted_doublets = scrub.call_doublets(threshold=0.5)
            scrubletdubs=cells[predicted_doublets]
    except:
            scrubletdubs=[]
    try:    
        import doubletdetection
        clf = doubletdetection.BoostClassifier()
        labels = clf.fit(adata.X).predict()
        dddubs=cells[np.array([bool(x) for x in labels])]
    except:
        dddubs=[]
    doublets=list(set(dddubs).union(set(scrubletdubs)))
    #print(nondoublets)
    return(doublets)

#Carry out brain atlas marker expresion, preprocessing if very complicated to process class and subclasses of markers
#Uses score_genes
def marker_analysis(adata,variables=['leiden','region'],markerpath='https://docs.google.com/spreadsheets/d/e/2PACX-1vTz5a6QncpOOO-f3FHW2Edomn7YM5mOJu4z_y07OE3Q4TzcRr14iZuVyXWHv8rQuejzhhPlEBBH1y0V/pub?gid=1154528422&single=true&output=tsv',subclass=True,plotall=True,save=False,prefix=''):
    sc.set_figure_params(color_map="Purples")
    import random
    markerpath=os.path.expanduser(markerpath)
    markers=pd.read_csv(markerpath,sep="\t")
    markers[markers.keys()[0]]=[str(x) for x in markers[markers.keys()[0]]]
    markers[markers.keys()[2]]=[str(x).split(',') for x in markers[markers.keys()[2]]]
    markers[markers.keys()[3]]=[str(x).split(';') for x in markers[markers.keys()[3]]]
    markers[markers.keys()[3]]=[[str(x).split(',') for x in y] for y in markers[markers.keys()[3]]]
    uniqueClasses=set([y for x in markers[markers.keys()[2]] for y in x if y!='nan'])
    uniqueSubClasses=set([z for x in markers[markers.keys()[3]] for y in x for z in y if z!='nan'])
    comboClasses=[]
    #print(markers)
    markers[markers.keys()[2]]=[[y.lstrip() for y in x ]for x in markers[markers.keys()[2]]]
    markers[markers.keys()[3]]=[[[z.lstrip() for z in  y] for y in x] for x in markers[markers.keys()[3]]]
    if subclass:
        for i in range(markers.shape[0]):
            rowlist=[]
            for j in range(len(markers[markers.keys()[2]][i])):
                for k in markers[markers.keys()[3]][i][j]:
                    rowlist.append(re.sub('^ ','',' '.join(filter(lambda x: x != 'nan',[k,markers[markers.keys()[2]][i][j]]))))
            comboClasses.append(rowlist)
    else:
        for i in range(markers.shape[0]):
            rowlist=[]
            for j in range(len(markers[markers.keys()[2]][i])):
                rowlist.append(markers[markers.keys()[2]][i][j])
            comboClasses.append(rowlist)

    markers['fullclass']=comboClasses
    markers.set_index(markers.keys()[0],inplace=True,drop=False)
    markers=markers.loc[ [x for x in markers[markers.keys()[0]] if x in adata.var_names],:]
    uniqueFullClasses=set([y for x in markers['fullclass'] for y in x if y!='nan'])
    from collections import defaultdict
    markerDict = defaultdict(list)

    for x in uniqueFullClasses:
        for y in markers[markers.keys()[0]]:
            if x in markers.loc[y,'fullclass']:
                markerDict[x].append(y)
    markerDictClass = defaultdict(list)
    for x in uniqueClasses:
        for y in markers[markers.keys()[0]]:
            if x in markers.loc[y,'fullclass']:
                markerDictClass[x].append(y)

    markerPlotGroups=[]
    for k in markerDict.keys():
        if len(markerDict[k])>1:
            sc.tl.score_genes(adata,gene_list=markerDict[k],score_name='mkrscore'+k,gene_pool= markerDict[k]+random.sample(adata.var.index.tolist(),min(1000,adata.var.index.shape[0])),use_raw=False)
            markerPlotGroups.append('mkrscore'+k)
    adata.uns['marker_groups']=list(markerDict.keys())
    for tag in variables:
        pd.DataFrame(adata.obs.groupby(tag).describe()).to_csv(os.path.join(sc.settings.figdir, tag+prefix+"MarkerSumStats.csv"))

    sc.pl.umap(adata, color=markerPlotGroups,save=prefix+"_Marker_Group")

    #sc.pl.violin(adata, markerPlotGroups, groupby='leiden',save=prefix+"_Marker_Group_violins")
    if plotall:
        for i in markerDictClass:
            genes=sorted(markerDictClass[i])
            genes=[g for g in genes if g in adata.var.index]
            sc.pl.umap(adata, color=genes,save=prefix+"_"+str(i)+"_Marker",use_raw=False)
    #adata.uns['markers']=markers
    adata.obs['subclassname']=[re.sub('mkrscore','',x) for x in adata.obs.loc[:,['mkrscore' in x for x in adata.obs.columns]].astype('float').idxmax(axis=1)]
    def most_frequent(List): 
        return max(set(List), key = List.count) 
    classlist=[]
    for c in adata.obs['subclassname']:
        fullbool=[c in x for x in markers['fullclass']]
        flatclass=[item for sublist in markers.loc[fullbool,'type'] for item in sublist]
        classlist.append(most_frequent(flatclass))
    adata.obs['classname']=classlist
    sc.pl.umap(adata,color=['classname'],save='classname')
    sc.pl.umap(adata,color=['subclassname'],save='subclassname')
    #adata.uns['operations']=np.append(adata.uns['operations'],inspect.stack()[0][3])
    return(adata)

def dirichlet_marker_analysis(adata,markerpath='https://docs.google.com/spreadsheets/d/e/2PACX-1vTz5a6QncpOOO-f3FHW2Edomn7YM5mOJu4z_y07OE3Q4TzcRr14iZuVyXWHv8rQuejzhhPlEBBH1y0V/pub?gid=1154528422&single=true&output=tsv'):
    sc.set_figure_params(color_map="Purples")
    import random
    markerpath=os.path.expanduser(markerpath)
    markers=pd.read_csv(markerpath,sep="\t")
    markers[markers.keys()[0]]=[str(x) for x in markers[markers.keys()[0]]]
    markers[markers.keys()[2]]=[str(x).split(',') for x in markers[markers.keys()[2]]]
    markers[markers.keys()[3]]=[str(x).split(';') for x in markers[markers.keys()[3]]]
    markers[markers.keys()[3]]=[[str(x).split(',') for x in y] for y in markers[markers.keys()[3]]]
    uniqueClasses=set([y for x in markers[markers.keys()[2]] for y in x if y!='nan'])
    uniqueSubClasses=set([z for x in markers[markers.keys()[3]] for y in x for z in y if z!='nan'])
    comboClasses=[]
    print(markers)
    for i in range(markers.shape[0]):
        rowlist=[]
        for j in range(len(markers[markers.keys()[2]][i])):
            for k in markers[markers.keys()[3]][i][j]:
                rowlist.append(' '.join(filter(lambda x: x != 'nan',[k,markers[markers.keys()[2]][i][j]])))
        comboClasses.append(rowlist)
    markers['fullclass']=comboClasses
    markers.set_index(markers.keys()[0],inplace=True,drop=False)
    markers=markers.loc[ [x for x in markers[markers.keys()[0]] if x in adata.var_names],:]
    uniqueFullClasses=set([y for x in markers['fullclass'] for y in x if y!='nan'])
    from collections import defaultdict
    markerDict = defaultdict(list)

    for x in uniqueFullClasses:
        for y in markers[markers.keys()[0]]:
            if x in markers.loc[y,'fullclass']:
                markerDict[x].append(y)
    markerDictClass = defaultdict(list)
    for x in uniqueClasses:
        for y in markers[markers.keys()[0]]:
            if x in markers.loc[y,'fullclass']:
                markerDictClass[x].append(y)

    markerPosterior = pd.DataFrame()
    for k in markerDict.keys():
        if len(markerDict[k])>1:
            markerPosterior=pd.concat([markerPosterior,pd.DataFrame(np.mean(adata[:,np.array(markerDict[k])].varm['gene_topic'],axis=0),columns=[k])],axis=1)

    markerPosterior.to_csv(os.path.join(sc.settings.figdir,"MarkerPosteriorMeans.csv"))

#Calculate Hierarchical dirichlet process model for expression set
def sc_hdp(adata,alpha=1,eta=.01,gamma=1,eps=1e-5,save=False):
    import gensim
    import scipy
    import itertools
    def moving_average(data,window_width):
        cumsum_vec = np.cumsum(np.insert(data, 0, 0))
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        return(ma_vec)

    def table_top_words(word_topic, feature_names, n_top_words):
        df=[]
        for topic_idx, topic in enumerate(word_topic):
            print(topic_idx)
            df.append([feature_names[i]
                                for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(df)
        return(pd.DataFrame(df))

    #A bit of a hack because normally this is returned nice and sparse, but you don't know which topics are included
    #Works, slow, only method that works
    def get_doc_topic(corpus, model,eps=0.0):
        doc_topic = list()
        for doc in corpus:
            doc_topic.append(model.__getitem__(doc, eps=eps))
            #Convert index,value tuples to sparse matrix, then dense
        ii=[[i]*len(v) for i,v in enumerate(doc_topic)]
        ii=list(itertools.chain(*ii))
        jj=[j for j,_ in itertools.chain(*doc_topic)]
        data=[d for _,d in itertools.chain(*doc_topic)]
        return(scipy.sparse.csr_matrix((data, (ii, jj))).todense(),list(set(jj)))


    def get_topic_to_wordids(model,eps=0.0):
        p = list()
        if hasattr(model, 'm_T'):
            for topicid in range(model.m_T):
                topic = model.m_lambda[topicid]
                topic = topic / topic.sum() # normalize to probability dist
                p.append(topic)
            return(np.array(p).T,list(range(model.m_T)))
        else:
            for topicid in range(model.num_terms):
                topic = model.get_term_topics(topicid,minimum_probability=eps)
                p.append(topic)
            ii=[[i]*len(v) for i,v in enumerate(p)]
            ii=list(itertools.chain(*ii))
            jj=[j for j,_ in itertools.chain(*p)]
            data=[d for _,d in itertools.chain(*p)]
            return(scipy.sparse.csr_matrix((data, (ii, jj))).todense(),list(set(jj)))

    #Merges 2 2D dataframes into 3d, then collapses them every other column (Name1,Val1,Name2,Val2)
    #word_topic must be np array, not np matrix
    def table_top_words(word_topic, feature_names):
        return(pd.DataFrame(np.stack([feature_names[(-word_topic).argsort(axis=0)],word_topic[(-word_topic).argsort(axis=0)][:,:,0]],2).reshape(word_topic.shape[0],-1)))

    #def table_top_words(word_topic, feature_names, n_top_words):
    #    df=[]
    #    for topic_idx, topic in enumerate(word_topic.T):
    #        print(topic_idx)
    #        df.append([feature_names[i]
    #                            for i in topic.argsort()[:-n_top_words - 1:-1]])
    #print(df)
    #return(pd.DataFrame(df))

    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    adata=adata[:,adata.var.index.argsort()]
    model = gensim.models.HdpModel(corpus=gensim.matutils.Sparse2Corpus( adata.X.T),id2word=gensim.corpora.dictionary.Dictionary([adata.var.index.tolist()]),alpha=alpha,gamma=gamma,eta=eta)
    print(model)
    print(model.lda_alpha)
    print(model.lda_beta)
    modelLDA=model.suggested_lda_model()
    #modelLDA=model
    #Will drop topics even with eps = 0.0 ... idk why
    doc_topic,included_topics=get_doc_topic(gensim.matutils.Sparse2Corpus(adata.X.T),modelLDA,eps=eps)
    #Included topics from words seems unnecessary empirically, but will catch in variable anyway
    word_topic,included_topics_from_words=get_topic_to_wordids(modelLDA,eps=0.0)
    doc_topic=doc_topic[:,intersection(included_topics,included_topics_from_words)]
    word_topic=word_topic[:,intersection(included_topics,included_topics_from_words)]
    #Slim down the topics to the ones that are well represented in cells (sum > eps)
    included_topics=np.array(np.sum(doc_topic,axis=0)/sum(np.sum(doc_topic,axis=0))>eps)[0]
    doc_topic=doc_topic[:,included_topics]
    word_topic=word_topic[:,included_topics]
    adata.varm['gene_topic']=word_topic
    adata.obsm['cell_topic']=doc_topic
    if 'operations' in adata.uns.keys():
        adata.uns['operations']=np.append(adata.uns['operations'],inspect.stack()[0][3])
    if save:
        save_adata(adata,sc.settings.figdir)
    table_top_words(np.array(word_topic),adata.var.index).to_csv(os.path.join(sc.settings.figdir,"TopicMarkers.txt"))
    pl=seaborn.barplot(x=list(range(doc_topic.shape[1])),y=np.sum(np.array(doc_topic),axis=0)).get_figure()
    pl.savefig(os.path.join(sc.settings.figdir,"TopicWeightBarplot.png"))
    plt.clf()
    return(adata)

#Calculate non-negative matrix factorization of a dataset
def sc_nmf(adata,n_components=12,save=True):
    import sklearn
    from sklearn import decomposition
    def moving_average(data,window_width):
        cumsum_vec = np.cumsum(np.insert(data, 0, 0))
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        return(ma_vec)
    def table_top_words(model, feature_names, n_top_words):
        df=[]
        for topic_idx, topic in enumerate(model.components_):
            print(topic_idx)
            df.append([feature_names[i]
                                for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(df)
        return(pd.DataFrame(df))

    adata=adata[adata.obs.n_counts.argsort()[::-1],:]
    nmfM=sklearn.decomposition.NMF(n_components=n_components,verbose=1)
    nmfM.fit(adata.X)
    doc_topic=nmfM.transform(adata.X)
    adata.uns['nmf_model']=nmfM
    for i in range(doc_topic.shape[1]):
        adata.obs['nmf_'+str(i)]=doc_topic[:,i]

    for i in range(nmfM.components_.shape[0]):
        adata.var['nmf_'+str(i)]=np.log10(nmfM.components_[i,:])
    import seaborn
    hm=seaborn.heatmap(doc_topic).get_figure()
    hm.savefig(os.path.join(sc.settings.figdir,"TopicHeatmap.png"))
    plt.clf()
    plt.plot(range(doc_topic.shape[0]),np.std( doc_topic,axis=1))
    plt.title("SD of topic allocation in cells")
    plt.xlabel("RankedCells")
    plt.ylabel("SD")
    plt.savefig(os.path.join(sc.settings.figdir,"TopicCellSD"))
    plt.clf()
    convolvedSD=moving_average(np.std( doc_topic,axis=1),300)
    plt.plot(range(len(convolvedSD)),convolvedSD)
    plt.title("Moving Average Sparsity (SD)")
    plt.xlabel("RankedCells")
    plt.ylabel("SD")
    plt.savefig(os.path.join(sc.settings.figdir,"TopicCellMASD.png"))
    plt.clf()
    plt.plot(range(adata.obs.shape[0]),np.log10(adata.obs.n_counts))
    plt.title("Log Counts per Cell")
    plt.xlabel("RankedCells")
    plt.title("Log Reads")
    plt.savefig(os.path.join(sc.settings.figdir,"LogCounts.png"))
    plt.clf()
    for i in range(doc_topic.shape[1]):
        convolved=moving_average(doc_topic[:,i],300)
        plt.plot(range(len(convolved)),convolved)
    plt.xlabel("RankedCells")
    plt.ylabel("Moving Avg Topic Probability")
    plt.savefig(os.path.join(sc.settings.figdir,"TopicMovingAvg.png"))
    plt.clf()
    table_top_words(nmfM,adata.var.index,40).to_csv(os.path.join(sc.settings.figdir,"TopicMarkers.txt"))
    adata.uns['operations']=np.append(adata.uns['operations'],inspect.stack()[0][3])
    if save:
        save_adata(adata,sc.settings.figdir)
    return(adata)

#Calculate latent dirichlet allocation of a dataset
def sc_lda(adata,n_components=12,topic_word_prior=None,doc_topic_prior=None,save=False):
    import sklearn
    from sklearn import decomposition
    def moving_average(data,window_width):
        cumsum_vec = np.cumsum(np.insert(data, 0, 0))
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        return(ma_vec)
    def table_top_words(word_topic, feature_names):
        return(pd.DataFrame(np.stack([feature_names[(-word_topic).argsort(axis=0)],word_topic[(-word_topic).argsort(axis=0)][:,:,0]],2).reshape(word_topic.shape[0],-1)))

    adata=adata[adata.obs.n_counts.argsort()[::-1],:]
    ldaM=sklearn.decomposition.LatentDirichletAllocation(n_components=n_components,topic_word_prior=topic_word_prior,doc_topic_prior=doc_topic_prior,learning_method="online",verbose=1,n_jobs=-1)
    ldaM.fit(adata.X)
    doc_topic=ldaM.transform(adata.X)
    adata.uns['lda_model']=ldaM
    for i in range(doc_topic.shape[1]):
        adata.obs['lda_'+str(i)]=doc_topic[:,i]

    for i in range(ldaM.components_.shape[0]):
        adata.var['lda_'+str(i)]=np.log10(ldaM.components_[i,:])
    import seaborn
    hm=seaborn.heatmap(doc_topic).get_figure()
    hm.savefig(os.path.join(sc.settings.figdir,"TopicHeatmap.png"))
    plt.clf()
    plt.plot(range(doc_topic.shape[0]),np.std( doc_topic,axis=1))
    plt.title("SD of topic allocation in cells")
    plt.xlabel("RankedCells")
    plt.ylabel("SD")
    plt.savefig(os.path.join(sc.settings.figdir,"TopicCellSD"))
    plt.clf()
    convolvedSD=moving_average(np.std( doc_topic,axis=1),300)
    plt.plot(range(len(convolvedSD)),convolvedSD)
    plt.title("Moving Average Sparsity (SD)")
    plt.xlabel("RankedCells")
    plt.ylabel("SD")
    plt.savefig(os.path.join(sc.settings.figdir,"TopicCellMASD.png"))
    plt.clf()
    plt.plot(range(adata.obs.shape[0]),np.log10(adata.obs.n_counts))
    plt.title("Log Counts per Cell")
    plt.xlabel("RankedCells")
    plt.title("Log Reads")
    plt.savefig(os.path.join(sc.settings.figdir,"LogCounts.png"))
    plt.clf()
    for i in range(doc_topic.shape[1]):
        convolved=moving_average(doc_topic[:,i],300)
        plt.plot(range(len(convolved)),convolved)
    plt.xlabel("RankedCells")
    plt.ylabel("Moving Avg Topic Probability")
    plt.savefig(os.path.join(sc.settings.figdir,"TopicMovingAvg.png"))
    plt.clf()
    table_top_words(ldaM.components_.T,adata.var.index).to_csv(os.path.join(sc.settings.figdir,"TopicMarkers.txt"))
    adata.uns['operations']=np.append(adata.uns['operations'],inspect.stack()[0][3])
    if save:
        save_adata(adata,sc.settings.figdir)
    return(adata)


def cellphonedb(adata,annotation_name):
    import subprocess
    df_expr_matrix = adata.X 
    df_expr_matrix = df_expr_matrix.T 
    df_expr_matrix = pd.DataFrame(df_expr_matrix.toarray()) # Set cell ids as columns 
    df_expr_matrix.columns = adata.obs.index # Genes should be either Ensembl IDs or gene names 
    df_expr_matrix.set_index(adata.raw.var.index, inplace=True) 
    savepath_counts=os.path.join(sc.settings.figdir,'cellphonedbInputExpression.txt')
    df_expr_matrix.to_csv(savepath_counts,sep='\t') # generating meta file 
    df_meta = pd.DataFrame(data={'Cell': list(adata.obs.index), 'cell_type': list(adata.obs[annotation_name])})
    df_meta.set_index('Cell',inplace=True) 
    savepath_meta=os.path.join(sc.settings.figdir,'cellphonedbInputMeta.txt')
    df_meta.to_csv(savepath_meta, sep='\t')
    subprocess.run('cellphonedb method statistical_analysis '+ str(savepath_meta) +' '+ str(savepath_counts),shell=True)


#From UMI tools package
def getKneeEstimate(cell_barcode_counts,
                    expect_cells=False,
                    cell_number=False,
                    plotfile_prefix=None,
                    force=True):
    ''' estimate the number of "true" cell barcodes
    input:
         cell_barcode_counts = dict(key = barcode, value = count)
         expect_cells (optional) = define the expected number of cells
         cell_number (optional) = define number of cell barcodes to accept
         plotfile_prefix = (optional) prefix for plots
    returns:
         List of true barcodes
    '''
    from umi_tools import umi_methods
    from collections import Counter
    from functools import partial
    from scipy.signal import argrelextrema
    from scipy.stats import gaussian_kde
    import umi_tools.Utilities as U
    import matplotlib.lines as mlines
    # very low abundance cell barcodes are filtered out (< 0.001 *
    # the most abundant)
    threshold = 0.001 * cell_barcode_counts.most_common(1)[0][1]

    counts = sorted(cell_barcode_counts.values(), reverse=True)
    counts_thresh = [x for x in counts if x > threshold]
    log_counts = np.log10(counts_thresh)

    # guassian density with hardcoded bw
    density = gaussian_kde(log_counts, bw_method=0.1)

    xx_values = 10000  # how many x values for density plot
    xx = np.linspace(log_counts.min(), log_counts.max(), xx_values)

    local_min = None

    if cell_number:  # we have a prior hard expectation on the number of cells
        threshold = counts[cell_number]

    else:
        local_mins = argrelextrema(density(xx), np.less)[0]
        local_mins_counts = []

        for poss_local_min in local_mins[::-1]:

            passing_threshold = sum([y > np.power(10, xx[poss_local_min])
                                     for x, y in cell_barcode_counts.items()])
            local_mins_counts.append(passing_threshold)

            if not local_min:   # if we have selected a local min yet
                if expect_cells:  # we have a "soft" expectation
                    if (passing_threshold > expect_cells * 0.01 and
                        passing_threshold <= expect_cells):
                        local_min = poss_local_min

                else:  # we have no prior expectation
                    # TS: In abscence of any expectation (either hard or soft),
                    # this set of heuristic thresholds are used to decide
                    # which local minimum to select.
                    # This is very unlikely to be the best way to achieve this!
                    if (poss_local_min >= 0.2 * xx_values and
                        (log_counts.max() - xx[poss_local_min] > 0.5 or
                         xx[poss_local_min] < log_counts.max()/2)):
                        local_min = poss_local_min

        if local_min is None and force:
            local_min = min(local_mins)

        if local_min is not None:
            threshold = np.power(10, xx[local_min])

    if cell_number or local_min is not None:
        final_barcodes = set([
            x for x, y in cell_barcode_counts.items() if y > threshold])
    else:
        final_barcodes = None

    if plotfile_prefix:

        # colour-blind friendly colours - https://gist.github.com/thriveth/8560036
        CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                          '#f781bf', '#a65628', '#984ea3',
                          '#999999', '#e41a1c', '#dede00']
        user_line = mlines.Line2D(
            [], [], color=CB_color_cycle[0], ls="dashed",
            markersize=15, label='User-defined')
        selected_line = mlines.Line2D(
            [], [], color=CB_color_cycle[0], ls="dashed", markersize=15, label='Selected')
        rejected_line = mlines.Line2D(
            [], [], color=CB_color_cycle[3], ls="dashed", markersize=15, label='Rejected')

        # make density plot
        fig = plt.figure()
        fig1 = fig.add_subplot(111)
        fig1.plot(xx, density(xx), 'k')
        fig1.set_xlabel("Count per cell (log10)")
        fig1.set_ylabel("Density")

        if cell_number:
            fig1.axvline(np.log10(threshold), ls="dashed", color=CB_color_cycle[0])
            lgd = fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                              handles=[user_line],
                              title="Cell threshold")

        elif local_min is None:  # no local_min was accepted
            for pos in xx[local_mins]:
                fig1.axvline(x=pos, ls="dashed", color=CB_color_cycle[3])
            lgd = fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                              handles=[selected_line, rejected_line],
                              title="Possible thresholds")
        else:
            for pos in xx[local_mins]:
                if pos == xx[local_min]:  # selected local minima
                    fig1.axvline(x=xx[local_min], ls="dashed", color=CB_color_cycle[0])
                else:
                    fig1.axvline(x=pos, ls="dashed", color=CB_color_cycle[3])

            lgd = fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                              handles=[selected_line, rejected_line],
                              title="Possible thresholds")

        fig.savefig("%s_cell_barcode_count_density.png" % plotfile_prefix,
                    bbox_extra_artists=(lgd,), bbox_inches='tight')

        # make knee plot
        fig = plt.figure()
        fig2 = fig.add_subplot(111)
        fig2.plot(range(0, len(counts)), np.cumsum(counts), c="black")

        xmax = len(counts)
        if local_min is not None:
            # reasonable maximum x-axis value
            xmax = min(len(final_barcodes) * 5, xmax)

        fig2.set_xlim((0 - (0.01 * xmax), xmax))
        fig2.set_xlabel("Rank")
        fig2.set_ylabel("Cumulative count")

        if cell_number:
            fig2.axvline(x=cell_number, ls="dashed", color=CB_color_cycle[0])
            lgd = fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                              handles=[user_line],
                              title="Cell threshold")

        elif local_min is None:  # no local_min was accepted
            for local_mins_count in local_mins_counts:
                fig2.axvline(x=local_mins_count, ls="dashed",
                             color=CB_color_cycle[3])
            lgd = fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                              handles=[selected_line, rejected_line],
                              title="Possible thresholds")

        else:
            for local_mins_count in local_mins_counts:
                if local_mins_count == len(final_barcodes):  # selected local minima
                    fig2.axvline(x=local_mins_count, ls="dashed",
                                 color=CB_color_cycle[0])
                else:
                    fig2.axvline(x=local_mins_count, ls="dashed",
                                 color=CB_color_cycle[3])

            lgd = fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                              handles=[selected_line, rejected_line],
                              title="Possible thresholds")

        fig.savefig("%s_cell_barcode_knee.png" % plotfile_prefix,
                    bbox_extra_artists=(lgd,), bbox_inches='tight')

        if local_min is not None:
            colours_selected = [CB_color_cycle[0] for x in range(0, len(final_barcodes))]
            colours_rejected = ["black" for x in range(0, len(counts)-len(final_barcodes))]
            colours = colours_selected + colours_rejected
        else:
            colours = ["black" for x in range(0, len(counts))]

        fig = plt.figure()
        fig3 = fig.add_subplot(111)
        fig3.scatter(x=range(1, len(counts)+1), y=counts,
                     c=colours, s=10, linewidths=0)
        fig3.loglog()
        fig3.set_xlim(0, len(counts)*1.25)
        fig3.set_xlabel('Barcode index')
        fig3.set_ylabel('Count')

        if cell_number:
            fig3.axvline(x=cell_number, ls="dashed", color=CB_color_cycle[0])
            lgd = fig3.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                              handles=[user_line],
                              title="Cell threshold")
        elif local_min is None:  # no local_min was accepted
            for local_mins_count in local_mins_counts:
                fig3.axvline(x=local_mins_count, ls="dashed",
                             color=CB_color_cycle[3])
            lgd = fig3.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                              handles=[selected_line, rejected_line],
                              title="Possible thresholds")
        else:
            for local_mins_count in local_mins_counts:
                if local_mins_count == len(final_barcodes):  # selected local minima
                    fig3.axvline(x=local_mins_count, ls="dashed",
                                 color=CB_color_cycle[0])
                else:
                    fig3.axvline(x=local_mins_count, ls="dashed",
                                 color=CB_color_cycle[3])

            lgd = fig3.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                              handles=[selected_line, rejected_line],
                              title="Possible thresholds")

        fig.savefig("%s_cell_barcode_counts.png" % plotfile_prefix,
                    bbox_extra_artists=(lgd,), bbox_inches='tight')

        if not cell_number:
            with U.openFile("%s_cell_thresholds.tsv" % plotfile_prefix, "w") as outf:
                outf.write("count\taction\n")
                for local_mins_count in local_mins_counts:
                    if local_min and local_mins_count == len(final_barcodes):
                        threshold_type = "Selected"
                    else:
                        threshold_type = "Rejected"

                    outf.write("%s\t%s\n" % (local_mins_count, threshold_type))

    return final_barcodes
