#!/usr/bin/env python
# coding: utf-8

import os
import sys
import re
import tqdm
import matplotlib
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
import scipy
import ray
import psutil
import exifread
import networkx as nx
from collections import defaultdict
import cv2
import pynndescent
import networkx
import tifffile
import shapely
import traceback
#import rolling_ball
import sklearn
import sklearn.cluster

ray.shutdown()

'''def subtract_background(image, radius=50, light_bg=False):
        from skimage.morphology import white_tophat, black_tophat, disk
        str_el = disk(radius)
        if light_bg:
            return(black_tophat(image, str_el))
        else:
            return(white_tophat(image, str_el))
'''

pd.set_option("display.max_rows", None, "display.max_columns", None)

def RollingBallIJ(infile,imagejpath='~/utils/Fiji.app/ImageJ-linux64',scriptpath='~/code/macaque-dev-brain/imaging/SubtractBackground.js'):
        import sys
        import subprocess
        from subprocess import PIPE, STDOUT
        cmd=[os.path.expanduser(imagejpath),os.path.expanduser(scriptpath),os.path.expanduser(infile)]
        cmd=' '.join(cmd)
        print(cmd,flush=True)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        (stdoutdata, stderrdata) = process.communicate()
        print(stdoutdata)      
        
#stitchchannel should be a string ['0','1','2','3']
#chosenstitchgroup corresponds to tissue slide position via MTS slide naming conventions (when in doubt choose 1)
#x lims correspond to proportion of image to stitch, defaults to stitching entire image
#save multipage combines channels for each tile and saves them, for easier stitching
#save stitched saves the full stitched image for each channel (at constant size to downsample image)
#Constant size approx 300 MB for 16bit stitched at constant_size=200000000
#use_gene_name requires MTS directory structuring and a channel to gene table
def CorrelateStitchImages(dirname,dirout,stitchchannel,chosenstitchgroup,x1lim=-.05,x2lim=1.05,y1lim=-.05,y2lim=1.05,save_stitched=True,save_multipage=True,constant_size=250000000,roll_ball=True,use_gene_name=True,save_merged=False):
    #dirname = os.path.expanduser('/wynton/group/ye/mtschmitz/images/MacaqueMotorCortex2/P2sagittal1_27_20200828/TR1.2020-09-03-01-35-13/')
    #Test params
    '''dirname = os.path.expanduser('/media/mt/Extreme SSD/MacaqueMotorCortex2/P2_OB_20200805/TR1.2020-08-06-23-10-18')
    dirout =os.path.expanduser('~/tmp/')
    stitchchannel='1'
    chosenstitchgroup='1'
    x1lim=0
    x2lim=1
    y1lim=0
    y2lim=1
    save_merged=False
    save_stitched=False
    save_multipage=True
    use_gene_name=True
    '''
    
    print(dirname,flush=True)
    ray.shutdown()
    num_cpus = 1#psutil.cpu_count(logical=False)
    #set number of cores to use
    print('cpus:',num_cpus)
    ray.init(num_cpus=num_cpus)

    x1lim=float(x1lim)
    x2lim=float(x2lim)
    y1lim=float(y1lim)
    y2lim=float(y2lim)
    chosenstitchgroup=re.sub('TR_',"1",chosenstitchgroup)
    chosenstitchgroup=re.sub('TR',"",chosenstitchgroup)

    protocol=[x for x in os.listdir(dirname) if '.scanprotocol' in x][0]
    minoverlap=0
    with open(os.path.join(dirname,protocol), 'r') as f:
        for line in f:
            try:
                #print(line)
                if 'MinOverlapPixel' in line:
                    minoverlap=float(line.split('>')[1].split('<')[0])*1.1
            except:
                pass

    n_locations=0     
    counting=False
    with open(os.path.join(dirname,protocol), 'r') as f:
        for line in f:
            try:
                if 'LocationIds' in line:
                    counting=~counting
                elif counting:
                    n_locations+=1
            except:
                pass        


    n_location=1
    counting=False
    reference=False
    shapes=defaultdict(list)
    with open(os.path.join(dirname,protocol), 'r') as f:
        for line in f:
            try:
                if '<d2p1:ScanLocation>' in line:
                    counting=~counting
                if counting and '<d10p1:_x>' in line:
                    x=float(line.split('>')[1].split('<')[0])
                if counting and '<d10p1:_y>' in line:
                    y=float(line.split('>')[1].split('<')[0])
                    shapes[str(n_location)].append((x,y))
                if '<d2p1:ReferencePoint ' in line:   
                    counting=False
                    reference=True
                if reference and '<d10p1:_x>' in line:
                    xref=float(line.split('>')[1].split('<')[0])
                if reference and '<d10p1:_y>' in line:
                    yref=float(line.split('>')[1].split('<')[0]) 
                    reference=False
                    shapes[str(n_location)]=[(x+xref,y+yref) for x,y in shapes[str(n_location)]]
                    n_location+=1
                    xref=0
                    yref=0
            except:
                pass        
            
    tags_to_get=['SizeX','SizeY','ActualPositionX','ActualPositionY','Run Index','Index','"field" Index','TheC','AreaGrid AreaGridIndex','Name','<Image ID="Image:" Name','ActualPositionZ']
    def get_tag(x,tp):
        try:
            return(re.search(x+'='+'"([A-Za-z0-9_\./\\-]*)"',tp).group(1))
        except:
            print(x)
            return('')

    @ray.remote
    def getTiffMetadata(f):
        f = open(f, 'rb')
        # Return Exif tags
        tags = exifread.process_file(f)
        tp=tags['Image ImageDescription'].values
        d={}
        for tag in tags_to_get:
            d[tag]=get_tag(tag,tp)
        return(d)

    data = []
    for fname in sorted(os.listdir(dirname)):
        if fname.endswith(".TIF"):
            fpath = os.path.join(dirname, fname)
            path = os.path.normpath(dirname)
            d=path.split(os.sep)[-2]
            data.append((fname,d,fpath))
    imageFiles = pd.DataFrame(data, columns=['FileName','DirName','Path'])

    #normalize positions so starts at 0,0
    #imageFiles['ActualPositionX']=imageFiles['ActualPositionX']-imageFiles['ActualPositionX'].min()
    #imageFiles['ActualPositionY']=imageFiles['ActualPositionY']-imageFiles['ActualPositionY'].min()
    imageFiles=imageFiles.loc[['_R_' in x for x in imageFiles['FileName']],:]
    print(imageFiles,flush=True)
    l=[]
    for i in imageFiles.index:
        f = imageFiles.loc[i,'Path']
        #d=getTiffMetadata(f)
        l.append(getTiffMetadata.remote(f))
        #print(d)
        #imageFiles.loc[i,d.keys()]=list(d.values())

    metadf=pd.DataFrame(ray.get(l))
    metadf.index=imageFiles.index
    imageFiles=imageFiles.join(metadf)
    ray.shutdown()

    imageFiles.rename(columns={'<Image ID="Image:" Name':'Channel Name'},inplace=True)

    imageFiles['ActualPositionX']=imageFiles['ActualPositionX'].astype(float)
    imageFiles['ActualPositionY']=imageFiles['ActualPositionY'].astype(float)
    #print(imageFiles['ActualPositionX'])
    #print(imageFiles['ActualPositionY'])
    imageFiles['SizeX']=imageFiles['SizeX'].astype(float)
    imageFiles['SizeY']=imageFiles['SizeY'].astype(float)
    imageFiles.sort_values(by=['"field" Index','Channel Name'],inplace=True)
    print(imageFiles,flush=True)
    #Max size
    chif=imageFiles#.loc[imageFiles['Channel Name']==stitchchannel,:]
    chif['x1pix']=0
    chif['x2pix']=0
    chif['y1pix']=0
    chif['y2pix']=0
    xsize=imageFiles['SizeX'].value_counts().idxmax()
    ysize=imageFiles['SizeY'].value_counts().idxmax()
    xend=len(chif.ActualPositionX.unique())*xsize
    yend=len(chif.ActualPositionY.unique())*ysize

    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    polygon = Polygon(shapes[chosenstitchgroup])
    #Buffer expands, scale doesn't work for expanding linear regions with no points
    buffgon = Polygon(polygon.buffer(1).exterior)
    #polygon=shapely.affinity.scale(polygon,xfact=1.2,yfact=1.2)
    inside=[buffgon.contains(Point(x,y)) for x,y in list(zip(chif['ActualPositionX'],chif['ActualPositionY']))]

    matplotlib.pyplot.scatter(list(chif['ActualPositionX']),list(chif['ActualPositionY']))
    matplotlib.pyplot.scatter([x[0] for x in shapes[chosenstitchgroup]], [x[1] for x in shapes[chosenstitchgroup]])
    matplotlib.pyplot.scatter(buffgon.exterior.coords.xy[0],buffgon.exterior.coords.xy[1])
    matplotlib.pyplot.savefig(os.path.join(dirout,'BufferPolygon.png'))
    matplotlib.pyplot.close()
    chif=chif.loc[inside,:]

    #assume adjacent images are taken sequentially
    xd=[]
    yd=[]
    for i in range(len(chif['"field" Index'].unique())-1):
        indi=chif['"field" Index']==list(chif['"field" Index'])[i]
        indip1=chif['"field" Index']==list(chif['"field" Index'])[i+1]
        xdiff=np.abs(list(chif.loc[indi,'ActualPositionX'])[0]-list(chif.loc[indip1,'ActualPositionX'])[0])
        ydiff=np.abs(list(chif.loc[indi,'ActualPositionY'])[0]-list(chif.loc[indip1,'ActualPositionY'])[0])
        xd.append(xdiff)
        yd.append(ydiff)
    xd=np.array(xd)
    yd=np.array(yd)
    #nonzero (>10) mode is the distance between images
    #x=plt.hist(xd,bins=50)
    #xmedian=x[1][1:][x[0][1:]>0][0]
    #x=plt.hist(yd,bins=50)
    #ymedian=x[1][1:][x[0][1:]>0][0]
    #use meanshift to 1d cluster linear displacements
    #then take the 2nd (first nonzero) distance as the grid displacement
    ms=sklearn.cluster.MeanShift(bandwidth=10).fit(xd.reshape(-1, 1)).labels_
    xmedian=sorted([xd[ms==x].mean() for x in set(ms)])[1]
    if xmedian<40:
        xmedian=sorted([xd[ms==x].mean() for x in set(ms)])[2]
    ms=sklearn.cluster.MeanShift(bandwidth=10).fit(yd.reshape(-1, 1)).labels_
    ymedian=sorted([yd[ms==x].mean() for x in set(ms)])[1]
    if ymedian<40:
        ymedian=sorted([yd[ms==x].mean() for x in set(ms)])[2]

    #for reference xsize-minoverlap=xmedian
    print('MEDIANS')
    print(xmedian,ymedian)
    plt.close()

    #normalize positions so starts at 0,0
    chif['ActualPositionX']=chif['ActualPositionX']-chif['ActualPositionX'].min()
    chif['ActualPositionY']=chif['ActualPositionY']-chif['ActualPositionY'].min()

    xorder=dict(zip(np.sort(chif['ActualPositionX'].unique()),np.sort(chif['ActualPositionX'].unique().argsort())))
    yorder=dict(zip(np.sort(chif['ActualPositionY'].unique()),np.sort(chif['ActualPositionY'].unique().argsort())))

    for i in chif.index:
        x=chif.loc[i,'ActualPositionX']/xmedian#xorder[chif.loc[i,'ActualPositionX']]
        y=chif.loc[i,'ActualPositionY']/ymedian#yorder[chif.loc[i,'ActualPositionY']]
        x1=int((xsize*x)-(x*minoverlap))
        x2=x1+int(xsize)
        y1=int((ysize*y)-(y*minoverlap))
        y2=y1+int(ysize)
        chif.loc[i,'x1pix']=x1
        chif.loc[i,'x2pix']=x2
        chif.loc[i,'y1pix']=y1
        chif.loc[i,'y2pix']=y2


    cf=chif.loc[chif['TheC']==stitchchannel,:]
    cf['Image']=None
    for i in tqdm.tqdm(cf.index):
        img=cv2.imread(cf.loc[i,'Path'])[:,:,0].T
        cf.loc[i,'Image']=[img]
    print(chif,flush=True)
    #print(cf.loc[2,'Image'].shape)
    #plt.imshow(cf.loc[2,'Image'],origin='lower')
    print('cf')
    print(cf,flush=True)
    print(np.array(cf))
    index=pynndescent.NNDescent(cf.loc[:,['x1pix','y1pix']],n_neighbors=9)
    nn=index.query(cf.loc[:,['x1pix','y1pix']],k=9)[0][:,1:]

    g=networkx.DiGraph().to_undirected()
    for i,x in enumerate(nn):
        for j in x:
            g.add_edge(i,j)
    g=g.to_undirected()
    edge_list=np.array(list(g.edges))
    choices=np.random.choice(range(len(edge_list)),size=min(100,len(edge_list)),replace=False)
    edges_to_optimize=edge_list[choices]

    x1i=np.where(cf.columns=='x1pix')[0][0]
    x2i=np.where(cf.columns=='x2pix')[0][0]
    y1i=np.where(cf.columns=='y1pix')[0][0]
    y2i=np.where(cf.columns=='y2pix')[0][0]
    imgi=np.where(cf.columns=='Image')[0][0]

    #could be modified with inner loop to get
    #average correlation of all channels
    #Downside of course is 4x processing time
    def correlateOffsets(x):
        lower_bound=80
        upper_bound=220
        xoffset,yoffset=x[0],x[1]
        if (xoffset>upper_bound) or (yoffset>upper_bound):
            corr=2-(upper_bound-max(xoffset,yoffset))
            print(xoffset,yoffset,corr,flush=True)
            return(corr)
        if (xoffset<lower_bound) or (yoffset<lower_bound):
            corr=2+(lower_bound-min(xoffset,yoffset))
            print(xoffset,yoffset,corr,flush=True)
            return(corr)
        for i in cf.index:
            x=cf.loc[i,'ActualPositionX']/xmedian#xorder[chif.loc[i,'ActualPositionX']]
            y=cf.loc[i,'ActualPositionY']/ymedian#yorder[chif.loc[i,'ActualPositionY']]
            x1=int((xsize*x)-x*int(xoffset))
            x2=x1+int(xsize)
            y1=int((ysize*y)-y*int(yoffset))
            y2=y1+int(ysize)
            cf.loc[i,'x1pix']=x1
            cf.loc[i,'x2pix']=x2
            cf.loc[i,'y1pix']=y1
            cf.loc[i,'y2pix']=y2

        i_vect=[]
        j_vect=[]
        
        for i,j in edges_to_optimize:    
            ix1i=cf.iloc[i,x1i]
            iy1i=cf.iloc[i,y1i]
            ix2i=cf.iloc[i,x2i]
            iy2i=cf.iloc[i,y2i]
            jx1i=cf.iloc[j,x1i]
            jy1i=cf.iloc[j,y1i]
            jx2i=cf.iloc[j,x2i]
            jy2i=cf.iloc[j,y2i]

            p = Polygon([(ix1i,iy1i),
                         (ix2i,iy1i),
                         (ix2i,iy2i),
                         (ix1i,iy2i)])

            q = Polygon([(jx1i,jy1i),
                         (jx2i,jy1i),
                         (jx2i,jy2i),
                         (jx1i,jy2i)])

            if p.intersects(q):
                pqi=p.intersection(q)
                #x1,y1,x2,y2
                bounds=pqi.bounds
                ix1b,iy1b,ix2b,iy2b=bounds[0]-ix1i,bounds[1]-iy1i,bounds[2]-ix2i,bounds[3]-iy2i
                jx1b,jy1b,jx2b,jy2b=bounds[0]-jx1i,bounds[1]-jy1i,bounds[2]-jx2i,bounds[3]-jy2i

                rxi=np.intersect1d(np.arange(ix1i,ix2i),np.arange(bounds[0],bounds[2]))-ix1i
                rxj=np.intersect1d(np.arange(jx1i,jx2i),np.arange(bounds[0],bounds[2]))-jx1i
                ryi=np.intersect1d(np.arange(iy1i,iy2i),np.arange(bounds[1],bounds[3]))-iy1i
                ryj=np.intersect1d(np.arange(jy1i,jy2i),np.arange(bounds[1],bounds[3]))-jy1i
                rxi=rxi.astype(int)
                rxj=rxj.astype(int)
                ryi=ryi.astype(int)
                ryj=ryj.astype(int)

                if len(rxi)>0 and len(rxj)>0:
                    i_vect.append(list(cf.iloc[i,imgi][rxi,ryi[:,np.newaxis]].flatten()))
                    j_vect.append(list(cf.iloc[j,imgi][rxj,ryj[:,np.newaxis]].flatten()))
                    #plt.imshow(cf.iloc[i,imgi][rxi,ryi[:,np.newaxis]],origin='lower')
                    #plt.show()
                    #plt.imshow(cf.iloc[j,imgi][rxj,ryj[:,np.newaxis]],origin='lower')
                    #plt.show()
                #x,y = p.exterior.xy
                #plt.plot(x,y)
                #x,y = q.exterior.xy
                #plt.plot(x,y)
                #x,y = pqi.exterior.xy
                #plt.plot(x,y)
                #plt.show()

        i_vect = np.array([item for sublist in i_vect for item in sublist])
        j_vect = np.array([item for sublist in j_vect for item in sublist])
        approx_inds=np.random.choice(list(range(len(i_vect-1))),size=min(400000,len(i_vect-2)),replace=False)
        print(approx_inds,flush=True)
        corr=1-np.corrcoef(i_vect[approx_inds],j_vect[approx_inds])[1,0]
        #corr=1-scipy.stats.spearmanr(i_vect,j_vect)[0]
        if np.isnan(corr):
            corr=2
        print(xoffset,yoffset,corr,flush=True)
        return(corr)

    #opt=scipy.optimize.brute(correlateOffsets,(slice(30,500),slice(30,500)),full_output=True,disp=False,workers=num_cpus)
    #opt=scipy.optimize.minimize(correlateOffsets,(minoverlap*2.5,minoverlap*2.5),method='Nelder-Mead',options={'xtol':.49,'ftol':.01})
    opt=scipy.optimize.basinhopping(correlateOffsets,(minoverlap*2.5,minoverlap*2.5),niter=7,stepsize=200,minimizer_kwargs={'method':'Nelder-Mead','options':{'xtol':.49,'ftol':.01}})
    print('final:',opt)
    chif['final_identifier']=chif['TheC']
    if use_gene_name: 
        l=[]
        for x in chif['FileName']:
            if x.startswith('L_'):
                l.append('1')
                continue
            if x.startswith('R_'):
                l.append('2')
                continue
            if '_TR_' in x:
                l.append('1')
                continue
            if '_TD_' in x:
                l.append('1')
                continue
            if 'TR' in x or 'TD' in x:
                groupnum=re.sub('TR|TD','',re.search('TD[0-9]|TR[0-9]',x).group(0))
                l.append(groupnum)  
                continue
            l.append(chosenstitchgroup)

        chif['stitchgroup']=l
        tmppath=os.path.expanduser('~/imagingmetadata.csv')
        if os.path.exists(tmppath):
            refdf=pd.read_csv(tmppath,sep='\t')
        else:
            refdf=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSYbvCJpS-GfRKuGgs2IBH7MD1KtDPDqs7ePqQJ1PyrMKp7f7z7ZpY4WtMFGPxU4mWbnRHgBl4PtaeH/pub?output=tsv&gid=1520792104',sep='\t')
            refdf.to_csv(tmppath,sep='\t')

        refdf.rename(columns={'Channel0':'0','Channel1':'1','Channel2':'2','Channel3':'3'},inplace=True)
        chif=pd.merge(chif, refdf,  how='left', left_on=['DirName','stitchgroup'], right_on = ['DirName','SlidePosition.1isL'])
        chif['gene']=list([chif.loc[chif.index[i],x] for i,x in enumerate(chif['TheC'])])
        chif['final_identifier']=chif['gene']
    else:
        chif['final_identifier']=chif['TheC']


    xoffset=opt.x[0]
    yoffset=opt.x[1]
    #xoffset=218
    #yoffset=209

    for i in chif.index:
        x=chif.loc[i,'ActualPositionX']/xmedian#xorder[chif.loc[i,'ActualPositionX']]
        y=chif.loc[i,'ActualPositionY']/ymedian#yorder[chif.loc[i,'ActualPositionY']]
        x1=int((xsize*x)-x*int(xoffset))
        x2=x1+int(xsize)
        y1=int((ysize*y)-y*int(yoffset))
        y2=y1+int(ysize)
        chif.loc[i,'x1pix']=x1
        chif.loc[i,'x2pix']=x2
        chif.loc[i,'y1pix']=y1
        chif.loc[i,'y2pix']=y2
    print('before min subtract')    
    print(chif)
    xmin=chif['x1pix'].min()
    ymin=chif['y1pix'].min()

    chif['x1pix']=chif['x1pix']-xmin
    chif['x2pix']=chif['x2pix']-xmin
    chif['y1pix']=chif['y1pix']-ymin
    chif['y2pix']=chif['y2pix']-ymin

    def write_stitchy(chdf,infile,keyname='FileName'):    
        with open(infile, 'w') as the_file:
            the_file.write('dim = 2\n')
            for i in chdf.index:
                cur=chdf.loc[i,:]
                the_file.write(cur[keyname]+'; ; ('+ str(cur['x1pix'])+','+str(cur['y1pix'])+') \n')

    #for imagej merge on the fly
    for c in chif['final_identifier'].unique():
        cf=chif.loc[chif['final_identifier']==c,:]
        infile=os.path.join(dirout,str(chosenstitchgroup)+'_'+stitchchannel+'.stitchy')
        write_stitchy(cf,infile,keyname='FileName')
    
    print(chif)
    #Or write whole file         
    if save_stitched:
        for c in chif['final_identifier'].unique():
            cf=chif.loc[chif['final_identifier']==c,:]
            cf['Image']=None
            for i in cf.index:
                img=cv2.imread(cf.loc[i,'Path'])[:,:,0].T
                cf.loc[i,'Image']=[img]
            newimg=np.zeros((int(np.nanmax(cf['x2pix'])),int(np.nanmax(cf['y2pix']))), np.uint8)
            divisor=np.zeros((int(np.nanmax(cf['x2pix'])),int(np.nanmax(cf['y2pix']))), np.uint8)
            for i in cf.index:
                x1,x2,y1,y2=cf.loc[i,['x1pix','x2pix','y1pix','y2pix']]
                newimg[x1:x2,y1:y2]+=cf.loc[i,'Image']
                divisor[x1:x2,y1:y2]+=1
                
            cur_size=newimg.shape[0]*newimg.shape[1]
            if constant_size<cur_size:
                scale_percent = np.sqrt(constant_size/cur_size)  # percent of original size
                width = int(newimg.shape[1] * scale_percent)
                height = int(newimg.shape[0] * scale_percent)
                dim = (width, height)
                # resize image
                newimg = cv2.resize(newimg, dim, interpolation = cv2.INTER_LINEAR)
                divisor = cv2.resize(divisor, dim, interpolation = cv2.INTER_AREA)
                
            im=np.nan_to_num(np.divide(newimg,divisor).T,nan=0).astype(np.uint8)
            #from PIL import Image
            print('background subbing',flush=True)
            #import skimage
            #from skimage import morphology
            #im=im-skimage.morphology.rolling_ball(im,radius=100)
            #subtract_background(im,radius=100,light_bg=False)
            im=im.astype(np.uint16)
            tifffile.imsave(os.path.join(dirout,c+'_stitched.TIF'),im,compress=6)
            if roll_ball:
                RollingBallIJ(os.path.join(dirout,c+'_stitched.TIF'))
            print('background subbed',flush=True)

    '''
    if save_merged:
        imgs={}
        for c in sorted(chif['final_identifier'].unique()):
            cf=chif.loc[chif['final_identifier']==c,:]
            cf['Image']=None
            for i in tqdm.tqdm(cf.index):
                img=cv2.imread(cf.loc[i,'Path'])[:,:,0].T
                cf.loc[i,'Image']=[img]
            newimg=np.zeros((int(cf['x2pix'].max()),int(cf['y2pix'].max())), np.uint8)
            divisor=np.zeros((int(cf['x2pix'].max()),int(cf['y2pix'].max())), np.uint8)
            for i in cf.index:
                x1,x2,y1,y2=cf.loc[i,['x1pix','x2pix','y1pix','y2pix']]
                newimg[x1:x2,y1:y2]+=cf.loc[i,'Image']
                divisor[x1:x2,y1:y2]+=1
                imgs[c]=np.nan_to_num(np.divide(newimg,divisor).T,nan=0).astype(np.uint8)
        tifffile.imsave(os.path.join(dirout,'merged_stitched.TIF'),list(imgs.values()),metadata={'Test':'YES,No','Value':100},compress=6)
    '''

    if save_multipage:
        l=[]
        for f in tqdm.tqdm(cf['"field" Index'].unique()):
            print(f,flush=True)
            cf=chif.loc[chif['"field" Index']==f,:]
            cf=cf.sort_values(by='final_identifier',axis=0)
            cf['Image']=None
            for i in cf.index:
                img=cv2.imread(cf.loc[i,'Path'])[:,:,0]
                cf.loc[i,'Image']=[img]
            #print(cf)
            imgs={}
            for i in cf.index:
                imgs[cf.loc[i,'final_identifier']]=cf.loc[i,'Image']#
            metadata={'channel_names':','.join(list(imgs.keys()))}
            metadata.update(cf.loc[i,['DirName','SizeX','SizeY','ActualPositionX','ActualPositionY','"field" Index','ActualPositionZ','x1pix','y1pix']].astype(str).to_dict())
            tifffile.imsave(os.path.join(dirout,f+'_merged.TIF'),list(imgs.values()),
                            metadata=metadata,
                            compress=6)
            l.append([f+'_merged.TIF',cf.loc[i,'x1pix'],cf.loc[i,'y1pix']])
        infile=os.path.join(dirout,'_'.join(list(imgs.keys()))+'_merged.stitchy')
        write_stitchy(pd.DataFrame(l,columns=['FileName','x1pix','y1pix']),infile,keyname='FileName')
