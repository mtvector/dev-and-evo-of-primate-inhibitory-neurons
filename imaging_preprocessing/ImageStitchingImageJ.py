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
from IPython.core.display import display, HTML
import exifread
import networkx as nx
import sklearn
from sklearn import metrics
import matplotlib.path as mpltPath
from collections import defaultdict


#Test params
#dirname = os.path.expanduser('/wynton/group/ye/mtschmitz/images/MacaqueMotorCortex/E80-20180214_230_20200708#/scan.2020-07-09-00-00-28/')
#dirout =os.path.expanduser('~/tmp/')
#stitchchannel='DAPI'
#chosenstitchgroup='1'
#x1lim=0
#x2lim=1
#y1lim=0
#y2lim=1


def ImageJStitchImages(dirname,dirout,stitchchannel,chosenstitchgroup,x1lim,x2lim,y1lim,y2lim,minmax_displace,abs_displace):
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
    print(shapes)
    outline = mpltPath.Path(shapes[chosenstitchgroup])
    print(outline)
    outline=outline.transformed(matplotlib.transforms.Affine2D().scale(1.2))
    print(outline)
    tags_to_get=['SizeX','SizeY','ActualPositionX','ActualPositionY','Run Index','Index','"field" Index','AreaGrid AreaGridIndex','Name','<Image ID="Image:" Name','ActualPositionZ']
    def get_tag(x,tp):
        return(re.search(x+'='+'"([A-Za-z0-9_\./\\-]*)"',tp).group(1))

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
    print(imageFiles['ActualPositionX'])
    print(imageFiles['ActualPositionY'])
    imageFiles['SizeX']=imageFiles['SizeX'].astype(float)
    imageFiles['SizeY']=imageFiles['SizeY'].astype(float)
    imageFiles.sort_values(by=['"field" Index','Channel Name'],inplace=True)


    #Max size
    chif=imageFiles.loc[imageFiles['Channel Name']==stitchchannel,:]
    chif['x1pix']=0
    chif['x2pix']=0
    chif['y1pix']=0
    chif['y2pix']=0
    xsize=imageFiles['SizeX'].value_counts().idxmax()
    ysize=imageFiles['SizeY'].value_counts().idxmax()
    xend=len(chif.ActualPositionX.unique())*xsize
    yend=len(chif.ActualPositionY.unique())*ysize

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
    #nonzero median is the distance between images
    xmedian=np.median(xd[xd>5])
    ymedian=np.median(yd[yd>5])
    #for reference xsize-minoverlap=xmedian
    print('MEDIANS')
    print(xmedian,ymedian)
    
    import shapely
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    polygon = Polygon(shapes[chosenstitchgroup])
    #Buffer expands, scale doesn't work for expanding linear regions with no points
    buffgon = Polygon(polygon.buffer(min(xmedian,ymedian)).exterior)
    #polygon=shapely.affinity.scale(polygon,xfact=1.2,yfact=1.2)
    inside=[buffgon.contains(Point(x,y)) for x,y in list(zip(chif['ActualPositionX'],chif['ActualPositionY']))]
    
    matplotlib.pyplot.scatter(list(chif['ActualPositionX']),list(chif['ActualPositionY']))
    matplotlib.pyplot.scatter([x[0] for x in shapes[chosenstitchgroup]], [x[1] for x in shapes[chosenstitchgroup]])
    matplotlib.pyplot.scatter(buffgon.exterior.coords.xy[0],buffgon.exterior.coords.xy[1])
    matplotlib.pyplot.savefig(os.path.join(dirout,'BufferPolygon.png'))
    #inside = outline.contains_points(list(zip(chif['ActualPositionX'],chif['ActualPositionY'])))
    print(inside)
    print(list(zip(chif['ActualPositionX'],chif['ActualPositionY'])))
    chif=chif.loc[inside,:]
    
    #normalize positions so starts at 0,0
    chif['ActualPositionX']=chif['ActualPositionX']-chif['ActualPositionX'].min()
    chif['ActualPositionY']=chif['ActualPositionY']-chif['ActualPositionY'].min()
    
    xorder=dict(zip(np.sort(chif['ActualPositionX'].unique()),np.sort(chif['ActualPositionX'].unique().argsort())))
    yorder=dict(zip(np.sort(chif['ActualPositionY'].unique()),np.sort(chif['ActualPositionY'].unique().argsort())))

    for i in chif.index:
        x=chif.loc[i,'ActualPositionX']/xmedian#xorder[chif.loc[i,'ActualPositionX']]
        y=chif.loc[i,'ActualPositionY']/ymedian#yorder[chif.loc[i,'ActualPositionY']]
        print(x,y)
        x1=int((xsize*x)-(x*minoverlap))
        x2=x1+int(xsize)
        y1=int((ysize*y)-(y*minoverlap))
        y2=y1+int(ysize)
        chif.loc[i,'x1pix']=x1
        chif.loc[i,'x2pix']=x2
        chif.loc[i,'y1pix']=y1
        chif.loc[i,'y2pix']=y2


    #Incase specific stitchgroups are denoted in my normal way
    '''l=[]
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

    ###Getting stitch group
    if len(l)!=imageFiles.shape[0]:
        #add estimate pixel vals to df
        for i in chif.index:
            x=xorder[chif.loc[i,'ActualPositionX']]
            y=yorder[chif.loc[i,'ActualPositionY']]
            print(x,y)
            x1=int((xsize*x)-(x*minoverlap))
            x2=x1+int(xsize)
            y1=int((ysize*y)-(y*minoverlap))
            y2=y1+int(ysize)
            chif.loc[i,'x1pix']=x1
            chif.loc[i,'x2pix']=x2
            chif.loc[i,'y1pix']=y1
            chif.loc[i,'y2pix']=y2

  
        mat=sklearn.metrics.pairwise.euclidean_distances(chif.loc[:,['ActualPositionX','ActualPositionY']])
        np.fill_diagonal(mat,9999999999)
        sortmat=mat.copy()
        sortmat.sort(1)    
        chif['stitchgroup']='nan'
        #1.1 is larger than longest acceptable hypotenuse given reasonable aspect ratios
        G=nx.from_numpy_matrix((mat<=sortmat[:,1].max()).astype(int))
        co=nx.algorithms.components.connected_components(G)
        for i,sg in enumerate(co):
            chif.loc[chif.index[list(sg)],'stitchgroup']=str(i+1)
        
        
        
        
    else:
        chif['stitchgroup']=l

    
    imageFiles['stitchgroup']='nan'
    for i in chif['"field" Index']:
        imageFiles.loc[imageFiles['"field" Index']==i,'stitchgroup']=list(chif.loc[chif['"field" Index']==i,'stitchgroup'])[0]

    chif=chif.loc[chif['stitchgroup']==chosenstitchgroup,:]   
    imageFiles=imageFiles.loc[imageFiles['stitchgroup']==chosenstitchgroup,:]

    tmppath=os.path.expanduser('~/imagingmetadata.csv')
    if os.path.exists(tmppath):
        refdf=pd.read_csv(tmppath,sep='\t')
    else:
        refdf=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSYbvCJpS-GfRKuGgs2IBH7MD1KtDPDqs7ePqQJ1PyrMKp7f7z7ZpY4WtMFGPxU4mWbnRHgBl4PtaeH/pub?output=tsv&gid=1520792104',sep='\t')
        refdf.to_csv(tmppath,sep='\t')

    refdf.rename(columns={'Channel0':'GFP','Channel1':'DAPI','Channel2':'CY5','Channel3':'RFP'},inplace=True)
    imageFiles=pd.merge(imageFiles, refdf,  how='left', left_on=['DirName','stitchgroup'], right_on = ['DirName','SlidePosition.1isL'])
    imageFiles['gene']=list([imageFiles.loc[i,x] for i,x in enumerate(imageFiles['Channel Name'])])
    imageFiles['Channel Name']=imageFiles['gene']

    xpixdict=dict(zip(chif['"field" Index'],chif['x1pix']))
    ypixdict=dict(zip(chif['"field" Index'],chif['y1pix']))

    imageFiles=imageFiles.loc[imageFiles['"field" Index'].isin(chif['"field" Index']),:]
    imageFiles['x1pix']='nan'
    imageFiles['y1pix']='nan'
    imageFiles['x1pix']=[xpixdict[x] for x in imageFiles['"field" Index']]
    imageFiles['y1pix']=[ypixdict[x] for x in imageFiles['"field" Index']]

    
    if imageFiles.shape[0]<2:
        print('No Images!')
        return(None)

    imageFiles['xind']=[xorder[x] for x in imageFiles['ActualPositionX']]
    imageFiles['yind']=[yorder[x] for x in imageFiles['ActualPositionY']]


    # In[135]:


    channeldfs={}
    for c in imageFiles['Channel Name'].unique():
        channeldfs[c]=imageFiles.loc[imageFiles['Channel Name']==c,:]
        channeldfs[c].index=channeldfs[c]['"field" Index']
    '''
    #Take ordered list of items and break into blocks of size blocksize overlapping by 1
    #Used to break image into subblocks for faster stitching
    def array_to_overlapping_blocks(A,blocksize):
        import numpy as np
        size = blocksize
        step = blocksize-1
        return(np.array([A[i : i + size] for i in range(0, len(A), step)]))

        # In[14]:
    print('PIXEL POSITIONS')
    print('minoverlap',minoverlap)
    print(chif.loc[:,['x1pix','x2pix','y1pix','y2pix']],flush=True)
    
    
    xmax=chif['x2pix'].max()
    ymax=chif['y2pix'].max()

    inrange=(chif['x2pix']/xmax > x1lim)&(chif['x1pix']/xmax < x2lim)&(chif['y2pix']/ymax >y1lim)&(chif['y1pix']/ymax <y2lim)

    chif=chif.loc[inrange,:]

    xmin=chif['x1pix'].min()
    ymin=chif['y1pix'].min()

    chif['x1pix']=chif['x1pix']-xmin
    chif['x2pix']=chif['x2pix']-xmin
    chif['y1pix']=chif['y1pix']-ymin
    chif['y2pix']=chif['y2pix']-ymin
    xorder=dict(zip(np.sort(chif['ActualPositionX'].unique()),np.sort(chif['ActualPositionX'].unique().argsort())))
    yorder=dict(zip(np.sort(chif['ActualPositionY'].unique()),np.sort(chif['ActualPositionY'].unique().argsort())))
    chif['xind']=[xorder[x] for x in chif['ActualPositionX']]
    chif['yind']=[yorder[x] for x in chif['ActualPositionY']]

    #xblocks=array_to_overlapping_blocks(chif['xind'].unique(),blocksize)
    #yblocks=array_to_overlapping_blocks(chif['yind'].unique(),blocksize)
    

    infile=os.path.join(dirout,str(chosenstitchgroup)+'_'+stitchchannel+'.stitchy')
    def ijstitch(chdf,infile):
        #conda create -n imagej -c conda-forge openjdk=8 pyimagej
        #conda install -c conda-forge maven
        #conda install -c conda-forge pyjnius
        #conda install -c conda-forge imglyb
        import imagej
        ij = imagej.init(os.path.expanduser('~/utils/Fiji.app/'), new_instance=True)
        #ij = imagej.init('sc.fiji:fiji',new_instance=True)
        plugin = 'Grid/Collection stitching'
        args = {'type': '[Positions from file]', 'order': '[Defined by TileConfiguration]',
                'directory':'[]','layout_file':infile, 'regression_threshold': '0.0001','fusion_method':'[Linear Blending]',
                'max/avg_displacement_threshold': '0.20', 'absolute_displacement_threshold': '0.30','subpixel_accuracy':True, 
                #'downsample_tiles':True,'x':'.25', 'y':'.25','interpolation':'Bicubic average',"width":'512','height':'384','outputFile':infile+'.registered.txt',
                'compute_overlap': True,'use_virtual_input_images':True, 'computation_parameters': '[Save memory (but be slower)]','image_output': '[Fuse and display]'}
        #ij.py.run_plugin(plugin, args,ij1_style=False)
        #ij.py.run_macro('run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by TileConfiguration] directory=[] layout_file=/home/mt/tmp/HOECHST.stitchy fusion_method=Average regression_threshold=0 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 compute_overlap subpixel_accuracy downsample_tiles use_virtual_input_images computation_parameters=[Save memory (but be slower)] image_output=[Fuse and display] x=.25 y=.25 width=512 height=384 interpolation=Bicubic average");\n')
        #ij.py.run_script(language='js',script='importClass(Packages.ij.IJ);IJ.run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by TileConfiguration] directory=[] layout_file=/home/mt/tmp/HOECHST.stitchy fusion_method=[Linear Blending] regression_threshold=.01 max/avg_displacement_threshold=.250 absolute_displacement_threshold=.35 compute_overlap subpixel_accuracy downsample_tiles use_virtual_input_images computation_parameters=[Save memory (but be slower)] image_output=[Fuse and display] x=.2 y=.2 width=410 height=307 interpolation=Bicubic average");')
        ij.getContext().dispose()    

    def callstitching(infile,imagejpath='~/utils/Fiji.app/ImageJ-linux64',scriptpath='~/code/macaque-dev-brain/imaging/Stitching.js'):
        import sys
        import subprocess
        from subprocess import PIPE, STDOUT
        cmd=[os.path.expanduser(imagejpath),os.path.expanduser(scriptpath),os.path.expanduser(infile),str(minmax_displace),str(abs_displace)]
        cmd=' '.join(cmd)
        print(cmd,flush=True)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        (stdoutdata, stderrdata) = process.communicate()
        print(stdoutdata)

    def write_stitchy(chdf,infile):    
        with open(infile, 'w') as the_file:
            the_file.write('dim = 2\n')
            for i in chdf.index:
                cur=chdf.loc[i,:]
                the_file.write(cur['Path']+'; ; ('+ str(cur['x1pix'])+','+str(cur['y1pix'])+') \n')

    def read_stitchy(chdf,infile):
        print('read_stitchy:')
        print(infile)
        df=pd.read_csv(infile,skiprows=4,header=None,sep=';')
        df.loc[:,['ActualPositionX','ActualPositionY']]=[re.sub('\(|\)','',x).split(',') for x in df[2]]
        df.index=df[0]
        print(df)
        df.loc[:,['xind','yind']]=0
        for i in df.index:
            df.loc[i,['xind','yind']]=chdf.loc[chdf['FileName']==i,['xind','yind']].values[0]
        return(df)

    def do_ij_stitching(chdf,infile):
        print('do_ij_stitching')
        write_stitchy(chdf,infile)
        #ijstitch(chdf,infile)
        callstitching(infile)
        
        #This would have to be changed for non-evos microscopes
        with open(infile+'.registered.txt', "r") as sources:
            lines = sources.readlines()
        for d in ['d0','d1','d2','d3']:
            with open(re.sub('DAPI',d,infile)+str(minmax_displace)+'_'+str(abs_displace)+'.registered.txt', "w") as sources:
                for line in lines:
                    sources.write(re.sub('d1.TIF', d+'.TIF', line))
        return(read_stitchy(chdf,infile+'.registered.txt'))


    df=do_ij_stitching(chif,infile)
    with open(os.path.join(dirout,'dummyfile.dummy'), 'w') as fp: 
        pass
    ray.shutdown()
    return(df,imageFiles)

if __name__ == '__main__':
    ImageJStitchImages(sys.argv[1:])
