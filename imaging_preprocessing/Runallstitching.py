#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
#import subprocess
import sys
sys.path.insert(1, os.path.expanduser('~/code/macaque-dev-brain/imaging/'))
import ImageStitching
import random
import traceback
# In[2]:


import ray
ray.shutdown()
ray.shutdown()
ray.shutdown()


# In[ ]:


toppath = os.path.expanduser(sys.argv[1])

dirlist=os.listdir(os.path.join(toppath))
random.shuffle(dirlist)
dirlist=[f for f in dirlist if os.path.isdir(os.path.join(toppath,f))]
for f in dirlist:
    dirlist2=os.listdir(os.path.join(toppath,f))
    dirlist2=[d for d in dirlist2 if os.path.isdir(os.path.join(toppath,f,d))]
    for d in dirlist2:
        dirname=os.path.join(toppath,f,d)
        numunstitched=np.sum(['_R_p' in x for x in os.listdir(dirname)])
        ###LOOP THROUGH image areas
        for i in range(4):
            i+=1
            if numunstitched>10:
                print(dirname)
                #if not 'sagittal' in dirname:
                #    continue
                #if not os.path.exists(os.path.join(dirname,str(i))):
                #    os.mkdir(os.path.join(dirname,str(i)))
                #ImageStitching.CorrelateStitchImages(dirname,os.path.join(dirname,str(i)),'1',str(i))
                try:
                    if not os.path.exists(os.path.join(dirname,str(i))):
                        os.mkdir(os.path.join(dirname,str(i)))
                    print(os.path.join(dirname,str(i)))
                    allfiles=[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(dirname)) for f in fn]
                    if not any(['merged.stitchy' in x for x in allfiles]):
                        ImageStitching.CorrelateStitchImages(dirname,os.path.join(dirname,str(i)),'0',str(i))
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    print('fail')

