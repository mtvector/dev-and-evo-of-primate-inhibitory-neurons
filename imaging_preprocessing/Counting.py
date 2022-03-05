import math
from ij import IJ, ImagePlus
from ij.process import FloatProcessor
from ij.process import ImageProcessor as IP
import ij
import csv
import os
import ij.plugin.HyperStackConverter as HSC
from time import sleep
import sys
from ij.gui import Toolbar
from ij import WindowManager
import ij.gui.OvalRoi as OvalRoi
import ij.gui.TextRoi as TextRoi
import re
from ij import IJ
from ij.io import Opener
from ij.plugin import Concatenator
from jarray import array
import ij.gui.GenericDialog as GenericDialog 
import java.awt.Font as Font
import java.awt.Color as Color
import ij.plugin.Duplicator as dup
import re
from ij.io import TiffDecoder
import time
from ij.io import Opener
from ij import *
from ij.plugin.frame import RoiManager as rm
import gc
gc.collect()
IJ.setForegroundColor(254, 254, 254);
IJ.run("Line Width...", "line=14");

IJ.run("Collect Garbage");
inputDir1 = IJ.getDirectory("Choose image directory! ")
#inputDir1 = "/home/mt/Downloads/1/"
fileList1 = os.listdir(inputDir1); 
###tilestring=getString("Which Tilescan", "1")

project_prefix='Striatum_'

FileList=sorted(os.listdir(inputDir1))[::-1]
print(FileList)
stitchedFiles=[x for x in FileList if "stitched.TIF" in x]
stitchedNames=[x.split('_')[0] for x in stitchedFiles]

mergedFiles=[x for x in FileList if "merged.TIF" in x]

imp = Opener.openUsingBioFormats(os.path.join(inputDir1,mergedFiles[0]))
imp.close()


for im in stitchedFiles:
    if not "hoechst" in im.lower():
        IJ.open(inputDir1+im)
        IJ.run("8-bit", "")
        imp = IJ.getImage()
        imp.setTitle(im.split('_')[0])
IJ.run("Images to Stack", "name=Stack title=[] use")
imp = IJ.getImage()
#IJ.run("Multiply...", "value=2.0")
IJ.beep()
gd = GenericDialog("Which to threshold on?")
gd.addChoice("Gene:",stitchedNames,stitchedNames[0])
gd.showDialog()
mask_channel_name=gd.getChoices().get(0).getSelectedItem()
print(mask_channel_name)
mask_channel=1
for i,n in enumerate(stitchedNames):
	mask_channel=i+1
	if n == mask_channel_name:
		break
	

imp.setSlice(mask_channel)
IJ.run("Subtract Background...", "rolling=50")
IJ.run(imp, "Auto Threshold", "method=MaxEntropy white")
IJ.run(imp, "Watershed", "slice")
#IJ.run(imp, "Analyze Particles...", "size=80-81 circularity=0.20-1.00 display include summarize add slice")
IJ.run(imp, "Analyze Particles...", "size=60-350 circularity=0.20-1.00 display include summarize add slice")

rmg = rm.getInstance()
if not rmg:
  rmg = rm()
#rm.runCommand('reset')

ip = imp.getProcessor()
#rm.setVisible(False) 
 
# Instead of multimeasure command in RoiManager, get an array of ROIs.
ra = rmg.getRoisAsArray()
 
# loop through ROI array and do measurements. 
# here is only listing mean intensity of ROIs
# if you want more, see 
# http://rsbweb.nih.gov/ij/developer/api/ij/process/ImageStatistics.html
for r in ra:
	ip.setRoi(r)
	istats = ip.getStatistics()
	print istats.mean

print(rmg)

myWait = ij.gui.WaitForUserDialog('Select Region to count')
myWait.show()
imp.setSlice(mask_channel)
rmg.addRoi(imp.getRoi())
rmg.runCommand(imp,"Fill")

rmg.setSelectedIndexes(list(range(len(ra))))
rmg.runCommand(imp,"Combine")
rmg.addRoi(imp.getRoi())

ra = rmg.getRoisAsArray()
rmg.setSelectedIndexes([len(ra)-2,len(ra)-1])
#rm.runCommand(imp,"Deselect")
rmg.runCommand(imp,"AND")
rmg.runCommand('reset')
rmg.addRoi(imp.getRoi())
rmg.setSelectedIndexes([0])
rmg.runCommand(imp,"Split")
rmg.setSelectedIndexes([0])
rmg.runCommand(imp,"Delete")
rmg.runCommand(imp,"Deselect")
ra = rmg.getRoisAsArray()
print(list(range(len(ra))))
rmg.setSelectedIndexes(list(range(len(ra))))
rt = rmg.multiMeasure(imp)
rt.show("Results")
IJ.saveAs("Results",os.path.join(inputDir1,project_prefix+stitchedNames[mask_channel-1]+'_mask_stats.csv'))
IJ.run("Make Composite", "display=Composite")
IJ.run("Select All")
IJ.run("Stack to RGB")
IJ.run("Scale...", "x=0.2 y=0.2 interpolation=Bilinear average")
imp= IJ.getImage()
IJ.saveAs(imp, "png", os.path.join(inputDir1,project_prefix+stitchedNames[mask_channel-1]+'_mask_stats_guide.png'))
rmg.runCommand("Save", os.path.join(inputDir1,project_prefix+stitchedNames[mask_channel-1]+'_RoiSet.zip'))
IJ.run("Close")
IJ.run("Close")
IJ.run("Close")
IJ.run("Collect Garbage")
#IJ.deleteRows(0, 4550)
#gd = GenericDialog("Clockwise Rotation?")
#gd.addChoice("How Many Degrees",["0","90","180",'270'],"0")
#gd.showDialog()
#rotation = gd.getChoices().get(0).getSelectedItem()

first=True
#IJ.run("Stack to Images")
#IJ.run("Images to Stack", "name=Stack title=[] use")
#IJ.run("Cell Counter", "")



'''
while True:
	imp = IJ.getImage()
	
	IJ.run("Make Composite", "display=Composite")
	IJ.run("Select All")
	#if first:
		#IJ.run("Multiply...", "value=2.0")
	IJ.run("Stack to RGB")
	IJ.selectWindow("Stack")
	IJ.run("Stack to Images")
	IJ.run("Images to Stack", "name=Stack title=[] use")
	IJ.run("Collect Garbage")
	gc.collect()
	IJ.run("Cell Counter", "")
'''	
	