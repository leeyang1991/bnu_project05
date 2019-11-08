# coding=gbk

import os
import arcpy
from arcpy.sa import *


this_root = 'D:/project05/'


def mk_dir( dir, force=False):
    if not os.path.isdir(dir):
        if force == True:
            os.makedirs(dir)
        else:
            os.mkdir(dir)


def resample30m():
    ndvi_8km_dir = this_root+'NDVI\\tif\\'
    ndvi_0_5_dir = this_root+'NDVI\\tif_resample_0.5\\'
    mk_dir(ndvi_0_5_dir)
    for f in os.listdir(ndvi_8km_dir):
        if f.endswith('.tif'):
            print(f)
            arcpy.Resample_management(ndvi_8km_dir+f,ndvi_0_5_dir+f,"0.5","NEAREST")
            
def main():

    resample30m()

if __name__ == '__main__':
    main()