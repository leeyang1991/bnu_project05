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
    ndvi_8km_dir = this_root+'PDSI\\tif\\'
    ndvi_0_5_dir = this_root+'PDSI\\tif_resample_0.5\\'
    mk_dir(ndvi_0_5_dir)
    for f in os.listdir(ndvi_8km_dir):
        if f.endswith('.tif'):
            print(f)
            arcpy.Resample_management(ndvi_8km_dir+f,ndvi_0_5_dir+f,"0.5","NEAREST")



def define_swe_projection():
    fdir = r'D:\project05\GLOBSWE\tif\SWE_avg\\'
    fdir = r'D:\project05\GLOBSWE\tif\SWE_max\\'
    for f in os.listdir(fdir):
        print f
        sr = arcpy.SpatialReference("North Pole Lambert Azimuthal Equal Area")
        arcpy.DefineProjection_management(fdir+f, sr)

def re_projection_swe():
    # f = r'D:\project05\GLOBSWE\tif\SWE_avg\198202.tif'
    # outf = r'D:\project05\GLOBSWE\test123.tif'
    # prj = r'D:\project05\MXD'
    fdir = r'D:\project05\GLOBSWE\tif\SWE_max\\'
    outdir = r'D:\project05\GLOBSWE\re_proj\SWE_max\\'
    for f in os.listdir(fdir):
        if f.endswith('.tif'):
            print f
            in_f = fdir+f
            outf = outdir+f
            arcpy.ProjectRaster_management(in_f, outf,
                                           r"C:\Program Files (x86)\ArcGIS\Desktop10.2\Reference Systems\georef1.prj", "BILINEAR", "0.5",
                                           "#", "#", "#")
    pass


def main():

    # define_swe_projection()
    re_projection_swe()
if __name__ == '__main__':
    main()