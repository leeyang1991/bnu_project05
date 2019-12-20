# coding=gbk
# from osgeo import gdal
import numpy as np
# import osr, ogr
# import gdal
# from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import re
import os
# import shpbuffer as sb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
from pyhdf.SD import SD, SDC
from mpl_toolkits.basemap import Basemap
import to_raster
import h5py
from tqdm import tqdm

this_root = 'D:/project05/'
def read_hdf():
    out_dir = this_root+'NDVI\\tif\\'
    fdir = this_root+'NDVI\\GIMMS_NDVI\\'
    for f in tqdm(os.listdir(fdir)):
    # for f in os.listdir(fdir):
        if not f.endswith('.hdf'):
            continue
        # if not f == 'ndvi3g_geo_v1_2015_0106.hdf':
        #     continue
        # print(f)
        year = f.split('.')[0].split('_')[-2]
        # print(year)
        f = h5py.File(fdir+f, 'r')
        # print(len(f))
        # continue
        base_month = f['time'][0]
        for i in range(len(f)):
            arr = f['ndvi'][i]
            lon = f['lon']
            lat = f['lat']
            time = f['time'][i]
            month = base_month+(time-base_month)/0.5
            # print(time_)
            # continue
            month = int(month)
            date = year+'%02d'%month
            # print(date)
            # continue
            newRasterfn = out_dir+'{}.tif'.format(date)
            longitude_start = lon[0]
            latitude_start = lat[0]
            pixelWidth = lon[1]-lon[0]
            pixelHeight = lat[1]-lat[0]
            arr = np.array(arr,dtype=float)
            # print(arr.dtype)
            grid = arr > - 10000
            arr[np.logical_not(grid)] = -999999
            # plt.imshow(arr)
            # plt.show()
            to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,arr)
            pass
def main():
    read_hdf()

if __name__ == '__main__':
    main()