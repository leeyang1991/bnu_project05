# coding=gbk
'''
author: LiYang
Date: 20191112
Location: zzq BeiJing
Desctiption: PDSI TERRACLIMATE .nc pre-process
'''
import os
import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
# import subprocess
from scipy import interpolate
import shutil
from scipy import stats
import lon_lat_to_address
import multiprocessing
import copy
import to_raster
import pandas
import seaborn as sns
import datetime
from netCDF4 import Dataset

this_root = 'D:\\project05\\'


def nc_to_npz(nc,npz_out):
    # print(nc)
    # exit()
    ncin = Dataset(nc, 'r')

    # print ncin.variables

        # print(i)
    # exit()
    lat = ncin['lat']
    lon = ncin['lon']
    longitude_grid_distance = abs((lon[-1] - lon[0]) / (len(lon) - 1))
    latitude_grid_distance = -abs((lat[-1] - lat[0]) / (len(lat) - 1))
    longitude_start = lon[0]
    latitude_start = lat[0]

    time = ncin.variables['time']

    # print(time)
    # exit()
    # time_bounds = ncin.variables['time_bounds']
    # print(time_bounds)
    start = datetime.datetime(1900, 0o1, 0o1)
    # a = start + datetime.timedelta(days=5459)
    # print(a)
    # print(len(time_bounds))
    # print(len(time))
    # for i in time:
    #     print(i)
    # exit()
    nc_dic = {}
    flag = 0

    valid_year = []
    for i in range(1982,2016):
        valid_year.append(str(i))

    for i in tqdm(list(range(len(time)))):

        flag += 1
        # print(time[i])
        date = start + datetime.timedelta(days=int(time[i]))
        year = str(date.year)
        month = '%02d'%date.month
        # day = '%02d'%date.day
        date_str = year+month
        if not date_str[:4] in valid_year:
            continue
        # print(date_str)
        # print(date_str[:4])

        # continue
        grid = ncin.variables['spei'][i][::-1]
        grid = np.array(grid)
        # grid = np.ma.masked_where(grid>1000,grid)
        # plt.imshow(grid,'RdBu',vmin=-3,vmax=3)
        # plt.colorbar()
        # plt.show()
        nc_dic[date_str] = grid


    # print(len(nc_dic))
    # print(flag)
    np.savez(npz_out,**nc_dic)


def nc_to_tif(nc,outdir):
    # outdir = this_root+'SPEI\\tif\\'
    import analysis
    analysis.Tools().mk_dir(outdir,force=1)
    ncin = Dataset(nc, 'r')
    # print(ncin.variables)
    # exit()
    lat = ncin['lat']
    lon = ncin['lon']
    pixelWidth = lon[1]-lon[0]
    pixelHeight = lat[1]-lat[0]
    longitude_start = lon[0]
    latitude_start = lat[0]

    time = ncin.variables['time']

    # print(time)
    # exit()
    # time_bounds = ncin.variables['time_bounds']
    # print(time_bounds)
    start = datetime.datetime(1900, 0o1, 0o1)
    # a = start + datetime.timedelta(days=5459)
    # print(a)
    # print(len(time_bounds))
    # print(len(time))
    # for i in time:
    #     print(i)
    # exit()
    # nc_dic = {}
    flag = 0

    # valid_year = []
    # for i in range(1982, 2016):
    #     valid_year.append(str(i))
    # for i in time:
    #     print(i)
    # exit()
    for i in range(len(time)):

        flag += 1
        # print(time[i])
        date = start + datetime.timedelta(days=int(time[i]))
        year = str(date.year)
        month = '%02d' % date.month
        # day = '%02d'%date.day
        date_str = year + month
        # if not date_str[:4] in valid_year:
        #     continue
        # print(date_str)
        # exit()
        ndv = np.nan
        arr = ncin.variables['def'][i]
        for name,variable in list(ncin.variables.items()):
            for var in variable.ncattrs():
                if var == 'missing_value':
                    ndv = variable.getncattr(var)
        if np.isnan(ndv):
            raise IOError('no key missing_value')
        arr = np.array(arr)
        #### mask ####
        grid = arr == 32768
        # print ndv
        arr[grid] = -999999
        # arr[grid] = np.nan
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()
        newRasterfn = outdir+date_str+'.tif'
        to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,arr)
        # grid = np.ma.masked_where(grid>1000,grid)
        # plt.imshow(arr,'RdBu',vmin=-3,vmax=3)
        # plt.colorbar()
        # plt.show()
        # nc_dic[date_str] = arr
        # exit()

def main():
    # n=12
    # n='%02d'%n
    this_root = r'D:\project05\CWD\\'
    for year in tqdm(list(range(1982,2016))):
        nc = this_root + 'download\\TerraClimate_def_{}.nc'.format(year)
        out_dir = this_root+'tif\\'
        nc_to_tif(nc,out_dir)

if __name__ == '__main__':
    main()