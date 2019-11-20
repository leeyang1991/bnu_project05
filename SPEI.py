# coding=gbk
'''
author: LiYang
Date: 20191108
Location: zzq BeiJing
Desctiption: SPEI pre-process
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
import requests
import analysis

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
    start = datetime.datetime(1900, 01, 01)
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

    for i in tqdm(range(len(time))):

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
    analysis.Tools().mk_dir(outdir)
    ncin = Dataset(nc, 'r')

    lat = ncin['lat'][::-1]
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
    start = datetime.datetime(1900, 01, 01)
    # a = start + datetime.timedelta(days=5459)
    # print(a)
    # print(len(time_bounds))
    # print(len(time))
    # for i in time:
    #     print(i)
    # exit()
    # nc_dic = {}
    flag = 0

    valid_year = []
    for i in range(1982, 2016):
        valid_year.append(str(i))

    for i in tqdm(range(len(time))):

        flag += 1
        # print(time[i])
        date = start + datetime.timedelta(days=int(time[i]))
        year = str(date.year)
        month = '%02d' % date.month
        # day = '%02d'%date.day
        date_str = year + month
        if not date_str[:4] in valid_year:
            continue
        # print(date_str)
        arr = ncin.variables['spei'][i][::-1]
        arr = np.array(arr)
        grid = arr < 100
        arr[np.logical_not(grid)] = -999999
        newRasterfn = outdir+date_str+'.tif'
        to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,arr)
        # grid = np.ma.masked_where(grid>1000,grid)
        # plt.imshow(arr,'RdBu',vmin=-3,vmax=3)
        # plt.colorbar()
        # plt.show()
        # nc_dic[date_str] = arr
        # exit()


def downloadFILE(url,name):
    resp = requests.get(url=url,stream=True)
    #stream=True的作用是仅让响应头被下载，连接保持打开状态，
    content_size = int(resp.headers['Content-Length'])/1024        #确定整个安装包的大小
    with open(name, "wb") as f:
        # for data in tqdm(iterable=resp.iter_content(1024),total=content_size,unit='k',desc=name):
        for data in resp.iter_content(1024):
            f.write(data)

def kernel_download_spei(params):
    i,outdir = params
    interval = '%02d' % i
    url = 'http://digital.csic.es/bitstream/10261/153475/{}/spei{}.nc'.format(i + 2, interval)
    print url
    downloadFILE(url, outdir + 'spei{}.nc'.format('%02d'%i))


def download_spei():
    outdir = this_root + 'SPEI\\download_from_web\\'
    params = []
    for i in range(1,25):
        params.append([i,outdir])
    analysis.MUTIPROCESS(kernel_download_spei,params).run(6,'t')



def main():
    # n=12
    # n='%02d'%n
    # nc = this_root+'SPEI\\download_from_web\\spei{}.nc'.format(n)
    # out_dir = this_root+'SPEI\\tif\\SPEI_{}\\'.format(n)
    # npz = this_root+'SPEI\\npz\\spei_3'
    # nc_to_npz(nc,npz)
    # nc_to_tif(nc,out_dir)
    download_spei()

if __name__ == '__main__':
    main()