# coding=gbk
'''
author: LiYang
Date: 20191108
Location: zzq BeiJing
Desctiption: SPEI pre-process
'''
import os
import zipfile
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

def downloadFILE(url,name):
    resp = requests.get(url=url,stream=True)
    #stream=True的作用是仅让响应头被下载，连接保持打开状态，
    content_size = int(resp.headers['Content-Length'])/1024        #确定整个安装包的大小
    with open(name, "wb") as f:
        # for data in tqdm(iterable=resp.iter_content(1024),total=content_size,unit='k',desc=name):
        for data in resp.iter_content(1024):
            f.write(data)


def kernel_download(params):
    year,date = params
    # url = 'http://www.globsnow.info/swe/archive_v2.0/'
    url = 'http://www.globsnow.info/swe/archive_v2.0/{}/L3B_monthly_SWE/' \
          'GlobSnow_SWE_L3B_monthly_{}_v2.0.nc.gz'.format(year,date)
    # print url
    name = '{}.gz'.format(date)
    out_dir = this_root+'GLOBSWE\\download\\'
    f = out_dir+name

    analysis.Tools().mk_dir(out_dir,force=True)
    if check_zip(f):
        downloadFILE(url, f)


def download():
    year_list = []
    for y in range(1982,2016):
        year_list.append(str(y))
    date_list = []
    for y in year_list:
        for mon in range(1,13):
            mon = '%02d'%mon
            date = y+mon
            date_list.append(date)
    params = []
    for i in date_list:
        y = i[:4]
        date = i
        params.append([y,date])
    # for i in params:
    #     kernel_download(i)
    analysis.MUTIPROCESS(kernel_download,params).run(408,'t',desc='downloading...')


def check_zip(path):
    '''
    :param path: *.zip
    :return: True 下载
    :return: False 不下载
    '''
    if not os.path.isfile(path):
        return True
    ZipFile = zipfile.ZipFile
    BadZipfile = zipfile.BadZipfile
    try:
        with ZipFile(path) as zf:
            return False

    except BadZipfile:
        os.remove(path)
        return True


def main():
    download()


if __name__ == '__main__':
    main()