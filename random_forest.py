# coding=gbk
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import datetime
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import sklearn
import scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import math
import analysis
from tqdm import tqdm

this_root = 'D:\\project05\\'

def prepare_Y():

    # 1 drought periods
    out_dir = this_root+'random_forest\\'
    analysis.Tools().mk_dir(out_dir)
    print 'loading f_recovery_time...'
    f_recovery_time = this_root+'arr\\recovery_time\\composite_3_modes\\composite_3_mode_recovery_time.npy'
    recovery_time = dict(np.load(f_recovery_time).item())
    print 'done'
    Y = {}
    flag = 0
    for pix in tqdm(recovery_time):
        vals = recovery_time[pix]
        for r_time,mark,date_range in vals:
            if r_time == None:  #r_time 为 TRUE
                continue
            flag += 1
            start = date_range[0]
            end = start + r_time
            key = pix+'_'+mark+'_'+'{}.{}'.format(start,end)
            print key
            Y[key] = r_time
    print flag
    # flag=1192218
    # flag=198075
    np.save(out_dir+'Y',Y)


def cal_monthly_mean(fdir,outdir):
    # outdir = this_root + 'TMP\\mon_mean_tif\\'
    # fdir = this_root + 'TMP\\tif\\'

    analysis.Tools().mk_dir(outdir)

    for m in tqdm(range(1, 13)):
        if m in range(6,10):
            continue
        arrs_sum = 0.
        for y in range(1982, 2016):
            date = '{}{}'.format(y, '%02d' % m)
            tif = fdir + date + '.tif'
            if not os.path.isfile(tif):
                continue
            arr, originX, originY, pixelWidth, pixelHeight = analysis.to_raster.raster2array(tif)
            arrs_sum += arr
        mean_arr = arrs_sum / len(range(1982, 2016))
        mean_arr = np.array(mean_arr, dtype=float)
        grid = mean_arr <= 0
        mean_arr[grid] = np.nan
        analysis.DIC_and_TIF().arr_to_tif(mean_arr, outdir + '%02d.tif' % m)



def prepare_X(x):
    Y_dic = dict(np.load(this_root+'random_forest\\Y.npy').item())
    per_pix_dir = this_root+'{}\\per_pix\\'.format(x)
    mean_dir = this_root+'{}\\mon_mean_tif\\'.format(x)

    # 1 加载所有原始数据
    all_dic = {}
    for f in tqdm(os.listdir(per_pix_dir), desc='1/3 loading per_pix_dir ...'):
        dic = dict(np.load(per_pix_dir+f).item())
        for pix in dic:
            all_dic[pix] = dic[pix]

    # 2 加载月平均数据
    mean_dic = {}
    for m in tqdm(range(1,13),desc='2/3 loading monthly mean ...'):
        m = '%02d'%m
        arr,originX,originY,pixelWidth,pixelHeight = analysis.to_raster.raster2array(mean_dir+m+'.tif')
        arr_dic = analysis.DIC_and_TIF().spatial_arr_to_dic(arr)
        mean_dic[m] = arr_dic

    # 3 找干旱事件对应的X的距平的平均或求和
    X = {}
    for key in tqdm(Y_dic,desc='3/3 generate X dic ...'):
        split_key = key.split('_')
        pix,mark,date_range = split_key
        split_date_range = date_range.split('.')
        start = split_date_range[0]
        end = split_date_range[1]
        start = int(start)
        end = int(end)
        drought_range = range(start, end)
        # print pix,mark,drought_range
        vals = all_dic[pix]
        selected_val = []
        for dr in drought_range:
            mon = dr % 12 + 1
            mon = '%02d'%mon
            mon_mean = mean_dic[mon][pix]
            val = vals[dr]
            juping = val - mon_mean
            selected_val.append(juping)
        if x == 'TMP':
            juping_mean = np.mean(selected_val)
        else:
            juping_mean = np.sum(selected_val)
        X[key] = juping_mean

    np.save(this_root+'random_forest\\{}'.format(x),X)




def check_ndvi():
    fdir = this_root+'NDVI\\per_pix_anomaly\\'
    key = '165.403'
    for f in tqdm(os.listdir(fdir)):
        if not '012' in f:
            continue
        dic = dict(np.load(fdir+f).item())
        if key in dic:
            val = dic[key]
            val = analysis.SMOOTH().forward_window_smooth(val)
            plt.plot(val)
            plt.show()


def main():
    # prepare_Y()
    prepare_X('PRE')
    # check_ndvi()
    # outdir = this_root + 'GLOBSWE\\monthly_SWE_max\\'
    # fdir = this_root + 'GLOBSWE\\tif\\SWE_max\\'
    # cal_monthly_mean(fdir,outdir)
    pass

if __name__ == '__main__':
    main()