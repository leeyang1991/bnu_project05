# coding=utf-8
'''
author: LiYang
Date: 20191107
Location: zzq BeiJing
Desctiption
'''
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal
import time
import to_raster
import ogr, os, osr
from tqdm import tqdm

import datetime
import lon_lat_to_address
from scipy import stats, linalg
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
import imageio
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl

import multiprocessing
from multiprocessing.pool import ThreadPool as TPool
import copy_reg
import types
from scipy.stats import gamma as gam
import math

this_root = 'D:\\project05\\'
# this_root = 'D:\\ly\\project05\\'
# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 显示负号
plt.rcParams['axes.unicode_minus'] = False


landcover_types_dic = {
1:'ENF',
2:'EBF',
3:'DNF',
4:'DBF',
5:'MF',
6:'Close Shrublands',
7:'Open Shrublands',
8:'Woody Savannas',
9:'Savannas',
10:'Grasslands',
11:'Wetlands',
12:'Croplands',
13:'Urban',
14:'Natural Vegetation',
15:'Snow and Ice',
16:'Barren',
17:'Water',
18:'Ocean',
}



class Tools:
    '''
    小工具
    '''

    def __init__(self):
        pass

    def mk_dir(self, dir, force=False):

        if not os.path.isdir(dir):
            if force == True:
                os.makedirs(dir)
            else:
                os.mkdir(dir)

    def interp_1d(self, val,threashold):
        if len(val) == 0 or np.std(val) == 0:
            return [None]

        # 1、插缺失值
        x = []
        val_new = []
        flag = 0
        for i in range(len(val)):
            if val[i] >= threashold:
                flag += 1.
                index = i
                x = np.append(x, index)
                val_new = np.append(val_new, val[i])
        if flag / len(val) < 0.3:
            return [None]
        # if flag == 0:
        #     return
        interp = interpolate.interp1d(x, val_new, kind='nearest', fill_value="extrapolate")

        xi = range(len(val))
        yi = interp(xi)

        # 2、利用三倍sigma，去除离群值
        # print(len(yi))
        val_mean = np.mean(yi)
        sigma = np.std(yi)
        n = 3
        yi[(val_mean - n * sigma) > yi] = -999999
        yi[(val_mean + n * sigma) < yi] = 999999
        bottom = val_mean - n * sigma
        top = val_mean + n * sigma
        # plt.scatter(range(len(yi)),yi)
        # print(len(yi),123)
        # plt.scatter(range(len(yi)),yi)
        # plt.plot(yi)
        # plt.show()
        # print(len(yi))

        # 3、插离群值
        xii = []
        val_new_ii = []

        for i in range(len(yi)):
            if -999999 < yi[i] < 999999:
                index = i
                xii = np.append(xii, index)
                val_new_ii = np.append(val_new_ii, yi[i])

        interp_1 = interpolate.interp1d(xii, val_new_ii, kind='nearest', fill_value="extrapolate")

        xiii = range(len(val))
        yiii = interp_1(xiii)

        # for i in range(len(yi)):
        #     if yi[i] == -999999:
        #         val_new_ii = np.append(val_new_ii, bottom)
        #     elif yi[i] == 999999:
        #         val_new_ii = np.append(val_new_ii, top)
        #     else:
        #         val_new_ii = np.append(val_new_ii, yi[i])

        return yiii

    def interp_1d_1(self, val,threshold):
        # 不插离群值 只插缺失值
        if len(val) == 0 or np.std(val) == 0:
            return [None]

        # 1、插缺失值
        x = []
        val_new = []
        flag = 0
        for i in range(len(val)):
            if val[i] >= threshold:
                flag += 1.
                index = i
                x = np.append(x, index)
                val_new = np.append(val_new, val[i])
        if flag / len(val) < 0.3:
            return [None]
        interp = interpolate.interp1d(x, val_new, kind='nearest', fill_value="extrapolate")

        xi = range(len(val))
        yi = interp(xi)


        return yi



    def interp_nan(self,val):
        if len(val) == 0 or np.std(val) == 0:
            return [None]

        # 1、插缺失值
        x = []
        val_new = []
        flag = 0
        for i in range(len(val)):
            if not np.isnan(val[i]):
                flag += 1.
                index = i
                x = np.append(x, index)
                val_new = np.append(val_new, val[i])
        if flag / len(val) < 0.3:
            return [None]
        interp = interpolate.interp1d(x, val_new, kind='nearest', fill_value="extrapolate")

        xi = range(len(val))
        yi = interp(xi)


        return yi

        pass



    def smooth_convolve(self, x, window_len=11, window='hanning'):
        """
        1d卷积滤波
        smooth the data using a window with requested size.
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the beginning and end part of the output signal.
        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
        output:
            the smoothed signal
        example:
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        see also:
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """
        x = np.array(x)

        if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."

        if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        # return y
        return y[(window_len / 2 - 1):-(window_len / 2)]

    def smooth(self, x):
        # 后窗滤波
        # 滑动平均
        x = np.array(x)
        temp = 0
        new_x = []
        for i in range(len(x)):
            if i + 3 == len(x):
                break
            temp += x[i] + x[i + 1] + x[i + 2] + x[i + 3]
            new_x.append(temp / 4.)
            temp = 0
        return np.array(new_x)

    def forward_window_smooth(self, x, window=3):
        # 前窗滤波
        # window = window-1

        return SMOOTH().forward_window_smooth(x, window)

    def detrend_dic(self, dic):
        dic_new = {}
        for key in dic:
            vals = dic[key]
            if len(vals) == 0:
                dic_new[key] = []
                continue
            vals_new = signal.detrend(vals)
            dic_new[key] = vals_new

        return dic_new

    # def arrs_mean(self,arrs):
    #     for i in range(len(arrs[0])):
    #         for j in range
    #
    #
    #     pass


    def arr_mean(self, arr, threshold):
        grid = arr > threshold
        arr_mean = np.mean(arr[np.logical_not(grid)])
        return arr_mean

    def arr_mean_greater(self, arr, threshold):
        # mask greater
        grid_nan = np.isnan(arr)
        grid = np.logical_not(grid_nan)
        # print(grid)
        arr[np.logical_not(grid)] = 255
        grid = arr < threshold
        arr_mean = np.mean(arr[np.logical_not(grid)])
        return arr_mean

    def rename(self):
        fdir = this_root + 'GPP\\smooth_per_pix\\'
        flist = os.listdir(fdir)
        for f in flist:
            print(f)
            f_new = f.replace('gpp_', '')
            os.rename(fdir + f, fdir + f_new)

    def gen_lag_arr(self, arr1, arr2, lag):
        # example
        # 谁提前谁 0
        # 谁滞后谁 -1
        # lag 为正：arr1提前，arr2滞后
        # arr1 = [1,2,3,4]
        # arr2 = [1,2,3,4]

        # lag = 1
        # ret arr1 =   [1,2,3,4]
        # ret arr2 = [1,2,3,4]

        # lag = -1
        # arr1 = [1,2,3,4]
        # arr2 =   [1,2,3,4]

        lag = int(lag)
        arr1 = list(arr1)
        arr2 = list(arr2)
        if lag > 0:
            for _ in range(lag):
                arr1.pop(0)
                arr2.pop(-1)
        elif lag < 0:
            lag = -lag
            for _ in range(lag):
                arr1.pop(0)
                arr2.pop(-1)
        else:
            pass
        return arr1, arr2

    def gen_lag_arr_multiple_arrs(self, arrs, lag):
        # 谁提前谁 0
        # 谁滞后谁 -1
        lag = int(lag)
        arr_list = []
        for arr in arrs:
            arr_list.append(list(arr))
        if lag > 0:
            for _ in range(lag):
                for i, arr in enumerate(arr_list):
                    if i == 0:
                        arr.pop(0)
                        continue
                    arr.pop(-1)

            pass
        elif lag < 0:
            for _ in range(lag):
                for i, arr in enumerate(arr_list):
                    if i == 0:
                        arr.pop(0)
                        continue
                    arr.pop(-1)

        else:
            pass

        # print()
        # for i in arr_list:
        #     print(i)
        # time.sleep(1)
        # print('**********')

        return arr_list
        pass

    def partial_corr(self, df):
        """
        Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
        for the remaining variables in C.
        Parameters
        ----------
        df : array-like, shape (n, p)
            Array with the different variables. Each column of C is taken as a variable
        Returns
        -------
        P : array-like, shape (p, p)
            P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
            for the remaining variables in C.
        """

        C = np.asarray(df)
        p = C.shape[1]
        P_corr = np.zeros((p, p), dtype=np.float)
        for i in range(p):
            P_corr[i, i] = 1
            for j in range(i + 1, p):
                idx = np.ones(p, dtype=np.bool)
                idx[i] = False
                idx[j] = False
                beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
                beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

                res_j = C[:, j] - C[:, idx].dot(beta_i)
                res_i = C[:, i] - C[:, idx].dot(beta_j)

                corr = stats.pearsonr(res_i, res_j)[0]
                P_corr[i, j] = corr
                P_corr[j, i] = corr

        return P_corr

    def pix_to_address(self, pix):
        # 只适用于单个像素查看，不可大量for循环pix，存在磁盘重复读写现象
        if not os.path.isfile(this_root + 'arr\\pix_to_address_history.npy'):
            np.save(this_root + 'arr\\pix_to_address_history.npy', {0: 0})

        lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        # print(pix)
        lon, lat = lon_lat_dic[pix]
        print(lon, lat)

        history_dic = dict(np.load(this_root + 'arr\\pix_to_address_history.npy').item())

        if pix in history_dic:
            # print(history_dic[pix])
            return lon, lat, history_dic[pix]
        else:

            address = lon_lat_to_address.lonlat_to_address(lon, lat).decode('utf-8')
            key = pix
            val = address
            history_dic[key] = val
            np.save(this_root + 'arr\\pix_to_address_history.npy', history_dic)
            return lon, lat, address

    def pick_vals_from_2darray(self, array, index):
        # 2d
        ################# check zone #################
        # plt.imshow(array)
        # for r,c in index:
        #     # print(r,c)
        #     array[r,c] = 100
        # #     # exit()
        # plt.figure()
        # plt.imshow(array)
        # plt.show()
        ################# check zone #################
        picked_val = []
        for r, c in index:
            val = array[r, c]
            picked_val.append(val)
        picked_val = np.array(picked_val)
        return picked_val
        pass

    def pick_vals_from_1darray(self, arr, index):
        # 1d
        picked_vals = []
        for i in index:
            picked_vals.append(arr[i])
        return picked_vals

    def pick_min_indx_from_1darray(self, arr, indexs):
        min_index = 99999
        min_val = 99999
        # plt.plot(arr)
        # plt.show()
        for i in indexs:
            val = arr[i]
            # print val
            if val < min_val:
                min_val = val
                min_index = i
        return min_index

    def pick_growing_season_vals(self, arr, spei_min_index, growing_season_range):
        # 获取当前生长季的index
        year = int(spei_min_index / 12)
        select_vals = []
        select_index = []
        for i in range(-5, 6):
            s_index = spei_min_index + i
            s_mon = s_index % 12 + 1
            s_year = int(s_index / 12)
            if s_year == year and s_mon in growing_season_range:
                select_index.append(spei_min_index + i)
                select_vals.append(arr[spei_min_index + i])

            # s_mon = mon + i
            # if s_mon in growing_season_range:
            #     select_index.append(spei_min_index+i)
            #     select_vals.append(arr[spei_min_index+i])
        return select_index, select_vals

    def get_sta_position(self):
        f = open(this_root + 'conf\\sta_pos.csv', 'r')
        f.readline()
        lines = f.readlines()
        sta_pos_dic = {}
        for line in lines:
            line = line.split(',')
            sta_num = str(int(float(line[0])))
            sta_pos_dic[sta_num] = [float(line[1]), float(line[2])]
        return sta_pos_dic

    def point_to_shp(self, inputlist, outSHPfn):
        '''

        :param inputlist:

        # input list format
        # [[lon,lat,val],
        #      ...,
        # [lon,lat,val]]

        :param outSHPfn:
        :return:
        '''

        if len(inputlist) > 0:
            outSHPfn = outSHPfn + '.shp'
            fieldType = ogr.OFTReal
            # Create the output shapefile
            shpDriver = ogr.GetDriverByName("ESRI Shapefile")
            if os.path.exists(outSHPfn):
                shpDriver.DeleteDataSource(outSHPfn)
            outDataSource = shpDriver.CreateDataSource(outSHPfn)
            outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPoint)
            idField1 = ogr.FieldDefn('val', fieldType)
            outLayer.CreateField(idField1)
            for i in range(len(inputlist)):
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(inputlist[i][0], inputlist[i][1])
                featureDefn = outLayer.GetLayerDefn()
                outFeature = ogr.Feature(featureDefn)
                outFeature.SetGeometry(point)
                outFeature.SetField('val', inputlist[i][2])
                # 加坐标系
                spatialRef = osr.SpatialReference()
                spatialRef.ImportFromEPSG(4326)
                spatialRef.MorphToESRI()
                file = open(outSHPfn[:-4] + '.prj', 'w')
                file.write(spatialRef.ExportToWkt())
                file.close()

                outLayer.CreateFeature(outFeature)
                outFeature.Destroy()
            outFeature = None

    def gen_hi_scatter_plot_range_index(self, n):
        # 把HI按分成n份，分别存储pix位置
        Tools().mk_dir(this_root + 'arr\\gen_hi_scatter_plot_range_index\\')
        out_dic_name = this_root + 'arr\\gen_hi_scatter_plot_range_index\\' + str(n) + '.npy'
        if os.path.isfile(out_dic_name):
            return dict(np.load(out_dic_name).item())
        else:
            print(out_dic_name + ' is not existed, generating...')
            HI = np.load(this_root + 'arr\\fusion_HI\\fusion.npy')
            # arr = np.load(r'E:\project02\arr\plot_z_score_length\GPP\level1.npy')

            min_max = np.linspace(0.2, 1.6, n)

            # 建立空字典
            index_dic = {}
            for i in range(len(min_max)):
                if i + 1 == len(min_max):
                    break
                min_ = min_max[i]
                index_dic[min_] = []

            for i in tqdm(range(len(min_max))):
                if i + 1 == len(min_max):
                    break
                min_ = min_max[i]
                max_ = min_max[i + 1]
                pix_grid1 = min_ < HI
                pix_grid2 = HI < max_
                grid_and = np.logical_and(pix_grid1, pix_grid2)

                # selected = []
                for ii in range(len(grid_and)):
                    for jj in range(len(grid_and[ii])):
                        if grid_and[ii][jj]:
                            index_dic[min_].append([ii, jj])
            np.save(out_dic_name, index_dic)
            return index_dic

    def gen_lon_lat_address_dic(self):
        sta_pos_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())

        # sequential
        # sta_add_dic = {}
        # for pix in tqdm(sta_pos_dic):
        # # for pix in sta_pos_dic:
        #     lon,lat = sta_pos_dic[pix]
        #     # print(lat,lon)
        #     add = lon_lat_to_address.lonlat_to_address(lon, lat).decode('utf-8')
        #     # print(pix,lon,lat)
        #     # print(add)
        #     sta_add_dic[pix] = add
        # np.save(this_root+'conf\\sta_add_dic',sta_add_dic)

        # parallel
        params = []
        for pix in sta_pos_dic:
            params.append([sta_pos_dic, pix])
        result = MUTIPROCESS(self.kernel_gen_lon_lat_address_dic, params).run(process=50, process_or_thread='t',
                                                                              text='downloading')
        np.save(this_root + 'conf\\sta_add_dic', result)

    def kernel_gen_lon_lat_address_dic(self, params):
        sta_pos_dic, pix = params
        lon, lat = sta_pos_dic[pix]
        # print(lat,lon)
        add = lon_lat_to_address.lonlat_to_address(lon, lat).decode('utf-8')
        return add
        # print(pix,lon,lat)
        # print(add)

    def _cal_mean_false(self, array):
        # print array
        grid = array > 2000.
        picked_vals = []
        # print grid
        for i in range(len(grid)):
            if grid[i]:
                picked_vals.append(array[i])
        # print picked_vals
        # exit()
        if len(picked_vals) > 0:
            return np.mean(picked_vals)
        else:
            return None

    def _pick_growing_season(self, arr, hemi):
        growing_season = Recovery_time_winter().get_growing_months(hemi)
        picked_vals = []
        for i in range(len(arr)):
            mon = i % 12 + 1
            if mon in growing_season:
                picked_vals.append(arr[i])
        return np.array(picked_vals)

    def mask_ndvi_arr(self):
        # 1 cal NDVI mean
        fdir = this_root + 'NDVI\\per_pix\\'
        mask_arr = this_root + 'arr\\NDVI_mask_arr'
        pix_lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        spatial_dic = {}
        for f in tqdm(os.listdir(fdir)):
            dic = dict(np.load(fdir + f).item())
            for pix in dic:
                hemi = Recovery_time_winter().return_hemi(pix, pix_lon_lat_dic)

                val = dic[pix]
                val = np.array(val, dtype=float)
                picked_vals = self._pick_growing_season(val, hemi)
                mean = self._cal_mean_false(picked_vals)
                if mean:
                    spatial_dic[pix] = mean
                else:
                    spatial_dic[pix] = np.nan
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()
        print '\nsaving...'
        np.save(mask_arr, arr)

        # arr_list = []
        # for y in range(1982,2016):
        #     for m in range(5,10):
        #         date = str(y)+'%02d'%m
        #         f = fdir+date+'.tif'
        #         array,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(f)
        #         array = np.array(array,dtype=float)
        #
        #         arr_list.append(array)
        #         grid = array < - 999
        #         # array[grid] = np.nan
        #         plt.imshow(array)
        #         plt.show()


    def check_ndvi_perpix(self):
        fdir = this_root + 'NDVI\\per_pix\\'
        mask_arr = this_root + 'arr\\NDVI_mask_arr'
        pix_lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        spatial_dic = {}
        for f in tqdm(os.listdir(fdir)):
            dic = dict(np.load(fdir + f).item())
            for pix in dic:
                hemi = Recovery_time_winter().return_hemi(pix, pix_lon_lat_dic)

                val = dic[pix]
                val = np.array(val, dtype=float)
                picked_vals = self._pick_growing_season(val, hemi)
                mean = self._cal_mean_false(picked_vals)

        pass


    def filter_NDVI_valid_pix(self):
        mask_ndvi_arr = np.load(this_root + 'NDVI\\NDVI_growing_season_mean.npy')

        valid_pix = []
        for i in range(len(mask_ndvi_arr)):
            for j in range(len(mask_ndvi_arr[0])):
                val = mask_ndvi_arr[i][j]
                if not np.isnan(val):
                    pix = '%03d.%03d' % (i, j)
                    valid_pix.append(pix)
        valid_pix = set(valid_pix)

        return valid_pix
        pass

    def filter_tropical_valid_pix(self,arr):
        pass

class SMOOTH:
    '''
    一些平滑算法
    '''

    def __init__(self):

        pass

    def interp_1d(self, val):
        if len(val) == 0:
            return [None]

        # 1、插缺失值
        x = []
        val_new = []
        flag = 0
        for i in range(len(val)):
            if val[i] >= -10:
                flag += 1.
                index = i
                x = np.append(x, index)
                val_new = np.append(val_new, val[i])
        if flag / len(val) < 0.9:
            return [None]
        interp = interpolate.interp1d(x, val_new, kind='nearest', fill_value="extrapolate")

        xi = range(len(val))
        yi = interp(xi)

        # 2、利用三倍sigma，去除离群值
        # print(len(yi))
        val_mean = np.mean(yi)
        sigma = np.std(yi)
        n = 3
        yi[(val_mean - n * sigma) > yi] = -999999
        yi[(val_mean + n * sigma) < yi] = 999999
        bottom = val_mean - n * sigma
        top = val_mean + n * sigma
        # plt.scatter(range(len(yi)),yi)
        # print(len(yi),123)
        # plt.scatter(range(len(yi)),yi)
        # plt.plot(yi)
        # plt.show()
        # print(len(yi))

        # 3、插离群值
        xii = []
        val_new_ii = []

        for i in range(len(yi)):
            if -999999 < yi[i] < 999999:
                index = i
                xii = np.append(xii, index)
                val_new_ii = np.append(val_new_ii, yi[i])

        interp_1 = interpolate.interp1d(xii, val_new_ii, kind='nearest', fill_value="extrapolate")

        xiii = range(len(val))
        yiii = interp_1(xiii)

        # for i in range(len(yi)):
        #     if yi[i] == -999999:
        #         val_new_ii = np.append(val_new_ii, bottom)
        #     elif yi[i] == 999999:
        #         val_new_ii = np.append(val_new_ii, top)
        #     else:
        #         val_new_ii = np.append(val_new_ii, yi[i])

        return yiii

    def smooth_convolve(self, x, window_len=11, window='hanning'):
        """
        1d卷积滤波
        smooth the data using a window with requested size.
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the beginning and end part of the output signal.
        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
        output:
            the smoothed signal
        example:
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        see also:
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """
        x = np.array(x)

        if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."

        if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        # return y
        return y[(window_len / 2 - 1):-(window_len / 2)]

    def smooth(self, x):
        # 后窗滤波
        # 滑动平均
        x = np.array(x)
        temp = 0
        new_x = []
        for i in range(len(x)):
            if i + 3 == len(x):
                break
            temp += x[i] + x[i + 1] + x[i + 2] + x[i + 3]
            new_x.append(temp / 4.)
            temp = 0
        return np.array(new_x)

    def forward_window_smooth(self, x, window=3):
        # 前窗滤波
        # window = window-1
        # 不改变数据长度

        if window < 0:
            raise IOError('window must be greater than 0')
        elif window == 0:
            return x
        else:
            pass

        x = np.array(x)

        new_x = np.array([])
        # plt.plot(x)
        # plt.show()
        for i in range(len(x)):
            if i - window < 0:
                new_x = np.append(new_x, x[i])
            else:
                temp = 0
                for w in range(window):
                    temp += x[i - w]
                smoothed = temp / float(window)
                new_x = np.append(new_x, smoothed)
        return new_x

    def filter_3_sigma(self, arr_list):
        sum_ = []
        for i in arr_list:
            if i >= 0:
                sum_.append(i)
        sum_ = np.array(sum_)
        val_mean = np.mean(sum_)
        sigma = np.std(sum_)
        n = 3
        sum_[(val_mean - n * sigma) > sum_] = -999999
        sum_[(val_mean + n * sigma) < sum_] = -999999

        # for i in
        return sum_

        pass


class DIC_and_TIF:
    '''
    字典转tif
    tif转字典
    '''

    def __init__(self):

        pass

    def per_pix_dic_to_spatial_tif(self, mode, folder):

        outfolder = this_root + mode + '\\' + folder + '_tif\\'
        Tools().mk_dir(outfolder)
        tif_template = this_root + 'conf\\tif_template.tif'
        _, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        fdir = this_root + mode + '\\' + folder + '\\'
        flist = os.listdir(fdir)
        for f in flist:
            print(f)

        spatial_dic = {}
        for f in tqdm(flist):
            pix_dic = dict(np.load(fdir + f).item())
            for pix in pix_dic:
                vals = pix_dic[pix]
                spatial_dic[pix] = vals
        x = []
        y = []
        for key in spatial_dic:
            key_split = key.split('.')
            x.append(key_split[0])
            y.append(key_split[1])
        row = len(set(x))
        col = len(set(y))

        for date in tqdm(range(len(spatial_dic['0000.0000']))):
            spatial = []
            for r in range(row):
                temp = []
                for c in range(col):
                    key = '%03d.%03d' % (r, c)
                    val_pix = spatial_dic[key][date]
                    temp.append(val_pix)
                spatial.append(temp)
            spatial = np.array(spatial)
            grid = np.isnan(spatial)
            grid = np.logical_not(grid)
            spatial[np.logical_not(grid)] = -999999
            to_raster.array2raster(outfolder + '%03d.tif' % date, originX, originY, pixelWidth, pixelHeight, spatial)
            # plt.imshow(spatial)
            # plt.colorbar()
            # plt.show()

        # x = []
        # y = []
        # for key in spatial_dic:
        #     key_split = key.split('.')
        #     x.append(key_split[0])
        #     y.append(key_split[1])
        # row = len(set(x))
        # col = len(set(y))
        # spatial = []
        # all_vals = []
        # for r in tqdm(range(row)):
        #     temp = []
        #     for c in range(col):
        #         key = '%03d.%03d' % (r, c)
        #         val_pix = spatial_dic[key]
        #         temp.append(val_pix)
        #         all_vals.append(val_pix)
        #     spatial.append(temp)

        pass

    def arr_to_tif(self, array, newRasterfn):
        # template
        tif_template = this_root + 'conf\\tif_template.tif'
        _, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        grid_nan = np.isnan(array)
        grid = np.logical_not(grid_nan)
        array[np.logical_not(grid)] = -999999
        to_raster.array2raster(newRasterfn, originX, originY, pixelWidth, pixelHeight, array)
        pass

    def arr_to_tif_GDT_Byte(self, array, newRasterfn):
        # template
        tif_template = this_root + 'conf\\tif_template.tif'
        _, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        grid_nan = np.isnan(array)
        grid = np.logical_not(grid_nan)
        array[np.logical_not(grid)] = 255
        to_raster.array2raster_GDT_Byte(newRasterfn, originX, originY, pixelWidth, pixelHeight, array)
        pass


    def spatial_arr_to_dic(self,arr):

        pix_dic = {}
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                pix = '%03d.%03d'%(i,j)
                val = arr[i][j]
                pix_dic[pix] = val

        return pix_dic


    def pix_dic_to_spatial_arr(self, spatial_dic):

        # x = []
        # y = []
        # for key in spatial_dic:
        #     key_split = key.split('.')
        #     x.append(key_split[0])
        #     y.append(key_split[1])
        # row = len(set(x))
        # col = len(set(y))
        tif_template = this_root + 'conf\\tif_template.tif'
        arr_template, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        row = len(arr_template)
        col = len(arr_template[0])
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = '%03d.%03d' % (r, c)
                if key in spatial_dic:
                    val_pix = spatial_dic[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        # hist = []
        # for v in all_vals:
        #     if not np.isnan(v):
        #         if 00<v<1.5:
        #             hist.append(v)

        spatial = np.array(spatial,dtype=float)
        return spatial
        # # plt.figure()
        # # plt.hist(hist,bins=100)
        # # plt.title(str(set_level))
        # plt.figure()
        # # spatial = np.ma.masked_where(spatial<0,spatial)
        # # spatial = np.ma.masked_where(spatial>2,spatial)
        # # plt.imshow(spatial,'RdBu_r',vmin=0.7 ,vmax=1.3)
        # plt.imshow(spatial, 'RdBu_r')
        # plt.colorbar()
        # # plt.title(str(set_level))
        # plt.show()

    def pix_dic_to_tif(self, spatial_dic, out_tif):

        x = []
        y = []
        for key in spatial_dic:
            key_split = key.split('.')
            x.append(key_split[0])
            y.append(key_split[1])
        row = len(set(x))
        col = len(set(y))
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = '%03d.%03d' % (r, c)
                if key in spatial_dic:
                    val_pix = spatial_dic[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        spatial = np.array(spatial)
        self.arr_to_tif(spatial, out_tif)

    def spatial_tif_to_lon_lat_dic(self):
        tif_template = this_root + 'conf\\SPEI.tif'
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        # print(originX, originY, pixelWidth, pixelHeight)
        # exit()
        pix_to_lon_lat_dic = {}
        for i in tqdm(range(len(arr))):
            for j in range(len(arr[0])):
                pix = '%03d.%03d' % (i, j)
                lon = originX + pixelWidth * j
                lat = originY + pixelHeight * i
                pix_to_lon_lat_dic[pix] = [lon, lat]
        print('saving')
        np.save(this_root + 'arr\\pix_to_lon_lat_dic', pix_to_lon_lat_dic)

    def void_spatial_dic(self):
        tif_template = this_root + 'conf\\tif_template.tif'
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        void_dic = {}
        for row in range(len(arr)):
            for col in range(len(arr[row])):
                key = '%03d.%03d' % (row, col)
                void_dic[key] = []
        return void_dic


class MUTIPROCESS:
    '''
    可对类内的函数进行多进程并行
    由于GIL，多线程无法跑满CPU，对于不占用CPU的计算函数可用多线程
    并行计算加入进度条
    '''

    def __init__(self, func, params):
        self.func = func
        self.params = params
        copy_reg.pickle(types.MethodType, self._pickle_method)
        pass

    def _pickle_method(self, m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)

    def run(self, process=-9999, process_or_thread='p', **kwargs):
        '''
        # 并行计算加进度条
        :param func: input a kenel_function
        :param params: para1,para2,para3... = params
        :param process: number of cpu
        :param thread_or_process: multi-thread or multi-process,'p' or 't'
        :param kwargs: tqdm kwargs
        :return:
        '''
        if 'text' in kwargs:
            kwargs['desc'] = kwargs['text']
            del kwargs['text']

        if process > 0:
            if process_or_thread == 'p':
                pool = multiprocessing.Pool(process)
            elif process_or_thread == 't':
                pool = TPool(process)
            else:
                raise IOError('process_or_thread key error, input keyword such as "p" or "t"')

            results = list(tqdm(pool.imap(self.func, self.params), total=len(self.params), **kwargs))
            pool.close()
            pool.join()
            return results
        else:
            if process_or_thread == 'p':
                pool = multiprocessing.Pool()
            elif process_or_thread == 't':
                pool = TPool()
            else:
                raise IOError('process_or_thread key error, input keyword such as "p" or "t"')

            results = list(tqdm(pool.imap(self.func, self.params), total=len(self.params), **kwargs))
            pool.close()
            pool.join()
            return results


class KDE_plot:

    def __init__(self):

        pass

    def reverse_colourmap(self, cmap, name='my_cmap_r'):
        """
        In:
        cmap, name
        Out:
        my_cmap_r
        Explanation:
        t[0] goes from 0 to 1
        row i:   x  y0  y1 -> t[0] t[1] t[2]
                       /
                      /
        row i+1: x  y0  y1 -> t[n] t[1] t[2]
        so the inverse should do the same:
        row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                       /
                      /
        row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
        """
        reverse = []
        k = []

        for key in cmap._segmentdata:
            k.append(key)
            channel = cmap._segmentdata[key]
            data = []

            for t in channel:
                data.append((1 - t[0], t[2], t[1]))
            reverse.append(sorted(data))

        LinearL = dict(zip(k, reverse))
        my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
        return my_cmap_r

    def makeColours(self, vals, cmap, reverse=0):
        norm = []
        for i in vals:
            norm.append((i - np.min(vals)) / (np.max(vals) - np.min(vals)))
        colors = []
        cmap = plt.get_cmap(cmap)
        if reverse:
            cmap = self.reverse_colourmap(cmap)
        else:
            cmap = cmap

        for i in norm:
            colors.append(cmap(i))
        return colors

    def plot_scatter(self, val1, val2, cmap='magma', reverse=0, s=0.3, title=''):

        kde_val = np.array([val1, val2])
        print('doing kernel density estimation... ')
        densObj = kde(kde_val)
        dens_vals = densObj.evaluate(kde_val)
        colors = self.makeColours(dens_vals, cmap, reverse=reverse)
        plt.figure()
        plt.title(title)
        plt.scatter(val1, val2, c=colors, s=s)


class Pre_Process:
    def __init__(self):
        fdir = this_root+'CCI\\0.5\\tif\\'
        per_pix = this_root+'CCI\\0.5\\per_pix\\'
        # anomaly = this_root+'NDVI\\per_pix_anomaly\\'
        # # Tools().mk_dir(outdir)
        self.data_transform(fdir,per_pix)
        # self.cal_anomaly(per_pix,anomaly)

        # self.check_ndvi_anomaly()
        # self.check_per_pix(per_pix)

        pass

    def do_data_transform(self):
        father_dir = this_root + 'SPEI\\tif\\'
        for spei_dir in os.listdir(father_dir):
            print spei_dir + '\n'
            interval = spei_dir[-2:]

            spei_dir_ = spei_dir.upper()[:4] + '_' + interval
            outdir = this_root + 'SPEI\\per_pix\\' + spei_dir_ + '\\'
            print outdir
            Tools().mk_dir(outdir)
            self.data_transform(father_dir + spei_dir + '\\', outdir)

    def data_transform(self, fdir, outdir):
        # 不可并行，内存不足
        Tools().mk_dir(outdir)
        # 将空间图转换为数组
        # per_pix_data
        flist = os.listdir(fdir)
        date_list = []
        for f in flist:
            if f.endswith('.tif'):
                date = f.split('.')[0]
                date_list.append(date)
        date_list.sort()
        all_array = []
        for d in tqdm(date_list, 'loading...'):
            # for d in date_list:
            for f in flist:
                if f.endswith('.tif'):
                    if f.split('.')[0] == d:
                        # print(d)
                        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(fdir + f)
                        all_array.append(array)

        row = len(all_array[0])
        col = len(all_array[0][0])

        void_dic = {}
        void_dic_list = []
        for r in range(row):
            for c in range(col):
                void_dic['%03d.%03d' % (r, c)] = []
                void_dic_list.append('%03d.%03d' % (r, c))

        # print(len(void_dic))
        # exit()
        for r in tqdm(range(row), 'transforming...'):
            for c in range(col):
                for arr in all_array:
                    val = arr[r][c]
                    void_dic['%03d.%03d' % (r, c)].append(val)

        # for i in void_dic_list:
        #     print(i)
        # exit()
        flag = 0
        temp_dic = {}
        for key in tqdm(void_dic_list, 'saving...'):
            flag += 1
            # print('saving ',flag,'/',len(void_dic)/100000)
            temp_dic[key] = void_dic[key]
            if flag % 10000 == 0:
                # print('\nsaving %02d' % (flag / 10000)+'\n')
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

    def kernel_cal_anomaly(self, params):
        fdir, f, save_dir = params
        pix_dic = dict(np.load(fdir + f).item())
        anomaly_pix_dic = {}
        for pix in pix_dic:
            ####### one pix #######
            vals = pix_dic[pix]
            # 清洗数据
            vals = Tools().interp_1d_1(vals,-3000)

            if len(vals) == 1:
                anomaly_pix_dic[pix] = []
                continue
            climatology_means = []
            climatology_std = []
            # vals = signal.detrend(vals)
            for m in range(1, 13):
                one_mon = []
                for i in range(len(pix_dic[pix])):
                    mon = i % 12 + 1
                    if mon == m:
                        one_mon.append(pix_dic[pix][i])
                mean = np.mean(one_mon)
                std = np.std(one_mon)
                climatology_means.append(mean)
                climatology_std.append(std)

            # 算法1
            # pix_anomaly = {}
            # for m in range(1, 13):
            #     for i in range(len(pix_dic[pix])):
            #         mon = i % 12 + 1
            #         if mon == m:
            #             this_mon_mean_val = climatology_means[mon - 1]
            #             this_mon_std_val = climatology_std[mon - 1]
            #             if this_mon_std_val == 0:
            #                 anomaly = -999999
            #             else:
            #                 anomaly = (pix_dic[pix][i] - this_mon_mean_val) / float(this_mon_std_val)
            #             key_anomaly = i
            #             pix_anomaly[key_anomaly] = anomaly
            # arr = pandas.Series(pix_anomaly)
            # anomaly_list = arr.to_list()
            # anomaly_pix_dic[pix] = anomaly_list

            # 算法2
            pix_anomaly = []
            for i in range(len(vals)):
                mon = i % 12
                std_ = climatology_std[mon]
                mean_ = climatology_means[mon]
                if std_ == 0:
                    anomaly = np.nan
                else:
                    anomaly = (vals[i] - mean_) / std_

                pix_anomaly.append(anomaly)
            # pix_anomaly = Tools().interp_1d_1(pix_anomaly,-100)
            # plt.plot(pix_anomaly)
            # plt.show()
            anomaly_pix_dic[pix] = pix_anomaly

        np.save(save_dir + f, anomaly_pix_dic)

    def cal_anomaly(self,fdir,save_dir):
        # fdir = this_root + 'NDVI\\per_pix\\'
        # save_dir = this_root + 'NDVI\\per_pix_anomaly\\'
        Tools().mk_dir(save_dir)
        flist = os.listdir(fdir)
        # flag = 0
        params = []
        for f in flist:
            # print(f)
            params.append([fdir, f, save_dir])

        # for p in params:
        #     print(p[1])
        #     self.kernel_cal_anomaly(p)
        MUTIPROCESS(self.kernel_cal_anomaly, params).run(process=6, process_or_thread='p',
                                                         text='calculating anomaly...')


    def check_ndvi_anomaly(self):
        fdir = this_root + 'NDVI\\per_pix\\'
        for f in os.listdir(fdir):
            dic = dict(np.load(fdir+f).item())

            for pix in tqdm(dic):
                val = dic[pix]
                std = np.std(val)
                if std == 0 or len(val) == 0:
                    continue
                # print val
                val = Tools().interp_1d_1(val,-3000)
                # print val
                if len(val) == 1:
                    continue
                plt.plot(val)
                plt.grid()
                plt.show()
        pass

    def check_per_pix(self,fdir):

        for f in os.listdir(fdir):
            dic = dict(np.load(fdir+f).item())
            for pix in dic:
                val = dic[pix]
                print pix,val


class Pick_Single_events():
    def __init__(self):
        # self.check_global_dic()
        # self.check_events()
        # interval = 3
        # self.pick_non_growing_season_events(interval)
        # self.pick_post_growing_season_events(interval)
        # self.check_growing_season()
        pass

    def run(self, interval):
        #
        # 1 找single event 前后24个月无严重干旱事件
        # 耗时：13 min
        self.pick(interval)

        # 2 在single event 的基础上，找在生长季的干旱事件，分南北半球和热带
        # 耗时：15 min
        self.pick_non_growing_season_events(interval)
        self.pick_pre_growing_season_events(interval)
        self.pick_post_growing_season_events(interval)
        # self.pick_growing_season_events(interval)

        # 3 合成南北半球和热带区域的生长季事件
        # self.composite_global_growing(interval)
        self.composite_global_non_growing(interval)
        self.composite_global_pre_growing(interval)
        self.composite_global_post_growing(interval)

    def run1(self,interval):
        self.pick_non_growing_season_events(interval)
        self.pick_pre_growing_season_events(interval)
        self.pick_post_growing_season_events(interval)


        pass


    def pick_plot(self):
        # 作为pick展示
        # 前36个月和后36个月无极端干旱事件
        n = 36
        spei_dir = this_root + 'SPEI\\per_pix\\SPEI_03\\'
        for f in os.listdir(spei_dir):
            if '015' not in f:
                continue
            print(f)
            spei_dic = dict(np.load(spei_dir + f).item())
            for pix in spei_dic:

                spei = spei_dic[pix]
                spei = Tools().interp_1d(spei)
                if len(spei) == 1 or spei[0] == -999999:
                    continue
                spei = Tools().forward_window_smooth(spei, 3)
                params = [spei, pix, n]
                events_dic, key = self.kernel_find_drought_period(params)

                events_4 = []  # 严重干旱事件
                for i in events_dic:
                    level, date_range = events_dic[i]
                    # print(level,date_range)
                    if level == 4:
                        events_4.append(date_range)

                for i in range(len(events_4)):
                    spei_v = self.get_spei_vals(spei, events_4[i])
                    plt.plot(events_4[i], spei_v, c='black', zorder=99)

                    if i - 1 < 0:  # 首次事件
                        if events_4[i][0] - n < 0 or events_4[i][-1] + n >= len(spei):  # 触及两边则忽略
                            continue
                        if len(events_4) == 1:
                            spei_v = self.get_spei_vals(spei, events_4[i])
                            plt.plot(events_4[i], spei_v, linewidth=6, c='g')
                        elif events_4[i][-1] + n <= events_4[i + 1][0]:
                            spei_v = self.get_spei_vals(spei, events_4[i])
                            plt.plot(events_4[i], spei_v, linewidth=6, c='g')
                        continue

                    # 最后一次事件
                    if i + 1 >= len(events_4):
                        if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= len(spei):
                            spei_v = self.get_spei_vals(spei, events_4[i])
                            plt.plot(events_4[i], spei_v, linewidth=6, c='g')

                        break

                    # 中间事件
                    if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= events_4[i + 1][0]:
                        spei_v = self.get_spei_vals(spei, events_4[i])
                        plt.plot(events_4[i], spei_v, linewidth=6, c='g')

                #################### PLOT ##################
                print(pix)
                lon, lat, add = Tools().pix_to_address(pix)
                print(add)
                plt.plot(spei, 'r')
                plt.title(add + '_{}.{}'.format(lon, lat))
                plt.plot(range(len(spei)), [-2] * len(spei), '--', c='black')
                plt.plot(range(len(spei)), [-0.5] * len(spei), '--', c='black')
                plt.grid()
                plt.show()
                print('******')
                #################### PLOT ##################
        pass

    def get_spei_vals(self, spei, indxs):
        picked_vals = []
        for i in indxs:
            picked_vals.append(spei[i])
        return picked_vals

    def get_min_spei_index(self, spei, indexs):
        min_index = 99999
        min_val = 99999
        for i in indexs:
            val = spei[i]
            if val < min_val:
                min_val = val
                min_index = i
        return min_index

    def index_to_mon(self, ind):

        # base_date = '198201'
        mon = ind % 12 + 1
        return mon

        pass

    def pick(self, interval, mode='SPEI'):
        # 前n个月和后n个月无极端干旱事件
        n = 24
        spei_dir = this_root + mode + '\\per_pix\\' + 'SPEI_{:0>2d}\\'.format(interval)
        # spei_dir = this_root+'PDSI\\per_pix\\'
        out_dir = this_root + mode + '\\single_events_{}\\'.format(n) + 'SPEI_{:0>2d}\\'.format(interval)
        # out_dir = this_root+'PDSI\\single_events\\'
        Tools().mk_dir(out_dir, force=True)
        for f in tqdm(os.listdir(spei_dir), 'file...'):
            # if not '005' in f:
            #     continue
            spei_dic = dict(np.load(spei_dir + f).item())
            single_event_dic = {}
            for pix in tqdm(spei_dic, f):
                spei = spei_dic[pix]
                spei = Tools().interp_1d(spei)
                if len(spei) == 1 or spei[0] == -999999:
                    single_event_dic[pix] = []
                    continue
                spei = Tools().forward_window_smooth(spei, 3)
                params = [spei, pix, mode]
                events_dic, key = self.kernel_find_drought_period(params)
                # for i in events_dic:
                #     print i,events_dic[i]
                # exit()
                events_4 = []  # 严重干旱事件
                for i in events_dic:
                    level, date_range = events_dic[i]
                    if level == 4:
                        events_4.append(date_range)

                # # # # # # # # # # # # # # # # # # # # # # #
                # 不筛选单次事件（前后n个月无干旱事件）
                single_event_dic[pix] = events_4
                # print events_4
                # # # # # # # # # # # # # # # # # # # # # # #

                # # # # # # # # # # # # # # # # # # # # # # #
                # # 筛选单次事件（前后n个月无干旱事件）
                # single_event = []
                # for i in range(len(events_4)):
                #     if i - 1 < 0:  # 首次事件
                #         if events_4[i][0] - n < 0 or events_4[i][-1] + n >= len(spei):  # 触及两边则忽略
                #             continue
                #         if len(events_4) == 1:
                #             single_event.append(events_4[i])
                #         elif events_4[i][-1] + n <= events_4[i + 1][0]:
                #             single_event.append(events_4[i])
                #         continue
                #
                #     # 最后一次事件
                #     if i + 1 >= len(events_4):
                #         if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= len(spei):
                #             single_event.append(events_4[i])
                #         break
                #
                #     # 中间事件
                #     if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= events_4[i + 1][0]:
                #         single_event.append(events_4[i])
                # single_event_dic[pix] = single_event
                # # # # # # # # # # # # # # # # # # # # # # #
            np.save(out_dir + f, single_event_dic)

    # def kernel_find_drought_period(self, params):
    #     # 根据不同干旱程度查找干旱时期
    #     pdsi = params[0]
    #     key = params[1]
    #     mode = params[2]
    #     drought_month = []
    #     for i, val in enumerate(pdsi):
    #         # if val < -0.5:# SPEI
    #         if val < -1:  # PDSI
    #             drought_month.append(i)
    #         else:
    #             drought_month.append(-99)
    #     # plt.plot(drought_month)
    #     # plt.show()
    #     events = []
    #     event_i = []
    #     for ii in drought_month:
    #         if ii > -99:
    #             event_i.append(ii)
    #         else:
    #             if len(event_i) > 3:
    #                 events.append(event_i)
    #                 event_i = []
    #             else:
    #                 event_i = []
    #     # print(len(pdsi))
    #     # print(event_i)
    #     if len(event_i) > 3:
    #         events.append(event_i)
    #
    #     # print(events)
    #
    #     # 去除两头小于0的index
    #     # events_new = []
    #     # for event in events:
    #     #     print(event)
    #     # exit()
    #
    #     flag = 0
    #     events_dic = {}
    #
    #     # 取两个端点
    #     for i in events:
    #         # print(i)
    #         # 去除两端pdsi值小于-0.5
    #         if 0 in i or len(pdsi) - 1 in i:
    #             continue
    #         new_i = []
    #         for jj in i:
    #             # print(jj)
    #             if jj - 1 >= 0:
    #                 new_i.append(jj - 1)
    #             else:
    #                 pass
    #         new_i.append(i[-1])
    #         if i[-1] + 1 < len(pdsi):
    #             new_i.append(i[-1] + 1)
    #         # print(new_i)
    #         # exit()
    #         flag += 1
    #         vals = []
    #         for j in new_i:
    #             try:
    #                 vals.append(pdsi[j])
    #             except:
    #                 print(j)
    #                 print('error')
    #                 print(new_i)
    #                 exit()
    #         # print(vals)
    #
    #         # if 0 in new_i:
    #         # SPEI
    #         if mode == 'SPEI':
    #             min_val = min(vals)
    #             if -1 <= min_val < -.5:
    #                 level = 1
    #             elif -1.5 <= min_val < -1.:
    #                 level = 2
    #             elif -2 <= min_val < -1.5:
    #                 level = 3
    #             elif min_val <= -2.:
    #                 level = 4
    #             else:
    #                 print('error')
    #                 print(vals)
    #                 print(min_val)
    #                 time.sleep(1)
    #                 continue
    #
    #         # PDSI
    #         elif mode == 'PDSI':
    #             min_val = min(vals)
    #             if -2 <= min_val < -1:
    #                 level = 1
    #             elif -3 <= min_val < -2:
    #                 level = 2
    #             elif -4 <= min_val < -3:
    #                 level = 3
    #             elif min_val <= -4.:
    #                 level = 4
    #             else:
    #                 print('error')
    #                 print(vals)
    #                 print(min_val)
    #                 time.sleep(1)
    #                 continue
    #
    #         else:
    #             raise IOError('mode {} error'.format(mode))
    #
    #         events_dic[flag] = [level, new_i]
    #         # print(min_val)
    #         # plt.plot(vals)
    #         # plt.show()
    #     # for key in events_dic:
    #     #     # print key,events_dic[key]
    #     #     if 0 in events_dic[key][1]:
    #     #         print(events_dic[key])
    #     # exit()
    #     return events_dic, key

    def split_winter(self):
        # 筛选 -30度 ~ 30度之间为无冬季
        # -30度以下生长季为11-3月
        # 30度以上生长季为5-9月
        pix_lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        save_dir = this_root + 'arr\\split_winter\\'
        Tools().mk_dir(save_dir)

        north_hemi_pixs = {}
        south_hemi_pixs = {}
        tropical_pixs = {}
        for pix in tqdm(pix_lon_lat_dic):
            lon, lat = pix_lon_lat_dic[pix]
            if lat <= -30:
                south_hemi_pixs[pix] = [lon, lat]
            elif -30 < lat < 30:
                tropical_pixs[pix] = [lon, lat]
            else:
                north_hemi_pixs[pix] = [lon, lat]

        np.save(save_dir + 'north_hemi_pixs', north_hemi_pixs)
        np.save(save_dir + 'south_hemi_pixs', south_hemi_pixs)
        np.save(save_dir + 'tropical_pixs', tropical_pixs)

        pass

    def pick_growing_season_events(self, interval):
        # north: 5-9
        # south: 11-3
        # tropical: 1-12

        interval = '%02d' % interval

        out_dir = this_root + 'SPEI\\pick_growing_season_events\\SPEI_{}\\'.format(interval)
        Tools().mk_dir(out_dir, force=True)
        print 'loading hemi pix'
        north_hemi_pix = dict(np.load(this_root + 'arr\\split_winter\\north_hemi_pixs.npy').item())
        south_hemi_pix = dict(np.load(this_root + 'arr\\split_winter\\south_hemi_pixs.npy').item())
        tropical_pix = dict(np.load(this_root + 'arr\\split_winter\\tropical_pixs.npy').item())
        print 'done'
        SPEI_dir = this_root + 'SPEI\\per_pix\\SPEI_{}\\'.format(interval)
        single_event_dir = this_root + 'SPEI\\single_events_24\\SPEI_{}\\'.format(interval)
        hemi_pix_dic = {'north_hemi_pix': north_hemi_pix, 'south_hemi_pix': south_hemi_pix,
                        'tropical_pix': tropical_pix}
        for pixes_dic in hemi_pix_dic:
            if pixes_dic == 'north_hemi_pix':
                growing_date_range = range(5, 10)
            elif pixes_dic == 'south_hemi_pix':
                growing_date_range = [11, 12, 1, 2, 3]
            else:
                growing_date_range = range(1, 13)
            hemi_dic = {}
            for f in tqdm(os.listdir(single_event_dir)):
                dic = dict(np.load(single_event_dir + f).item())
                spei_dic = dict(np.load(SPEI_dir + f).item())
                for pix in dic:
                    # print pix
                    # print dic[pix]
                    if pix in hemi_pix_dic[pixes_dic]:
                        val = dic[pix]
                        spei = spei_dic[pix]
                        smooth_window = 3
                        spei = Tools().forward_window_smooth(spei, smooth_window)

                        if len(val) > 0:
                            # print val
                            # plt.plot(spei)
                            # plt.show()
                            selected_date_range = []
                            for date_range in val:
                                # picked_vals = self.get_spei_vals(spei,date_range)
                                min_index = self.get_min_spei_index(spei, date_range)
                                mon = self.index_to_mon(min_index)
                                if mon in growing_date_range:
                                    # hemi_dic[pix] = date_range
                                    selected_date_range.append(date_range)
                            hemi_dic[pix] = selected_date_range
            np.save(out_dir + pixes_dic, hemi_dic)


    def remove_vals_from_list(self,arr):
        a = range(1,13)
        for i in arr:
            a.remove(i)
        return a


    def pick_non_growing_season_events(self, interval):
        interval = '%02d' % interval
        out_dir = this_root + 'SPEI\\pick_non_growing_season_events\\SPEI_{}\\'.format(interval)
        Tools().mk_dir(out_dir, force=True)
        tropical_pix = np.load(this_root+'NDVI\\tropical_pix.npy')
        growing_season_daterange = dict(np.load(this_root+'NDVI\\growing_season_index.npy').item())

        SPEI_dir = this_root + 'SPEI\\per_pix\\SPEI_{}\\'.format(interval)
        single_event_dir = this_root + 'SPEI\\single_events_24\\SPEI_{}\\'.format(interval)

        non_growing_season = {}
        for f in tqdm(os.listdir(single_event_dir)):
            dic = dict(np.load(single_event_dir + f).item())
            spei_dic = dict(np.load(SPEI_dir + f).item())
            for pix in dic:
                # exit()
                val = dic[pix]
                if len(val) == 0:
                    continue
                spei = spei_dic[pix]
                smooth_window = 3
                spei = Tools().forward_window_smooth(spei, smooth_window)

                if pix in tropical_pix:
                    growing_daterange = range(1,13)
                    # print growing_daterange
                elif pix in growing_season_daterange:
                    growing_daterange = growing_season_daterange[pix]
                    growing_daterange = self.remove_vals_from_list(growing_daterange)
                else:
                    growing_daterange = []

                selected_date_range = []
                for date_range in val:
                    # picked_vals = self.get_spei_vals(spei,date_range)
                    min_index = self.get_min_spei_index(spei, date_range)
                    mon = self.index_to_mon(min_index)
                    if mon in growing_daterange:
                        # hemi_dic[pix] = date_range
                        selected_date_range.append(date_range)
                non_growing_season[pix] = selected_date_range

        np.save(out_dir+'global',non_growing_season)


    def _pre(self,arr):
        # 南半球 生长季跨年
        if 1 in arr and 12 in arr:
            if list(arr) == [1, 2, 3, 4, 12]:
                pre = [1, 2, 12]
            elif list(arr) == [1, 2, 3, 11, 12]:
                pre = [11, 12, 1]
            elif list(arr) == [1, 2, 10, 11, 12]:
                pre = [10, 11, 12]
            elif list(arr) == [1, 9, 10, 11, 12]:
                pre = [9, 10, 11]
            else:
                print arr
                raise IOError('error')
        # 北半球 生长季不跨年
        else:
            pre = arr[:3]

        return pre

    def pick_pre_growing_season_events(self, interval):
        interval = '%02d' % interval
        out_dir = this_root + 'SPEI\\pick_pre_growing_season_events\\SPEI_{}\\'.format(interval)
        Tools().mk_dir(out_dir, force=True)
        tropical_pix = np.load(this_root + 'NDVI\\tropical_pix.npy')
        growing_season_daterange = dict(np.load(this_root + 'NDVI\\growing_season_index.npy').item())

        SPEI_dir = this_root + 'SPEI\\per_pix\\SPEI_{}\\'.format(interval)
        single_event_dir = this_root + 'SPEI\\single_events_24\\SPEI_{}\\'.format(interval)

        non_growing_season = {}
        for f in tqdm(os.listdir(single_event_dir)):
            dic = dict(np.load(single_event_dir + f).item())
            spei_dic = dict(np.load(SPEI_dir + f).item())
            for pix in dic:
                # exit()
                val = dic[pix]
                if len(val) == 0:
                    continue
                spei = spei_dic[pix]
                smooth_window = 3
                spei = Tools().forward_window_smooth(spei, smooth_window)

                if pix in tropical_pix:
                    growing_daterange = range(1, 13)
                    # print growing_daterange
                elif pix in growing_season_daterange:
                    growing_daterange = growing_season_daterange[pix]
                    growing_daterange = self._pre(growing_daterange)
                else:
                    growing_daterange = []
                # if len(growing_daterange)>0:
                #     print growing_daterange
                selected_date_range = []
                for date_range in val:
                    # picked_vals = self.get_spei_vals(spei,date_range)
                    min_index = self.get_min_spei_index(spei, date_range)
                    mon = self.index_to_mon(min_index)
                    if mon in growing_daterange:
                        # hemi_dic[pix] = date_range
                        selected_date_range.append(date_range)
                non_growing_season[pix] = selected_date_range

        np.save(out_dir + 'global', non_growing_season)


    def _post(self,arr):
        # 南半球 生长季跨年
        if 1 in arr and 12 in arr:
            if list(arr) == [1, 2, 3, 4, 12]:
                post = [2, 3, 4]
            elif list(arr) == [1, 2, 3, 11, 12]:
                post = [1, 2, 3]
            elif list(arr) == [1, 2, 10, 11, 12]:
                post = [1, 2, 12]
            elif list(arr) == [1, 9, 10, 11, 12]:
                post = [1, 11, 12]
            else:
                print arr
                raise IOError('error')
        # 北半球 生长季不跨年
        else:
            post = arr[-3:]

        return post

        pass


    def pick_post_growing_season_events(self, interval):
        interval = '%02d' % interval
        out_dir = this_root + 'SPEI\\pick_post_growing_season_events\\SPEI_{}\\'.format(interval)
        Tools().mk_dir(out_dir, force=True)
        tropical_pix = np.load(this_root + 'NDVI\\tropical_pix.npy')
        growing_season_daterange = dict(np.load(this_root + 'NDVI\\growing_season_index.npy').item())

        SPEI_dir = this_root + 'SPEI\\per_pix\\SPEI_{}\\'.format(interval)
        single_event_dir = this_root + 'SPEI\\single_events_24\\SPEI_{}\\'.format(interval)

        non_growing_season = {}
        for f in tqdm(os.listdir(single_event_dir)):
            dic = dict(np.load(single_event_dir + f).item())
            spei_dic = dict(np.load(SPEI_dir + f).item())
            for pix in dic:
                # exit()
                val = dic[pix]
                if len(val) == 0:
                    continue
                spei = spei_dic[pix]
                smooth_window = 3
                spei = Tools().forward_window_smooth(spei, smooth_window)

                if pix in tropical_pix:
                    growing_daterange = range(1, 13)
                    # print growing_daterange
                elif pix in growing_season_daterange:
                    growing_daterange = growing_season_daterange[pix]
                    growing_daterange = self._post(growing_daterange)
                else:
                    growing_daterange = []
                # if len(growing_daterange)>0:
                #     print growing_daterange
                selected_date_range = []
                for date_range in val:
                    # picked_vals = self.get_spei_vals(spei,date_range)
                    min_index = self.get_min_spei_index(spei, date_range)
                    mon = self.index_to_mon(min_index)
                    if mon in growing_daterange:
                        # hemi_dic[pix] = date_range
                        selected_date_range.append(date_range)
                non_growing_season[pix] = selected_date_range

        np.save(out_dir + 'global', non_growing_season)

    # def composite_global_growing(self, interval):
    #     interval = '%02d' % interval
    #     outdir = this_root + 'SPEI\\pick_growing_season_events\\SPEI_{}\\'.format(interval)
    #     dic_north = this_root + 'SPEI\\pick_growing_season_events\\SPEI_{}\\north_hemi_pix.npy'.format(interval)
    #     dic_south = this_root + 'SPEI\\pick_growing_season_events\\SPEI_{}\\south_hemi_pix.npy'.format(interval)
    #     dic_tropical = this_root + 'SPEI\\pick_growing_season_events\\SPEI_{}\\tropical_pix.npy'.format(interval)
    #     north = dict(np.load(dic_north.format(interval)).item())
    #     south = dict(np.load(dic_south.format(interval)).item())
    #     tropical = dict(np.load(dic_tropical.format(interval)).item())
    #     global_dic = {}
    #     for zone in [north, south, tropical]:
    #         for pix in zone:
    #             global_dic[pix] = zone[pix]
    #
    #     # global_dic_valid = {}
    #     # for pix in global_dic:
    #     #     val = global_dic[pix]
    #     #     if len(val)>0:
    #     #         global_dic_valid[pix] = 1
    #     # arr = DIC_and_TIF().pix_dic_to_spatial_arr(global_dic_valid)
    #     # plt.imshow(arr)
    #     # plt.show()
    #     np.save(outdir + 'global_pix', global_dic)

    # def composite_global_non_growing(self, interval):
    #     interval = '%02d' % interval
    #     outdir = this_root + 'SPEI\\pick_non_growing_season_events\\SPEI_{}\\'.format(interval)
    #     dic_north = this_root + 'SPEI\\pick_non_growing_season_events\\SPEI_{}\\north_hemi_pix.npy'.format(interval)
    #     dic_south = this_root + 'SPEI\\pick_non_growing_season_events\\SPEI_{}\\south_hemi_pix.npy'.format(interval)
    #     dic_tropical = this_root + 'SPEI\\pick_non_growing_season_events\\SPEI_{}\\tropical_pix.npy'.format(interval)
    #     north = dict(np.load(dic_north.format(interval)).item())
    #     south = dict(np.load(dic_south.format(interval)).item())
    #     tropical = dict(np.load(dic_tropical.format(interval)).item())
    #     global_dic = {}
    #     for zone in [north, south, tropical]:
    #         for pix in zone:
    #             global_dic[pix] = zone[pix]
    #
    #     # global_dic_valid = {}
    #     # for pix in global_dic:
    #     #     val = global_dic[pix]
    #     #     if len(val)>0:
    #     #         global_dic_valid[pix] = 1
    #     # arr = DIC_and_TIF().pix_dic_to_spatial_arr(global_dic_valid)
    #     # plt.imshow(arr)
    #     # plt.show()
    #     np.save(outdir + 'global_pix', global_dic)

    # def composite_global_pre_growing(self, interval):
    #     interval = '%02d' % interval
    #     outdir = this_root + 'SPEI\\pick_pre_growing_season_events\\SPEI_{}\\'.format(interval)
    #     dic_north = this_root + 'SPEI\\pick_pre_growing_season_events\\SPEI_{}\\north_hemi_pix.npy'.format(interval)
    #     dic_south = this_root + 'SPEI\\pick_pre_growing_season_events\\SPEI_{}\\south_hemi_pix.npy'.format(interval)
    #     dic_tropical = this_root + 'SPEI\\pick_pre_growing_season_events\\SPEI_{}\\tropical_pix.npy'.format(interval)
    #     north = dict(np.load(dic_north.format(interval)).item())
    #     south = dict(np.load(dic_south.format(interval)).item())
    #     tropical = dict(np.load(dic_tropical.format(interval)).item())
    #     global_dic = {}
    #     for zone in [north, south, tropical]:
    #         for pix in zone:
    #             global_dic[pix] = zone[pix]
    #
    #     # global_dic_valid = {}
    #     # for pix in global_dic:
    #     #     val = global_dic[pix]
    #     #     if len(val)>0:
    #     #         global_dic_valid[pix] = 1
    #     # arr = DIC_and_TIF().pix_dic_to_spatial_arr(global_dic_valid)
    #     # plt.imshow(arr)
    #     # plt.show()
    #     np.save(outdir + 'global_pix', global_dic)

    # def composite_global_post_growing(self, interval):
    #     interval = '%02d' % interval
    #     outdir = this_root + 'SPEI\\pick_post_growing_season_events\\SPEI_{}\\'.format(interval)
    #     dic_north = this_root + 'SPEI\\pick_post_growing_season_events\\SPEI_{}\\north_hemi_pix.npy'.format(interval)
    #     dic_south = this_root + 'SPEI\\pick_post_growing_season_events\\SPEI_{}\\south_hemi_pix.npy'.format(interval)
    #     dic_tropical = this_root + 'SPEI\\pick_post_growing_season_events\\SPEI_{}\\tropical_pix.npy'.format(interval)
    #     north = dict(np.load(dic_north.format(interval)).item())
    #     south = dict(np.load(dic_south.format(interval)).item())
    #     tropical = dict(np.load(dic_tropical.format(interval)).item())
    #     global_dic = {}
    #     for zone in [north, south, tropical]:
    #         for pix in zone:
    #             global_dic[pix] = zone[pix]
    #
    #     # global_dic_valid = {}
    #     # for pix in global_dic:
    #     #     val = global_dic[pix]
    #     #     if len(val)>0:
    #     #         global_dic_valid[pix] = 1
    #     # arr = DIC_and_TIF().pix_dic_to_spatial_arr(global_dic_valid)
    #     # plt.imshow(arr)
    #     # plt.show()
    #     np.save(outdir + 'global_pix', global_dic)

    def check_global_dic(self):
        fdir = r'D:\project05\SPEI\\single_events_24\SPEI_03\\'
        pix_dic = {}
        for f in tqdm(os.listdir(fdir)):
            dic = dict(np.load(fdir + f).item())
            for pix in dic:
                pix_dic[pix] = 0
        valid_label = []
        for i in range(1, 13):
            valid_label.append('%02d' % i)
        fdir = r'D:\project05\SPEI\\single_events_24\\'
        for folder in tqdm(os.listdir(fdir)):
            label = folder.split('_')[1]
            if not label in valid_label:
                continue
            for f in os.listdir(fdir + folder):
                dic = dict(np.load(fdir + folder + '\\' + f).item())
                for pix in dic:
                    val = dic[pix]
                    if len(val) > 0:
                        pix_dic[pix] += 1.
                    else:
                        pass
                # else:
                #     pix_dic[pix] = 0
                #
        # f = fdir+'global_pix.npy'
        # dic = dict(np.load(f).item())
        # pixdic = {}
        # for pix in dic:
        #     if len(dic[pix]) > 0:
        #         pixdic[pix] = 1
        #     else:
        #         pixdic[pix] = 0

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(pix_dic)
        arr = np.array(arr)
        grid = arr == 0
        arr[grid] = 255
        tif = this_root + 'tif\\plot_gen_recovery_time\\20191218\\single_events.tif'
        DIC_and_TIF().arr_to_tif_GDT_Byte(arr, tif)
        # plt.imshow(arr)
        # plt.show()


    def kernel_check_events(self,params):
        
        fdir,f = params
        spatial_dic = {}
        dic = dict(np.load(fdir + f).item())
        for pix in dic:
            vals = dic[pix]
            if vals[0] == -999999:
                spatial_dic[pix] = -999999
                continue
            # lon, lat, address = Tools().pix_to_address(pix)

            # title = '{}_{}_{}_{}'.format(pix, lon, lat, address)
            # plt.title(title)

            # plt.plot(vals)

            vals = Tools().interp_1d_1(vals)
            # vals = Tools().interp_1d(vals)
            min_v = min(vals)
            spatial_dic[pix] = min_v
            # print min_v
            # plt.plot(vals)
            # plt.ylim(-3 , 3)
            # plt.show()
        return spatial_dic


    def check_events(self):
        # 挑出长时间序列 SPEI 1 - 12 小于-2的像素

        fdir = this_root+'SPEI\\per_pix\\SPEI_06\\'


        params = []
        for f in tqdm(os.listdir(fdir)):
            params.append([fdir,f])
        
        results = MUTIPROCESS(self.kernel_check_events,params).run()
        spatial_dic = {}
        for sd in tqdm(results):
            for pix in sd:
                spatial_dic[pix] = sd[pix]

        np.save(this_root+'arr\\temp_delete_06',spatial_dic)
        spatial_dic = dict(np.load(this_root+'arr\\temp_delete_06.npy').item())
        # print spatial_dic
        # for i in spatial_dic:
        #     print i
        #     print spatial_dic[i]
        # spatial_dic = np.array(spatial_dic)
        spatial = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        grid = spatial > -2
        spatial[grid] = -999999
        DIC_and_TIF().arr_to_tif(spatial,this_root+'tif\\check_spei_06.tif')
        # print spatial
        # plt.imshow(spatial)
        # plt.show()


        pass

    def check_growing_season(self):
        interval = '03'
        out_dir = this_root + 'SPEI\\pick_non_growing_season_events\\SPEI_{}\\'.format(interval)
        growing_season = dict(np.load(out_dir+'global.npy').item())
        spatial_dic = {}
        for pix in growing_season:
            spatial_dic[pix] = 1
            print growing_season[pix]
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        plt.show()
        pass


    def composite_3_modes(self):


        pass


# class Recovery_time:
#
#     def __init__(self):
#         interval = 'PDSI'
#         self.plot_recovery_time()
#         # 1 生成per_pix恢复期
#         # self.gen_recovery_time(interval)
#         # self.plot_recovery_period_per_pix()
#         # self.per_pix_dic_to_spatial_tif(interval)
#         # 2 合成SPEI 3 6 9 12 和 PDSI恢复期
#         # self.composite_recovery_time()
#         # 3 绘制composite图
#         # self.plot_composite_recovery_time()
#
#         pass
#
#     def plot_recovery_time(self):
#         # 恢复期示意图，逐个像元看
#         n = 24
#         interval = '%02d'%3
#         spei_dir = this_root + 'SPEI\\per_pix\\SPEI_{}\\'.format(interval)
#         ndvi_dir = this_root + 'NDVI\\per_pix_anomaly\\'
#         single_dir = this_root+'SPEI\\single_events_24\\SPEI_{}\\'.format(interval)
#         for f in tqdm(os.listdir(spei_dir)):
#             if '015' not in f:
#                 continue
#             spei_dic = dict(np.load(spei_dir + f).item())
#             ndvi_dic = dict(np.load(ndvi_dir + f).item())
#             ndvi_dic = Tools().detrend_dic(ndvi_dic)
#             single_dic = dict(np.load(single_dir + f).item())
#             for pix in spei_dic:
#                 spei = spei_dic[pix]
#                 ndvi = ndvi_dic[pix]
#                 single = single_dic[pix]
#                 if len(single) == 0:
#                     continue
#                 spei = Tools().interp_1d(spei)
#                 ndvi = Tools().interp_1d(ndvi)
#                 if len(spei) == 1 or spei[0] == -999999 or ndvi[0] == 0 or len(ndvi) == 1:
#                     continue
#                 smooth_window = 3
#                 spei = Tools().forward_window_smooth(spei, smooth_window)
#                 ndvi = Tools().forward_window_smooth(ndvi, smooth_window)
#                 plt.plot(spei,'r')
#                 plt.plot(ndvi,'g')
#                 for event in single:
#                     start = event[0]
#                     pre_index = []
#                     pre_ndvi_val = []
#                     for i in range(n):
#                         ind = start-i
#                         pre_index.append(ind)
#                         pre_ndvi_val.append(ndvi[ind])
#
#                     # pre_ndvi_mean = np.mean(pre_ndvi_val)
#                     # print(pre_ndvi_mean)
#                     ndvi_drought = []
#                     spei_drought = []
#                     for i in event:
#                         ndvi_drought.append(ndvi[i])
#                         spei_drought.append(spei[i])
#                     plt.plot(event,ndvi_drought,'g',linewidth=4,zorder=99)
#                     plt.plot(event,spei_drought,'r',linewidth=4,zorder=99)
#                     plt.plot(event,[0]*len(event),'--',c='black')
#
#                     # 搜索
#                     max_index = event[-1]
#                     mode_vals = ndvi
#                     # max_val = pre_ndvi_mean
#                     max_val = 0
#                     date_range = event
#                     for search_i in range(len(mode_vals)):
#                         if (max_index + search_i) >= len(mode_vals):
#                             break
#                         search_v = mode_vals[max_index + search_i]
#                         # 设置缓冲区
#                         buffer = 0.0
#                         max_val_buffer_range = [max_val - buffer, max_val + buffer]
#                         if mode_vals[max_index] > max_val:
#                             plt.scatter(max_index, search_v, marker='v', s=50, zorder=99,
#                                         alpha=0.7)
#                             # plt.scatter(recovery_date[-1], recovery_veg_val[-1], marker='^', s=50,
#                             #             zorder=99, alpha=0.7)
#                             xmin = event[0] - 15
#                             xmax = max_index + 15
#                             break
#                         # print(search_v)
#
#                         # if search_v > max_val:
#                         # print((search_v > max_val) or (max_val_buffer_range[0] < search_v < max_val_buffer_range[1]))
#                         # print(search_v > max_val) or (max_val_buffer_range[0] < search_v < max_val_buffer_range[1] and search_i > 5)
#                         if (search_v > max_val) or (
#                                 max_val_buffer_range[0] < search_v < max_val_buffer_range[1] and search_i > 2):
#                             # print(search_v, date_range[-1])
#                             # print(search_v,max_index+search_i)
#                             if (max_index + search_i) <= date_range[-1]:
#                                 continue
#
#                             start = date_range[-1]
#                             if max_index >= start:
#                                 start = max_index
#
#                             end = max_index + search_i + 1
#                             recovery_date = range(start, end)
#
#                             # print(recovery_date)
#                             recovery_veg_val = []
#                             for rd in recovery_date:
#                                 rvv = mode_vals[rd]
#                                 recovery_veg_val.append(rvv)
#                             plt.plot(recovery_date, recovery_veg_val, '--',c='black',linewidth=4, zorder=50, alpha=0.5)
#                             plt.scatter(recovery_date[0], recovery_veg_val[0], marker='v', s=50, zorder=99,
#                                         alpha=0.7)
#                             plt.scatter(recovery_date[-1], recovery_veg_val[-1], marker='^', s=50,
#                                         zorder=99, alpha=0.7)
#                             xmin = event[0] - 15
#                             xmax = end + 15
#
#                             break
#                     # print(in_drought_dic)
#                     # print('****')
#
#                     # exit()
#
#                     # plt.plot(mode_vals_smooth,'--',c='g')
#                 # plt.plot(range(len(mode_vals)), [0.0] * len(mode_vals), '--', c='black')
#                 # plt.plot(range(len(mode_vals)), [-1.0] * len(mode_vals), '--', c='black')
#                 # plt.plot(range(len(mode_vals)), [-1.5] * len(mode_vals), '--', c='black')
#                 plt.plot(range(len(mode_vals)), [-2.0] * len(mode_vals), '--', c='black')
#
#                 plt.plot([], [], c='r', linewidth=4, label='Drought Event')
#                 plt.plot([], [], '--', c='black', linewidth=2, label='Recovery Periods')
#                 plt.scatter([], [], c='black', marker='v', label='Recovery Start Point')
#                 plt.scatter([], [], c='black', marker='^', label='Recovery End Point')
#                 # plt.legend([(p2[0],p1[0]),],['Recovery Period'])
#                 # plt.legend([p2, p1], ['Stuff'])
#                 plt.legend(loc='upper left')
#
#                 xtick = []
#                 for iii in np.arange(smooth_window, len(mode_vals) + smooth_window):
#                     year = 1982 + iii / 12
#                     mon = (iii) % 12 + 1
#                     mon = '%02d' % mon
#                     xtick.append('{}.{}'.format(year, mon))
#                 plt.xticks(range(len(xtick))[::3], xtick[::3], rotation=90)
#                 # minx =
#                 plt.xlim(xmin, xmax)
#                 plt.grid()
#                 plt.show()
#
#
#     def get_min_spei_val_and_index(self,event,spei):
#         min_val = 99999
#         min_index = 99999
#         for i in event:
#             val = spei[i]
#             if val < min_val:
#                 min_val = val
#                 min_index = i
#         mon = min_index % 12 + 1
#         return mon
#
#
#     def gen_recovery_time(self,interval):
#         # 无效值
#         ndv = -999999
#         # 生成逐个像元恢复期
#         n = 24 # 选取干旱事件前n个月的NDVI平均值，
#         if interval == 'PDSI':
#             spei_dir = this_root + 'PDSI\\per_pix\\'
#             single_dir = this_root + 'PDSI\\single_events\\'
#             out_dir = this_root + 'PDSI\\recovery_time_per_pix\\'
#
#         else:
#             spei_dir = this_root + 'SPEI\\per_pix\\SPEI_{:0>2d}\\'.format(interval)
#             single_dir = this_root+'SPEI\\single_events_{}\\'.format(n)+'SPEI_{:0>2d}\\'.format(interval)
#             out_dir = this_root + 'SPEI\\recovery_time_per_pix_{}\\'.format(n) + 'SPEI_{:0>2d}\\'.format(interval)
#
#         ndvi_dir = this_root + 'NDVI\\per_pix_anomaly\\'
#         Tools().mk_dir(out_dir)
#         for f in tqdm(os.listdir(spei_dir)):
#             spei_dic = dict(np.load(spei_dir + f).item())
#             ndvi_dic = dict(np.load(ndvi_dir + f).item())
#             # NDVI detrend
#             ndvi_dic = Tools().detrend_dic(ndvi_dic)
#             single_dic = dict(np.load(single_dir + f).item())
#             recovery_time_list_dic = {}
#             for pix in spei_dic:
#                 spei = spei_dic[pix]
#                 ndvi = ndvi_dic[pix]
#                 single = single_dic[pix]
#                 if len(single) == 0:
#                     recovery_time_list_dic[pix] = ndv
#                     continue
#                 # 数据清洗，去除-999999 和3倍sigma
#                 spei = Tools().interp_1d(spei)
#                 ndvi = Tools().interp_1d(ndvi)
#                 if len(spei) == 1 or spei[0] == -999999 or ndvi[0] == 0 or len(ndvi) == 1:
#                     recovery_time_list_dic[pix] = ndv
#                     continue
#                 # 前窗滤波
#                 smooth_window = 3
#                 spei = Tools().forward_window_smooth(spei, smooth_window)
#                 ndvi = Tools().forward_window_smooth(ndvi, smooth_window)
#                 recovery_time_list = []
#                 for event in single:
#                     ####################################
#                     ####################################
#                     ####################################
#                     # 选取干旱事件前n个月的NDVI平均值
#                     start = event[0]
#                     pre_index = []
#                     pre_ndvi_val = []
#                     for i in range(n):
#                         ind = start-i
#                         pre_index.append(ind)
#                         pre_ndvi_val.append(ndvi[ind])
#                     pre_ndvi_mean = np.mean(pre_ndvi_val)
#                     # 选取干旱事件前n个月的NDVI平均值
#                     ####################################
#                     ####################################
#                     ####################################
#                     # print(pre_ndvi_mean)
#                     # 获取spei最低点和开始点的月份, 为了看事件发生在那个月
#                     spei_min_mon = self.get_min_spei_val_and_index(event,spei)
#
#                     ndvi_drought = []
#                     spei_drought = []
#                     for i in event:
#                         ndvi_drought.append(ndvi[i])
#                         spei_drought.append(spei[i])
#                     # 搜索
#                     max_index = event[-1]
#                     mode_vals = ndvi
#                     # max_val = 0 # 设置anomaly为0时恢复
#                     max_val = pre_ndvi_mean # 设置anomaly为前n个月平均值
#                     date_range = event
#                     # 如果最后一个月的值大于阈值则恢复期为0
#                     if mode_vals[max_index] > max_val:
#                         recovery_time = 0
#                         recovery_time_list.append([str(spei_min_mon),recovery_time])
#                         break
#                     for search_i in range(len(mode_vals)):
#                         if (max_index + search_i) >= len(mode_vals):
#                             break
#                         search_v = mode_vals[max_index + search_i]
#                         # 设置缓冲区
#                         buffer = 0.0
#                         max_val_buffer_range = [max_val - buffer, max_val + buffer]
#
#                         if (search_v > max_val) or (
#                                 max_val_buffer_range[0] < search_v < max_val_buffer_range[1] and search_i > 2):
#                             # print(search_v, date_range[-1])
#                             # print(search_v,max_index+search_i)
#                             if (max_index + search_i) <= date_range[-1]:
#                                 continue
#
#                             start = date_range[-1]
#                             if max_index >= start:
#                                 start = max_index
#
#                             end = max_index + search_i + 1
#                             recovery_date = range(start, end)
#                             recovery_time = len(recovery_date)
#                             recovery_time_list.append([str(spei_min_mon), recovery_time])
#
#                             break
#                 recovery_time_list_dic[pix] = recovery_time_list
#
#
#             np.save(out_dir+f,recovery_time_list_dic)
#
#
#     def plot_recovery_period_per_pix(self):
#         fdir = this_root+'SPEI\\recovery_time_per_pix\\'
#
#         hist = []
#         mon_hist = []
#         for f in tqdm(os.listdir(fdir)):
#
#             dic = dict(np.load(fdir+f).item())
#             for pix in dic:
#                 val = dic[pix]
#                 if type(val) == list:
#                     for i in val:
#                         mon, recovery = i
#                         mon = int(mon)
#                         # if recovery > 25 or recovery == 0:
#                         if recovery > 25:
#                             continue
#                         # print(mon,recovery)
#                         hist.append(recovery)
#                         mon_hist.append(mon)
#         # hist
#         plt.hist(hist,bins=23)
#         plt.figure()
#         plt.hist(mon_hist,bins=12)
#         plt.show()
#
#
#
#         pass
#
#
#
#     def per_pix_dic_to_spatial_tif(self):
#
#         outfolder = this_root+'SPEI\\recovery_tif\\'
#         Tools().mk_dir(outfolder)
#         tif_template = this_root+'conf\\SPEI.tif'
#         _, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
#         fdir = this_root+'SPEI\\recovery_time_per_pix_24\\'
#         flist = os.listdir(fdir)
#
#         spatial_dic = {}
#         for f in tqdm(flist):
#             pix_dic = dict(np.load(fdir + f).item())
#             for pix in pix_dic:
#                 vals = pix_dic[pix]
#                 if type(vals) == list:
#                     recovery_sum = []
#                     for i in vals:
#                         mon, recovery = i
#                         mon = int(mon)
#                         if not mon in range(5,10):
#                             continue
#                         recovery_sum.append(recovery)
#                     recovery_mean = np.mean(recovery_sum)
#                     spatial_dic[pix] = recovery_mean
#                 else:
#                     spatial_dic[pix] = -999999
#         x = []
#         y = []
#         for key in spatial_dic:
#             key_split = key.split('.')
#             x.append(key_split[0])
#             y.append(key_split[1])
#         row = len(set(x))
#         col = len(set(y))
#
#         spatial = []
#         for r in range(row):
#             temp = []
#             for c in range(col):
#                 key = '%03d.%03d' % (r, c)
#                 val_pix = spatial_dic[key]
#                 temp.append(val_pix)
#             spatial.append(temp)
#         spatial = np.array(spatial)
#         grid = np.isnan(spatial)
#         grid = np.logical_not(grid)
#         spatial[np.logical_not(grid)] = -999999
#         to_raster.array2raster(outfolder+'recovery_time1.tif',originX, originY, pixelWidth, pixelHeight,spatial)
#             # plt.imshow(spatial)
#             # plt.colorbar()
#             # plt.show()
#
#     def composite_recovery_time(self):
#         # for interval in [3,6,9,12]:
#         fdir = this_root+'SPEI\\recovery_time_per_pix_24\\SPEI_03\\'
#         # outdir = this_root+'arr\\recovery_time_per_pix\\composite_SPEI_PDSI\\'
#         outdir = this_root+'arr\\recovery_time_per_pix\\composite_SPEI\\'
#         Tools().mk_dir(outdir,force=1)
#         flist = os.listdir(fdir)
#         for f in tqdm(flist):
#             void_dic = {}
#             fdir_ = this_root+'SPEI\\recovery_time_per_pix_24\\SPEI_03\\'
#             dic_ = dict(np.load(fdir_ + f).item())
#
#             for pix in dic_:
#                 void_dic[pix] = []
#
#             for interval in [3,6,9,12]:
#                 interval = '%02d'%interval
#                 fdir = this_root+'SPEI\\recovery_time_per_pix_24\\SPEI_{}\\'.format(interval)
#                 dic = dict(np.load(fdir+f).item())
#                 for pix in dic:
#                     vals = dic[pix]
#                     void_dic[pix].append(vals)
#             # 加PDSI
#             # pdsi_fdir_ = this_root+'PDSI\\recovery_time_per_pix\\'
#             # dic = dict(np.load(pdsi_fdir_ + f).item())
#             # for pix in dic:
#             #     vals = dic[pix]
#             #     void_dic[pix].append(vals)
#
#             np.save(outdir+f,void_dic)
#
#         pass
#
#     def plot_composite_recovery_time(self):
#         # fdir = this_root+'arr\\recovery_time_per_pix\\composite_SPEI\\'
#         fdir = this_root+'arr\\recovery_time_per_pix\\composite_SPEI_PDSI\\'
#         outfolder = this_root+'SPEI\\recovery_tif\\'
#         outf = outfolder+'recovery_PDSI_SPEI.tif'
#         # outf = outfolder+'recovery_SPEI.tif'
#         tif_template = this_root + 'conf\\SPEI.tif'
#         _, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
#
#         spatial_dic = {}
#         for f in tqdm(os.listdir(fdir)):
#             dic = dict(np.load(fdir+f).item())
#             # print(f)
#             for pix in dic:
#                 _vals = dic[pix]
#                 val_sum = []
#                 for vals in _vals:
#                     if type(vals) == list:
#                         for val in vals:
#                             mon,recovery_time = val
#                             mon = int(mon)
#                             val_sum.append(recovery_time)
#                     else:
#                         pass
#                 if len(val_sum) > 0:
#                     val_mean = np.mean(val_sum)
#                     spatial_dic[pix] = val_mean
#                 else:
#                     spatial_dic[pix] = -999999
#
#         x = []
#         y = []
#         for key in spatial_dic:
#             key_split = key.split('.')
#             x.append(key_split[0])
#             y.append(key_split[1])
#         row = len(set(x))
#         col = len(set(y))
#
#         spatial = []
#         for r in range(row):
#             temp = []
#             for c in range(col):
#                 key = '%03d.%03d' % (r, c)
#                 val_pix = spatial_dic[key]
#                 temp.append(val_pix)
#             spatial.append(temp)
#         spatial = np.array(spatial)
#         grid = np.isnan(spatial)
#         grid = np.logical_not(grid)
#         spatial[np.logical_not(grid)] = -999999
#         to_raster.array2raster(outf, originX, originY, pixelWidth, pixelHeight, spatial)


class Recovery_time_winter:
    '''
    考虑冬季的恢复期
    '''

    def __init__(self):
        # self.plot_recovery_time(3)


        # self.statistic_recovery_p_t_swe(interval)
        # self.plot_gen_recovery_time()
        pass

    def run(self):
        # 1 生成recovery time 1-24

        mode = ['pick_non_growing_season_events',
                'pick_pre_growing_season_events',
                'pick_post_growing_season_events'
                ]
        for m in mode:
            print m
            param = []
            for interval in range(1,13):
                param.append([interval,m])
            MUTIPROCESS(self.gen_recovery_time,param).run(6)
            # self.gen_recovery_time(interval)
            # 2 合成 spei 1-24
            self.composite_recovery_time(m)
            # 3 选择参数
            # in_or_out = 'in'
            # in_or_out = 'out'
            # in_or_out = 'all'
            # 4 出tif图
            self.plot_composite_recovery_time(m)
        pass


    def run1(self):
        self.check_events()


    def return_hemi(self, pix, pix_lon_lat_dic):
        lon, lat = pix_lon_lat_dic[pix]
        if lat <= -30:
            return 's'
        elif -30 < lat < 30:
            return 't'
        else:
            return 'n'

    def get_growing_months(self, hemi):
        if hemi == 'n':
            growing_date_range = range(5, 10)
        elif hemi == 's':
            growing_date_range = [11, 12, 1, 2, 3]
        elif hemi == 't':
            growing_date_range = range(1, 13)
        else:
            raise IOError('hemi {} error'.format(hemi))
        return growing_date_range

    def get_non_growing_months(self, hemi):
        if hemi == 'n':
            growing_date_range = [1, 2, 3, 4, 10, 11, 12]
        elif hemi == 's':
            growing_date_range = [4, 5, 6, 7, 8, 9, 10]
        elif hemi == 't':
            growing_date_range = range(1, 13)
        else:
            raise IOError('hemi {} error'.format(hemi))
        return growing_date_range

    def plot_recovery_time(self, interval):
        '''
        画示意图
        :param interval: SPEI_{interval}
        :return:
        '''
        pix_lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        # 1 加载事件
        interval = '%02d' % interval
        events = dict(
            np.load(this_root + 'SPEI\\pick_pre_growing_season_events\\SPEI_{}\\global_pix.npy'.format(interval)).item())
        # 2 加载NDVI
        ndvi_dir = this_root + 'NDVI\\per_pix_anomaly\\'
        spei_dir = this_root + 'SPEI\\per_pix\\SPEI_{}\\'.format(interval)
        pre_dir = this_root + 'PRE\\per_pix_anomaly\\'
        tmp_dir = this_root + 'TMP\\per_pix_anomaly\\'
        swe_dir = this_root + 'GLOBSWE\\per_pix_SWE_max_anomaly\\'
        for f in os.listdir(ndvi_dir):
            ############################
            if not '005' in f:
                continue
            ############################

            ndvi_dic = dict(np.load(ndvi_dir + f).item())
            ndvi_dic = Tools().detrend_dic(ndvi_dic)
            spei_dic = dict(np.load(spei_dir + f).item())
            pre_dic = dict(np.load(pre_dir + f).item())
            tmp_dic = dict(np.load(tmp_dir + f).item())
            swe_dic = dict(np.load(swe_dir + f).item())

            for pix in ndvi_dic:
                if pix in events:

                    ndvi = ndvi_dic[pix]
                    spei = spei_dic[pix]
                    pre = pre_dic[pix]
                    tmp = tmp_dic[pix]
                    swe = swe_dic[pix]
                    event = events[pix]

                    smooth_window = 3

                    ndvi = Tools().forward_window_smooth(ndvi, smooth_window)
                    spei = Tools().forward_window_smooth(spei, smooth_window)
                    pre = Tools().forward_window_smooth(pre, smooth_window)
                    tmp = Tools().forward_window_smooth(tmp, smooth_window)
                    # swe = Tools().forward_window_smooth(swe,smooth_window)
                    swe = np.array(swe)
                    grid = swe > -999
                    swe[np.logical_not(grid)] = np.nan
                    hemi = self.return_hemi(pix, pix_lon_lat_dic)
                    growing_date_range = self.get_growing_months(hemi)  # return [5,6,7,8,9], [11,12,1,2,3], [1-12]
                    for date_range in event:
                        if len(ndvi) < 300:
                            continue
                        # 1 挑出此次干旱事件的NDVI和SPEI值
                        ndvi_picked_vals = Tools().pick_vals_from_1darray(ndvi, date_range)
                        spei_picked_vals = Tools().pick_vals_from_1darray(spei, date_range)
                        # date_range_post = []
                        # for i in range(10):

                        # 2 挑出此次干旱事件SPEI最低的索引
                        min_spei_indx = Tools().pick_min_indx_from_1darray(spei, date_range)
                        # 3 在此次干旱事件SPEI最低索引的周围搜索NDVI的索引和值
                        # 在当前生长季搜索
                        growing_index, growing_vals = Tools().pick_growing_season_vals(ndvi, min_spei_indx,
                                                                                       growing_date_range)
                        # 4 搜索恢复期
                        # 4.1 获取growing season NDVI的最小值
                        min_ndvi_indx = Tools().pick_min_indx_from_1darray(ndvi, growing_index)
                        min_ndvi = min(growing_vals)
                        # 4.2 搜索恢复到正常情况的时间，recovery_time：恢复期； mark：'in', 'out', 'tropical'
                        recovery_time, mark = self.search(ndvi, min_ndvi_indx, growing_date_range)
                        print recovery_time, mark
                        recovery_date_range = range(min_ndvi_indx, min_ndvi_indx + recovery_time + 1)
                        recovery_ndvi = Tools().pick_vals_from_1darray(ndvi, recovery_date_range)

                        tmp_pre_date_range = []
                        for i in recovery_date_range:
                            tmp_pre_date_range.append(i)
                        for i in date_range:
                            tmp_pre_date_range.append(i)
                        tmp_pre_date_range = list(set(tmp_pre_date_range))
                        tmp_pre_date_range.sort()
                        pre_picked_vals = Tools().pick_vals_from_1darray(pre, tmp_pre_date_range)
                        tmp_picked_vals = Tools().pick_vals_from_1darray(tmp, tmp_pre_date_range)
                        if len(swe) == 0:
                            continue
                        swe_picked_vals = Tools().pick_vals_from_1darray(swe, tmp_pre_date_range)

                        plt.figure(figsize=(8, 6))
                        plt.plot(tmp_pre_date_range, pre_picked_vals, '--', c='blue', label='precipitation')
                        plt.plot(tmp_pre_date_range, tmp_picked_vals, '--', c='cyan', label='temperature')
                        plt.plot(tmp_pre_date_range, swe_picked_vals, '--', c='black', linewidth=2, label='SWE',
                                 zorder=99)
                        plt.plot(recovery_date_range, recovery_ndvi, c='g', linewidth=6, label='Recovery Period')
                        plt.plot(date_range, spei_picked_vals, c='r', linewidth=6,
                                 label='SPEI_{} Event'.format(interval))
                        plt.plot(range(len(ndvi)), ndvi, '--', c='g', zorder=99, label='ndvi')
                        plt.plot(range(len(spei)), spei, '--', c='r', zorder=99, label='SPEI_{}'.format(interval))
                        # plt.plot(growing_index,growing_vals,c='g',linewidth=6)
                        plt.legend()

                        minx = 9999
                        maxx = -9999

                        for ii in recovery_date_range:
                            if ii > maxx:
                                maxx = ii
                            if ii < minx:
                                minx = ii

                        for ii in date_range:
                            if ii > maxx:
                                maxx = ii
                            if ii < minx:
                                minx = ii
                        # print date_range[0]-5,recovery_date_range[-1]+5

                        xtick = []
                        for iii in np.arange(len(ndvi)):
                            year = 1982 + iii / 12
                            mon = iii % 12 + 1
                            mon = '%02d' % mon
                            xtick.append('{}.{}'.format(year, mon))
                        # plt.xticks(range(len(xtick))[::3], xtick[::3], rotation=90)
                        plt.xticks(range(len(xtick)), xtick, rotation=90)
                        plt.grid()
                        plt.xlim(minx - 5, maxx + 5)

                        lon, lat, address = Tools().pix_to_address(pix)
                        plt.title('lon:{} lat:{} address:{}'.format(lon, lat, address))
                        plt.show()

        pass

    def kernel_gen_recovery_time(self, params):

        pass

    def gen_recovery_time(self,params):
        '''
        生成全球恢复期
        :param interval: SPEI_{interval}
        :return:
        '''
        interval, mode = params
        pix_lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        interval = '%02d' % interval
        out_dir = this_root + 'arr\\{}\\SPEI_{}\\'.format(mode,interval)
        Tools().mk_dir(out_dir, force=True)
        # 1 加载事件
        # interval = '%02d' % interval
        events = dict(
            np.load(this_root + 'SPEI\\{}\\SPEI_{}\\global_pix.npy'.format(mode,interval)).item())
        # 2 加载NDVI
        ndvi_dir = this_root + 'NDVI\\per_pix_anomaly\\'
        spei_dir = this_root + 'SPEI\\per_pix\\SPEI_{}\\'.format(interval)
        for f in os.listdir(ndvi_dir):
            ndvi_dic = dict(np.load(ndvi_dir + f).item())
            ndvi_dic = Tools().detrend_dic(ndvi_dic)
            spei_dic = dict(np.load(spei_dir + f).item())
            recovery_time_dic = {}
            for pix in ndvi_dic:
                if pix in events:
                    ndvi = ndvi_dic[pix]
                    spei = spei_dic[pix]
                    event = events[pix]
                    smooth_window = 3
                    ndvi = Tools().forward_window_smooth(ndvi, smooth_window)
                    spei = Tools().forward_window_smooth(spei, smooth_window)
                    # 输入Pixel：001.001 输出s:south, n:north ,t: tropical
                    hemi = self.return_hemi(pix, pix_lon_lat_dic)
                    # 输入s,n,t 输出 [5,6,7,8,9], [11,12,1,2,3], [1-12]
                    growing_date_range = self.get_growing_months(hemi)  # return [5,6,7,8,9], [11,12,1,2,3], [1-12]
                    recovery_time_result = []
                    for date_range in event:
                        if len(ndvi) < 300:
                            continue
                        # 1 挑出此次干旱事件的NDVI和SPEI值 （画图需要）
                        # spei_picked_vals = Tools().pick_vals_from_1darray(spei, date_range)
                        # 2 挑出此次干旱事件SPEI最低的索引
                        min_spei_indx = Tools().pick_min_indx_from_1darray(spei, date_range)
                        # 3 在此次干旱事件SPEI最低索引的周围搜索NDVI的索引和值
                        # 在当前生长季搜索
                        growing_index, growing_vals = Tools().pick_growing_season_vals(ndvi, min_spei_indx,
                                                                                       growing_date_range)
                        # 4 搜索恢复期
                        # 4.1 获取growing season NDVI的最小值
                        min_ndvi_indx = Tools().pick_min_indx_from_1darray(ndvi, growing_index)
                        # 4.2 搜索恢复到正常情况的时间，recovery_time：恢复期； mark：'in', 'out', 'tropical'
                        recovery_time, mark = self.search(ndvi, min_ndvi_indx, growing_date_range)
                        # recovery_time, mark = self.search_non_growing_season(ndvi, min_ndvi_indx)
                        recovery_time_result.append([recovery_time, mark])
                    recovery_time_dic[pix] = recovery_time_result
                else:
                    recovery_time_dic[pix] = []
            np.save(out_dir + f, recovery_time_dic)
        pass

    def search(self, ndvi, min_ndvi_indx, growing_date_range):
        # if ndvi[min_ndvi_indx] >= 0:  # 如果在生长季中，NDVI最小值大于0，则恢复期为0个月
        #     return 0,'in'
        for i in range(len(ndvi)):
            if (min_ndvi_indx + i) >= len(ndvi):  # 到头了
                return None, None
            search_ = min_ndvi_indx + i
            search_v = ndvi[search_]
            if search_v >= 0:
                recovery_time = i
                end_mon = search_ % 12 + 1

                if len(growing_date_range) <= 10:  # 存在冬季的地区
                    if end_mon in growing_date_range and recovery_time <= 5:  # 在当年内恢复
                        return recovery_time, 'in'  # 在生长季恢复
                    else:
                        return recovery_time, 'out'  # 不在生长季恢复
                else:  # 不存在冬季的地区
                    return recovery_time, 'tropical'


    def search_non_growing_season(self, ndvi, min_ndvi_indx):
        # if ndvi[min_ndvi_indx] >= 0:  # 如果在生长季中，NDVI最小值大于0，则恢复期为0个月
        #     return 0,'in'
        for i in range(len(ndvi)):
            if (min_ndvi_indx + i) >= len(ndvi):  # 到头了
                break
            search_ = min_ndvi_indx + i
            search_v = ndvi[search_]
            if search_v >= 0:
                recovery_time = i
                # end_mon = search_ % 12 + 1
                return recovery_time, None
                # if len(growing_date_range) <= 10:  # 存在冬季的地区
                #     if end_mon in growing_date_range and recovery_time <= 5: # 在当年内恢复
                #         return recovery_time,'in'  # 在生长季恢复
                #     else:
                #         return recovery_time,'out'  # 不在生长季恢复
                # else:  # 不存在冬季的地区
                #     return recovery_time,'tropical'
        return None, None

    def plot_gen_recovery_time(self):
        '''
        看全球的结果
        :return:
        '''
        interval = 3
        interval = '%02d' % interval
        fdir = this_root + 'arr\\gen_recovery_time\\SPEI_{}\\'.format(interval)
        out_tif_dir = this_root + 'tif\\plot_gen_recovery_time\\'
        Tools().mk_dir(out_tif_dir)
        out_tif = out_tif_dir + 'out_winter.tif'
        global_recovery = {}
        for f in tqdm(os.listdir(fdir)):
            # #################
            # if not '015' in f:
            #     continue
            # #################

            dic = dict(np.load(fdir + f).item())
            for pix in dic:
                events = dic[pix]
                if len(events) > 0:
                    recovery_sum = []
                    for recovery, mark in events:
                        # if mark == 'in' or mark == 'tropical':
                        if mark == 'out' or mark == 'tropical':
                            recovery_sum.append(recovery)
                    if len(recovery_sum) > 0:
                        recovery_mean = int(np.mean(recovery_sum))
                    else:
                        recovery_mean = -999999
                else:
                    recovery_mean = -999999
                global_recovery[pix] = recovery_mean

        DIC_and_TIF().pix_dic_to_tif(global_recovery, out_tif)
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(global_recovery)
        # arr = np.ma.masked_where(arr < -999,arr)
        # plt.imshow(arr,'jet',vmin=0,vmax=6)
        # plt.colorbar()
        # plt.show()
        pass

    def composite_recovery_time(self,mode):
        '''
        合成SPEI 1 - 24 的recovery time
        :return:
        '''
        fdir = this_root + 'arr\\{}\\'.format(mode)
        out_dir = this_root + 'arr\\{}_composite_recovery_time\\'.format(mode)
        Tools().mk_dir(out_dir)
        void_dic = DIC_and_TIF().void_spatial_dic()
        for folder in os.listdir(fdir):
            for f in os.listdir(fdir + folder):
                dic = dict(np.load(fdir + folder + '\\' + f).item())
                for pix in dic:
                    recovery_events = dic[pix]
                    for event in recovery_events:
                        void_dic[pix].append(event)
        # print '\nsaving...'
        np.save(out_dir + 'composite', void_dic)
        # exit()
        pass

    def plot_composite_recovery_time(self,mode):
        # in_or_out = 'in', or 'out'
        composite_recovery = dict(np.load(this_root + 'arr\\{}_composite_recovery_time\\composite.npy'.format(mode)).item())
        out_tif_dir = this_root + 'tif\\{}_plot_gen_recovery_time\\'.format(mode)
        Tools().mk_dir(out_tif_dir)
        out_tif = out_tif_dir + 'global.tif'
        global_recovery = {}
        for pix in composite_recovery:
            # val = composite_recovery[key]
            events = composite_recovery[pix]
            if len(events) > 0:
                recovery_sum = []
                for recovery, mark in events:
                    # if mark == 'in' or mark == 'tropical':
                    # if mark == in_or_out or mark == 'tropical':
                    if recovery != None:
                        recovery_sum.append(recovery)
                if len(recovery_sum) > 0:
                    recovery_mean = np.mean(recovery_sum)
                else:
                    recovery_mean = -999999
            else:
                recovery_mean = -999999
            global_recovery[pix] = recovery_mean

        DIC_and_TIF().pix_dic_to_tif(global_recovery, out_tif)

    def _count_nan(self, array):
        flag = 0.
        for i in array:
            if np.isnan(i):
                flag += 1.
        ratio = flag / len(array)

        return ratio

    def _pick_winter_indexs(self, min_indx):
        print min_indx
        # print len(arr)
        print min_indx % 12 + 1
        picked_index = []
        for i in range(12):
            indx = min_indx + i
            mon = indx % 12 + 1
            if mon in [11, 12, 1, 2, 3]:
                picked_index.append(indx)

            pass

        return arr
        pass

    def statistic_recovery_p_t_swe(self, interval):

        pix_lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        # 1 加载事件
        interval = '%02d' % interval
        events = dict(
            np.load(this_root + 'SPEI\\pick_growing_season_events\\SPEI_{}\\global_pix.npy'.format(interval)).item())
        # 2 加载NDVI
        ndvi_dir = this_root + 'NDVI\\per_pix_anomaly\\'
        spei_dir = this_root + 'SPEI\\per_pix\\SPEI_{}\\'.format(interval)
        pre_dir = this_root + 'PRE\\per_pix\\'  # 原始值
        tmp_dir = this_root + 'TMP\\per_pix\\'  # 原始值
        swe_dir = this_root + 'GLOBSWE\\per_pix\\SWE_max_408\\'  # 原始值
        for f in os.listdir(ndvi_dir):
            ############################
            if not '006' in f:
                continue
            ############################

            ndvi_dic = dict(np.load(ndvi_dir + f).item())
            ndvi_dic = Tools().detrend_dic(ndvi_dic)
            spei_dic = dict(np.load(spei_dir + f).item())
            pre_dic = dict(np.load(pre_dir + f).item())
            tmp_dic = dict(np.load(tmp_dir + f).item())
            swe_dic = dict(np.load(swe_dir + f).item())

            for pix in ndvi_dic:
                if pix in events:

                    ndvi = ndvi_dic[pix]
                    spei = spei_dic[pix]
                    pre = pre_dic[pix]
                    tmp = tmp_dic[pix]
                    swe = swe_dic[pix]
                    event = events[pix]

                    smooth_window = 3

                    ndvi = Tools().forward_window_smooth(ndvi, smooth_window)
                    spei = Tools().forward_window_smooth(spei, smooth_window)
                    pre = Tools().forward_window_smooth(pre, smooth_window)
                    tmp = Tools().forward_window_smooth(tmp, smooth_window)
                    # swe = Tools().forward_window_smooth(swe,smooth_window)
                    swe = np.array(swe, dtype=float)
                    grid = swe > -999
                    # print swe
                    swe[np.logical_not(grid)] = np.nan
                    # print swe
                    ratio = self._count_nan(swe)
                    if ratio == 1:
                        continue
                    hemi = self.return_hemi(pix, pix_lon_lat_dic)
                    growing_date_range = self.get_growing_months(hemi)  # return [5,6,7,8,9], [11,12,1,2,3], [1-12]
                    for date_range in event:
                        if len(ndvi) < 300:
                            continue
                        # 1 挑出此次干旱事件的NDVI和SPEI值
                        ndvi_picked_vals = Tools().pick_vals_from_1darray(ndvi, date_range)
                        spei_picked_vals = Tools().pick_vals_from_1darray(spei, date_range)
                        # date_range_post = []
                        # for i in range(10):

                        # 2 挑出此次干旱事件SPEI最低的索引
                        min_spei_indx = Tools().pick_min_indx_from_1darray(spei, date_range)
                        # 3 在此次干旱事件SPEI最低索引的周围搜索NDVI的索引和值
                        # 在当前生长季搜索
                        growing_index, growing_vals = Tools().pick_growing_season_vals(ndvi, min_spei_indx,
                                                                                       growing_date_range)
                        # 4 搜索恢复期
                        # 4.1 获取growing season NDVI的最小值
                        min_ndvi_indx = Tools().pick_min_indx_from_1darray(ndvi, growing_index)
                        min_ndvi = min(growing_vals)
                        # 4.2 搜索恢复到正常情况的时间，recovery_time：恢复期； mark：'in', 'out', 'tropical'
                        recovery_time, mark = self.search(ndvi, min_ndvi_indx, growing_date_range)
                        print recovery_time, mark
                        # 4.3 搜索 during winter vals
                        winter_swe = self._pick_winter_vals(min_spei_indx, swe)
                        recovery_date_range = range(min_ndvi_indx, min_ndvi_indx + recovery_time + 1)
                        recovery_ndvi = Tools().pick_vals_from_1darray(ndvi, recovery_date_range)

                        tmp_pre_date_range = []
                        for i in recovery_date_range:
                            tmp_pre_date_range.append(i)
                        for i in date_range:
                            tmp_pre_date_range.append(i)
                        tmp_pre_date_range = list(set(tmp_pre_date_range))
                        tmp_pre_date_range.sort()
                        pre_picked_vals = Tools().pick_vals_from_1darray(pre, tmp_pre_date_range)
                        tmp_picked_vals = Tools().pick_vals_from_1darray(tmp, tmp_pre_date_range)
                        if len(swe) == 0:
                            continue
                        swe_picked_vals = Tools().pick_vals_from_1darray(swe, tmp_pre_date_range)

                        plt.figure(figsize=(8, 6))
                        plt.plot(tmp_pre_date_range, pre_picked_vals, '--', c='blue', label='precipitation')
                        plt.plot(tmp_pre_date_range, tmp_picked_vals, '--', c='cyan', label='temperature')
                        plt.plot(tmp_pre_date_range, swe_picked_vals, '--', c='black', linewidth=2, label='SWE',
                                 zorder=99)
                        plt.plot(recovery_date_range, recovery_ndvi, c='g', linewidth=6, label='Recovery Period')
                        plt.plot(date_range, spei_picked_vals, c='r', linewidth=6,
                                 label='SPEI_{} Event'.format(interval))
                        plt.plot(range(len(ndvi)), ndvi, '--', c='g', zorder=99, label='ndvi')
                        plt.plot(range(len(spei)), spei, '--', c='r', zorder=99, label='SPEI_{}'.format(interval))
                        plt.plot(range(len(swe)), swe, '--', c='black', linewidth=2, zorder=99, label='SWE')
                        # plt.plot(growing_index,growing_vals,c='g',linewidth=6)
                        plt.legend()

                        minx = 9999
                        maxx = -9999

                        for ii in recovery_date_range:
                            if ii > maxx:
                                maxx = ii
                            if ii < minx:
                                minx = ii

                        for ii in date_range:
                            if ii > maxx:
                                maxx = ii
                            if ii < minx:
                                minx = ii
                        # print date_range[0]-5,recovery_date_range[-1]+5

                        xtick = []
                        for iii in np.arange(len(ndvi)):
                            year = 1982 + iii / 12
                            mon = iii % 12 + 1
                            mon = '%02d' % mon
                            xtick.append('{}.{}'.format(year, mon))
                        # plt.xticks(range(len(xtick))[::3], xtick[::3], rotation=90)
                        plt.xticks(range(len(xtick)), xtick, rotation=90)
                        plt.grid()
                        plt.xlim(minx - 5, maxx + 5)

                        lon, lat, address = Tools().pix_to_address(pix)
                        plt.title('lon:{} lat:{} address:{}'.format(lon, lat, address))
                        plt.show()

    def check_events(self):
        mode = ['pick_non_growing_season_events',
                'pick_pre_growing_season_events',
                'pick_post_growing_season_events'
                ]
        png_dir = this_root+'png\\check_events\\'
        Tools().mk_dir(png_dir,True)
        for m in mode:
            for interval in tqdm(range(1,13)):
                events = dict(
                    np.load(this_root + 'SPEI\\{}\\SPEI_{}\\global_pix.npy'.format(m, '%02d'%interval)).item())
                new_dic = {}
                for key in events:
                    new_dic[key] = 1
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(new_dic)
                title = '{}_{}'.format(m,interval)
                plt.imshow(arr)
                plt.title(title)
                plt.savefig(png_dir+title+'.png',dpi=144)

                # exit()
        pass



class Winter:
    '''
    主要思想：
    1、计算每个NDVI像素每个月的多年平均值
    2、计算值大于3000的月的个数
    3、如果大于3000的个数大于10，则没有冬季，反之则有冬季
    4、选出冬季date range
    '''
    def __init__(self):


        # self.cal_monthly_mean()
        # self.count_num()
        # self.get_grow_season_index()
        self.composite_growing_season_pix()
        # self.check_pix()
        pass

    def cal_monthly_mean(self):

        outdir = this_root+'NDVI\\mon_mean_tif\\'
        Tools().mk_dir(outdir)
        fdir = this_root+'NDVI\\tif_resample_0.5\\'
        for m in range(1,13):
            arrs_sum = 0.
            for y in range(1982,2016):
                date = '{}{}'.format(y,'%02d'%m)
                tif = fdir+date+'.tif'
                arr,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(tif)
                arrs_sum += arr
            mean_arr = arrs_sum/len(range(1982,2016))
            mean_arr = np.array(mean_arr,dtype=float)
            DIC_and_TIF().arr_to_tif(mean_arr,outdir+'%02d.tif'%m)

    def count_num(self):
        fdir = this_root + 'NDVI\\mon_mean_tif\\'
        pix_lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        arrs = []
        month = range(1,13)
        for m in month:
            tif = fdir+'%02d.tif'%m
            arr,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(tif)
            arrs.append(arr)

        row = len(arrs[0])
        col = len(arrs[0][0])

        winter_count = []
        winter_pix = []
        for i in tqdm(range(row)):
            temp = []
            for j in range(col):
                flag = 0
                for arr in arrs:
                    val = arr[i][j]
                    if val>4500:
                        flag += 1.
                if flag >=10:
                    winter_pix.append('%03d.%03d'%(i,j))
                temp.append(flag)
            winter_count.append(temp)

        np.save(this_root+'NDVI\\tropical_pix',winter_pix)


        ##### show #####
        winter_count = np.array(winter_count)
        winter_count = np.ma.masked_where(winter_count<10,winter_count)
        plt.imshow(winter_count,'jet')
        plt.colorbar()
        plt.show()
        pass


    def max_5_vals(self,vals):

        vals = np.array(vals)
        # 从小到大排序，获取索引值
        a = np.argsort(vals)
        maxvs = []
        maxv_ind = []
        for i in a[-5:][::-1]:
            maxvs.append(vals[i])
            maxv_ind.append(i)
        # 南半球
        if 0 in maxv_ind or 1 in maxv_ind:
            if 9 in maxv_ind:
                growing_season = [0, 8, 9, 10, 11]
            elif 10 in maxv_ind:
                growing_season = [0, 1, 2, 10, 11]
            elif 11 in maxv_ind:
                growing_season = [0, 1, 2, 3, 11]
            else:
                mid = int(np.mean(maxv_ind))
                growing_season = [mid-2,mid-1,mid,mid+1,mid+2]
        # 北半球
        else:
            mid = int(np.mean(maxv_ind))
            growing_season = [mid-2,mid-1,mid,mid+1,mid+2]
        growing_season = np.array(growing_season) + 1
        return growing_season

    def get_grow_season_index(self):
        tropical_pix = np.load(this_root+'NDVI\\tropical_pix.npy')
        fdir = this_root + 'NDVI\\mon_mean_tif\\'
        arrs = []
        month = range(1, 13)
        for m in month:
            tif = fdir + '%02d.tif' % m
            arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif)
            arrs.append(arr)

        row = len(arrs[0])
        col = len(arrs[0][0])

        winter_dic = {}
        for i in tqdm(range(row)):
            for j in range(col):
                # if i < 150:
                #     continue
                pix = '%03d.%03d' % (i, j)
                if pix in tropical_pix:
                   continue
                vals = []
                for arr in arrs:
                    val = arr[i][j]
                    vals.append(val)
                if vals[0] > -10000:
                    std = np.std(vals)
                    if std == 0:
                        continue
                    growing_season = self.max_5_vals(vals)
                    # print growing_season
                    # plt.plot(vals)
                    # plt.grid()
                    # plt.show()
                    winter_dic[pix] = growing_season
        np.save(this_root+'NDVI\\growing_season_index',winter_dic)

        pass

    def composite_growing_season_pix(self):
        growing_season_index = dict(np.load(this_root + 'NDVI\\growing_season_index.npy').item())
        tropical_pix = np.load(this_root + 'NDVI\\tropical_pix.npy')
        pix_dic = {}
        for pix in tropical_pix:
            # pix_dic[pix] = 2
            pix_dic[pix] = range(1,13)
        # for pix in growing_season_index:
        #     pix_dic[pix] = growing_season_index[pix]
        np.save(this_root+'NDVI\\global_growing_season',pix_dic)

        for pix in pix_dic:
            print pix,pix_dic[pix]

        pass

    def check_pix(self):
        growing_season_index = dict(np.load(this_root+'NDVI\\growing_season_index.npy').item())
        tropical_pix = np.load(this_root+'NDVI\\tropical_pix.npy')
        pix_dic = {}
        for pix in tropical_pix:
            pix_dic[pix] = 2

        for pix in growing_season_index:
            pix_dic[pix] = 1
            print growing_season_index[pix]

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(pix_dic)

        plt.imshow(arr)
        plt.show()

        pass

class Recovery_time_winter_2:
    '''
    :arg
    改进growing season
    '''

    def __init__(self):

        self.valid_pix()
        pass


    def run(self):

        mode = ['pick_non_growing_season_events',
                'pick_pre_growing_season_events',
                'pick_post_growing_season_events'
                ]
        for m in mode:
            print m
            param = []
            for interval in range(1,4):
                param.append([interval,m])
                # self.gen_recovery_time([interval,m])
            MUTIPROCESS(self.gen_recovery_time,param).run(6)
            # self.gen_recovery_time(interval)
            # 2 合成 spei 1-24
            self.composite_recovery_time(m)
            # 4 出tif图
            # self.plot_composite_recovery_time(m)
        pass


    def run1(self):
        # self.recovery_latitude()
        # self.composite_3_mode_recovery_time()
        # self.gen_composite_recovery_time_tif()
        # self.recovery_latitude_3mode()
        # self.recovery_landcover_3mode()
        # self.recovery_latitude_mix()
        # self.recovery_landcover_mix()
        pass

    def valid_pix(self):
        self.ndvi_valid_pix = Tools().filter_NDVI_valid_pix()
        self.tropical_pix = np.load(this_root + 'NDVI\\tropical_pix.npy')


    def return_hemi(self, pix, pix_lon_lat_dic):
        lon, lat = pix_lon_lat_dic[pix]
        if lat <= -30:
            return 's'
        elif -30 < lat < 30:
            return 't'
        else:
            return 'n'

    def get_growing_months(self, hemi):
        if hemi == 'n':
            growing_date_range = range(5, 10)
        elif hemi == 's':
            growing_date_range = [11, 12, 1, 2, 3]
        elif hemi == 't':
            growing_date_range = range(1, 13)
        else:
            raise IOError('hemi {} error'.format(hemi))
        return growing_date_range

    def get_non_growing_months(self, hemi):
        if hemi == 'n':
            growing_date_range = [1, 2, 3, 4, 10, 11, 12]
        elif hemi == 's':
            growing_date_range = [4, 5, 6, 7, 8, 9, 10]
        elif hemi == 't':
            growing_date_range = range(1, 13)
        else:
            raise IOError('hemi {} error'.format(hemi))
        return growing_date_range

    def plot_recovery_time(self, interval):
        '''
        画示意图
        :param interval: SPEI_{interval}
        :return:
        '''
        pix_lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        # 1 加载事件
        interval = '%02d' % interval
        events = dict(
            np.load(this_root + 'SPEI\\pick_pre_growing_season_events\\SPEI_{}\\global_pix.npy'.format(interval)).item())
        # 2 加载NDVI
        ndvi_dir = this_root + 'NDVI\\per_pix_anomaly\\'
        spei_dir = this_root + 'SPEI\\per_pix\\SPEI_{}\\'.format(interval)
        pre_dir = this_root + 'PRE\\per_pix_anomaly\\'
        tmp_dir = this_root + 'TMP\\per_pix_anomaly\\'
        swe_dir = this_root + 'GLOBSWE\\per_pix_SWE_max_anomaly\\'
        for f in os.listdir(ndvi_dir):
            ############################
            if not '005' in f:
                continue
            ############################

            ndvi_dic = dict(np.load(ndvi_dir + f).item())
            # ndvi_dic = Tools().detrend_dic(ndvi_dic)
            spei_dic = dict(np.load(spei_dir + f).item())
            pre_dic = dict(np.load(pre_dir + f).item())
            tmp_dic = dict(np.load(tmp_dir + f).item())
            swe_dic = dict(np.load(swe_dir + f).item())

            for pix in ndvi_dic:
                if pix in events:

                    ndvi = ndvi_dic[pix]
                    spei = spei_dic[pix]
                    pre = pre_dic[pix]
                    tmp = tmp_dic[pix]
                    swe = swe_dic[pix]
                    event = events[pix]

                    smooth_window = 3

                    ndvi = Tools().forward_window_smooth(ndvi, smooth_window)
                    spei = Tools().forward_window_smooth(spei, smooth_window)
                    pre = Tools().forward_window_smooth(pre, smooth_window)
                    tmp = Tools().forward_window_smooth(tmp, smooth_window)
                    # swe = Tools().forward_window_smooth(swe,smooth_window)
                    swe = np.array(swe)
                    grid = swe > -999
                    swe[np.logical_not(grid)] = np.nan
                    hemi = self.return_hemi(pix, pix_lon_lat_dic)
                    growing_date_range = self.get_growing_months(hemi)  # return [5,6,7,8,9], [11,12,1,2,3], [1-12]
                    for date_range in event:
                        if len(ndvi) < 300:
                            continue
                        # 1 挑出此次干旱事件的NDVI和SPEI值
                        ndvi_picked_vals = Tools().pick_vals_from_1darray(ndvi, date_range)
                        spei_picked_vals = Tools().pick_vals_from_1darray(spei, date_range)
                        # date_range_post = []
                        # for i in range(10):

                        # 2 挑出此次干旱事件SPEI最低的索引
                        min_spei_indx = Tools().pick_min_indx_from_1darray(spei, date_range)
                        # 3 在此次干旱事件SPEI最低索引的周围搜索NDVI的索引和值
                        # 在当前生长季搜索
                        growing_index, growing_vals = Tools().pick_growing_season_vals(ndvi, min_spei_indx,
                                                                                       growing_date_range)
                        # 4 搜索恢复期
                        # 4.1 获取growing season NDVI的最小值
                        min_ndvi_indx = Tools().pick_min_indx_from_1darray(ndvi, growing_index)
                        min_ndvi = min(growing_vals)
                        # 4.2 搜索恢复到正常情况的时间，recovery_time：恢复期； mark：'in', 'out', 'tropical'
                        recovery_time, mark = self.search(ndvi, min_ndvi_indx, growing_date_range)
                        print recovery_time, mark
                        recovery_date_range = range(min_ndvi_indx, min_ndvi_indx + recovery_time + 1)
                        recovery_ndvi = Tools().pick_vals_from_1darray(ndvi, recovery_date_range)

                        tmp_pre_date_range = []
                        for i in recovery_date_range:
                            tmp_pre_date_range.append(i)
                        for i in date_range:
                            tmp_pre_date_range.append(i)
                        tmp_pre_date_range = list(set(tmp_pre_date_range))
                        tmp_pre_date_range.sort()
                        pre_picked_vals = Tools().pick_vals_from_1darray(pre, tmp_pre_date_range)
                        tmp_picked_vals = Tools().pick_vals_from_1darray(tmp, tmp_pre_date_range)
                        if len(swe) == 0:
                            continue
                        swe_picked_vals = Tools().pick_vals_from_1darray(swe, tmp_pre_date_range)

                        plt.figure(figsize=(8, 6))
                        plt.plot(tmp_pre_date_range, pre_picked_vals, '--', c='blue', label='precipitation')
                        plt.plot(tmp_pre_date_range, tmp_picked_vals, '--', c='cyan', label='temperature')
                        plt.plot(tmp_pre_date_range, swe_picked_vals, '--', c='black', linewidth=2, label='SWE',
                                 zorder=99)
                        plt.plot(recovery_date_range, recovery_ndvi, c='g', linewidth=6, label='Recovery Period')
                        plt.plot(date_range, spei_picked_vals, c='r', linewidth=6,
                                 label='SPEI_{} Event'.format(interval))
                        plt.plot(range(len(ndvi)), ndvi, '--', c='g', zorder=99, label='ndvi')
                        plt.plot(range(len(spei)), spei, '--', c='r', zorder=99, label='SPEI_{}'.format(interval))
                        # plt.plot(growing_index,growing_vals,c='g',linewidth=6)
                        plt.legend()

                        minx = 9999
                        maxx = -9999

                        for ii in recovery_date_range:
                            if ii > maxx:
                                maxx = ii
                            if ii < minx:
                                minx = ii

                        for ii in date_range:
                            if ii > maxx:
                                maxx = ii
                            if ii < minx:
                                minx = ii
                        # print date_range[0]-5,recovery_date_range[-1]+5

                        xtick = []
                        for iii in np.arange(len(ndvi)):
                            year = 1982 + iii / 12
                            mon = iii % 12 + 1
                            mon = '%02d' % mon
                            xtick.append('{}.{}'.format(year, mon))
                        # plt.xticks(range(len(xtick))[::3], xtick[::3], rotation=90)
                        plt.xticks(range(len(xtick)), xtick, rotation=90)
                        plt.grid()
                        plt.xlim(minx - 5, maxx + 5)

                        lon, lat, address = Tools().pix_to_address(pix)
                        plt.title('lon:{} lat:{} address:{}'.format(lon, lat, address))
                        plt.show()

        pass


    def get_growing_season_range(self,pix):


        return []

    def gen_recovery_time(self,params):
        '''
        生成全球恢复期
        :param interval: SPEI_{interval}
        :return:
        '''
        interval, mode = params
        # pix_lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        growing_season_daterange = dict(np.load(this_root+'NDVI\\global_growing_season.npy').item())
        interval = '%02d' % interval
        out_dir = this_root + 'arr\\recovery_time\\{}\\SPEI_{}\\'.format(mode,interval)
        Tools().mk_dir(out_dir, force=True)
        # 1 加载事件
        # interval = '%02d' % interval
        events = dict(
            np.load(this_root + 'SPEI\\{}\\SPEI_{}\\global.npy'.format(mode,interval)).item())
        # 2 加载NDVI
        ndvi_dir = this_root + 'NDVI\\per_pix_anomaly\\'
        spei_dir = this_root + 'SPEI\\per_pix\\SPEI_{}\\'.format(interval)
        for f in os.listdir(ndvi_dir):
            # if not '005' in f:
            #     continue
            ndvi_dic = dict(np.load(ndvi_dir + f).item())
            # ndvi_dic = Tools().detrend_dic(ndvi_dic)
            spei_dic = dict(np.load(spei_dir + f).item())
            recovery_time_dic = {}
            for pix in ndvi_dic:
                if pix in events and pix in growing_season_daterange:
                    growing_date_range =growing_season_daterange[pix]
                    ndvi = ndvi_dic[pix]
                    spei = spei_dic[pix]
                    event = events[pix]
                    smooth_window = 3
                    # ndvi = Tools().forward_window_smooth(ndvi, smooth_window)
                    spei = Tools().forward_window_smooth(spei, smooth_window)
                    # 输入Pixel：001.001 输出s:south, n:north ,t: tropical
                    # hemi = self.return_hemi(pix, pix_lon_lat_dic)
                    # 输入s,n,t 输出 [5,6,7,8,9], [11,12,1,2,3], [1-12]
                    # growing_date_range = self.get_growing_months(hemi)  # return [5,6,7,8,9], [11,12,1,2,3], [1-12]
                    recovery_time_result = []
                    for date_range in event:

                        ndvi = np.array(ndvi)
                        grid = ndvi < -100
                        ndvi[grid] = np.nan
                        ndvi = Tools().interp_nan(ndvi)

                        spei = np.array(spei)
                        grid_1 = spei < -100
                        spei[grid_1] = np.nan
                        spei = Tools().interp_nan(spei)

                        if len(ndvi) < 300:
                            continue
                        # 1 挑出此次干旱事件的NDVI和SPEI值 （画图需要）
                        # spei_picked_vals = Tools().pick_vals_from_1darray(spei, date_range)
                        # 2 挑出此次干旱事件SPEI最低的索引
                        min_spei_indx = Tools().pick_min_indx_from_1darray(spei, date_range)
                        # 3 在此次干旱事件SPEI最低索引的周围搜索NDVI的索引和值
                        # 在当前生长季搜索
                        growing_index, growing_vals = Tools().pick_growing_season_vals(ndvi, min_spei_indx,
                                                                                       growing_date_range)
                        # 4 搜索恢复期
                        # 4.1 获取growing season NDVI的最小值
                        min_ndvi_indx = Tools().pick_min_indx_from_1darray(ndvi, growing_index)
                        # print min_ndvi_indx
                        # print min_spei_indx
                        # 4.2 搜索恢复到正常情况的时间，recovery_time：恢复期； mark：'in', 'out', 'tropical'
                        recovery_time, mark, recovery_date_range = self.search(ndvi, min_ndvi_indx, growing_date_range)
                        # recovery_time, mark = self.search_non_growing_season(ndvi, min_ndvi_indx)
                        recovery_time_result.append([recovery_time, mark, recovery_date_range])


                        ################# plot ##################
                        # print recovery_time, mark
                        # print growing_date_range
                        # recovery_date_range = range(min_ndvi_indx, min_ndvi_indx + recovery_time + 1)
                        # recovery_ndvi = Tools().pick_vals_from_1darray(ndvi, recovery_date_range)
                        #
                        # tmp_pre_date_range = []
                        # for i in recovery_date_range:
                        #     tmp_pre_date_range.append(i)
                        # for i in date_range:
                        #     tmp_pre_date_range.append(i)
                        # tmp_pre_date_range = list(set(tmp_pre_date_range))
                        # tmp_pre_date_range.sort()
                        # # pre_picked_vals = Tools().pick_vals_from_1darray(pre, tmp_pre_date_range)
                        # # tmp_picked_vals = Tools().pick_vals_from_1darray(tmp, tmp_pre_date_range)
                        # # if len(swe) == 0:
                        # #     continue
                        # # swe_picked_vals = Tools().pick_vals_from_1darray(swe, tmp_pre_date_range)
                        #
                        # plt.figure(figsize=(8, 6))
                        # # plt.plot(tmp_pre_date_range, pre_picked_vals, '--', c='blue', label='precipitation')
                        # # plt.plot(tmp_pre_date_range, tmp_picked_vals, '--', c='cyan', label='temperature')
                        # # plt.plot(tmp_pre_date_range, swe_picked_vals, '--', c='black', linewidth=2, label='SWE',
                        # #          zorder=99)
                        # plt.plot(recovery_date_range, recovery_ndvi, c='g', linewidth=6, label='Recovery Period')
                        # plt.plot(date_range, spei_picked_vals, c='r', linewidth=6,
                        #          label='SPEI_{} Event'.format(interval))
                        # plt.plot(range(len(ndvi)), ndvi, '--', c='g', zorder=99, label='ndvi')
                        # plt.plot(range(len(spei)), spei, '--', c='r', zorder=99, label='SPEI_{}'.format(interval))
                        # plt.legend()
                        #
                        # minx = 9999
                        # maxx = -9999
                        #
                        # for ii in recovery_date_range:
                        #     if ii > maxx:
                        #         maxx = ii
                        #     if ii < minx:
                        #         minx = ii
                        #
                        # for ii in date_range:
                        #     if ii > maxx:
                        #         maxx = ii
                        #     if ii < minx:
                        #         minx = ii
                        # # print date_range[0]-5,recovery_date_range[-1]+5
                        #
                        # xtick = []
                        # for iii in np.arange(len(ndvi)):
                        #     year = 1982 + iii / 12
                        #     mon = iii % 12 + 1
                        #     mon = '%02d' % mon
                        #     xtick.append('{}.{}'.format(year, mon))
                        # # plt.xticks(range(len(xtick))[::3], xtick[::3], rotation=90)
                        # plt.xticks(range(len(xtick)), xtick, rotation=90)
                        # plt.grid()
                        # plt.xlim(minx - 5, maxx + 5)
                        #
                        # lon, lat, address = Tools().pix_to_address(pix)
                        # plt.title('lon:{} lat:{} address:{}'.format(lon, lat, address))
                        # plt.show()
                        #################plot##################


                    recovery_time_dic[pix] = recovery_time_result
                else:
                    recovery_time_dic[pix] = []
            np.save(out_dir + f, recovery_time_dic)
        pass

    def search(self, ndvi, min_ndvi_indx, growing_date_range):
        # if ndvi[min_ndvi_indx] >= 0:  # 如果在生长季中，NDVI最小值大于0，则恢复期为0个月
        #     return 0,'in'
        for i in range(len(ndvi)):
            if (min_ndvi_indx + i) >= len(ndvi):  # 到头了
                return None, None, None
            search_ = min_ndvi_indx + i
            search_v = ndvi[search_]
            if search_v >= 0:
                recovery_time = i
                end_mon = search_ % 12 + 1
                recovery_date_range = range(min_ndvi_indx,min_ndvi_indx+i+1)
                if len(growing_date_range) <= 10:  # 存在冬季的地区
                    if end_mon in growing_date_range:  # 在当年内恢复
                        if recovery_time <= 5:
                            return recovery_time, 'in',recovery_date_range  # 在生长季恢复
                        else:
                            return recovery_time,'out',recovery_date_range  # 不在生长季恢复
                    else:
                        continue  # 继续搜索
                else:  # 不存在冬季的地区
                    return recovery_time, 'tropical', recovery_date_range


    def search_non_growing_season(self, ndvi, min_ndvi_indx):
        # if ndvi[min_ndvi_indx] >= 0:  # 如果在生长季中，NDVI最小值大于0，则恢复期为0个月
        #     return 0,'in'
        for i in range(len(ndvi)):
            if (min_ndvi_indx + i) >= len(ndvi):  # 到头了
                break
            search_ = min_ndvi_indx + i
            search_v = ndvi[search_]
            if search_v >= 0:
                recovery_time = i
                # end_mon = search_ % 12 + 1
                return recovery_time, None
                # if len(growing_date_range) <= 10:  # 存在冬季的地区
                #     if end_mon in growing_date_range and recovery_time <= 5: # 在当年内恢复
                #         return recovery_time,'in'  # 在生长季恢复
                #     else:
                #         return recovery_time,'out'  # 不在生长季恢复
                # else:  # 不存在冬季的地区
                #     return recovery_time,'tropical'
        return None, None

    def plot_gen_recovery_time(self):
        '''
        看全球的结果
        :return:
        '''
        interval = 3
        interval = '%02d' % interval
        fdir = this_root + 'arr\\gen_recovery_time\\SPEI_{}\\'.format(interval)
        out_tif_dir = this_root + 'tif\\plot_gen_recovery_time\\'
        Tools().mk_dir(out_tif_dir)
        out_tif = out_tif_dir + 'out_winter.tif'
        global_recovery = {}
        for f in tqdm(os.listdir(fdir)):
            # #################
            # if not '015' in f:
            #     continue
            # #################

            dic = dict(np.load(fdir + f).item())
            for pix in dic:
                events = dic[pix]
                if len(events) > 0:
                    recovery_sum = []
                    for recovery, mark in events:
                        # if mark == 'in' or mark == 'tropical':
                        if mark == 'out' or mark == 'tropical':
                            recovery_sum.append(recovery)
                    if len(recovery_sum) > 0:
                        recovery_mean = int(np.mean(recovery_sum))
                    else:
                        recovery_mean = -999999
                else:
                    recovery_mean = -999999
                global_recovery[pix] = recovery_mean

        DIC_and_TIF().pix_dic_to_tif(global_recovery, out_tif)
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(global_recovery)
        # arr = np.ma.masked_where(arr < -999,arr)
        # plt.imshow(arr,'jet',vmin=0,vmax=6)
        # plt.colorbar()
        # plt.show()
        pass

    def composite_recovery_time(self,mode):
        '''
        合成SPEI 1 - 24 的recovery time
        :return:
        '''
        fdir = this_root + 'arr\\recovery_time\\{}\\'.format(mode)
        out_dir = this_root + 'arr\\recovery_time\\{}_composite_recovery_time\\'.format(mode)
        Tools().mk_dir(out_dir)
        void_dic = DIC_and_TIF().void_spatial_dic()
        interval_range = []
        for interval in range(1,4):
            interval_range.append()
        for folder in os.listdir(fdir):
            for f in os.listdir(fdir + folder):
                dic = dict(np.load(fdir + folder + '\\' + f).item())
                for pix in dic:
                    recovery_events = dic[pix]
                    for event in recovery_events:
                        void_dic[pix].append(event)
        # print '\nsaving...'
        np.save(out_dir + 'composite', void_dic)
        # exit()
        pass

    def plot_composite_recovery_time(self,mode):
        # in_or_out = 'in', or 'out'
        composite_recovery = dict(np.load(this_root + 'arr\\{}_composite_recovery_time\\composite.npy'.format(mode)).item())
        out_tif_dir = this_root + 'tif\\{}_plot_gen_recovery_time\\'.format(mode)
        Tools().mk_dir(out_tif_dir)
        out_tif = out_tif_dir + 'global.tif'
        global_recovery = {}
        for pix in composite_recovery:
            # val = composite_recovery[key]
            events = composite_recovery[pix]
            if len(events) > 0:
                recovery_sum = []
                for recovery, mark in events:
                    # if mark == 'in' or mark == 'tropical':
                    # if mark == in_or_out or mark == 'tropical':
                    if recovery != None:
                        recovery_sum.append(recovery)
                if len(recovery_sum) > 0:
                    recovery_mean = np.mean(recovery_sum)
                else:
                    recovery_mean = -999999
            else:
                recovery_mean = -999999
            global_recovery[pix] = recovery_mean

        DIC_and_TIF().pix_dic_to_tif(global_recovery, out_tif)

    def _count_nan(self, array):
        flag = 0.
        for i in array:
            if np.isnan(i):
                flag += 1.
        ratio = flag / len(array)

        return ratio

    def _pick_winter_indexs(self, min_indx):
        print min_indx
        # print len(arr)
        print min_indx % 12 + 1
        picked_index = []
        for i in range(12):
            indx = min_indx + i
            mon = indx % 12 + 1
            if mon in [11, 12, 1, 2, 3]:
                picked_index.append(indx)

            pass

        return arr
        pass

    def statistic_recovery_p_t_swe(self, interval):

        pix_lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        # 1 加载事件
        interval = '%02d' % interval
        events = dict(
            np.load(this_root + 'SPEI\\pick_growing_season_events\\SPEI_{}\\global_pix.npy'.format(interval)).item())
        # 2 加载NDVI
        ndvi_dir = this_root + 'NDVI\\per_pix_anomaly\\'
        spei_dir = this_root + 'SPEI\\per_pix\\SPEI_{}\\'.format(interval)
        pre_dir = this_root + 'PRE\\per_pix\\'  # 原始值
        tmp_dir = this_root + 'TMP\\per_pix\\'  # 原始值
        swe_dir = this_root + 'GLOBSWE\\per_pix\\SWE_max_408\\'  # 原始值
        for f in os.listdir(ndvi_dir):
            ############################
            if not '006' in f:
                continue
            ############################

            ndvi_dic = dict(np.load(ndvi_dir + f).item())
            ndvi_dic = Tools().detrend_dic(ndvi_dic)
            spei_dic = dict(np.load(spei_dir + f).item())
            pre_dic = dict(np.load(pre_dir + f).item())
            tmp_dic = dict(np.load(tmp_dir + f).item())
            swe_dic = dict(np.load(swe_dir + f).item())

            for pix in ndvi_dic:
                if pix in events:

                    ndvi = ndvi_dic[pix]
                    spei = spei_dic[pix]
                    pre = pre_dic[pix]
                    tmp = tmp_dic[pix]
                    swe = swe_dic[pix]
                    event = events[pix]

                    smooth_window = 3

                    ndvi = Tools().forward_window_smooth(ndvi, smooth_window)
                    spei = Tools().forward_window_smooth(spei, smooth_window)
                    pre = Tools().forward_window_smooth(pre, smooth_window)
                    tmp = Tools().forward_window_smooth(tmp, smooth_window)
                    # swe = Tools().forward_window_smooth(swe,smooth_window)
                    swe = np.array(swe, dtype=float)
                    grid = swe > -999
                    # print swe
                    swe[np.logical_not(grid)] = np.nan
                    # print swe
                    ratio = self._count_nan(swe)
                    if ratio == 1:
                        continue
                    hemi = self.return_hemi(pix, pix_lon_lat_dic)
                    growing_date_range = self.get_growing_months(hemi)  # return [5,6,7,8,9], [11,12,1,2,3], [1-12]
                    for date_range in event:
                        if len(ndvi) < 300:
                            continue
                        # 1 挑出此次干旱事件的NDVI和SPEI值
                        ndvi_picked_vals = Tools().pick_vals_from_1darray(ndvi, date_range)
                        spei_picked_vals = Tools().pick_vals_from_1darray(spei, date_range)
                        # date_range_post = []
                        # for i in range(10):

                        # 2 挑出此次干旱事件SPEI最低的索引
                        min_spei_indx = Tools().pick_min_indx_from_1darray(spei, date_range)
                        # 3 在此次干旱事件SPEI最低索引的周围搜索NDVI的索引和值
                        # 在当前生长季搜索
                        growing_index, growing_vals = Tools().pick_growing_season_vals(ndvi, min_spei_indx,
                                                                                       growing_date_range)
                        # 4 搜索恢复期
                        # 4.1 获取growing season NDVI的最小值
                        min_ndvi_indx = Tools().pick_min_indx_from_1darray(ndvi, growing_index)
                        min_ndvi = min(growing_vals)
                        # 4.2 搜索恢复到正常情况的时间，recovery_time：恢复期； mark：'in', 'out', 'tropical'
                        recovery_time, mark = self.search(ndvi, min_ndvi_indx, growing_date_range)
                        print recovery_time, mark
                        # 4.3 搜索 during winter vals
                        winter_swe = self._pick_winter_vals(min_spei_indx, swe)
                        recovery_date_range = range(min_ndvi_indx, min_ndvi_indx + recovery_time + 1)
                        recovery_ndvi = Tools().pick_vals_from_1darray(ndvi, recovery_date_range)

                        tmp_pre_date_range = []
                        for i in recovery_date_range:
                            tmp_pre_date_range.append(i)
                        for i in date_range:
                            tmp_pre_date_range.append(i)
                        tmp_pre_date_range = list(set(tmp_pre_date_range))
                        tmp_pre_date_range.sort()
                        pre_picked_vals = Tools().pick_vals_from_1darray(pre, tmp_pre_date_range)
                        tmp_picked_vals = Tools().pick_vals_from_1darray(tmp, tmp_pre_date_range)
                        if len(swe) == 0:
                            continue
                        swe_picked_vals = Tools().pick_vals_from_1darray(swe, tmp_pre_date_range)

                        plt.figure(figsize=(8, 6))
                        plt.plot(tmp_pre_date_range, pre_picked_vals, '--', c='blue', label='precipitation')
                        plt.plot(tmp_pre_date_range, tmp_picked_vals, '--', c='cyan', label='temperature')
                        plt.plot(tmp_pre_date_range, swe_picked_vals, '--', c='black', linewidth=2, label='SWE',
                                 zorder=99)
                        plt.plot(recovery_date_range, recovery_ndvi, c='g', linewidth=6, label='Recovery Period')
                        plt.plot(date_range, spei_picked_vals, c='r', linewidth=6,
                                 label='SPEI_{} Event'.format(interval))
                        plt.plot(range(len(ndvi)), ndvi, '--', c='g', zorder=99, label='ndvi')
                        plt.plot(range(len(spei)), spei, '--', c='r', zorder=99, label='SPEI_{}'.format(interval))
                        plt.plot(range(len(swe)), swe, '--', c='black', linewidth=2, zorder=99, label='SWE')
                        # plt.plot(growing_index,growing_vals,c='g',linewidth=6)
                        plt.legend()

                        minx = 9999
                        maxx = -9999

                        for ii in recovery_date_range:
                            if ii > maxx:
                                maxx = ii
                            if ii < minx:
                                minx = ii

                        for ii in date_range:
                            if ii > maxx:
                                maxx = ii
                            if ii < minx:
                                minx = ii
                        # print date_range[0]-5,recovery_date_range[-1]+5

                        xtick = []
                        for iii in np.arange(len(ndvi)):
                            year = 1982 + iii / 12
                            mon = iii % 12 + 1
                            mon = '%02d' % mon
                            xtick.append('{}.{}'.format(year, mon))
                        # plt.xticks(range(len(xtick))[::3], xtick[::3], rotation=90)
                        plt.xticks(range(len(xtick)), xtick, rotation=90)
                        plt.grid()
                        plt.xlim(minx - 5, maxx + 5)

                        lon, lat, address = Tools().pix_to_address(pix)
                        plt.title('lon:{} lat:{} address:{}'.format(lon, lat, address))
                        plt.show()

    def check_spei_events(self):
        mode = ['pick_non_growing_season_events',
                'pick_pre_growing_season_events',
                'pick_post_growing_season_events'
                ]
        png_dir = this_root+'png\\check_events\\'
        Tools().mk_dir(png_dir,True)
        for m in mode:
            for interval in tqdm(range(1,13)):
                events = dict(
                    np.load(this_root + 'SPEI\\{}\\SPEI_{}\\global_pix.npy'.format(m, '%02d'%interval)).item())
                new_dic = {}
                for key in events:
                    new_dic[key] = 1
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(new_dic)
                title = '{}_{}'.format(m,interval)
                plt.imshow(arr)
                plt.title(title)
                plt.savefig(png_dir+title+'.png',dpi=144)

                # exit()
        pass



    def composite_3_mode_recovery_time(self):
        mode = [
            'pick_non_growing_season_events',
            'pick_pre_growing_season_events',
            'pick_post_growing_season_events'
                ]

        out_dir = this_root+'arr\\recovery_time\\composite_3_modes\\'
        Tools().mk_dir(out_dir)
        void_dic = DIC_and_TIF().void_spatial_dic()
        for m in mode:
            fdir = this_root+'arr\\recovery_time\\{}_composite_recovery_time\\'.format(m)
            f = fdir+'composite.npy'
            dic = dict(np.load(f).item())
            for pix in tqdm(dic,desc=m):
                vals = dic[pix]
                for val in vals:
                    void_dic[pix].append(val)
        print '\nsaving...'
        np.save(out_dir+'composite_3_mode_recovery_time',void_dic)


    def gen_composite_recovery_time_tif(self):

        out_tif = this_root+'tif\\recovery_time\\recovery_time_mix.tif'
        dic = dict(np.load(this_root+'arr\\recovery_time\\composite_3_modes\\composite_3_mode_recovery_time.npy').item())

        spatial_dic = {}
        for pix in tqdm(dic):
            if not pix in self.ndvi_valid_pix:
                continue
            vals = dic[pix]
            sum_recovery_time = 0.
            flag = 0.
            if len(vals) > 0:
                for recovery_time,mark,recovery_range in vals:
                    if recovery_time != None:  # 当 recovery_time 不为 None 时
                        sum_recovery_time += recovery_time
                        flag += 1.
            if flag != 0:
                mean_recovery_time = sum_recovery_time/flag
                spatial_dic[pix] = mean_recovery_time
            else:
                spatial_dic[pix] = np.nan

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().arr_to_tif(arr,out_tif)

    def recovery_latitude_3mode(self):
        # 统计不同纬度的恢复期
        mode = [
            'pick_non_growing_season_events',
            'pick_pre_growing_season_events',
            'pick_post_growing_season_events'
        ]

        lats = [-50, -20, 20, 50, 90][::-1]
        # lats = [0,80,140,]
        lats_new = []
        for lat in lats:
            lat = (90 - lat) * 2
            lats_new.append(lat)
        lats = np.array(lats_new, dtype=int)
        ranges = []
        for i in range(len(lats)):
            if i + 1 == len(lats):
                break
            range_i = [lats[i], lats[i + 1]]
            ranges.append(range_i)
            print range_i


        for m in mode:
            tif = this_root+'tif\\recovery_time\\{}_plot_gen_recovery_time\\global.tif'.format(m)
            arr,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(tif)
            grid = arr < 0
            arr[grid] = np.nan

            grid1 = arr > 18
            arr[grid1] = np.nan

            # plt.imshow(arr,'jet')
            # plt.colorbar()
            # plt.show()
            # exit()
            lats_selected = []
            for range_i in ranges:
                # print range_i
                selected_i = []
                for i in range(len(arr)):
                    if range_i[0] < i <= range_i[1]:
                        # print i
                        temp = []
                        for j in range(len(arr[0])):
                            pix = '%03d.%03d' % (i, j)
                            if pix in self.tropical_pix:
                                continue
                            val = arr[i][j]
                            temp.append(val)
                        selected_i.append(temp)

                selected_pix = []
                for j in selected_i:
                    for k in j:
                        if not np.isnan(k):
                            selected_pix.append(k)

                lats_selected.append(selected_pix)
            bar = []
            for i in lats_selected:
                if len(i) > 0:
                    bar.append(np.mean(i))
                else:
                    bar.append(np.nan)
            plt.figure()
            plt.bar(range(len(bar)), bar)
            # plt.plot(range(len(bar)),bar,label=m)
            plt.title('recovery time (months) {}'.format(m))
            # plt.xticks(range(len(bar))[10::10],lats_ticks[10::10])
            # plt.xticks(range(len(bar)),lats_ticks)
            # plt.ylim(20, 80)
            plt.legend()
            # plt.boxplot(lats_selected)
        plt.show()


    def kernel_recovery_landcover(self,params):
        landcover_dic,landcover_type,dic = params
        landcover_pix = landcover_dic[landcover_type]
        landcover_selected = []
        for pix in dic:
            if not pix in self.ndvi_valid_pix:
                continue
            if pix in self.tropical_pix:
                continue
            if not pix in landcover_pix:
                continue
            val = dic[pix]
            if np.isnan(val):
                continue
            landcover_selected.append(val)
        mean_landcover_selected = np.mean(landcover_selected)
        return mean_landcover_selected


        pass

    def recovery_landcover_3mode(self):
        # 统计不同植被类型的恢复期

        landcover_dic = dict(np.load(this_root + 'arr\\landcover_dic.npy').item())
        # mode_dic = self.load_data()
        mode = [
            'pick_non_growing_season_events',
            'pick_pre_growing_season_events',
            'pick_post_growing_season_events'
        ]
        for m in mode:
            tif = this_root + 'tif\\recovery_time\\{}_plot_gen_recovery_time\\global.tif'.format(m)
            arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif)

            grid = arr < 0
            arr[grid] = np.nan

            grid1 = arr > 18
            arr[grid1] = np.nan


            dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            params = []
            for landcover_type in landcover_dic:
                params.append([landcover_dic,landcover_type,dic])

            bars = MUTIPROCESS(self.kernel_recovery_landcover,params).run()


            plt.bar(range(len(bars)), bars)

            xticks = []
            for i in range(len(bars)):
                label = landcover_types_dic[i+1]
                xticks.append(label)
            plt.xticks(range(len(bars)),xticks,rotation=90)
            plt.title(m)
            plt.show()


    def recovery_latitude_mix(self):
        # 统计不同纬度的恢复期
        lats = [-50, -20, 20, 50, 90][::-1]
        # lats = [0,80,140,]
        lats_new = []
        for lat in lats:
            lat = (90 - lat) * 2
            lats_new.append(lat)
        lats = np.array(lats_new, dtype=int)
        ranges = []
        for i in range(len(lats)):
            if i + 1 == len(lats):
                break
            range_i = [lats[i], lats[i + 1]]
            ranges.append(range_i)
            print range_i

        tif = this_root + 'tif\\recovery_time\\recovery_time_mix.tif'
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif)
        grid = arr < 0
        arr[grid] = np.nan

        grid1 = arr > 18
        arr[grid1] = np.nan

        # plt.imshow(arr,'jet')
        # plt.colorbar()
        # plt.show()
        # exit()
        lats_selected = []
        for range_i in ranges:
            # print range_i
            selected_i = []
            for i in range(len(arr)):
                if range_i[0] < i <= range_i[1]:
                    # print i

                    selected_i.append(arr[i])

            selected_pix = []
            for j in selected_i:
                for k in j:
                    if not np.isnan(k):
                        selected_pix.append(k)

            lats_selected.append(selected_pix)
        bar = []
        for i in lats_selected:
            if len(i) > 0:
                bar.append(np.mean(i))
            else:
                bar.append(np.nan)
        plt.figure()
        plt.bar(range(len(bar)), bar)
        # plt.plot(range(len(bar)),bar,label=m)
        plt.title('recovery time (months)')
        # plt.xticks(range(len(bar))[10::10],lats_ticks[10::10])
        # plt.xticks(range(len(bar)),lats_ticks)
        # plt.ylim(20, 80)
        plt.legend()
        plt.show()


        pass


    def kernel_recovery_landcover_mix(self,params):
        landcover_dic,landcover_type,dic = params
        landcover_pix = landcover_dic[landcover_type]
        landcover_selected = []
        for pix in dic:
            if not pix in self.ndvi_valid_pix:
                continue
            if not pix in landcover_pix:
                continue
            val = dic[pix]
            if np.isnan(val):
                continue
            landcover_selected.append(val)
        mean_landcover_selected = np.mean(landcover_selected)
        return mean_landcover_selected


    def recovery_landcover_mix(self):
        # 统计不同植被类型的恢复期

        landcover_dic = dict(np.load(this_root + 'arr\\landcover_dic.npy').item())
        # mode_dic = self.load_data()
        tif = this_root + 'tif\\recovery_time\\recovery_time_mix.tif'
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif)

        grid = arr < 0
        arr[grid] = np.nan

        grid1 = arr > 18
        arr[grid1] = np.nan


        dic = DIC_and_TIF().spatial_arr_to_dic(arr)
        params = []
        for landcover_type in landcover_dic:
            params.append([landcover_dic,landcover_type,dic])

        bars = MUTIPROCESS(self.kernel_recovery_landcover_mix,params).run()


        plt.bar(range(len(bars)), bars)

        xticks = []
        for i in range(len(bars)):
            label = landcover_types_dic[i+1]
            xticks.append(label)
        plt.xticks(range(len(bars)),xticks,rotation=90)
        plt.show()





class Statistic:
    def __init__(self):
        self.histogram()
        pass

    def get_u_sig(self,val):
        '''
        获取数据的均值μ和标准差σ
        :return: μ,σ
        '''
        u = sum(val) / float(len(val))
        sigma = np.std(val)
        return u, sigma

    def normal_distribution(self, x,u,sig):
        '''
        :param x: input x,single value, not list or tuple
        :return: 正态分布函数值
        '''
        x = float(x)
        norm = (1 / (math.sqrt(2 * math.pi) * sig)) * math.exp(-((x - u) ** 2) / (2 * sig ** 2))
        return norm

    def fit_normal(self,val):
        val_min = min(val)
        val_max = max(val)
        x = np.linspace(val_min, val_max, 1000)
        u, sigma = self.get_u_sig(val)
        y = []
        for i in range(len(x)):
            y.append(self.normal_distribution(x[i], u, sigma))
        # plt.hist(val, bins=11, normed=1, color='b')
        plt.plot(x, y)


    def histogram(self):
        mode = [
            # 'pick_non_growing_season_events',
            'pick_pre_growing_season_events',
            'pick_post_growing_season_events'
                ]

        global_growing_season = dict(np.load(this_root+'NDVI\\global_growing_season.npy').item())

        flag = 0
        plt.figure(figsize=(15, 3))
        for m in mode:
            f = this_root+'arr\\{}_composite_recovery_time\\composite.npy'.format(m)
            dic = dict(np.load(f).item())
            in_hist = []
            out_hist = []
            tropical_hist = []
            for pix in tqdm(dic):
                if pix in global_growing_season:
                    vals = dic[pix]
                    if len(vals) > 0:
                        # print vals
                        for recovery,in_or_out in vals:
                            if in_or_out == 'in':
                                flag += 1
                                in_hist.append(recovery)
                            elif in_or_out == 'out':
                                flag += 1
                                if recovery < 48:
                                    out_hist.append(recovery)
                            elif in_or_out == 'tropical':
                                flag += 1
                                if recovery < 15:
                                    tropical_hist.append(recovery)
                            else:
                                pass

            # plt.suptitle(m)
            plt.subplot(131)
            plt.hist(in_hist, bins=5, alpha=0.5,label=m)
            plt.legend()
            plt.title('recovered in current growing season')
            plt.subplot(132)
            plt.hist(out_hist, bins=8, alpha=0.5,label=m)
            plt.legend()
            plt.title('not recovered in current growing season')
            plt.subplot(133)
            plt.hist(tropical_hist, bins=12, density=1,alpha=0.5)
            plt.title('tropical')
        print flag
        plt.show()


        # Gamma fit
        # plt.figure()
        # _, bins, patchs = plt.hist(hist, bins=12,density=1,alpha=0.3)
        # val_fit = gam.fit(hist)
        # pdf = gam.pdf(range(6), val_fit[0], 0, val_fit[2])
        # plt.plot(range(6), pdf, linewidth=1)

        # Normal fit
        # _, bins, patchs = plt.hist(hist, bins=8, density=1)
        # self.fit_normal(hist)
        pass






    def count_events(self):
        '''
        events with NDVI
        # events number = 1248579
        # events number = 747612
        :return:
        '''
        mode = ['pick_non_growing_season_events',
                'pick_pre_growing_season_events',
                'pick_post_growing_season_events'
                ]

        all_events_dic = DIC_and_TIF().void_spatial_dic()
        new_all_events = {}
        for key in all_events_dic:
            new_all_events[key] = 0.
        flag = 0
        for m in tqdm(mode):
            f = this_root + 'arr\\{}_composite_recovery_time\\composite.npy'.format(m)
            dic = dict(np.load(f).item())
            for pix in dic:
                vals = dic[pix]
                num_events = len(vals)
                new_all_events[pix] += num_events
                flag += num_events
        print flag
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(new_all_events)
        grid = arr==0
        arr[grid] = np.nan
        # DIC_and_TIF().arr_to_tif_GDT_Byte(arr,this_root+'tif\\drought_events.tif')
        # arr = np.ma.masked_where(arr==0,arr)
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()


    def count_events_SPEI(self):
        # events number = 988397
        all_events_dic = DIC_and_TIF().void_spatial_dic()
        new_all_events = {}
        for key in all_events_dic:
            new_all_events[key] = 0.

        flag = 0
        for interval in tqdm(range(1, 13)):
            fdir = this_root+'SPEI\\single_events_24\\SPEI_{}\\'.format('%02d'%interval)
            for f in os.listdir(fdir):
                dic = dict(np.load(fdir+f).item())
                for pix in dic:
                    vals = dic[pix]
                    # print vals
                    event_count = len(vals)
                    new_all_events[pix] += event_count
                    flag += event_count

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(new_all_events)
        grid = arr == 0
        arr[grid] = np.nan
        # DIC_and_TIF().arr_to_tif_GDT_Byte(arr, this_root + 'tif\\drought_events_SPEI.tif')
        print flag
        # arr = np.ma.masked_where(arr==0,arr)
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()
        pass


    # def


class RATIO:

    def __init__(self):

        self.valid_pix()

        pass

    def run(self):
        # self.cal_ratio()
        # self.plot_in_or_out()
        self.ratio_latitude()
        # self.load_data()
        # self.gen_landcover_dic()
        # self.ratio_landcover()

        pass


    def valid_pix(self):
        self.ndvi_valid_pix = Tools().filter_NDVI_valid_pix()
        self.tropical_pix = np.load(this_root + 'NDVI\\tropical_pix.npy')

    def load_data(self):

        mode = [
            'pick_non_growing_season_events',
            'pick_pre_growing_season_events',
            'pick_post_growing_season_events'
        ]

        mode_dic = {}
        for m in mode:
            f = this_root + 'arr\\{}_composite_recovery_time\\composite.npy'.format(m)
            dic = dict(np.load(f).item())
            mode_i = {}
            for pix in tqdm(dic,desc=m):
                vals = dic[pix]
                mode_i[pix] = vals
            mode_dic[m] = mode_i

        return mode_dic


    def cal_ratio(self):
        out_dir = this_root+'arr\\ratio\\'
        Tools().mk_dir(out_dir)

        mode = [
            'pick_non_growing_season_events',
            'pick_pre_growing_season_events',
            'pick_post_growing_season_events'
                ]
        ndvi_valid_pix = Tools().filter_NDVI_valid_pix()
        tropical_pix = np.load(this_root+'NDVI\\tropical_pix.npy')

        for m in mode:
            f = this_root + 'arr\\{}_composite_recovery_time\\composite.npy'.format(m)
            dic = dict(np.load(f).item())
            ratio_dic = {}
            for pix in tqdm(dic):
                if not pix in ndvi_valid_pix:
                    continue
                if pix in tropical_pix:
                    continue
                vals = dic[pix]
                if len(vals) == 0:
                    continue

                in_flag = 0.
                for recovery,in_or_out in vals:
                    # print recovery,in_or_out
                    if in_or_out == 'out':
                        in_flag += 1
                # if in_flag != 0:
                ratio = in_flag/len(vals)
                ratio_dic[pix] = int(ratio*100)
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(ratio_dic)
            np.save(out_dir+m,arr)
            plt.figure()
            plt.imshow(arr,'jet')
            plt.colorbar()
            plt.title(m)
        plt.show()

        pass

    def plot_in_or_out(self):
        mode = [
            # 'pick_non_growing_season_events',
            'pick_pre_growing_season_events',
            'pick_post_growing_season_events'
                ]
        ndvi_valid_arr = Tools().filter_NDVI_valid_pix()
        for m in mode:
            f = this_root + 'arr\\{}_composite_recovery_time\\composite.npy'.format(m)
            dic = dict(np.load(f).item())
            in_dic = {}
            out_dic = {}
            for pix in tqdm(dic):
                if not pix in ndvi_valid_arr:
                    continue
                vals = dic[pix]
                if len(vals) == 0:
                    continue
                in_list = []
                out_list = []
                for recovery, in_or_out in vals:
                    # print recovery,in_or_out
                    if in_or_out == 'in':
                        in_list.append(recovery)
                    if in_or_out == 'out':
                        out_list.append(recovery)
                mean_inlist = np.mean(in_list)
                mean_outlist = np.mean(out_list)
                in_dic[pix] = mean_inlist
                out_dic[pix] = mean_outlist
            arr_in = DIC_and_TIF().pix_dic_to_spatial_arr(in_dic)
            arr_out = DIC_and_TIF().pix_dic_to_spatial_arr(out_dic)
            plt.figure()
            plt.imshow(arr_in, 'jet',vmin=0,vmax=4)
            plt.colorbar()
            plt.title(m+'_IN')

            plt.figure()
            plt.imshow(arr_out, 'jet',vmin=7,vmax=17)
            plt.colorbar()
            plt.title(m+'_OUT')
            plt.show()

    def ratio_latitude(self):
        fdir = this_root + 'arr\\ratio\\'
        mode = [
            'pick_non_growing_season_events',
            'pick_pre_growing_season_events',
            'pick_post_growing_season_events'
        ]
        lats_ticks = []
        # lats = np.linspace(0,360,13)
        # lats = [
        #     [-20,20],
        #     [20,50],
        #     [50,90],
        #     [-50,-20]
        # ]
        lats = [-50,-20,20,50,90][::-1]
        # lats = [0,80,140,]
        lats_new = []
        for lat in lats:
            lat = (90-lat)*2
            lats_new.append(lat)
        lats = np.array(lats_new,dtype=int)
        ranges = []
        for i in range(len(lats)):
            if i+1 == len(lats):
                break
            range_i = [lats[i],lats[i+1]]
            ranges.append(range_i)
            print range_i
            # lat_tick = 90. - 0.5*range_i[0]
            # lats_ticks.append(lat_tick)

        for m in mode:
            arr = np.load(fdir+m+'.npy')
            lats_selected = []
            for range_i in ranges:
                # print range_i
                selected_i = []
                for i in range(len(arr)):
                    if range_i[0] < i <= range_i[1]:
                        # print i
                        selected_i.append(arr[i])

                selected_pix = []
                for j in selected_i:
                    for k in j:
                        if not np.isnan(k):
                            selected_pix.append(k)

                lats_selected.append(selected_pix)
            bar = []
            for i in lats_selected:
                if len(i) > 0:
                    bar.append(np.mean(i))
                else:
                    bar.append(np.nan)
            plt.figure()
            plt.bar(range(len(bar)),bar)
            # plt.plot(range(len(bar)),bar,label=m)
            plt.title('Recovered in current growing season Ratio {}'.format(m))
            # plt.xticks(range(len(bar))[10::10],lats_ticks[10::10])
            # plt.xticks(range(len(bar)),lats_ticks)
            plt.ylim(20,80)
            # plt.boxplot(lats_selected)
        plt.show()


    def gen_landcover_dic(self):

        # tif = r'D:\project05\landcover\tif\0.5\landcover_0.5.tif'
        # array,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(tif)
        # plt.imshow(array)
        # plt.colorbar()
        # plt.show()
        landcover_dir = this_root + 'landcover\\per_pix\\'
        landcover_dic = {}

        for i in range(18):
            landcover_dic[i] = []

        for f in tqdm(os.listdir(landcover_dir), desc='loading landcover'):
            dic = dict(np.load(landcover_dir + f).item())
            for pix in dic:
                val = dic[pix][0]
                landcover_dic[val].append(pix)
        # for type_ in range()
        landcover_dic_array = {}
        for i in landcover_dic:
            landcover_dic_array[i] = np.array(landcover_dic[i])

        np.save(this_root+'arr\\landcover_dic',landcover_dic_array)

    def ratio_landcover(self):
        landcover_dic = dict(np.load(this_root+'arr\\landcover_dic.npy').item())
        # mode_dic = self.load_data()
        fdir = this_root + 'arr\\ratio\\'
        mode = [
            # 'pick_non_growing_season_events',
            'pick_pre_growing_season_events',
            'pick_post_growing_season_events'
        ]
        for m in mode:
            arr = np.load(fdir + m + '.npy')
            dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            bars = []
            for landcover_type in tqdm(landcover_dic):
                landcover_pix = landcover_dic[landcover_type]
                landcover_selected = []
                for pix in dic:
                    if not pix in self.ndvi_valid_pix:
                        continue
                    if pix in self.tropical_pix:
                        continue
                    if not pix in landcover_pix:
                        continue
                    val = dic[pix]
                    if np.isnan(val):
                        continue
                    landcover_selected.append(val)
                bars.append(np.mean(landcover_selected))

            plt.bar(range(len(bars)),bars)
            plt.title(m)
            plt.show()



def kernel_run(param):
    interval = param
    # PRE_POST_NON_Recovery_time().run(interval, 'non-growing')
    # Pick_Single_events().run(interval)
    Pick_Single_events().run(interval)


def run():
    param = []
    for interval in range(1, 13):
        param.append(interval)
        # Pick_Single_events().run()
        # Recovery_time_winter(interval)
    MUTIPROCESS(kernel_run, param).run()


def main():
    # run()
    # Pre_Process()
    # Pick_Single_events()
    # Recovery_time_winter()
    Recovery_time_winter_2()
    # Statistic()
    # RATIO().run()






    pass


if __name__ == '__main__':
    main()
