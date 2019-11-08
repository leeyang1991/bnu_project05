# coding=utf-8
'''
author: LiYang
Date: 20190801
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
import log_process
import multiprocessing
import datetime
import lon_lat_to_address
from scipy import stats, linalg
import pandas
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
import imageio
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl

this_root = 'D:\\project05\\'
# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 显示负号
plt.rcParams['axes.unicode_minus'] = False




class Tools:
    def __init__(self):
        pass

    def mk_dir(self,dir,force=False):

        if not os.path.isdir(dir):
            if force == True:
                os.makedirs(dir)
            else:
                os.mkdir(dir)

    def interp_1d(self,val):
        # 1、插缺失值
        x = []
        val_new = []
        for i in range(len(val)):
            if val[i] >= -90:
                index = i
                x = np.append(x, index)
                val_new = np.append(val_new, val[i])

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

    def smooth(self,x):
        # 后窗滤波
        # 滑动平均
        x = np.array(x)
        temp = 0
        new_x = []
        for i in range(len(x)):
            if i+3 == len(x):
                break
            temp += x[i]+x[i+1]+x[i+2]+x[i+3]
            new_x.append(temp/4.)
            temp = 0
        return np.array(new_x)

    def forward_window_smooth(self,x,window=3):
        # 前窗滤波
        # window = window-1

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
            if i-window<0:
                continue
            temp = 0
            for w in range(window):
                temp += x[i-w]
            smoothed = temp/float(window)
            new_x = np.append(new_x,smoothed)
        return new_x

    def detrend_dic(self,dic):
        dic_new = {}
        for key in dic:
            vals = dic[key]
            vals_new = signal.detrend(vals)
            dic_new[key] = vals_new

        return dic_new


    def arr_mean(self,arr,threshold):
        grid = arr > threshold
        arr_mean = np.mean(arr[np.logical_not(grid)])
        return arr_mean

    def arr_mean_greater(self,arr,threshold):
        # mask greater
        grid_nan = np.isnan(arr)
        grid = np.logical_not(grid_nan)
        # print(grid)
        arr[np.logical_not(grid)] = 255
        grid = arr < threshold
        arr_mean = np.mean(arr[np.logical_not(grid)])
        return arr_mean

    def rename(self):
        fdir = this_root+'GPP\\smooth_per_pix\\'
        flist = os.listdir(fdir)
        for f in flist:
            print(f)
            f_new = f.replace('gpp_','')
            os.rename(fdir+f,fdir+f_new)



    def per_pix_dic_to_spatial_tif(self,mode,folder):

        outfolder = this_root+mode+'\\'+folder+'_tif\\'
        self.mk_dir(outfolder)
        tif_template = this_root+'conf\\tif_template.tif'
        _, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        fdir = this_root+mode+'\\'+folder+'\\'
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
            to_raster.array2raster(outfolder+'%03d.tif'%date,originX, originY, pixelWidth, pixelHeight,spatial)
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

    def spatial_tif_to_lon_lat_dic(self):
        tif_template = this_root + 'conf\\tif_template.tif'
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        # print(originX, originY, pixelWidth, pixelHeight)
        # exit()
        pix_to_lon_lat_dic = {}
        for i in tqdm(range(len(arr))):
            for j in range(len(arr[0])):
                pix = '%03d.%03d'%(i,j)
                lon = originX+pixelWidth*j
                lat = originY+pixelHeight*i
                pix_to_lon_lat_dic[pix] = [lon,lat]
        print('saving')
        np.save(this_root+'arr\\pix_to_lon_lat_dic',pix_to_lon_lat_dic)


    def gen_lag_arr(self,arr1,arr2,lag):
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
        return arr1,arr2


    def gen_lag_arr_multiple_arrs(self,arrs,lag):
        # 谁提前谁 0
        # 谁滞后谁 -1
        lag = int(lag)
        arr_list = []
        for arr in arrs:
            arr_list.append(list(arr))
        if lag > 0:
            for _ in range(lag):
                for i,arr in enumerate(arr_list):
                    if i == 0:
                        arr.pop(0)
                        continue
                    arr.pop(-1)

            pass
        elif lag < 0:
            for _ in range(lag):
                for i,arr in enumerate(arr_list):
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


    def partial_corr(self,df):
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

    def pix_dic_to_spatial_arr(self,spatial_dic):

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

        # hist = []
        # for v in all_vals:
        #     if not np.isnan(v):
        #         if 00<v<1.5:
        #             hist.append(v)

        spatial = np.array(spatial)
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

    def pix_to_address(self,pix):
        # 只适用于单个像素查看，不可大量for循环pix，存在磁盘重复读写现象
        if not os.path.isfile(this_root + 'arr\\pix_to_address_history.npy'):
            np.save(this_root + 'arr\\pix_to_address_history.npy',{0:0})
        else:
            history_dic = dict(np.load(this_root + 'arr\\pix_to_address_history.npy').item())

            if pix in history_dic:
                print(history_dic[pix])
                print('address from search history')
                return history_dic[pix]
            else:
                lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
                print(pix)
                print('address from baidu API')
                lon, lat = lon_lat_dic[pix]
                # address = lon_lat_to_address.lonlat_to_address(lon, lat)
                address = lon_lat_to_address.lonlat_to_address(lon, lat).decode('utf-8')
                key = pix
                val = address
                # address_history = {}
                history_dic[key] = val
                np.save(this_root + 'arr\\pix_to_address_history.npy',history_dic)
                return address

    def arr_to_tif_GDT_Byte(self,array,newRasterfn):
        # template
        tif_template = this_root + 'conf\\tif_template.tif'
        _, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        grid_nan = np.isnan(array)
        grid = np.logical_not(grid_nan)
        array[np.logical_not(grid)] = 255
        to_raster.array2raster_GDT_Byte(newRasterfn,originX, originY,pixelWidth,pixelHeight,array)
        pass

    def arr_to_tif(self,array,newRasterfn):
        # template
        tif_template = this_root + 'conf\\tif_template.tif'
        _, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        grid_nan = np.isnan(array)
        grid = np.logical_not(grid_nan)
        array[np.logical_not(grid)] = -999999
        to_raster.array2raster(newRasterfn,originX, originY,pixelWidth,pixelHeight,array)
        pass


    def pick_vals_from_array(self,array, index):

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
        for r,c in index:
            val = array[r,c]
            picked_val.append(val)
        picked_val = np.array(picked_val)
        return picked_val
        pass


    def filter_3_sigma(self,arr_list):
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

    def get_sta_position(self):
        f = open(this_root+'conf\\sta_pos.csv', 'r')
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


    def gen_hi_scatter_plot_range_index(self,n):
        # 把HI按分成n份，分别存储pix位置
        Tools().mk_dir(this_root + 'arr\\gen_hi_scatter_plot_range_index\\')
        out_dic_name = this_root + 'arr\\gen_hi_scatter_plot_range_index\\' + str(n)+'.npy'
        if os.path.isfile(out_dic_name):
            return dict(np.load(out_dic_name).item())
        else:
            print(out_dic_name+' is not existed, generating...')
            HI = np.load(this_root + 'arr\\fusion_HI\\fusion.npy')
            # arr = np.load(r'E:\project02\arr\plot_z_score_length\GPP\level1.npy')

            min_max = np.linspace(0.2, 1.6, n)

            #建立空字典
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
                            index_dic[min_].append([ii,jj])
            np.save(out_dic_name,index_dic)
            return index_dic


    def gen_lon_lat_address_dic(self):
        sta_pos_dic = np.load(this_root + 'conf\\sta_pos_dic.npz')

        sta_add_dic = {}
        for sta in tqdm(sta_pos_dic):
            lat,lon = sta_pos_dic[sta]
            # print(lat,lon)
            add = lon_lat_to_address.lonlat_to_address(lon, lat).decode('utf-8')
            # print(add)
            sta_add_dic[sta] = add

        np.save(this_root+'conf\\sta_add_dic',sta_add_dic)

    def do_multiprocess(self,func,params,process=6):
        pool = multiprocessing.Pool(process)
        results = list(tqdm(pool.imap(func, params), total=len(params), ncols=50))
        pool.close()
        pool.join()
        return results

    def data_transform(self,fdir,outdir):
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
        for d in tqdm(date_list,'loading...'):
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
        for r in tqdm(range(row),'transforming...'):
            for c in range(col):
                for arr in all_array:
                    val = arr[r][c]
                    void_dic['%03d.%03d' % (r, c)].append(val)

        # for i in void_dic_list:
        #     print(i)
        # exit()
        flag = 0
        temp_dic = {}
        for key in tqdm(void_dic_list,'outputting...'):
            flag += 1
            # print('saving ',flag,'/',len(void_dic)/100000)
            temp_dic[key] = void_dic[key]
            if flag % 10000 == 0:
                print('\nsaving %02d' % (flag / 10000)+'\n')
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)



def kernel_cal_anomaly(params):
    fdir,f ,save_dir= params
    pix_dic = dict(np.load(fdir + f).item())
    anomaly_pix_dic = {}
    for pix in pix_dic:
        ####### one pix #######
        climatology_means = []
        climatology_std = []
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
        # for i in range(len(pix_dic[pix])):





    np.save(save_dir + f, anomaly_pix_dic)


def cal_anomaly():
    fdir = this_root + 'NDVI\\per_pix\\'
    save_dir = this_root + 'NDVI\\per_pix_anomaly1\\'
    Tools().mk_dir(save_dir)
    flist = os.listdir(fdir)
    time_init = time.time()
    # flag = 0
    params = []
    for f in flist:
        params.append([fdir,f,save_dir])

    # for p in params:
    #     kernel_cal_anomaly(p)
    Tools().do_multiprocess(kernel_cal_anomaly,params)


def main():

    fdir = this_root+'SPEI\\tif\\'
    outdir = this_root+'SPEI\\per_pix\\'
    Tools().mk_dir(outdir)
    # Tools().data_transform(fdir,outdir)
    cal_anomaly()


if __name__ == '__main__':
    main()