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
from multiprocessing.pool import ThreadPool as TPool
import copy_reg
import types
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
        flag = 0
        for i in range(len(val)):
            if val[i] >= -10:
                flag+=1.
                index = i
                x = np.append(x, index)
                val_new = np.append(val_new, val[i])
        if flag/len(val) < 0.9:
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
        tif_template = this_root + 'conf\\SPEI.tif'
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

        lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        # print(pix)
        lon, lat = lon_lat_dic[pix]
        print(lon, lat)

        history_dic = dict(np.load(this_root + 'arr\\pix_to_address_history.npy').item())

        if pix in history_dic:
            # print(history_dic[pix])
            return lon,lat,history_dic[pix]
        else:

            address = lon_lat_to_address.lonlat_to_address(lon, lat).decode('utf-8')
            key = pix
            val = address
            history_dic[key] = val
            np.save(this_root + 'arr\\pix_to_address_history.npy',history_dic)
            return lon,lat,address

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
            params.append([sta_pos_dic,pix])
        result = MUTIPROCESS(self.kernel_gen_lon_lat_address_dic,params).run(process=50,process_or_thread='t',text='downloading')
        np.save(this_root + 'conf\\sta_add_dic', result)


    def kernel_gen_lon_lat_address_dic(self,params):
        sta_pos_dic,pix = params
        lon, lat = sta_pos_dic[pix]
        # print(lat,lon)
        add = lon_lat_to_address.lonlat_to_address(lon, lat).decode('utf-8')
        return add
        # print(pix,lon,lat)
        # print(add)




    def do_multiprocess(self,func,params,process=6,process_or_thread='p',**kwargs):
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

        if process_or_thread == 'p':
            pool = multiprocessing.Pool(process)
        elif process_or_thread == 't':
            pool = TPool()
        else:
            raise IOError('process_or_thread key error, input keyword such as "p" or "t"')

        results = list(tqdm(pool.imap(func, params), total=len(params),**kwargs))
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


class MUTIPROCESS:
    '''
    可对类内的函数进行多进程并行
    由于GIL，多线程无法跑满CPU，对于不占用CPU的计算函数可用多线程
    并行计算加入进度条
    '''
    def __init__(self,func,params):
        self.func = func
        self.params = params
        copy_reg.pickle(types.MethodType, self._pickle_method)
        pass

    def _pickle_method(self,m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)


    def run(self,process=6,process_or_thread='p',**kwargs):
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

        if process_or_thread == 'p':
            pool = multiprocessing.Pool(process)
        elif process_or_thread == 't':
            pool = TPool()
        else:
            raise IOError('process_or_thread key error, input keyword such as "p" or "t"')

        results = list(tqdm(pool.imap(self.func, self.params), total=len(self.params),**kwargs))
        pool.close()
        pool.join()
        return results



class KDE_plot:

    def __init__(self):

        pass

    def reverse_colourmap(self,cmap, name='my_cmap_r'):
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

    def plot_scatter(self,val1, val2, cmap='magma', reverse=0, s=0.3,title=''):

        kde_val = np.array([val1, val2])
        print('doing kernel density estimation... ')
        densObj = kde(kde_val)
        dens_vals = densObj.evaluate(kde_val)
        colors = self.makeColours(dens_vals, cmap, reverse=reverse)
        plt.figure()
        plt.title(title)
        plt.scatter(val1, val2, c=colors, s=s)



class Cal_anomaly:
    def __init__(self):
        pass

    def kernel_cal_anomaly(self,params):
        fdir,f ,save_dir= params
        pix_dic = dict(np.load(fdir + f).item())
        anomaly_pix_dic = {}
        for pix in pix_dic:
            ####### one pix #######
            vals = pix_dic[pix]
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
            for i in range(len(vals)):
                mon = i % 12
                std_ = climatology_std[mon]
                mean_ = climatology_means[mon]
                if std_ == 0:
                    anomaly = -999999
                else:
                    anomaly = (vals[i] - mean_)/std_

                pix_anomaly.append(anomaly)
            anomaly_pix_dic[pix] = pix_anomaly

        np.save(save_dir + f, anomaly_pix_dic)


    def cal_anomaly(self,):
        fdir = this_root + 'NDVI\\per_pix\\'
        save_dir = this_root + 'NDVI\\per_pix_anomaly2\\'
        Tools().mk_dir(save_dir)
        flist = os.listdir(fdir)
        time_init = time.time()
        # flag = 0
        params = []
        for f in flist:
            # print(f)
            params.append([fdir,f,save_dir])

        # for p in params:
        #     print(p[1])
        #     kernel_cal_anomaly(p)
        # Tools().do_multiprocess(self.kernel_cal_anomaly,params,process=2,process_or_thread='t',text='calculating anomaly...')
        MUTIPROCESS(self.kernel_cal_anomaly,params).run(process=6,process_or_thread='p',text='calculating anomaly...')


class Pick_Single_events():
    def __init__(self):

        pass

    def pick_plot(self):
        # 作为pick展示
        # 前36个月和后36个月无极端干旱事件
        n = 36
        spei_dir = this_root+'SPEI\\per_pix\\'
        for f in os.listdir(spei_dir):
            if '015' not in f:
                continue
            print(f)
            spei_dic = dict(np.load(spei_dir+f).item())
            for pix in spei_dic:

                spei = spei_dic[pix]
                spei = Tools().interp_1d(spei)
                if len(spei) == 1 or spei[0] == -999999:
                    continue
                spei = Tools().forward_window_smooth(spei, 3)
                params = [spei,pix,n]
                events_dic, key = self.kernel_find_drought_period(params)

                events_4 = [] # 严重干旱事件
                for i in events_dic:
                    level, date_range = events_dic[i]
                    # print(level,date_range)
                    if level == 4:
                        events_4.append(date_range)

                for i in range(len(events_4)):
                    spei_v = self.get_spei_vals(spei, events_4[i])
                    plt.plot(events_4[i], spei_v, c='black',zorder=99)

                    if i - 1 < 0:# 首次事件
                        if events_4[i][0] - n < 0 or events_4[i][-1] + n >= len(spei):# 触及两边则忽略
                            continue
                        if len(events_4) == 1:
                            spei_v = self.get_spei_vals(spei, events_4[i])
                            plt.plot(events_4[i], spei_v, linewidth=6, c='g')
                        elif events_4[i][-1] + n <= events_4[i+1][0]:
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
                    if events_4[i][0] - events_4[i-1][-1] >= n and events_4[i][-1] + n <= events_4[i+1][0]:
                        spei_v = self.get_spei_vals(spei, events_4[i])
                        plt.plot(events_4[i], spei_v, linewidth=6, c='g')

                #################### PLOT ##################
                print(pix)
                lon, lat, add = Tools().pix_to_address(pix)
                print(add)
                plt.plot(spei,'r')
                plt.title(add+'_{}.{}'.format(lon,lat))
                plt.plot(range(len(spei)),[-2]*len(spei),'--',c='black')
                plt.plot(range(len(spei)),[-0.5]*len(spei),'--',c='black')
                plt.grid()
                plt.show()
                print('******')
                #################### PLOT ##################
        pass


    def get_spei_vals(self,spei,indxs):
        picked_vals = []
        for i in indxs:
            picked_vals.append(spei[i])
        return picked_vals


    def pick(self):
        # 前36个月和后36个月无极端干旱事件
        n = 24
        spei_dir = this_root+'SPEI\\per_pix\\'
        out_dir = this_root+'SPEI\\single_events_{}\\'.format(n)
        Tools().mk_dir(out_dir)
        for f in tqdm(os.listdir(spei_dir)):
            spei_dic = dict(np.load(spei_dir+f).item())
            single_event_dic = {}
            for pix in spei_dic:
                spei = spei_dic[pix]
                spei = Tools().interp_1d(spei)
                if len(spei) == 1 or spei[0] == -999999:
                    single_event_dic[pix] = []
                    continue
                spei = Tools().forward_window_smooth(spei, 3)
                params = [spei,pix,n]
                events_dic, key = self.kernel_find_drought_period(params)

                events_4 = [] # 严重干旱事件
                for i in events_dic:
                    level, date_range = events_dic[i]
                    if level == 4:
                        events_4.append(date_range)
                single_event = []
                for i in range(len(events_4)):
                    if i - 1 < 0:# 首次事件
                        if events_4[i][0] - n < 0 or events_4[i][-1] + n >= len(spei):# 触及两边则忽略
                            continue
                        if len(events_4) == 1:
                            single_event.append(events_4[i])
                        elif events_4[i][-1] + n <= events_4[i+1][0]:
                            single_event.append(events_4[i])
                        continue

                    # 最后一次事件
                    if i + 1 >= len(events_4):
                        if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= len(spei):
                            single_event.append(events_4[i])
                        break

                    # 中间事件
                    if events_4[i][0] - events_4[i-1][-1] >= n and events_4[i][-1] + n <= events_4[i+1][0]:
                        single_event.append(events_4[i])
                single_event_dic[pix] = single_event
            np.save(out_dir+f,single_event_dic)


    def kernel_find_drought_period(self,params):
        # 根据不同干旱程度查找干旱时期
        pdsi = params[0]
        key = params[1]
        n = params[2]
        drought_month = []
        for i, val in enumerate(pdsi):
            if val < -0.5:
                drought_month.append(i)
            else:
                drought_month.append(-99)
        # plt.plot(drought_month)
        # plt.show()
        events = []
        event_i = []
        for ii in drought_month:
            if ii > -99:
                event_i.append(ii)
            else:
                if len(event_i) > 3:
                    events.append(event_i)
                    event_i = []
                else:
                    event_i = []
        # print(len(pdsi))
        # print(event_i)
        if len(event_i) > 3:
            events.append(event_i)

        # print(events)

        # 去除两头小于0的index
        # events_new = []
        # for event in events:
        #     print(event)
        # exit()

        flag = 0
        events_dic = {}

        # 取两个端点
        for i in events:
            # print(i)
            # 去除两端pdsi值小于-0.5
            if 0 in i or len(pdsi) - 1 in i:
                continue
            new_i = []
            for jj in i:
                # print(jj)
                if jj - 1 >= 0:
                    new_i.append(jj - 1)
                else:
                    pass
            new_i.append(i[-1])
            if i[-1] + 1 < len(pdsi):
                new_i.append(i[-1] + 1)
            # print(new_i)
            # exit()
            flag += 1
            vals = []
            for j in new_i:
                try:
                    vals.append(pdsi[j])
                except:
                    print(j)
                    print('error')
                    print(new_i)
                    exit()
            # print(vals)

            # if 0 in new_i:

            min_val = min(vals)
            if -1 <= min_val < -.5:
                level = 1
            elif -1.5 <= min_val < -1.:
                level = 2
            elif -2 <= min_val < -1.5:
                level = 3
            elif min_val <= -2.:
                level = 4
            else:
                print('error')
                print(vals)
                print(min_val)
                time.sleep(1)
                continue
            events_dic[flag] = [level, new_i]
            # print(min_val)
            # plt.plot(vals)
            # plt.show()
        # for key in events_dic:
        #     # print key,events_dic[key]
        #     if 0 in events_dic[key][1]:
        #         print(events_dic[key])
        # exit()
        return events_dic, key

    def pick_extreme_events(self):


        pass

def main():

    # fdir = this_root+'SPEI\\tif\\'
    # outdir = this_root+'SPEI\\per_pix\\'
    # Tools().mk_dir(outdir)
    # Tools().data_transform(fdir,outdir)
    # Cal_anomaly().cal_anomaly()

    Pick_Single_events().pick()
    # Tools().pix()
    # Tools().pix()
    # Tools().gen_lon_lat_address_dic()





    pass



if __name__ == '__main__':
    main()