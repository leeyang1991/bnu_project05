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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import math
import analysis
from tqdm import tqdm

this_root = 'D:\\project05\\'


class Prepare:
    def __init__(self):
        pass

    def run(self):
        mode = [
            # 'pick_non_growing_season_events',
            # 'pick_pre_growing_season_events',
            # 'pick_post_growing_season_events',
            'mix'
        ]
        for m in mode:
            # 1 因变量 Y
            print m
            self.prepare_Y(m)
            # 2 自变量 X
            # 2.1 计算月平均 tif 1-12 月
            # outdir = this_root + 'GLOBSWE\\monthly_SWE_max\\'
            # fdir = this_root + 'GLOBSWE\\tif\\SWE_max\\'
            # self.cal_monthly_mean(fdir,outdir)
            # 2.2 根据字典Y 的key 生成 X 字典， key为对应标签
            # X in ['TMP','PRE','CCI','SWE']
            for x in ['TMP','PRE','CCI','SWE']:
                print x
                self.prepare_X(x,m)
            # self.check_ndvi()
            # 3 检查 X
            # self.check_X('SWE')
        pass

    def prepare_Y(self,mode):

        # 1 drought periods
        if mode != 'mix':
            out_dir = this_root+'random_forest\\{}\\'.format(mode)
            analysis.Tools().mk_dir(out_dir)
            print 'loading f_recovery_time...'
            f_recovery_time = this_root+'arr\\recovery_time\\{}_composite_recovery_time\\composite.npy'.format(mode)
        else:
            out_dir = this_root + 'random_forest\\{}\\'.format(mode)
            analysis.Tools().mk_dir(out_dir)
            print 'loading f_recovery_time...'
            f_recovery_time = this_root + 'arr\\recovery_time\\composite_3_modes\\composite_3_mode_recovery_time.npy'
        recovery_time = dict(np.load(f_recovery_time).item())
        print 'done'
        Y = {}
        flag = 0
        for pix in tqdm(recovery_time):
            vals = recovery_time[pix]
            # print vals
            for r_time,mark,date_range,drought_range in vals:
                if r_time == None:  #r_time 为 TRUE
                    continue
                flag += 1
                start = date_range[0]
                end = start + r_time
                key = pix+'_'+mark+'_'+'{}.{}'.format(start,end)
                # print key
                Y[key] = r_time
        # print flag
        # flag=1192218
        # flag=198075
        np.save(out_dir+'Y',Y)


    def cal_monthly_mean(self,fdir,outdir):
        # outdir = this_root + 'TMP\\mon_mean_tif\\'
        # fdir = this_root + 'TMP\\tif\\'

        analysis.Tools().mk_dir(outdir)

        for m in tqdm(range(1, 13)):
            # if m in range(6,10):
            #     continue
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



    def prepare_X(self,x,mode):
        Y_dic = dict(np.load(this_root+'random_forest\\{}\\Y.npy'.format(mode)).item())
        if x in ['TMP','PRE']:
            per_pix_dir = this_root+'{}\\per_pix\\'.format(x)
            mean_dir = this_root+'{}\\mon_mean_tif\\'.format(x)
        elif x == 'CCI':
            per_pix_dir = this_root + 'CCI\\0.5\\per_pix\\'
            mean_dir = this_root + 'CCI\\0.5\\monthly_mean\\'
        elif x == 'SWE':
            per_pix_dir = this_root + 'GLOBSWE\\per_pix\\SWE_max_408\\'
            mean_dir = this_root + 'GLOBSWE\\monthly_SWE_max\\'
        else:
            raise IOError('x error')
        # 1 加载所有原始数据
        all_dic = {}
        for f in tqdm(os.listdir(per_pix_dir), desc='1/3 loading per_pix_dir ...'):
            dic = dict(np.load(per_pix_dir+f).item())
            for pix in dic:
                all_dic[pix] = dic[pix]

        # 2 加载月平均数据

        if x == 'SWE':
            month_range = [1,2,3,4,5,10,11,12]
        else:
            month_range = range(1,13)
        mean_dic = {}
        for m in tqdm(month_range,desc='2/3 loading monthly mean ...'):
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
                if not mon in month_range:
                    continue
                mon = '%02d'%mon
                mon_mean = mean_dic[mon][pix]
                if mon_mean < -9999:
                    continue
                val = vals[dr]
                if val < -9999:
                    continue
                juping = val - mon_mean
                selected_val.append(juping)
            if len(selected_val) > 0:
                if x == 'TMP' or x == 'CCI':
                    juping_mean = np.mean(selected_val)
                else:
                    juping_mean = np.sum(selected_val)
            else:
                juping_mean = np.nan
            X[key] = juping_mean

        np.save(this_root+'random_forest\\{}\\{}'.format(mode,x),X)


    def check_ndvi(self):
        # 看NDVI单像素
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


    def check_X(self,x):
        X = dict(np.load(this_root + 'random_forest\\{}.npy'.format(x)).item())
        Y = dict(np.load(this_root + 'random_forest\\Y.npy').item())
        x = []
        y = []
        for key in X:
            val = X[key]

            if not np.isnan(val):
                split_key = key.split('_')
                pix, mark, date_range = split_key
                if not mark == 'out':
                    continue
                valy = Y[key]
                valx = X[key]
                if valy > 18:
                    continue
                x.append(valx)
                y.append(valy)

                # if mark == 'in':
                #     hist.append(val)
                #     pass
                # if mark == 'out':
                #     hist.append(val)
        plt.scatter(y,x)
        plt.grid()
        plt.show()



class RF_train:

    def __init__(self):
        mode = [
            # 'pick_non_growing_season_events',
            # 'pick_pre_growing_season_events',
            # 'pick_post_growing_season_events'
            'mix'
        ]

        for m in mode:
            self.random_forest_train(m)
        # self.latitude()
        pass


    def latitude(self):
        lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        lats = [-50, -20, 20, 50, 90]

        for i in range(len(lats)):
            if i+1 == len(lats):
                break
            start_lat = lats[i]
            end_lat = lats[i+1]

            selected_pix = []
            for pix in lon_lat_dic:
                lon,lat = lon_lat_dic[pix]
                if lat > start_lat and lat < end_lat:
                    selected_pix.append(pix)
            print selected_pix





        # lats = [0,80,140,]
        # lats_new = []
        # for lat in lats:
        #     lat = (90 - lat) * 2
        #     lats_new.append(lat)


    def load_variable(self,mode,args):

        print args
        fdir = this_root+'random_forest\\{}\\'.format(mode)
        print 'loading variables ...'
        Y_dic = dict(np.load(fdir+'Y.npy').item())
        pre_dic = dict(np.load(fdir+'PRE.npy').item())
        tmp_dic = dict(np.load(fdir+'TMP.npy').item())
        swe_dic = dict(np.load(fdir+'SWE.npy').item())
        cci_dic = dict(np.load(fdir+'CCI.npy').item())
        print 'done'


        keys = []
        for key in Y_dic:
            split_key = key.split('_')
            pix = split_key[0]
            mark_ = split_key[1]
            date_range = split_key[2]
            if mark_ in args:
                key_ = pix+'_'+mark_+'_'+date_range
                keys.append(key_)
        nan = False
        Y = []
        X = []
        for key in keys:
            y = Y_dic[key]
            if y > 18 or y == 0:
                continue
            pre = pre_dic[key]
            tmp = tmp_dic[key]
            cci = cci_dic[key]
            swe = swe_dic[key]
            if np.isnan(pre):
                pre = nan
            if np.isnan(tmp):
                tmp = nan
            if np.isnan(cci):
                cci = nan
            if np.isnan(swe):
                swe = nan

            if 'in' in args:
                Y.append(y)
                X.append([pre,tmp,cci])
            elif 'out' in args:
                Y.append(y)
                X.append([pre, tmp, cci, swe])
            else:
                Y.append(y)
                X.append([pre, tmp, cci])

        return X,Y

        pass

    def random_forest_train(self,mode):
        outdir = this_root+'AI\\RF\\'
        mode_dic = {
            'pick_non_growing_season_events': 'None Growing Season',
                'pick_pre_growing_season_events': 'Early Growing Season',
                'pick_post_growing_season_events': 'Late Growing Season',
                'mix': 'Mix'
                }
        # arg = ['in']
        # arg = ['in','tropical']
        # arg = ['out']
        arg = ['in','tropical','out']
        title = '{} {}'.format(mode_dic[mode],' and '.join(arg))
        print title
        out_pdf = outdir+title+'.pdf'

        # exit()
        X, Y = self.load_variable(mode,arg)
        # X = pd.DataFrame(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

        # clf = RandomForestClassifier(max_depth=None, min_samples_split=2,random_state = 0)
        # clf = sklearn.ensemble.RandomForestRegressor(n_estimators=10, max_depth=None,min_samples_split = 2, random_state = 0)
        # clf = RandomForestRegressor(n_estimators=2000,min_samples_split=1000)
        # clf = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=1, verbose=2)
        # clf = RandomForestRegressor()
        clf = RandomForestClassifier()
        clf.fit(X_train, Y_train)

        importances = clf.feature_importances_
        print importances
        y_min = min(importances)
        y_max = max(importances)
        offset = (y_max-y_min)

        y_min = y_min-offset*0.3
        y_max = y_max+offset*0.3

        plt.ylim(y_min,y_max)
        plt.bar(range(len(importances)),importances,width=0.3)
        plt.xticks(range(len(importances)),['P','T','CCI','SWE'])
        plt.title(title)
        # plt.show()
        plt.savefig(out_pdf)
        plt.close()
        # std = np.std([tree.feature_importances_ for tree in clf.estimators_],
        #              axis=0)
        # indices = np.argsort(importances)[::-1]
        #
        # # Print the feature ranking
        # print("Feature ranking:")
        #
        # for f in range(4):
        #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        #
        # # Plot the feature importances of the forest
        # plt.figure()
        # plt.title("Feature importances")
        # plt.bar(range(4), importances[indices],
        #         color="r", yerr=std[indices], align="center")
        # plt.xticks(range(4), indices)
        # plt.xlim([-1, 4])
        # plt.show()




        # exit()
        # clf.fe
        # y_pred = clf.predict(X_test)
        # r = scipy.stats.pearsonr(Y_test, y_pred)
        # r2 = sklearn.metrics.r2_score(Y_test, y_pred)
        # mse = sklearn.metrics.mean_squared_error(Y_test, y_pred)
        # print('r2:%s\nmse:%s\nr:%s' % (r2, mse, r))
        # analysis.KDE_plot().plot_scatter(Y_test, y_pred,s=40)
        # plt.figure()
        # plt.scatter(Y_test, y_pred)
        # # plt.xlim(-3,3)
        # # plt.ylim(-3,3)
        # plt.show()
        pass


def main():
    # Prepare().run()
    RF_train()
    pass

if __name__ == '__main__':
    main()