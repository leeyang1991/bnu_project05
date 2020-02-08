# coding=gbk

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from analysis import *



class Prepare:
    def __init__(self):
        # self.check_Y()
        # self.prepare_X()
        pass


    def run(self):
        # self.prepare_Y()
        # x = ['TMP', 'PRE', 'CCI', 'SWE']
        # MUTIPROCESS(self.prepare_X,x).run()
        self.prepare_NDVI()
        pass

    def prepare_Y(self):
        # config
        out_dir = this_root+'new_2020\\random_forest\\'
        Tools().mk_dir(out_dir)
        # 1 drought periods
        print '1. loading recovery time'
        f_recovery_time = this_root+'new_2020\\arr\\recovery_time_composite\\composite.npy'
        recovery_time = dict(np.load(f_recovery_time).item())
        print 'done'
        Y = {}
        flag = 0
        for pix in tqdm(recovery_time):
            vals = recovery_time[pix]
            # print vals
            # continue
            for r_time,mark,recovery_date_range,drought_range,eln in vals:
                if r_time == None:  #r_time 为 TRUE
                    continue
                flag += 1
                drought_start = drought_range[0]
                recovery_start = recovery_date_range[0]
                key = pix+'~'+mark+'~'+eln+'~'+'{}.{}'.format(drought_start,recovery_start)
                # print key
                Y[key] = r_time
        # print flag
        # flag=1192218
        # flag=198075
        np.save(out_dir+'Y',Y)


    def check_Y(self):
        print 'loading Y'
        f = this_root+'new_2020\\random_forest\\Y.npy'
        dic = dict(np.load(f).item())
        for key in dic:
            print key,dic[key]

    def prepare_X(self, x):
        # x = ['TMP','PRE','CCI','SWE']
        out_dir = this_root+'new_2020\\random_forest\\'
        Y_dic = dict(np.load(this_root + 'new_2020\\random_forest\\Y.npy').item())
        if x in ['TMP', 'PRE']:
            per_pix_dir = this_root + '{}\\per_pix\\'.format(x)
            mean_dir = this_root + '{}\\mon_mean_tif\\'.format(x)
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
            dic = dict(np.load(per_pix_dir + f).item())
            for pix in dic:
                all_dic[pix] = dic[pix]

        # 2 加载月平均数据

        if x == 'SWE':
            month_range = [1, 2, 3, 4, 5, 10, 11, 12]
        else:
            month_range = range(1, 13)
        mean_dic = {}
        for m in tqdm(month_range, desc='2/3 loading monthly mean ...'):
            m = '%02d' % m
            arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(mean_dir + m + '.tif')
            arr_dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            mean_dic[m] = arr_dic


        # 3 找干旱事件对应的X的距平的平均或求和
        X = {}
        for key in tqdm(Y_dic, desc='3/3 generate X dic ...'):
            split_key = key.split('~')
            pix, mark, eln, date_range = split_key
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
                mon = '%02d' % mon
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

        np.save(out_dir + '{}'.format(x), X)


    def prepare_NDVI(self):
        out_dir = this_root + 'new_2020\\random_forest\\'
        # y
        Y_dic = dict(np.load(this_root + 'new_2020\\random_forest\\Y.npy').item())
        # x
        per_pix_dir = this_root + 'NDVI\\per_pix_anomaly_smooth\\'
        # 1 加载x数据
        all_dic = {}
        # print 'loading Y ...'
        for f in tqdm(os.listdir(per_pix_dir), desc='1/2 loading per_pix_dir ...'):
            dic = dict(np.load(per_pix_dir + f).item())
            for pix in dic:
                vals = dic[pix]
                all_dic[pix] = vals
        # 3 找干旱事件对应的X的距平的平均或求和
        X_two_month_early_vals_mean = {}
        X_NDVI_change = {}
        for key in tqdm(Y_dic, desc='2/2 generate X dic ...'):
            split_key = key.split('~')
            pix, mark, eln, date_range = split_key
            split_date_range = date_range.split('.')
            drought_start = int(split_date_range[0])
            recovery_start = int(split_date_range[1])
            # drought_range = range(drought_start, recovery_start)
            vals = all_dic[pix]
            if drought_start <= recovery_start:
                if recovery_start - 2 >= 0:
                    two_month_early_vals_mean = (vals[recovery_start - 2] + vals[recovery_start - 1])/2.
                    NDVI_change = vals[recovery_start] - two_month_early_vals_mean
                    X_two_month_early_vals_mean[key] = two_month_early_vals_mean
                    X_NDVI_change[key] = NDVI_change
        # print flag
        # print flag1
        np.save(out_dir + 'two_month_early_vals_mean', X_two_month_early_vals_mean)
        np.save(out_dir + 'NDVI_change', X_NDVI_change)

        pass



class RF_train:

    def __init__(self):

        pass


    def run(self):
        lc_pix = self.gen_landcover_pixes()
        for lc in lc_pix:
            pixes = lc_pix[lc]
            self.random_forest_train(pixes)

        pass


    def __split_keys(self,key):
        pix, mark, enl, date_range = key.split('~')
        drought_start, recovery_start = date_range.split('.')
        drought_start = int(drought_start)
        recovery_start = int(recovery_start)
        return pix, mark, enl, date_range, drought_start, recovery_start


    def gen_landcover_pixes(self):
        landuse_types = [[1, 2, 3, 4, 5], [6, 7], [8, 9], 10, 12]
        labels = ['Forest', 'Shrublands',
                  'Savannas', 'Grasslands', 'Croplands']
        landuse_class_dic = Water_balance().gen_landuse_zonal_index()

        landuse_dic = {}
        for landuse in range(len(landuse_types)):
            #     # print 'landuse',landuse
            lc_label = labels[landuse]
            if type(landuse_types[landuse]) == int:
                landuse_index = landuse_class_dic[landuse_types[landuse]]
            elif type(landuse_types[landuse]) == list:
                landuse_index = []
                for lt in landuse_types[landuse]:
                    for ll in landuse_class_dic[lt]:
                        landuse_index.append(ll)
            else:
                landuse_index = None
                raise IOError('landuse type error')
            landuse_dic[lc_label] = landuse_index
        return landuse_dic

    def load_variable(self,selected_pix=()):

        fdir = this_root+'new_2020\\random_forest\\'
        print 'loading variables ...'
        Y_dic = dict(np.load(fdir+'Y.npy').item())
        pre_dic = dict(np.load(fdir+'PRE.npy').item())
        tmp_dic = dict(np.load(fdir+'TMP.npy').item())
        swe_dic = dict(np.load(fdir+'SWE.npy').item())
        cci_dic = dict(np.load(fdir+'CCI.npy').item())
        NDVI_change_dic = dict(np.load(fdir+'NDVI_change.npy').item())
        two_month_early_vals_mean_dic = dict(np.load(fdir+'two_month_early_vals_mean.npy').item())

        selected_keys = []
        for key in tqdm(Y_dic):
            pix, mark, enl, date_range, drought_start, recovery_start = self.__split_keys(key)
            # print pix
            if selected_pix == ():
                selected_keys.append(key)
            else:
                if pix in selected_pix:
                    selected_keys.append(key)
        nan = False
        Y = []
        X = []
        pix_dic = {}
        flag = 0
        for key in tqdm(selected_keys):
            y = Y_dic[key]
            if y > 18:
                continue
            try:
                pre = pre_dic[key]
                tmp = tmp_dic[key]
                cci = cci_dic[key]
                swe = swe_dic[key]
                ndvi_change = NDVI_change_dic[key]
                two_month_early_vals_mean = two_month_early_vals_mean_dic[key]
            except:
                continue
            _list = [pre,tmp,cci,swe,ndvi_change,two_month_early_vals_mean]
            _list_new = []
            for _l in _list:
                if np.isnan(_l):
                    _list_new.append(nan)
                else:
                    _list_new.append(_l)
            if False in _list_new:
                continue
            pre, tmp, cci, swe, ndvi_change, two_month_early_vals_mean = _list_new

            # print [pre, tmp, cci, swe, ndvi_change, two_month_early_vals_mean]
            pix, mark, enl, date_range, drought_start, recovery_start = self.__split_keys(key)
            pix_dic[pix] = 1
            X.append([pre, tmp, cci, swe, ndvi_change, two_month_early_vals_mean])
            Y.append(y)
            flag += 1
        print 'selected pixes: {}'.format(flag)
        selected_pix_spatial = DIC_and_TIF().pix_dic_to_spatial_arr(pix_dic)
        return X,Y,selected_pix_spatial

        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(valid_pix)
        # plt.imshow(arr,cmap='gray')
        # plt.show()
        # pass

    def random_forest_train(self,selected_pix=()):
        # outdir = this_root+'AI\\RF\\'
        # title = '{} {}'.format(mode_dic[mode],' and '.join(arg))
        # print title
        # out_pdf = outdir+title+'.pdf'

        # exit()
        X, Y, selected_pix_spatial = self.load_variable(selected_pix)
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
        print sum(importances)
        y_min = min(importances)
        y_max = max(importances)
        offset = (y_max-y_min)

        y_min = y_min-offset*0.3
        y_max = y_max+offset*0.3

        plt.ylim(y_min,y_max)
        plt.bar(range(len(importances)),importances,width=0.3)
        # plt.xticks(range(len(importances)),['P','T','CCI','SWE'])
        # plt.title(title)
        plt.figure()
        plt.imshow(selected_pix_spatial,cmap='gray')

        plt.show()
        # plt.savefig(out_pdf)
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
    RF_train().run()
    pass


if __name__ == '__main__':
    main()