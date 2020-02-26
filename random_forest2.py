# coding=gbk

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from analysis import *
import matplotlib.patches as patches


class Prepare:
    def __init__(self):
        # self.check_Y()
        # self.prepare_X()
        pass


    def run(self):
        # self.prepare_Y()
        # x = ['TMP', 'PRE', 'CCI', 'SWE']
        # MUTIPROCESS(self.prepare_X,x).run()
        # self.prepare_NDVI()
        # self.abs_X()
        self.minus_X()
        pass



    def abs_X(self):

        fdir = this_root + 'new_2020\\random_forest\\'
        abs_fdir = this_root+'new_2020\\random_forest_abs\\'
        Tools().mk_dir(abs_fdir)

        pre_dic = dict(np.load(fdir + 'PRE.npy').item())
        tmp_dic = dict(np.load(fdir + 'TMP.npy').item())
        swe_dic = dict(np.load(fdir + 'SWE.npy').item())
        cci_dic = dict(np.load(fdir + 'CCI.npy').item())
        NDVI_change_dic = dict(np.load(fdir + 'NDVI_change.npy').item())
        two_month_early_vals_mean_dic = dict(np.load(fdir + 'two_month_early_vals_mean.npy').item())


        new_pre_dic = {}
        new_tmp_dic = {}
        new_swe_dic = {}
        new_cci_dic = {}
        new_NDVI_change_dic = {}
        new_two_month_early_vals_mean_dic = {}

        for key in tqdm(two_month_early_vals_mean_dic):
            try:
                pre = pre_dic[key]
                tmp = tmp_dic[key]
                cci = cci_dic[key]
                swe = swe_dic[key]
                ndvi_change = NDVI_change_dic[key]
                two_month_early_vals_mean = two_month_early_vals_mean_dic[key]
            except:
                continue
            new_pre_dic[key] = abs(pre)
            new_tmp_dic[key] = abs(tmp)
            new_cci_dic[key] = abs(cci)
            new_swe_dic[key] = abs(swe)
            new_NDVI_change_dic[key] = abs(ndvi_change)
            new_two_month_early_vals_mean_dic[key] = abs(two_month_early_vals_mean)

        np.save(abs_fdir+'PRE.npy',new_pre_dic)
        np.save(abs_fdir+'TMP.npy',new_tmp_dic)
        np.save(abs_fdir+'CCI.npy',new_cci_dic)
        np.save(abs_fdir+'SWE.npy',new_swe_dic)
        np.save(abs_fdir+'NDVI_change.npy',new_NDVI_change_dic)
        np.save(abs_fdir+'two_month_early_vals_mean.npy',new_two_month_early_vals_mean_dic)





    def minus_X(self):

        fdir = this_root + 'new_2020\\random_forest\\'
        abs_fdir = this_root+'new_2020\\random_forest_minus\\'
        Tools().mk_dir(abs_fdir)

        pre_dic = dict(np.load(fdir + 'PRE.npy').item())
        tmp_dic = dict(np.load(fdir + 'TMP.npy').item())
        swe_dic = dict(np.load(fdir + 'SWE.npy').item())
        cci_dic = dict(np.load(fdir + 'CCI.npy').item())
        NDVI_change_dic = dict(np.load(fdir + 'NDVI_change.npy').item())
        two_month_early_vals_mean_dic = dict(np.load(fdir + 'two_month_early_vals_mean.npy').item())


        new_pre_dic = {}
        new_tmp_dic = {}
        new_swe_dic = {}
        new_cci_dic = {}
        new_NDVI_change_dic = {}
        new_two_month_early_vals_mean_dic = {}

        for key in tqdm(two_month_early_vals_mean_dic):
            try:
                pre = pre_dic[key]
                tmp = tmp_dic[key]
                cci = cci_dic[key]
                swe = swe_dic[key]
                ndvi_change = NDVI_change_dic[key]
                two_month_early_vals_mean = two_month_early_vals_mean_dic[key]
            except:
                continue
            new_pre_dic[key] = -pre
            new_tmp_dic[key] = tmp
            new_cci_dic[key] = -cci
            new_swe_dic[key] = -swe
            new_NDVI_change_dic[key] = -ndvi_change
            new_two_month_early_vals_mean_dic[key] = two_month_early_vals_mean

        np.save(abs_fdir+'PRE.npy',new_pre_dic)
        np.save(abs_fdir+'TMP.npy',new_tmp_dic)
        np.save(abs_fdir+'CCI.npy',new_cci_dic)
        np.save(abs_fdir+'SWE.npy',new_swe_dic)
        np.save(abs_fdir+'NDVI_change.npy',new_NDVI_change_dic)
        np.save(abs_fdir+'two_month_early_vals_mean.npy',new_two_month_early_vals_mean_dic)





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




class RF_train_events:
    '''
    Random Forest based on events
    '''
    def __init__(self):

        pass


    def run(self):
        # lc_pix = self.gen_landcover_pixes()
        # for lc in lc_pix:
        #     print lc
        #     pixes = lc_pix[lc]
        #     self.random_forest_train(pixes)
        # self.load_variable()
        # self.do_partition()
        # self.check_partition()
        self.do_random_forest_train()
        pass


    def __split_keys(self,key):
        pix, mark, eln, date_range = key.split('~')
        drought_start, recovery_start = date_range.split('.')
        drought_start = int(drought_start)
        recovery_start = int(recovery_start)
        return pix, mark, eln, date_range, drought_start, recovery_start


    def landcover_partition(self):
        # lc组合3
        landuse_types = [[1, 2, 3, 4, 5], [6, 7, 8, 9], 10]
        labels = ['Forest', 'Shrublands_Savanna', 'Grasslands']

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

    def koppen_partition(self):
        koppen_dic = Koppen().do_reclass()
        return koppen_dic
        pass


    def cross_koppen_landuse(self):
        HI_tif = this_root + 'tif\\HI\\HI.tif'
        outf = this_root+'arr\\cross_koppen_landuse_pix'
        HI_arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(HI_tif)
        HI_arr[HI_arr > 2.] = np.nan

        landuse_dic = self.landcover_partition()
        koppen_dic = self.koppen_partition()

        cross_pix = {}
        for lc_i in landuse_dic:
            scatter_labels = []
            for kop_i in koppen_dic:
                lc_pixs = landuse_dic[lc_i]
                lat_pixs = koppen_dic[kop_i]
                intersect = Water_balance().intersection(lc_pixs,lat_pixs)
                # print intersect
                if len(intersect) > 100:
                    key = lc_i + '.' + str(kop_i)
                    scatter_labels.append(key)
                    intersect_int = []
                    for str_pix in intersect:
                        r,c = str_pix.split('.')
                        r = int(r)
                        c = int(c)
                        intersect_int.append([r,c])
                    # 挑x轴
                    HI_picked_val = Tools().pick_vals_from_2darray(HI_arr,intersect_int)
                    HI_picked_val[HI_picked_val<0] = np.nan
                    # HI_picked_val[HI_picked_val>2] = np.nan
                    HI_mean,_ = Tools().arr_mean_nan(HI_picked_val)
                    cross_pix[key] = [intersect,HI_mean]
        np.save(outf,cross_pix)
        return cross_pix



    def parition(self,keys,pix_,mark_,eln_,desc=''):
        # pix_ = ['182.432','182.431']
        # mark_ = 'out'
        # eln_ = 'late'
        # keys = [
        #     ['182.432', 'out', 'late', '26.28', '26', '28'],
        #     ['182.431', 'out', 'late', '26.28', '26', '28'],
        # ]
        selected_keys = []
        for key in keys:
            pix, mark, eln, date_range, drought_start, recovery_start = self.__split_keys(key)
            if pix in pix_:
                if mark == mark_:
                    if eln in eln_:
                        selected_keys.append(key)
        return selected_keys


    def do_partition(self):
        fdir = this_root + 'new_2020\\random_forest\\'
        outf = this_root+'arr\\RF_partition'
        dic = dict(np.load(fdir + 'NDVI_change.npy').item())
        keys = []
        for key in dic:
            keys.append(key)
        keys = tuple(keys)
        cross_pix = self.cross_koppen_landuse()
        selected = {}
        for mark in ['in','out','tropical']:
            for eln in ['early','late','tropical']:
                condition_key = mark+'~'+eln
                cp_selected_keys = {}
                for cp in tqdm(cross_pix,desc=condition_key):
                    pix_,hi_mean = cross_pix[cp]
                    mark_, eln_ =mark, eln
                    search_keys = self.parition(keys, pix_, mark_, eln_)
                    cp_selected_keys[cp] = search_keys

                selected[condition_key] = cp_selected_keys
        np.save(outf,selected)


    def check_partition(self):
        f = this_root+'arr\\RF_partition.npy'
        dic = dict(np.load(f).item())
        keys = dic['in~early']['Forest.TA']
        for key in keys:
            print key


    def variable_partition(self):
        f = this_root + 'arr\\RF_partition.npy'
        dic = dict(np.load(f).item())
        for key in dic:
            print key
        pass



    def load_variable(self,partition_keys_dic,condition1,condition2):

        # fdir = this_root+'new_2020\\random_forest\\'
        # fdir = this_root+'new_2020\\random_forest_abs\\'
        fdir = this_root+'new_2020\\random_forest_minus\\'
        # print 'loading variables ...'
        Y_dic = dict(np.load(fdir+'Y.npy').item())
        pre_dic = dict(np.load(fdir+'PRE.npy').item())
        tmp_dic = dict(np.load(fdir+'TMP.npy').item())
        swe_dic = dict(np.load(fdir+'SWE.npy').item())
        cci_dic = dict(np.load(fdir+'CCI.npy').item())
        NDVI_change_dic = dict(np.load(fdir+'NDVI_change.npy').item())
        two_month_early_vals_mean_dic = dict(np.load(fdir+'two_month_early_vals_mean.npy').item())

        selected_keys = partition_keys_dic[condition1][condition2]
        nan = False
        Y = []
        X = []
        pix_dic = {}
        for key in selected_keys:
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

            if 'in' in condition1 or 'tropical' in condition1:
                _list = [pre,tmp,cci,ndvi_change,two_month_early_vals_mean]
                _list_new = []
                for _l in _list:
                    if np.isnan(_l):
                        _list_new.append(nan)
                    else:
                        _list_new.append(_l)
                pre, tmp, cci, ndvi_change, two_month_early_vals_mean = _list_new
                pix, mark, enl, date_range, drought_start, recovery_start = self.__split_keys(key)
                pix_dic[pix] = 1
                X.append([pre, tmp, cci, ndvi_change, two_month_early_vals_mean])
                Y.append(y)

            elif 'out' in condition1:
                _list = [pre, tmp, cci, swe, ndvi_change, two_month_early_vals_mean]
                _list_new = []
                for _l in _list:
                    if np.isnan(_l):
                        _list_new.append(nan)
                    else:
                        _list_new.append(_l)
                pre, tmp, cci, swe, ndvi_change, two_month_early_vals_mean = _list_new
                pix, mark, enl, date_range, drought_start, recovery_start = self.__split_keys(key)
                pix_dic[pix] = 1
                X.append([pre, tmp, cci, swe, ndvi_change, two_month_early_vals_mean])
                Y.append(y)
            else:
                raise IOError('error')
        selected_pix_spatial = DIC_and_TIF().pix_dic_to_spatial_arr(pix_dic)
        return X, Y, selected_pix_spatial

    def random_forest_train(self, X, Y, selected_pix_spatial,isplot=False):

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)
        clf = RandomForestRegressor()
        # clf = RandomForestClassifier()
        clf.fit(X_train, Y_train)

        importances = clf.feature_importances_
        y_pred = clf.predict(X_test)
        r_model = stats.pearsonr(Y_test, y_pred)[0]
        mse = sklearn.metrics.mean_squared_error(Y_test, y_pred)

        r_X = []
        for i in range(len(X_test[0])):
            corr_x = []
            corr_y = []
            for j in range(len(X_test)):
                if X_test[j][i] == False:
                    continue
                corr_x.append(X_test[j][i])
                corr_y.append(y_pred[j])
            # print corr_y
            r_c, p = stats.pearsonr(corr_x, corr_y)
            r_X.append(r_c)
            # print i, r_c, p
            # plt.scatter(corr_x, corr_y)
            # plt.show()
        #### plot ####
        if isplot:
            print importances
            print('mse:%s\nr:%s' % (mse, r_model))
            # 1 plot spatial
            plt.figure()
            plt.imshow(selected_pix_spatial,cmap='gray')

            # 2 plot importance
            plt.figure()
            y_min = min(importances)
            y_max = max(importances)
            offset = (y_max-y_min)

            y_min = y_min-offset*0.3
            y_max = y_max+offset*0.3

            plt.ylim(y_min,y_max)
            plt.bar(range(len(importances)),importances,width=0.3)
            KDE_plot().plot_scatter(Y_test, y_pred,s=10)
            plt.show()
        #### plot ####


        return importances,mse, r_model, Y_test, y_pred,r_X


    def kernel_do_random_forest_train(self,params):
        c1,c2,partition_keys_dic = params
        key = c1 + '-' + c2
        X, Y, selected_pix_spatial = self.load_variable(partition_keys_dic, c1, c2)
        # if len(X) < 100:
        #     result_dic[key] = None
        #     continue



        ################## debug
        # importances, mse, r, Y_test, y_pred, rX = self.random_forest_train(X, Y, selected_pix_spatial, isplot=False)

        try:
            importances, mse, r, Y_test, y_pred, rX = self.random_forest_train(X, Y, selected_pix_spatial, isplot=False)
            result = key,{'importances':importances, 'mse':mse, 'r':r, 'Y_test':Y_test, 'y_pred':y_pred,'rX':rX}
            return result
        except Exception as e:
            # print e,'error'
            return key,[]
        pass


    def do_random_forest_train(self):

        result_dic_arr = this_root+'arr\\RF_result_dic_arr_with_rX_minus'
        partition_keys_dic = dict(np.load(this_root + 'arr\\RF_partition.npy').item())
        condition1_list = [
                        'in~early',
                        'in~late',
                        'tropical~tropical',
                        'out~early',
                        'out~late'
                        ]
        condition2_list = []

        for i in partition_keys_dic['in~early']:
            condition2_list.append(i)


        # result_dic = {}

        params = []
        for c1 in condition1_list:
            for c2 in condition2_list:
                params.append([c1,c2,partition_keys_dic])
        # print params[1]
        # exit()
        ###1### debug
        # result = self.kernel_do_random_forest_train(params[1])
        ###1###

        ##### 2 run
        result = MUTIPROCESS(self.kernel_do_random_forest_train,params).run()
        np.save(result_dic_arr,result)
        ##### 2
        pass



class RF_train_pixels:
    '''
    Random Forest based on pixels
    '''
    def __init__(self):

        pass


    def run(self):

        pass


class Corelation_analysis():

    def __init__(self):
        pass

    def run(self):
        result_dic_arr = np.load(this_root + 'arr\\RF_result_dic_arr1.npy')

        for key in result_dic_arr:
            print key
            exit()





        pass





class Plot_RF_train_events_result:
    def __init__(self):

        pass


    def run(self):
        region_pix = dict(np.load(this_root+'arr\\cross_koppen_landuse_pix.npy').item())
        # f = this_root+'arr\\RF_result_dic_arr1.npy'
        f = this_root+'arr\\RF_result_dic_arr_with_rX_minus.npy'
        arr = np.load(f)
        x0list = []
        ylist = []
        size_list = []
        colors_list = []
        rX_colors_list = []

        self.__plot_grid()
        for key,result_dic in arr:
            print key
            try:
                importances = result_dic['importances']
                r = result_dic['r']
                rX = result_dic['rX']
                if np.isnan(r):
                    continue
            except:
                continue
            scatter_size = self.__importances_to_scatter_size(importances)
            rX_color = self.__rX_to_colors(rX)
            color = self.__r_to_color(r)
            x0,y = self.__get_scatter_position(key,region_pix)
            x0list.append(x0)
            ylist.append(y)
            colors_list.append(color)
            size_list.append(scatter_size)
            rX_colors_list.append(rX_color)


        # self.__plot_scatter(x0list,ylist,size_list,colors_list)
        self.__plot_scatter(x0list,ylist,size_list,rX_colors_list)
        # plt.scatter(xs,ys)
        plt.show()
        # key = 'in~early-Shrublands_Savanna.AH'
        # self.__plot_grid()
        pass

    # def run1(self):
    #     key = 'in~early-Shrublands_Savanna.AH'
    #     self.__get_scatter_position(key)

    def plot_colors_palette(self):
        cmap = sns.diverging_palette(236, 0, s=99, l=50, n=10, center="light")
        sns.palplot(cmap)
        plt.show()

    def plot_regions_arr(self):


        outpng_dir = this_root+'png\\plot_regions_arr\\'
        Tools().mk_dir(outpng_dir)
        partition_keys_dic = dict(np.load(this_root + 'arr\\RF_partition.npy').item())
        condition1_list = [
            'in~early',
            'in~late',
            'tropical~tropical',
            'out~early',
            'out~late'
        ]
        condition2_list = []

        for i in partition_keys_dic['in~early']:
            condition2_list.append(i)

        for c1 in tqdm(condition1_list):
            for c2 in condition2_list:
                key = c1 + '-' + c2
                X, Y, selected_pix_spatial = RF_train_events().load_variable(partition_keys_dic, c1, c2)
                # self.__get_scatter_position()
                plt.imshow(selected_pix_spatial)
                plt.title(key)
                plt.savefig(outpng_dir+key+'.png')
                plt.close()

        pass


    def __split_keys(self,key):
        pix, mark, eln, date_range = key.split('~')
        drought_start, recovery_start = date_range.split('.')
        drought_start = int(drought_start)
        recovery_start = int(recovery_start)
        return pix, mark, eln, date_range, drought_start, recovery_start


    def __plot_grid(self):
        plt.figure(figsize=(6*5+2, 17))
        # plot vertical lines

        X = range(6*5+2)

        # plot horizental lines

        Y = range(17)

        for x in X:
            plt.plot([x] * 2, [0, 17-1],c='black')
        for y in Y:
            plt.plot([0, 6*5+2-1],[y]*2,c='black')


        plt.axis("equal")
        # self.__plot_scatter()
        # plt.show()
        pass

    def __plot_scatter(self,x0list,ylist,slist,clist):
        # for i in range(len(x0list)):
        #     print x0list[i],ylist[i],slist[i],clist[i]
        for i in range(len(x0list)):
            for j in range(len(slist[i])):
                plt.scatter(x0list[i]+j, ylist[i], s=slist[i][j], c=clist[i][j])


    def __plot_matrix(self,x0list,ylist,slist,clist):
        for i in range(len(x0list)):
            for j in range(len(slist[i])):
                plt.scatter(x0list[i]+j, ylist[i], s=slist[i][j], c=clist[i][j])



    def __importances_to_scatter_size(self,importances_list):

        size_list = []
        for i in range(1,7):
            size = i * 80
            size_list.append(size)

        scatter_size_list = []
        a = np.argsort(importances_list)
        for i in a:
            scatter_size_list.append(size_list[i])
        return scatter_size_list
        pass



    def __rX_to_colors(self,rX_list):

        colors_list = []
        cmap = sns.diverging_palette(236, 0, s=99, l=50, n=10, center="light")

        for r in rX_list:
            if np.isnan(r):
                colors_list.append('cmap[5]')
            else:
                rr = int(round(round(r, 1) * 10. / 2. + 5., 0)) - 1
                c = cmap[rr]
                colors_list.append(c)

        return colors_list


    def __r_to_color(self,r):

        cmap = sns.diverging_palette(236, 0, s=99, l=50, n=10, center="light")
        r = round(r, 1)
        c = cmap[int(r * 10) - 1]
        return c
        pass



    def __get_scatter_position(self,key,region_pix):
        # key = 'in~early-Shrublands_Savanna.AH'

        # get Y coordinate
        region = key.split('-')[1]
        # region_pix = RF_train_events().cross_koppen_landuse()
        region_HI_list = []
        regions = []
        for region_i in region_pix:
            HI = region_pix[region_i][1]
            regions.append(region_i)
            region_HI_list.append(HI)
        # HI 排序
        region_sort_index = np.argsort(region_HI_list)
        # print region_sort_index
        # print region_HI_list
        # exit()
        region_sort_dic = {}
        for i in range(len(region_sort_index)):
            region_sort_dic[regions[i]] = region_sort_index[i]
        # for i in region_sort_dic:
        #     print i,region_sort_dic[i]
        # exit()
        y = region_sort_dic[region] + 0.5 # plus 0.5 means move the point to the center of a grid
        if 'in~early' in key:
            plt.text(-0.5, y, key, fontsize=12,horizontalalignment='right')
        # plt.text(-0.5, y, key, fontsize=12)

        # get X0 coordinate
        conditions = key.split('-')[0]
        if conditions == 'in~early':
            x0 = 0
        elif conditions == 'out~early':
            x0 = 0 + 6

        elif conditions == 'in~late':
            x0 = 0 + 6 + 7
        elif conditions == 'out~late':
            x0 = 0 + 6 + 7 + 6

        elif conditions == 'tropical~tropical':
            x0 = 0 + 6 + 7 + 6 + 7

        else:
            raise IOError('key error...')
        x0 = x0 + 0.5  # plus 0.5 means move the point to the center of a grid


        return x0,y

        pass




    def get_scatter_y(self):

        HI_tif = this_root + 'tif\\HI\\HI.tif'
        HI_arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(HI_tif)
        HI_arr[HI_arr > 2.] = np.nan

        # lc组合3
        landuse_types = [[1, 2, 3, 4, 5], [6, 7, 8, 9], 10]
        labels = ['Forest', 'Shrublands_Savanna', 'Grasslands']

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

        latitude_dic = Koppen().do_reclass()
        scatter_dic = {}
        for lc_i in landuse_dic:
            scatter_labels = []
            for lat_i in latitude_dic:
                lc_pixs = landuse_dic[lc_i]
                lat_pixs = latitude_dic[lat_i]
                intersect = Water_balance().intersection(lc_pixs,lat_pixs)
                # print intersect
                if len(intersect) > 100:
                    key = lc_i + '.' + str(lat_i)
                    scatter_labels.append(key)
                    intersect_int = []
                    for str_pix in intersect:
                        r,c = str_pix.split('.')
                        r = int(r)
                        c = int(c)
                        intersect_int.append([r,c])
                    # 挑x轴
                    HI_picked_val = Tools().pick_vals_from_2darray(HI_arr, intersect_int)
                    HI_picked_val[HI_picked_val < 0] = np.nan
                    # HI_picked_val[HI_picked_val>2] = np.nan
                    HI_mean, xerr = Tools().arr_mean_nan(HI_picked_val)
                    scatter_dic[key] = HI_mean
        for i in scatter_dic:
            print i,scatter_dic[i]



def main():

    # Prepare().run()
    # RF_train_events().run()
    Plot_RF_train_events_result().run()
    # Plot_RF_train_events_result().get_scatter_y()
    # Corelation_analysis().run()
    pass


if __name__ == '__main__':
    main()