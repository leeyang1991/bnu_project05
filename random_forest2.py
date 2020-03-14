# coding=gbk

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from analysis import *
import matplotlib.patches as patches

this_root_branch = this_root+'branch2020\\'
class Prepare:
    def __init__(self):
        self.this_class_arr = this_root_branch + 'Random_Forest\\arr\\Prepare\\'
        self.this_class_tif = this_root_branch + 'Random_Forest\\tif\\Prepare\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        pass


    def run(self):
        # 1.准备因变量 Y
        # self.prepare_Y()
        self.check_Y()
        # 2.准备自变量 X的 delta
        # x = ['TMP', 'PRE', 'CCI', 'SWE']
        # MUTIPROCESS(self.prepare_X,x).run()
        # 3.准备自变量 X 的标准差
        # self.prepare_X('CCI')
        # x = ['PRE_std', 'TMP_std', 'CCI_std', 'SWE_std']
        # for i in x:
        #     self.prepare_X_std(i)
        # 4.准备自变量 X 的平均值
        # x = ['PRE_mean','TMP_mean','CCI_mean','SWE_mean']
        # for i in x:
        #     self.prepare_X_mean(i)
        # 5.准备自变量 NDVI 的平均值和delta
        # self.prepare_NDVI()
        # 6.准备自变量 soil 的值，soil是常量
        # self.prepare_soil()
        # 7.准备自变量 bio diversity 的值，常量
        # self.prepare_bio_diversity()
        # 8.['PRE', 'CCI', 'SWE','NDVI'] 为亏损量，应为负值，需要加负号
        ########## 需再要手动覆盖 ###########
        # self.minus_X()
        pass

    def __split_keys(self,key):
        pix, mark, eln, date_range = key.split('~')
        drought_start, recovery_start = date_range.split('.')
        drought_start = int(drought_start)
        recovery_start = int(recovery_start)
        return pix, mark, eln, date_range, drought_start, recovery_start

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

        fdir = self.this_class_arr
        abs_fdir = fdir+'\\random_forest_minus\\'
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
        out_dir = self.this_class_arr+'\\'
        Tools().mk_dir(out_dir)
        # 1 drought periods
        print '1. loading recovery time'
        f_recovery_time = this_root_branch+'arr\\Recovery_time1\\recovery_time_composite\\composite.npy'
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
        f = this_root_branch+'Random_Forest\\arr\\Prepare\\CCI.npy'
        # f = r'D:\project05\new_2020\random_forest\Y.npy'
        dic = dict(np.load(f).item())
        # print len(dic)
        pix_dic = DIC_and_TIF().void_spatial_dic()
        for key in dic:
            # print key,dic[key]
            pix, mark, eln, date_range, drought_start, recovery_start = self.__split_keys(key)
            print pix, mark, eln, date_range, drought_start, recovery_start
            print dic[key]
            pix_dic[pix].append(1)
            exit()

        spatial_dic = {}
        for pix in pix_dic:
            val = pix_dic[pix]
            if len(val)>0:
                new_val = np.sum(val)
            else:
                new_val = np.nan
            spatial_dic[pix] = new_val

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr,cmap='jet',vmin=0,vmax=30)
        plt.colorbar()
        plt.show()



    def prepare_X(self, x):
        # x = ['TMP','PRE','CCI','SWE']
        out_dir = self.this_class_arr+''
        Y_dic = dict(np.load(self.this_class_arr + 'Y.npy').item())
        if x in ['TMP', 'PRE']:
            per_pix_dir = this_root + 'data\\{}\\per_pix\\'.format(x)
            mean_dir = this_root + 'data\\{}\\mon_mean_tif\\'.format(x)
        elif x == 'CCI':
            per_pix_dir = this_root + 'data\\CCI\\per_pix\\'
            mean_dir = this_root + 'data\\CCI\\\monthly_mean\\'
        elif x == 'SWE':
            per_pix_dir = this_root + 'data\\GLOBSWE\\per_pix\\SWE_max_408\\'
            mean_dir = this_root + 'data\\GLOBSWE\\monthly_SWE_max\\'
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

    def prepare_X_std(self,x):
        '''
        cv: 变异系数
        std: 标准差
        :return:
        '''
        # x = ['PRE_std','TMP_std','CCI_std','SWE_std']
        product = x.split('_')[0]
        if product == 'SWE':
            per_pix_dir = this_root + 'data\\GLOBSWE\\per_pix\\SWE_max_408\\'
        else:
            per_pix_dir = this_root + 'data\\{}\\per_pix\\'.format(product)

        out_dir = self.this_class_arr
        Tools().mk_dir(out_dir)
        Y_dic = dict(np.load(self.this_class_arr + 'Y.npy').item())
        all_dic = {}
        for f in tqdm(os.listdir(per_pix_dir), desc='1/2 loading per_pix_dir ...'):
            dic = dict(np.load(per_pix_dir + f).item())
            for pix in dic:
                all_dic[pix] = dic[pix]

        # 3 找干旱事件对应的X的std
        X = {}
        for key in tqdm(Y_dic, desc='2/2 generate X dic ...'):
            split_key = key.split('~')
            pix, mark, eln, date_range = split_key
            if product == 'SWE':
                if mark != 'out':
                    continue
            split_date_range = date_range.split('.')
            start = split_date_range[0]
            end = split_date_range[1]
            start = int(start)
            end = int(end)
            drought_range = range(start, end)
            # print pix,mark,drought_range
            # exit()
            vals = all_dic[pix]
            selected_val = []
            for dr in drought_range:
                val = vals[dr]
                if val < -9999:
                    continue
                selected_val.append(val)
            if len(selected_val) > 0:
                std = np.std(selected_val)
            else:
                std = np.nan
            X[key] = std

        np.save(out_dir + '{}'.format(x), X)
        pass



    def prepare_X_mean(self,x):
        '''
        cv: 变异系数
        std: 标准差
        :return:
        '''
        # x = ['PRE_std','TMP_std','CCI_std','SWE_std']
        product = x.split('_')[0]
        if product == 'SWE':
            per_pix_dir = this_root + 'data\\GLOBSWE\\per_pix\\SWE_max_408\\'
        else:
            per_pix_dir = this_root + 'data\\{}\\per_pix\\'.format(product)

        out_dir = self.this_class_arr
        Tools().mk_dir(out_dir)
        Y_dic = dict(np.load(self.this_class_arr + 'Y.npy').item())
        all_dic = {}
        for f in tqdm(os.listdir(per_pix_dir), desc='1/2 loading per_pix_dir ...'):
            dic = dict(np.load(per_pix_dir + f).item())
            for pix in dic:
                all_dic[pix] = dic[pix]

        # 3 找干旱事件对应的X的std
        X = {}
        for key in tqdm(Y_dic, desc='2/2 generate X dic ...'):
            split_key = key.split('~')
            pix, mark, eln, date_range = split_key
            if product == 'SWE':
                if mark != 'out':
                    continue
            split_date_range = date_range.split('.')
            start = split_date_range[0]
            end = split_date_range[1]
            start = int(start)
            end = int(end)
            drought_range = range(start, end)
            # print pix,mark,drought_range
            # exit()
            vals = all_dic[pix]
            selected_val = []
            for dr in drought_range:
                val = vals[dr]
                if val < -9999:
                    continue
                selected_val.append(val)
            if len(selected_val) > 0:
                mean = np.mean(selected_val)
            else:
                mean = np.nan
            X[key] = mean

        np.save(out_dir + '{}'.format(x), X)




    def prepare_NDVI(self):
        out_dir = self.this_class_arr + '\\'
        # y
        Y_dic = dict(np.load(self.this_class_arr + 'Y.npy').item())
        # x
        per_pix_dir = this_root + 'data\\NDVI\\per_pix_anomaly_smooth\\'
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


    def prepare_soil(self):
        out_dir = self.this_class_arr + '\\'
        Y_dic = dict(np.load(self.this_class_arr + 'Y.npy').item())
        sand_tif = this_root_branch+'tif\\HWSD\\T_SAND_resample.tif'
        silt_tif = this_root_branch+'tif\\HWSD\\T_SILT_resample.tif'
        clay_tif = this_root_branch+'tif\\HWSD\\T_CLAY_resample.tif'

        sand_arr = to_raster.raster2array(sand_tif)[0]
        silt_arr = to_raster.raster2array(silt_tif)[0]
        clay_arr = to_raster.raster2array(clay_tif)[0]

        sand_arr[sand_arr<-9999] = np.nan
        silt_arr[silt_arr<-9999] = np.nan
        clay_arr[clay_arr<-9999] = np.nan

        sand_dic = DIC_and_TIF().spatial_arr_to_dic(sand_arr)
        silt_dic = DIC_and_TIF().spatial_arr_to_dic(silt_arr)
        clay_dic = DIC_and_TIF().spatial_arr_to_dic(clay_arr)

        sand_x = {}
        silt_x = {}
        clay_x = {}

        for key in tqdm(Y_dic, desc='1/2 generate X dic ...'):
            split_key = key.split('~')
            pix, mark, eln, date_range = split_key
            sand = sand_dic[pix]
            silt = silt_dic[pix]
            clay = clay_dic[pix]

            if np.isnan(sand):
                continue
            if np.isnan(silt):
                continue
            if np.isnan(clay):
                continue
            sand_x[key] = sand
            silt_x[key] = silt
            clay_x[key] = clay

        np.save(out_dir+'sand',sand_x)
        np.save(out_dir+'silt',silt_x)
        np.save(out_dir+'clay',clay_x)

        pass


    def prepare_bio_diversity(self):
        out_dir = self.this_class_arr + '\\'
        Y_dic = dict(np.load(self.this_class_arr + 'Y.npy').item())
        bio_tif = this_root_branch + 'tif\\Bio_diversity\\bio_diversity_normalized.tif'
        bio_arr = to_raster.raster2array(bio_tif)[0]
        bio_arr[bio_arr < -9999] = np.nan
        bio_dic = DIC_and_TIF().spatial_arr_to_dic(bio_arr)
        bio_x = {}
        for key in tqdm(Y_dic, desc='1/2 generate X dic ...'):
            split_key = key.split('~')
            pix, mark, eln, date_range = split_key
            bio = bio_dic[pix]
            if np.isnan(bio):
                continue
            bio_x[key] = bio

        np.save(out_dir+'bio',bio_x)


        pass


class RF_train_events:
    '''
    Random Forest based on events
    '''
    def __init__(self):
        self.this_class_arr = this_root_branch + 'Random_Forest\\arr\\RF_train_events\\'
        self.this_class_tif = this_root_branch + 'Random_Forest\\tif\\RF_train_events\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        pass


    def run(self):
        # self.load_variable()
        # self.do_partition()
        # 1 分区 IN OUT EARLY LATE TROPICAL
        # self.check_partition()
        # 2 RF
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
        Ydic = Prepare().this_class_arr+'Y.npy'
        outf = self.this_class_arr+'RF_partition'
        dic = dict(np.load(Ydic).item())
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
        f = self.this_class_arr+'RF_partition.npy'
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

        fdir = Prepare().this_class_arr
        print 'loading variables ...'
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


    def load_variables_dir(self,fdir_dic,partition_keys_dic,condition1,condition2):
        '''
        load all variable in a single directory
        :return:
        '''

        selected_keys = partition_keys_dic[condition1][condition2]
        pix_dic = {}
        nan = False
        Y = []
        X = []
        for key in selected_keys:
            y = fdir_dic['Y'][key]
            if y > 18:
                continue

            if 'in' in condition1 or 'tropical' in condition1:
                try:
                    variables_names = [
                        'PRE', 'PRE_mean', 'PRE_std',
                        'TMP', 'TMP_mean', 'TMP_std',
                        'CCI', 'CCI_mean', 'CCI_std',
                        # 'SWE', 'SWE_mean', 'SWE_std',
                        'NDVI_change', 'two_month_early_vals_mean',
                        'sand', 'silt', 'clay',
                        'bio'
                    ]
                    variables_vals = []
                    for x in variables_names:
                        val = fdir_dic[x][key]
                        variables_vals.append(val)
                except Exception as e:
                    print e
                    continue
                _list = variables_vals
                _list_new = []
                for _l in _list:
                    if np.isnan(_l):
                        _list_new.append(nan)
                    else:
                        _list_new.append(_l)
                pix, mark, enl, date_range, drought_start, recovery_start = self.__split_keys(key)
                pix_dic[pix] = 1
                X.append(_list_new)
                Y.append(y)

            elif 'out' in condition1:
                try:
                    variables_names = [
                        'PRE', 'PRE_mean', 'PRE_std',
                        'TMP', 'TMP_mean', 'TMP_std',
                        'CCI', 'CCI_mean', 'CCI_std',
                        'SWE', 'SWE_mean', 'SWE_std',
                        'NDVI_change', 'two_month_early_vals_mean',
                        'sand', 'silt', 'clay',
                        'bio'
                    ]
                    variables_vals = []
                    for x in variables_names:
                        val = fdir_dic[x][key]
                        variables_vals.append(val)
                except Exception as e:
                    continue

                _list = variables_vals
                _list_new = []
                for _l in _list:
                    if np.isnan(_l):
                        _list_new.append(nan)
                    else:
                        _list_new.append(_l)
                pix, mark, enl, date_range, drought_start, recovery_start = self.__split_keys(key)
                pix_dic[pix] = 1
                X.append(_list_new)
                Y.append(y)
            else:
                raise IOError('error')
        selected_pix_spatial = DIC_and_TIF().pix_dic_to_spatial_arr(pix_dic)
        # print X
        # print Y
        # print len(X)
        # print len(Y)
        # plt.imshow(selected_pix_spatial)
        # plt.show()
        #
        # exit()

        return X, Y, selected_pix_spatial


        pass

    def random_forest_train(self, X, Y, selected_pix_spatial,isplot=False,title=''):

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
            out_png_dir = this_root+'png\\RF_importances\\'
            Tools().mk_dir(out_png_dir)
            # 1 plot spatial
            # plt.figure()
            # plt.imshow(selected_pix_spatial,cmap='gray')

            # 2 plot importance
            plt.subplot(211)
            plt.title(title)
            y_min = min(importances)
            y_max = max(importances)
            offset = (y_max-y_min)
            y_min = y_min-offset*0.3
            y_max = y_max+offset*0.3

            plt.ylim(y_min,y_max)
            plt.bar(range(len(importances)),importances,width=0.3)
            ax = plt.subplot(212)
            KDE_plot().plot_scatter(Y_test, y_pred,cmap='jet',s=10,ax=ax,linewidth=0)
            plt.axis('equal')
            plt.savefig(out_png_dir+title+'.png',ppi=300)
            plt.close()
            # plt.show()
        #### plot ####


        return importances,mse, r_model, Y_test, y_pred,r_X


    def kernel_do_random_forest_train(self,params):
        c1,c2,partition_keys_dic,fdir_dic = params
        key = c1 + '-' + c2
        # X, Y, selected_pix_spatial = self.load_variable(partition_keys_dic, c1, c2)

        X, Y, selected_pix_spatial = self.load_variables_dir(fdir_dic,partition_keys_dic, c1, c2)
        # exit()
        # if len(X) < 100:
        #     result_dic[key] = None
        #     continue



        ################## debug
        # importances, mse, r, Y_test, y_pred, rX = self.random_forest_train(X, Y, selected_pix_spatial, isplot=True,title=key)
        # print importances, mse, r, Y_test, y_pred, rX
        # exit()

        ################## run
        try:
            importances, mse, r, Y_test, y_pred, rX = self.random_forest_train(X, Y, selected_pix_spatial, isplot=False)
            result = key,{'importances':importances, 'mse':mse, 'r':r, 'Y_test':Y_test, 'y_pred':y_pred,'rX':rX}
            return result
        except Exception as e:
            # print e,'error'
            return key,[]
        pass


    def do_random_forest_train(self):

        out_result_dic = self.this_class_arr+'\\RF_result_dic_arr'
        partition_f = self.this_class_arr+'RF_partition.npy'
        partition_keys_dic = dict(np.load(partition_f).item())
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
        fdir = Prepare().this_class_arr
        fdir_dic = {}
        for f in tqdm(os.listdir(fdir), desc='loading each variable...'):
            x = f.split('.')[0]
            x_dic = dict(np.load(fdir + f).item())
            fdir_dic[x] = x_dic
        params = []
        for c1 in condition1_list:
            for c2 in condition2_list:
                params.append([c1,c2,partition_keys_dic,fdir_dic])
        ###1### debug
        results = []
        for p in tqdm(params):
            try:
                result = self.kernel_do_random_forest_train(p)
                results.append(result)
            except Exception as e:
                pass
                # print e
        np.save(out_result_dic,results)

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
        self.this_class_arr = this_root_branch + 'Random_Forest\\arr\\Plot_RF_train_events_result\\'
        self.this_class_tif = this_root_branch + 'Random_Forest\\tif\\Plot_RF_train_events_result\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        pass


    def run(self):
        self.plot_bar()

        pass

    def plot_bar(self):
        fdir = self.this_class_arr
        out_png_dir = this_root+'png\\Plot_RF_train_events_result\\'
        Tools().mk_dir(out_png_dir)
        for f in os.listdir(fdir):
            title = f.split('.')[0]
            dic = dict(np.load(fdir+f).item())
            max_x = dic['max_x']
            label = dic['max_key']
            plt.barh(label, max_x)
            plt.title(title)
            plt.savefig(out_png_dir+title+'.png',ppi=600)
            plt.close()

        pass

    def gen_bar_data(self):
        f = RF_train_events().this_class_arr + 'RF_result_dic_arr.npy'
        arr = np.load(f)
        # condition = 'in'
        # condition = 'out'
        condition = 'tropical'
        # condition2 = 'early'
        # condition2 = 'late'
        condition2 = 'tropical'

        variables_names_in = [
            'PRE', 'PRE_mean', 'PRE_std',
            'TMP', 'TMP_mean', 'TMP_std',
            'CCI', 'CCI_mean', 'CCI_std',
            # 'SWE', 'SWE_mean', 'SWE_std',
            'NDVI_change', 'two_month_early_vals_mean',
            'sand', 'silt', 'clay',
            'bio']
        # print len(variables_names_in)
        # exit()
        variables_names_out = [
            'PRE', 'PRE_mean', 'PRE_std',
            'TMP', 'TMP_mean', 'TMP_std',
            'CCI', 'CCI_mean', 'CCI_std',
            'SWE', 'SWE_mean', 'SWE_std',
            'NDVI_change', 'two_month_early_vals_mean',
            'sand', 'silt', 'clay',
            'bio']

        variables_names_out_dic = {}
        for var in variables_names_out:
            variables_names_out_dic[var] = []

        variables_names_in_dic = {}
        for var in variables_names_in:
            variables_names_in_dic[var] = []

        if condition == 'in':
            variables_names_dic = variables_names_in_dic
            variables_names_list = variables_names_in
        elif condition == 'out':
            variables_names_dic = variables_names_out_dic
            variables_names_list = variables_names_out
        elif condition == 'tropical':
            variables_names_dic = variables_names_in_dic
            variables_names_list = variables_names_in
        else:
            raise IOError
        for key,result_dic in arr:
            if not condition2 in key:
                continue
            if not condition in key:
                continue
            # print eln
            if len(result_dic) == 0:
                continue
            # print key
            for i in range(len(result_dic['importances'])):
                var_name = variables_names_list[i]
                val = result_dic['importances'][i]
                variables_names_dic[var_name].append(val)
                # print var_name,val

        x_list = []
        xerr_list = []
        key_list = []
        for i in variables_names_dic:
            val = variables_names_dic[i]
            mean,xerr = Tools().arr_mean_nan(val)
            key_list.append(i)
            x_list.append(mean)
            xerr_list.append(xerr/4.)
        x_list_sort = np.argsort(x_list)
        maxvs = []
        maxv_ind = []
        max_x = []
        max_xerr = []
        max_key = []
        for i in x_list_sort:
            maxvs.append(x_list_sort[i])
            maxv_ind.append(i)
            max_x.append(x_list[i])
            max_xerr.append(xerr_list[i])
            max_key.append(key_list[i])
        # plt.barh(key_list,x_list,xerr=xerr_list)
        # plt.figure()
        # plt.barh(max_key,max_x,xerr=max_xerr)
        # plt.show()
        np.save(self.this_class_arr+'{}_{}'.format(condition,condition2),{'max_key':max_key,'max_x':max_x,'max_xerr':max_xerr})

    def plot_scatter(self):
        region_pix = dict(np.load(this_root+'arr\\cross_koppen_landuse_pix.npy').item())
        # f = this_root+'arr\\RF_result_dic_arr1.npy'
        # f = this_root+'arr\\RF_result_dic_arr_with_rX_minus.npy'
        f = RF_train_events().this_class_arr+'RF_result_dic_arr.npy'
        arr = np.load(f)
        len_in_out_dic = {}
        for i in arr:
            region,content_dic = i
            if 'in' in region:
                len_in = len(content_dic['importances'])
                len_in_out_dic['in'] = len_in
                len_in_out_dic['tropical'] = len_in
            if 'out' in region:
                len_out = len(content_dic['importances'])
                len_in_out_dic['out'] = len_out
        print len_in_out_dic
        x0list = []
        ylist = []
        size_list = []
        colors_list = []
        rX_colors_list = []

        self.__plot_grid(len_in_out_dic)
        for key,result_dic in arr:
            # print key
            in_out = key.split('~')[0]
            # print in_out
            len_in_out = len_in_out_dic[in_out]
            # print len_in_out
            # pix, mark, eln, date_range, drought_start, recovery_start = self.__split_keys(key)
            # print eln
            # exit()
            try:
                importances = result_dic['importances']
                r = result_dic['r']
                rX = result_dic['rX']
                if np.isnan(r):
                    continue
            except:
                continue
            scatter_size = self.__importances_to_scatter_size(importances,len_in_out)
            rX_color = self.__rX_to_colors(rX)
            color = self.__r_to_color(r)
            x0,y = self.__get_scatter_position(key,region_pix,len_in_out_dic)
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


    def __plot_grid(self,len_in_out_dic):
        # plot vertical lines
        early_in = len_in_out_dic['in']
        late_in = len_in_out_dic['in']
        early_out = len_in_out_dic['out']
        late_out = len_in_out_dic['out']
        tropical = len_in_out_dic['in']
        X = range(early_in+late_in+early_out+late_out+tropical+4+1)
        # plot horizental lines
        Y = range(17)
        plt.figure(figsize=(25,5))
        for x in X:
            plt.plot([x] * 2, [0, 17-1],c='black',linewidth=0.4)
        for y in Y:
            plt.plot([0, early_in+late_in+early_out+late_out+tropical + 4],[y]*2,c='black',linewidth=0.4)

        plt.axis('off')
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



    def __importances_to_scatter_size(self,importances_list,len_in_out):

        size_list = []
        for i in range(1,len_in_out+1):
            size = i * 10
            size_list.append(size)

        scatter_size_list = []
        a = np.argsort(importances_list)
        for i in a:
            scatter_size_list.append(size_list[i])
        return scatter_size_list

    def __rX_to_colors(self,rX_list):

        colors_list = []
        cmap = sns.diverging_palette(236, 0, s=99, l=50, n=10, center="light")

        for r in rX_list:
            if np.isnan(r):
                colors_list.append(cmap[5])
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



    def __get_scatter_position(self,key,region_pix,len_in_out_dic):
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
        elif conditions == 'in~late':
            x0 = 0 + len_in_out_dic['in'] + 1
        elif conditions == 'out~early':
            x0 = 0 + len_in_out_dic['in'] + len_in_out_dic['in'] + 2
        elif conditions == 'out~late':
            x0 = 0 + len_in_out_dic['in'] + len_in_out_dic['in'] + len_in_out_dic['out'] + 3
        elif conditions == 'tropical~tropical':
            x0 = 0 + len_in_out_dic['in'] + len_in_out_dic['in'] + len_in_out_dic['out'] + len_in_out_dic['out'] + 4
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


class Plot_RF_Train_events_result_spatial:
    def __init__(self):

        pass


    def run(self):
        self.gen_zone_pixels()

        pass

    def run1(self):
        results_f = RF_train_events().this_class_arr+'RF_result_dic_arr.npy'
        results_dic = np.load(results_f)
        for i in results_dic:
            print i
            break


            # exit()
        pass

    def gen_zone_pixels(self):
        '''
        16 个分区的像素
        :return:
        '''
        partition_f = RF_train_events().this_class_arr+'RF_partition.npy'
        partition_dic = dict(np.load(partition_f).item())
        zones_dic = {}
        for i in partition_dic['in~late']:
            zones_dic[i] = []
        for z in tqdm(zones_dic):
            for condition in partition_dic:
                for key in partition_dic[condition][z]:
                    pix = key.split('~')[0]
                    # print z,key,pix
                    zones_dic[z].append(pix)

        flag = 0
        spatial_dic = {}
        for i in zones_dic:
            flag += 1
            print i,len(zones_dic[i])
            pixs = zones_dic[i]
            for pix in pixs:
                # print pix
                spatial_dic[pix] = flag
                # time.sleep(1)
        DIC_and_TIF().plot_back_ground_arr()
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        plt.show()



        pass



def main():

    # Prepare().run()
    # RF_train_events().run()
    # Plot_RF_train_events_result().run()
    # Corelation_analysis().run()
    Plot_RF_Train_events_result_spatial().run()
    pass


if __name__ == '__main__':
    main()