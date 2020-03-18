# coding=gbk
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from analysis import *

this_root_PLOT = this_root+'branch2020\\PLOT\\'
out_tif_dir = this_root_PLOT+'tif\\'
Tools().mk_dir(this_root_PLOT,force=True)
Tools().mk_dir(out_tif_dir,force=True)


def sleep(t=1):
    time.sleep(t)

def load_recovery_time_dic():
    # f = this_root+'branch2020\\arr\\Recovery_time1\\recovery_time_composite\\composite.npy'
    f = this_root+'branch2020\\Random_Forest\\arr\\Prepare\\Y.npy'
    recovery_dic = dict(np.load(f).item())
    return recovery_dic

def split_key(key):
    pix, mark, eln, date_range = key.split('~')
    drought_start, recovery_start = date_range.split('.')
    drought_start = int(drought_start)
    recovery_start = int(recovery_start)
    return pix, mark, eln, date_range, drought_start, recovery_start

def plot_events_spatial():
    recovery_dic = load_recovery_time_dic()
    void_spatial_dic = DIC_and_TIF().void_spatial_dic()
    for key in recovery_dic:
        pix, mark, eln, date_range, drought_start, recovery_start = split_key(key)
        void_spatial_dic[pix].append(1)

    events_spatial_dic = {}
    for pix in void_spatial_dic:
        events_num_list = void_spatial_dic[pix]
        if len(events_num_list) == 0:
            continue
        num = np.sum(events_num_list)
        events_spatial_dic[pix] = num
    spatial_arr = DIC_and_TIF().pix_dic_to_spatial_arr(events_spatial_dic)
    DIC_and_TIF().arr_to_tif(spatial_arr,out_tif_dir+'events_spatial.tif')

def plot_recovery_time_mix():
    recovery_dic = load_recovery_time_dic()
    void_spatial_dic = DIC_and_TIF().void_spatial_dic()
    for key in recovery_dic:
        pix, mark, eln, date_range, drought_start, recovery_start = split_key(key)
        recovery = recovery_dic[key]
        if recovery > 24:
            recovery = 24
        void_spatial_dic[pix].append(recovery)

    mean_recovery_time_dic = {}
    for pix in void_spatial_dic:
        mean_recovery_time_list = void_spatial_dic[pix]
        if len(mean_recovery_time_list) == 0:
            continue
        mean_recovery_time = np.mean(mean_recovery_time_list)
        mean_recovery_time_dic[pix] = mean_recovery_time
    spatial_arr = DIC_and_TIF().pix_dic_to_spatial_arr(mean_recovery_time_dic)
    DIC_and_TIF().arr_to_tif(spatial_arr, out_tif_dir + 'recovery_time_mix.tif')


def plot_recovery_time_in_and_out(in_out):
    print in_out
    recovery_dic = load_recovery_time_dic()
    void_spatial_dic = DIC_and_TIF().void_spatial_dic()
    for key in recovery_dic:
        pix, mark, eln, date_range, drought_start, recovery_start = split_key(key)
        if mark != in_out:
            continue
        recovery = recovery_dic[key]
        if recovery > 24:
            recovery = 24
        void_spatial_dic[pix].append(recovery)

    mean_recovery_time_dic = {}
    for pix in void_spatial_dic:
        mean_recovery_time_list = void_spatial_dic[pix]
        if len(mean_recovery_time_list) == 0:
            continue
        mean_recovery_time = np.mean(mean_recovery_time_list)
        mean_recovery_time_dic[pix] = mean_recovery_time
    spatial_arr = DIC_and_TIF().pix_dic_to_spatial_arr(mean_recovery_time_dic)
    DIC_and_TIF().arr_to_tif(spatial_arr, out_tif_dir + 'recovery_time_{}.tif'.format(in_out))



class Water_balance:

    def __init__(self):
        # self.cross_landuse_WB_recovery_time()
        # self.gen_latitude_zone_arr()
        pass
    def run(self):
        recovery_time_tif_in = out_tif_dir+'recovery_time_in.tif'
        recovery_time_tif_out = out_tif_dir+'recovery_time_out.tif'
        recovery_time_tif_mix = out_tif_dir+'recovery_time_mix.tif'
        plt.figure(figsize=(8,7))
        self.cross_landuse_WB_recovery_time(recovery_time_tif_in,title='IN')
        plt.figure(figsize=(8,7))
        self.cross_landuse_WB_recovery_time(recovery_time_tif_out,title='OUT')
        plt.figure(figsize=(8,7))
        self.cross_landuse_WB_recovery_time(recovery_time_tif_mix,title='MIX')
        plt.show()
        pass

    def gen_latitude_zone_arr(self):
        tif_template = this_root + 'conf\\tif_template.tif'
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)

        latitude_range = np.linspace(0,360,7)

        # print latitude_range
        # exit()
        latitude_zone = []
        for latitude_i in range(len(latitude_range)):
            if latitude_i+1 == len(latitude_range):
                break
            start = latitude_range[latitude_i]
            end = latitude_range[latitude_i+1]
            for i in range(len(arr)):
                if start < i < end:
                    temp = []
                    for j in range(len(arr[0])):
                        temp.append(latitude_i)
                    latitude_zone.append(temp)
        latitude_zone = np.array(latitude_zone)
        # plt.imshow(latitude_zone)
        # plt.show()
        latitude_zone_dic = {}
        for latitude_i in range(len(latitude_range)-1):
            latitude_zone_dic[latitude_i] = []

        for i in range(len(latitude_zone)):
            for j in range(len(latitude_zone)):
                val = latitude_zone[i][j]
                key = '%03d.%03d'%(i,j)
                latitude_zone_dic[val].append(key)

        # for lzd in latitude_zone_dic:
        #     print lzd
        #     print latitude_zone_dic[lzd]

        return latitude_zone_dic


    def gen_landuse_zonal_index(self):
        # [8,10,11,16]
        # grassland shrubland RainfedCropland SparseVegetation
        # landuse = this_root+'/landuse/3.tif'
        index_landuse_dic = this_root+'arr\\landcover_dic.npy'

        dic = dict(np.load(index_landuse_dic).item())
        return dic
        # if not os.path.isfile('index_landuse_dic'):
        #     arr,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(landuse_tif)
        #     arr = np.array(arr)
        #     print 'landuse',np.shape(arr)
        #
        #     index_landuse_dic = {}
        #     for c in range(1,18):
        #         print 'building landuse zonal index',c
        #         index_landuse = []
        #         for i in range(len(arr)):
        #             for j in range(len(arr[i])):
        #                 if arr[i,j] == c:
        #                     index_landuse.append((i,j))
        #         index_landuse_dic[c] = index_landuse
        #     fw = open('index_landuse_dic','w')
        #     fw.write(str(index_landuse_dic))
        #     fw.close()
        #     return index_landuse_dic
        # else:
        #     f = open('index_landuse_dic','r')
        #     index_landuse_dic = f.read()
        #     f.close()
        #     index_landuse_dic = eval(index_landuse_dic)
        #     return index_landuse_dic


    def gen_WB_zonal_index(self,wb,classes):
        max_val = 3
        min_val = 0
        # range_class = [vals]
        # classes=2
        range_class = np.linspace(min_val,max_val,classes)
        index_WB_dic = {}
        for c in tqdm(range(len(range_class)),desc='building WB zonal index'):
            if c == range(len(range_class))[-1]:
                break
            class_index = []
            for i in range(len(wb)):
                for j in range(len(wb[i])):
                    if range_class[c]<wb[i][j]<range_class[c+1]:
                        class_index.append('%03d.%03d'%(i,j))
            index_WB_dic[c] = class_index
        return index_WB_dic, range_class

    def intersection(self,lst1, lst2):
        return list(set(lst1) & set(lst2))

    def cross_landuse_WB_recovery_time(self,recovery_time_tif,title='',color_=None):
        # 花点图

        # 1、加载恢复期 (底图)
        ############################  Recovery Time  ############################
        # recovery_time_tif = this_root + 'tif\\recovery_time\\recovery_time_mix.tif'
        # title = 'Mix'

        # recovery_time_tif = this_root + 'tif\\recovery_time\\pick_non_growing_season_events_plot_gen_recovery_time\\global.tif'
        # title = 'None Growing Season'

        # recovery_time_tif = this_root + 'tif\\recovery_time\\pick_post_growing_season_events_plot_gen_recovery_time\\global.tif'
        # title = 'Late Growing Season'

        # recovery_time_tif = this_root + 'tif\\recovery_time\\pick_pre_growing_season_events_plot_gen_recovery_time\\global.tif'
        # title = 'Early Growing Season'
        ############################  Recovery Time  ############################

        ############################  in out  ############################
        # Early
        # recovery_time_tif = this_root + 'tif\\recovery_time\\in_or_out\\pick_pre_growing_season_events_in.tif'
        # title = 'Drought in Early Growing Season and Recovered IN Current Growing Season'
        # recovery_time_tif = this_root + 'tif\\recovery_time\\in_or_out\\pick_pre_growing_season_events_out.tif'
        # title = 'Drought in Early Growing Season and Recovered OUT Current Growing Season'
        # # Late
        # recovery_time_tif = this_root + 'tif\\recovery_time\\in_or_out\\pick_post_growing_season_events_in.tif'
        # title = 'Drought in Late Growing Season and Recovered IN Current Growing Season'
        # recovery_time_tif = this_root + 'tif\\recovery_time\\in_or_out\\pick_post_growing_season_events_out.tif'
        # title = 'Drought in Late Growing Season and Recovered OUT Current Growing Season'
        # # None
        # recovery_time_tif = this_root + 'tif\\recovery_time\\in_or_out\\pick_non_growing_season_events_in.tif'
        # title = 'Drought in None Growing Season and Recovered IN Current Growing Season'
        # recovery_time_tif = this_root + 'tif\\recovery_time\\in_or_out\\pick_non_growing_season_events_out.tif'
        # title = 'Drought in None Growing Season and Recovered OUT Current Growing Season'
        ############################  in out  ############################

        ############################  Ratio  ############################
        # recovery_time_tif = recovery_time_branch.Recovery_time1().this_class_tif + 'ratio\\late_ratio.tif'
        # title = 'Early Growing Season Ratio'

        # recovery_time_tif = this_root + 'tif\\Ratio\\pick_post_growing_season_events.tif'
        # title = 'Ratio of Overwinter in Late Growing Season'

        # recovery_time_tif = this_root + 'tif\\Ratio\\pick_non_growing_season_events.tif'
        # title = 'Drought in None Growing Season and Recovered IN Current Growing Season'

        ############################  Ratio  ############################


        ############################  recovery new 2020  ############################

        # recovery_time_tif = this_root + 'new_2020\\tif\\recovery_time\\early.tif'
        # title = 'Early Growing Season'

        # recovery_time_tif = this_root + 'new_2020\\tif\\recovery_time\\late.tif'
        # title = 'Late Growing Season'

        ############################  recovery new 2020  ############################


        ############################  ratio new 2020  ############################
        # recovery_time_tif = this_root + 'new_2020\\tif\\ratio\\early_ratio.tif'
        # title = 'Ratio of Overwinter in Early Growing Season'
        # recovery_time_tif = this_root + 'new_2020\\tif\\ratio\\late_ratio.tif'
        # title = 'Ratio of Overwinter in Late Growing Season'
        ############################  ratio new 2020  ############################

        ############################  recovery new 2020 in out ############################
        # early in
        # recovery_time_tif = this_root + 'new_2020\\tif\\recovery_time_in_out\\early_in_arr.tif'
        # title = 'Drought in Early Growing Season and Recovered IN Current Growing Season'
        # early out
        # recovery_time_tif = this_root + 'new_2020\\tif\\recovery_time_in_out\\early_out_arr.tif'
        # title = 'Drought in Early Growing Season and Recovered OUT Current Growing Season'

        # late in
        # recovery_time_tif = this_root + 'new_2020\\tif\\recovery_time_in_out\\late_in_arr.tif'
        # title = 'Drought in Late Growing Season and Recovered IN Current Growing Season'
        # late out
        # recovery_time_tif = this_root + 'new_2020\\tif\\recovery_time_in_out\\late_out_arr.tif'
        # title = 'Drought in Late Growing Season and Recovered OUT Current Growing Season'
        ############################  recovery new 2020 in out ############################



        recovery_time_arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(recovery_time_tif)
        # mask NDVI
        recovery_time_arr = NDVI().mask_arr_with_NDVI(recovery_time_arr)
        # recovery_time_arr = ''
        # plt.imshow(recovery_time_arr)
        # plt.show()
        # 2、生成 HI 分级别 dic
        HI_tif = this_root+'tif\\HI\\HI.tif'
        # HI_tif = r'D:\project05\branch2020\tif\Bio_diversity\bio_diversity_normalized.tif'
        HI_arr,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(HI_tif)
        HI_arr[HI_arr>2.] = np.nan
        # HI_class_dic, range_class = self.gen_WB_zonal_index(HI_arr, 12)

        # 3、landuse 字典
        # lc组合1
        # landuse_types = [1,2,3,4,5,[6,7],[8,9],10,12]
        # labels = ['ENF','EBF','DNF','DBF','MF','Shrublands',
        #           'Savannas','Grasslands','Croplands']
        # lc组合2
        # landuse_types = [[1, 2, 3, 4, 5], [6, 7], [8, 9], 10, 12]
        # labels = ['Forest', 'Shrublands',
        #           'Savannas', 'Grasslands', 'Croplands']
        # lc组合3
        landuse_types = [[1, 2, 3, 4, 5], [6, 7 ,8, 9], 10]
        labels = ['Forest', 'Shrublands_Savanna', 'Grasslands']


        landuse_class_dic = self.gen_landuse_zonal_index()

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

        # 4、生成纬度 dic
        # latitude_dic = self.gen_latitude_zone_arr()
        # 4.1 koppen zone
        latitude_dic = Koppen().do_reclass()
        # 5、交叉像素

        intersect_dic = {}
        flag = 0

        markers_dic = {'EBF':"X",
                       'Shrublands':"*",
                       'Forest':"X",
                      'MF':"s",
                      'DNF':"o",
                       'ENF':"^",
                      'Savannas':"D",
                      'DBF':"P",
                      'Croplands':"v",
                       'Grasslands':"p",
                       'Shrublands_Savanna':"D",
                       }
        # cmap = sns.color_palette('RdBu_r', len(latitude_dic))
        # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        # cmap = sns.color_palette(flatui)

        # color dic #
        cmap = sns.diverging_palette(236,0,s=99,l=50,n=len(latitude_dic), center="light")
        color_dic = {}
        cm = 0
        for lc in latitude_dic:
            # print lc,cm,cmap[cm]
            color_dic[lc] = cmap[cm]
            cm += 1
        # color dic #

        markers_flag = 0
        scatter_dic = {}
        for lc_i in landuse_dic:
            scatter_labels = []
            for lat_i in latitude_dic:
                lc_pixs = landuse_dic[lc_i]
                lat_pixs = latitude_dic[lat_i]
                # print len(lc_pixs)
                # print len(lat_pixs)
                # print lat_pixs
                # print '****'
                intersect = self.intersection(lc_pixs,lat_pixs)
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
                    HI_picked_val = Tools().pick_vals_from_2darray(HI_arr,intersect_int)
                    HI_picked_val[HI_picked_val<0] = np.nan
                    # HI_picked_val[HI_picked_val>2] = np.nan
                    HI_mean,xerr = Tools().arr_mean_nan(HI_picked_val)
                    # print 'HI_mean,xerr',HI_mean,xerr
                    # print xerr
                    # x.append(HI_mean)
                    # 挑y轴
                    recovery_picked_val = Tools().pick_vals_from_2darray(recovery_time_arr,intersect_int)
                    recovery_picked_val[recovery_picked_val<0] = np.nan
                    # recovery_picked_val[recovery_picked_val>18] = np.nan
                    # print recovery_picked_val
                    # exit()
                    # plt.imshow(recovery_picked_val)
                    # plt.show()
                    recovery_mean,yerr = Tools().arr_mean_nan(recovery_picked_val)
                    # print 'recovery_mean,yerr',recovery_mean,yerr
                    # y.append(recovery_mean)
                    if recovery_mean == None:
                        continue
                    scatter_dic[key] = [HI_mean,recovery_mean,xerr,yerr]
            # print scatter_labels
            # marker = markers[markers_flag]
            # plt.scatter(x,y,c=color_dic[],marker=marker)
            # markers_flag += 1
        X = []
        Y = []
        # sns.set(color_codes=True) # 背景
        # plt.figure(figsize=(8,7))
        for key in scatter_dic:
            lc,lat = key.split('.')
            # print key
            # lat = int(lat)
            # print lc,lat
            marker = markers_dic[lc]
            # color = cls_color_dic[lat]
            color = color_dic[lat]
            x,y,xerr,yerr = scatter_dic[key]
            # zorder : 图层顺序
            # plt.scatter(x,y,s=80,c='#'+color,marker=marker,edgecolors='black',linewidths=1,zorder=99,label=lc)
            # plt.scatter(x,y,s=80,c='#'+color,marker=marker,edgecolors='black',linewidths=1,zorder=99)
            plt.scatter(x,y,s=80,c=color,marker=marker,edgecolors='black',linewidths=1,zorder=99)
            # print x,y,xerr,yerr
            plt.errorbar(x,y,xerr=xerr/8.,yerr=yerr/8.,c='gray',zorder=0,alpha=0.5)
            X.append(x)
            Y.append(y)
        # plt.legend()
        plt.title(title)
        sns.regplot(X,Y,scatter=False,color=color_)
        a,b,r = Tools().linefit(X,Y)
        print 'r',r


def plot_ratio_of_winter():
    recovery_dic = load_recovery_time_dic()
    void_spatial_dic = DIC_and_TIF().void_spatial_dic()
    # a=['in','in','in','in','in','out']
    # in_ = a.count('in')
    # ratio = float(in_)/float(len(a))
    # print ratio
    # exit()
    for key in recovery_dic:
        pix, mark, eln, date_range, drought_start, recovery_start = split_key(key)
        if mark == 'in':
            void_spatial_dic[pix].append('in')
            pass
        elif mark == 'out':
            void_spatial_dic[pix].append('out')

    ratio_dic = {}
    for pix in void_spatial_dic:
        ratio_list = void_spatial_dic[pix]
        if len(ratio_list) == 0:
            continue
        in_num = float(ratio_list.count('out'))
        total_num = len(ratio_list)
        ratio = in_num/total_num
        ratio_dic[pix] = ratio
    spatial_arr = DIC_and_TIF().pix_dic_to_spatial_arr(ratio_dic)
    DIC_and_TIF().arr_to_tif(spatial_arr, out_tif_dir + 'ratio.tif')

    pass

def plot_time_series():
    recovery_f = this_root+'branch2020\\arr\\Recovery_time1\\recovery_time_composite\\composite.npy'
    recovery_dic = dict(np.load(recovery_f).item())
    year_dic = {}
    for y in range(1982,2016):
        year_dic[y] = []
    for key in tqdm(recovery_dic):
        events = recovery_dic[key]
        for event in events:
            recovery_time, mark, recovery_date_range, drought_date_range, eln = event
            # if recovery_time == None or recovery_time > 18:

            if recovery_time == None:
                continue
            if mark == 'in':
            # if 'tropical' in eln:
                start = recovery_date_range[0]
                year = int(start//12 + 1982)
                year_dic[year].append(recovery_time)


    ratio1 = []
    ratio2 = []
    ratio3 = []
    ratio4 = []
    for i in range(7):
        recovery_list = []

        selected_year = []
        for y in range(1982, 2016):
            if y in range(1981+5*i,1981+5*(i+1)):
                selected_year.append(y)
                for val in year_dic[y]:
                    recovery_list.append(val)
        print selected_year
        cls1 = []
        cls2 = []
        cls3 = []
        cls4 = []
        for recovery in recovery_list:
            if recovery <= 6:
                cls1.append(recovery)
            # elif 6 < recovery <= 9:
            elif 6 < recovery <= 12:
                cls2.append(recovery)
            elif 12 < recovery <= 24:
            # elif 12 < recovery <= 24:
                cls3.append(recovery)
            elif 24 < recovery:
            # elif 24 < recovery <= 18:
                cls4.append(recovery)
            else:
                raise IOError('recovery error')
        total = len(cls1) + len(cls2) + len(cls3) + len(cls4)
        r1 = float(len(cls1)) / total
        r2 = float(len(cls2)) / total
        r3 = float(len(cls3)) / total
        r4 = float(len(cls4)) / total
        ratio1.append(r1)
        ratio2.append(r2)
        ratio3.append(r3)
        ratio4.append(r4)
    ratio1 = np.array(ratio1)
    ratio2 = np.array(ratio2)
    ratio3 = np.array(ratio3)
    ratio4 = np.array(ratio4)
    plt.bar(range(len(ratio1)),ratio1)
    plt.bar(range(len(ratio1)),ratio2,bottom=ratio1)
    plt.bar(range(len(ratio1)),ratio3,bottom=ratio1+ratio2)
    plt.bar(range(len(ratio1)),ratio4,bottom=ratio1+ratio2+ratio3)

    plt.show()


class Ternary_plot:

    def __init__(self):
        pass



    def plot_scatter(self):
        this_root_branch = this_root + 'branch2020\\'
        # 1 load soil and recovery time data
        # recovery_tif = this_root_branch+'tif\\Recovery_time1\\recovery_time\\mix.tif'

        recovery_tif = out_tif_dir+'recovery_time_in.tif'
        vmin=0
        vmax=4

        # recovery_tif = out_tif_dir+'recovery_time_mix.tif'
        # vmin = 0
        # vmax = 9

        # recovery_tif = out_tif_dir+'recovery_time_out.tif'
        # vmin = 7
        # vmax = 13
        # recovery_tif = this_root_branch+'tif\\Recovery_time1\\recovery_time\\early.tif'
        tif_CLAY = this_root_branch+'\\tif\\HWSD\\T_CLAY_resample.tif'
        tif_SAND = this_root_branch+'\\tif\\HWSD\\T_SAND_resample.tif'
        tif_SILT = this_root_branch+'\\tif\\HWSD\\T_SILT_resample.tif'

        arr_clay = to_raster.raster2array(tif_CLAY)[0]
        arr_sand = to_raster.raster2array(tif_SAND)[0]
        arr_silt = to_raster.raster2array(tif_SILT)[0]
        arr_recovery = to_raster.raster2array(recovery_tif)[0]

        arr_clay[arr_clay<-999] = np.nan
        arr_sand[arr_sand<-999] = np.nan
        arr_silt[arr_silt<-999] = np.nan
        arr_recovery[arr_recovery<-999] = np.nan

        # 2 make empty points list dic
        points_list_dic = {}
        for i in range(len(arr_clay)):
            for j in range(len(arr_clay[0])):
                if np.isnan(arr_silt[i][j]):
                    continue
                silt = int(arr_silt[i][j])
                sand = int(arr_sand[i][j])
                clay = int(arr_clay[i][j])
                points_list_dic[(sand,silt,clay)] = []


        for i in range(len(arr_clay)):
            for j in range(len(arr_clay[0])):
                if np.isnan(arr_silt[i][j]):
                    continue
                silt = int(arr_silt[i][j])
                sand = int(arr_sand[i][j])
                clay = int(arr_clay[i][j])
                recovery = arr_recovery[i][j]
                if np.isnan(recovery):
                    continue
                if recovery > 18:
                    continue
                points_list_dic[(sand, silt, clay)].append(recovery)

        points_dic = {}
        for key in points_list_dic:
            vals = points_list_dic[key]
            mean,xerr = Tools().arr_mean_nan(vals)
            if not mean == None:
                points_dic[key] = mean

        # ternary.plt.figure(figsize=(11, 6))
        # figure = plt.figure(figsize=(16, 7))
        # plt.AxesSubplot
        # fig = figure.add_subplot(111)
        figure, ax = plt.subplots(figsize=(16, 7))
        fig, tax = ternary.figure(ax=ax,scale=100,permutation='120')
        # figure, ax = plt.subplots(figsize=(11, 6))

        # fig = ternary.plt.figure(figsize=(11, 6))
        # figure
        tax.boundary(linewidth=1.5)
        tax.gridlines(color="black", multiple=10)
        # tax.gridlines(color="blue", multiple=10, linewidth=0.5)

        tax.ticks(linewidth=1, multiple=10, offset=0.03,clockwise=True)
        # tax.ticks(linewidth=1, multiple=10, offset=0.03)
        fontsize = 12
        tax.left_axis_label("Clay", fontsize=fontsize,offset=0.1)
        tax.right_axis_label("Silt", fontsize=fontsize,offset=0.1)
        tax.bottom_axis_label("Sand", fontsize=fontsize,offset=0.1)
        plt.axis('equal')
        # tax.heatmap(data,style="triangular",vmin=0, vmax=100)
        points = []
        c_vals = []
        for point in points_dic:
            # print point
            points.append(point)
            c_vals.append(points_dic[point])
        # cmap = 'gist_heat_r'
        # cmap = 'hot_r'
        # cmap = 'afmhot_r'
        # cmap = 'magma_r'
        # cmap = 'YlGnBu'
        # cmap = 'hot'
        cmap = 'RdBu_r'
        # cmap = 'BrBG_r'
        tax.scatter(points,c=c_vals,cmap=cmap,colormap=cmap,s=16,vmin=vmin,marker='h',vmax=vmax,colorbar=True,alpha=1,linewidth=0)
        # tax.scatter(points,c=c_vals,cmap=cmap,colormap=cmap,s=16,marker='h',colorbar=True,alpha=1,linewidth=0)
        tax.boundary()
        plt.axis('off')
        # plt.colorbar(ax)
        plt.show()

        pass


    def soil_texture_condition(self,sand,silt,clay,zone):
        '''
        classified by the proportion of silt, sand and clay
        :return:
        '''
        c_heavy_clay = (sand<40 and silt<40 and clay>60)
        c_clay = (sand<=45 and silt<=40 and 40<=clay<=60)
        c_silty_clay = (sand<=20 and 40<=silt<=60 and 40<=clay<=60)
        c_sandy_clay = (45<=sand<=65 and silt<20 and 37<=clay<=55)
        c_sandy_clay_loam = (45<=sand<=80 and silt<=27 and 20<=clay<=37)
        c_clay_loam = (20<=sand<=45 and 16<=silt<=52 and 28<=clay<=40)
        c_silty_clay_loam = (sand<=20 and 40<=silt<=72 and 28<=clay<=40)
        c_loam = (22<=sand<=52 and 28<=silt<=50 and 8<=clay<=28)
        c_silt_loam = (sand<=50 and 50<=silt<=80 and 0<=clay<=28) or (sand<=8 and 80<=silt<=88 and 12<=clay<=20)
        c_silt = (sand<=20 and silt>=80 and clay<=12)
        # 有问题 TODO: need to be modified
        c_sandy_loam = (42<=sand<=85 and silt<=50 and clay<=20)#######
        c_sand = (sand>=85 and silt<=15 and clay<=10) #######
        c_loamy_sand = ()

        if zone == 'c_heavy_clay':
            return c_heavy_clay
        elif zone == 'c_clay':
            return c_clay
        elif zone == 'c_silty_clay':
            return c_silty_clay
        elif zone == 'c_sandy_clay':
            return c_sandy_clay
        elif zone == 'c_sandy_clay_loam':
            return c_sandy_clay_loam
        elif zone == 'c_clay_loam':
            return c_clay_loam
        elif zone == 'c_silty_clay_loam':
            return c_silty_clay_loam
        elif zone == 'c_loam':
            return c_loam
        elif zone == 'c_silt_loam':
            return c_silt_loam
        elif zone == 'c_silt':
            return c_silt
        else:
            raise IOError('input error')


        pass


    def plot_classify_soil(self):
        # 1 load soil and recovery time data
        # recovery_tif = this_root_branch+'tif\\Recovery_time1\\recovery_time\\mix.tif'
        recovery_tif = this_root_branch + 'tif\\Recovery_time1\\recovery_time\\late.tif'
        # recovery_tif = this_root_branch+'tif\\Recovery_time1\\recovery_time\\early.tif'
        tif_CLAY = this_root_branch + '\\tif\\HWSD\\T_CLAY_resample.tif'
        tif_SAND = this_root_branch + '\\tif\\HWSD\\T_SAND_resample.tif'
        tif_SILT = this_root_branch + '\\tif\\HWSD\\T_SILT_resample.tif'

        arr_clay = to_raster.raster2array(tif_CLAY)[0]
        arr_sand = to_raster.raster2array(tif_SAND)[0]
        arr_silt = to_raster.raster2array(tif_SILT)[0]
        arr_recovery = to_raster.raster2array(recovery_tif)[0]

        arr_clay[arr_clay < -999] = np.nan
        arr_sand[arr_sand < -999] = np.nan
        arr_silt[arr_silt < -999] = np.nan
        arr_recovery[arr_recovery < -999] = np.nan

        # 2 make empty points list dic
        points_list_dic = {}
        for i in range(len(arr_clay)):
            for j in range(len(arr_clay[0])):
                if np.isnan(arr_silt[i][j]):
                    continue
                silt = int(arr_silt[i][j])
                sand = int(arr_sand[i][j])
                clay = int(arr_clay[i][j])
                points_list_dic[(sand, silt, clay)] = []

        for i in range(len(arr_clay)):
            for j in range(len(arr_clay[0])):
                if np.isnan(arr_silt[i][j]):
                    continue
                silt = int(arr_silt[i][j])
                sand = int(arr_sand[i][j])
                clay = int(arr_clay[i][j])
                recovery = arr_recovery[i][j]
                if np.isnan(recovery):
                    continue
                if recovery > 18:
                    continue
                points_list_dic[(sand, silt, clay)].append(recovery)

        points_dic = {}
        for key in points_list_dic:
            vals = points_list_dic[key]
            mean, xerr = Tools().arr_mean_nan(vals)
            if not mean == None:
                points_dic[key] = mean

        zones = [
            'c_heavy_clay',
            'c_clay',
            'c_silty_clay',
            'c_sandy_clay',
            'c_sandy_clay_loam',
            'c_clay_loam',
            'c_silty_clay_loam',
            'c_loam',
            'c_silt_loam',
            'c_silt',
        ]
        zone_dic = {}
        for z in zones:
            pix_list = []
            for key in points_dic:
                sand,silt,clay = key
                val = points_dic[key]
                if self.soil_texture_condition(sand,silt,clay,z):
                    pix_list.append(val)
            pix_list_mean = np.mean(pix_list)
            # print z,pix_list_mean
            zone_dic[z] = pix_list_mean

        fig, tax = ternary.figure(scale=100, permutation='120')
        data = {}
        for zone in tqdm(zone_dic):
            if np.isnan(zone_dic[zone]):
                continue
            for i in range(101):
                for j in range(101):
                    for k in range(101):
                        if not i+j+k==100:
                            continue
                        if self.soil_texture_condition(i,j,k,zone):
                            data[(i,j,k)] = zone_dic[zone]
        tax.heatmap(data,cmap='RdBu_r', style="dual-triangular")
        tax.boundary(linewidth=1.5)
        tax.gridlines(color="black", multiple=10)

        tax.ticks(linewidth=1, multiple=10, offset=0.03, clockwise=True)
        fontsize = 12
        tax.left_axis_label("Clay", fontsize=fontsize, offset=0.1)
        tax.right_axis_label("Silt", fontsize=fontsize, offset=0.1)
        tax.bottom_axis_label("Sand", fontsize=fontsize, offset=0.1)
        plt.axis('equal')

        tax.show()




    def clockwise_and_counter_clockwise_example(self):
        '''
        make sure clockwise and counter-clockwise
        :return:
        '''
        sand = 20
        silt = 20
        clay = 60
        # sand, silt, clay to axis 0, 1, 2 i.e. bottom, right, left

        # method 1
        points = [
            # (sand, silt, clay), # clockwise=False
            (silt, clay, sand), # clockwise=True
        ]

        # or method 2
        # figure, tax = ternary.figure(scale=100, permutation='012')
        # figure, tax = ternary.figure(scale=100, permutation='012')  # clockwise=False
        figure, tax = ternary.figure(scale=100,permutation='120') # clockwise=True

        tax.boundary(linewidth=1.5)
        tax.gridlines(color="black", multiple=10)
        tax.ticks(linewidth=1, multiple=10, offset=0.03, clockwise=True)
        # tax.ticks(linewidth=1, multiple=10, offset=0.03)
        fontsize = 12
        tax.left_axis_label("Clay", fontsize=fontsize, offset=0.2)
        tax.right_axis_label("Silt", fontsize=fontsize, offset=0.2)
        tax.bottom_axis_label("Sand", fontsize=fontsize, offset=0.2)
        plt.axis('equal')
        tax.scatter(points, marker='D')
        tax.boundary()
        plt.show()


class Find_Threshold:
    '''
    plot recovery time scatter to find threshold
    '''
    def __init__(self):
        from mpl_toolkits.mplot3d import Axes3D
        pass

    def run(self):
        this_root_branch = this_root + 'branch2020\\'
        template_tif = this_root+'conf\\tif_template.tif'
        template_arr = to_raster.raster2array(template_tif)[0]
        template_arr[template_arr<-999]=np.nan

        # recovery_tif = out_tif_dir+'recovery_time_in.tif'
        # vmin=0;vmax=4
        # title = 'in'

        # recovery_tif = out_tif_dir + 'recovery_time_out.tif'
        # vmin = 7;vmax = 18
        # title = 'out'

        recovery_tif = out_tif_dir + 'recovery_time_mix.tif'
        vmin = 0;vmax = 18
        title = 'mix'

        recovery_arr = to_raster.raster2array(recovery_tif)[0]
        recovery_arr[recovery_arr<-999] = np.nan

        bio_tif = this_root_branch + 'tif\\Bio_diversity\\bio_diversity_normalized.tif'
        bio_arr = to_raster.raster2array(bio_tif)[0]
        bio_arr[bio_arr<-999] = np.nan

        rooting_tif = this_root+'data\\Global_effective_plant_rooting_depth\\Effective_Rooting_Depth.tif'
        rooting_arr = to_raster.raster2array(rooting_tif)[0]
        rooting_arr[rooting_arr<-999] = np.nan
        rooting_arr[rooting_arr>2] = np.nan

        landcover_tif = this_root+'data\\landcover\\tif\\0.5\\landcover_0.5.tif'
        landcover_arr = to_raster.raster2array(landcover_tif)[0]

        HI_tif = this_root + 'tif\\HI\\HI.tif'
        HI_arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(HI_tif)
        HI_arr[HI_arr > 2.] = np.nan
        HI_arr[HI_arr < -999] = np.nan
        x = []
        y = []
        z = []
        c = []
        marker = []

        for i in tqdm(range(len(template_arr))):
            for j in range(len(template_arr[0])):
                recovery = recovery_arr[i][j]
                if np.isnan(recovery):
                    continue
                if recovery > 24:
                    continue

                landcover = landcover_arr[i][j]
                if landcover in {1, 2, 3, 4, 5}:  # Forests
                    marker.append('X')
                elif landcover in {6, 7, 8, 9}:  # Shrubland and Savannas
                    marker.append('D')
                elif landcover == 10:  # Grassland
                    marker.append('p')
                else:
                    continue
                bio = bio_arr[i][j]
                rooting = rooting_arr[i][j]
                HI = HI_arr[i][j]
                if np.nan in {bio,rooting,HI,recovery}:
                    continue
                x.append(bio)
                y.append(rooting)
                z.append(HI)
                c.append(recovery)


        ############### plot un-filled circles ###############
        cmap = mpl.cm.get_cmap(name='RdBu_r', lut=None)
        std_c = []
        max_c = max(c)
        min_c = min(c)
        for i in c:
            std = (i-min_c)/(max_c-min_c)
            std_c.append(std)

        edge_color_list = []
        for i in std_c:
            edge_color_list.append(cmap(i))
        ############### plot un-filled circles ###############

        fig = plt.figure()
        ############### plot 3d ###############
        # ax = fig.gca(projection='3d')
        # ax.scatter(x,y,z,c=c,marker='o',s=1,cmap='jet',alpha=0.5,vmin=0,vmax=7)
        ############### plot 3d ###############

        ############### plot 2d ###############
        # plt.scatter(x,y,marker='.',c=c,s=4,cmap='RdBu_r',alpha=0.7,vmin=0,vmax=3)  # filled circles
        plt.scatter(x,y,marker='.',c=c,s=4,cmap='RdBu_r',alpha=0.7,vmin=vmin,vmax=vmax)  # filled circles
        # plt.scatter(x,y,marker='o',s=18,cmap='RdBu_r',alpha=0.5,vmin=0,vmax=4,facecolors='none', edgecolors=edge_color_list)  # un-filled circles
        ############### plot 2d ###############
        # c = '', edgecolors = 'g'
        plt.title(title)
        plt.show()

        ################## plot animation ##################
        # from matplotlib import animation
        # def animate(i):
        #     print i,'/',360
        #     ax.view_init(elev=10., azim=i)
        #     return fig,
        #
        # anim = animation.FuncAnimation(fig, animate, init_func=None,
        #                                frames=360, interval=20, blit=True)
        # anim.save(this_root+'png\\animation\\Find_Threshold_late.html', fps=30, extra_args=['-vcodec', 'libx264'])
        ################## plot animation ##################

        pass

    def landcover_dic(self):
        '''
        :return: dic['grassland']=('000.000','001.001',...,'nnn.nnn')
        '''
        index_landuse_dic = this_root + 'arr\\landcover_dic.npy'
        dic = dict(np.load(index_landuse_dic).item())
        landuse_class_dic = dic
        landuse_types = [[1, 2, 3, 4, 5], [6, 7, 8, 9], 10]
        labels = ['Forest', 'Shrublands_Savanna', 'Grasslands']
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
            landuse_dic[lc_label] = set(landuse_index)
        return landuse_dic



class RF:

    def __init__(self):
        self.this_class_arr = this_root_PLOT + 'Random_Forest\\arr\\RF\\'
        self.this_class_tif = this_root_PLOT + 'Random_Forest\\tif\\RF\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        self.this_root_branch = this_root + 'branch2020\\'
        pass

    def run(self):
        # self.load_variable()
        # 0 分区 do_partition
        # self.do_partition()
        # 1 分区 IN OUT EARLY LATE TROPICAL
        # self.check_partition()
        # 2 RF
        self.do_random_forest_train()
        # -1 check results
        # self.check_result()
        pass

    def __split_keys(self, key):
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
        outf = this_root + 'arr\\cross_koppen_landuse_pix'
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
                intersect = Water_balance().intersection(lc_pixs, lat_pixs)
                # print intersect
                if len(intersect) > 100:
                    key = lc_i + '.' + str(kop_i)
                    scatter_labels.append(key)
                    intersect_int = []
                    for str_pix in intersect:
                        r, c = str_pix.split('.')
                        r = int(r)
                        c = int(c)
                        intersect_int.append([r, c])
                    # 挑x轴
                    HI_picked_val = Tools().pick_vals_from_2darray(HI_arr, intersect_int)
                    HI_picked_val[HI_picked_val < 0] = np.nan
                    # HI_picked_val[HI_picked_val>2] = np.nan
                    HI_mean, _ = Tools().arr_mean_nan(HI_picked_val)
                    cross_pix[key] = [intersect, HI_mean]
        np.save(outf, cross_pix)
        return cross_pix

    def parition(self, keys, pix_, mark_, eln_, desc=''):
        # pix_ = ['182.432','182.431']
        # mark_ = 'out'
        # eln_ = 'late'
        # keys = [
        #     ['182.432', 'out', 'late', '26.28', '26', '28'],
        #     ['182.431', 'out', 'late', '26.28', '26', '28'],
        # ]
        selected_keys = []
        for key in keys:
            pix, mark, eln, date_range, recovery_start, recovery_end = self.__split_keys(key)
            if pix in pix_:
                if mark == mark_:
                    if eln in eln_:
                        selected_keys.append(key)
        return selected_keys

    def do_partition(self):
        Ydic = Prepare1().this_class_arr + 'Y.npy'
        outf = self.this_class_arr + 'RF_partition'
        dic = dict(np.load(Ydic).item())
        keys = []
        for key in dic:
            keys.append(key)
            # print key
        keys = tuple(keys)
        cross_pix = self.cross_koppen_landuse()
        selected = {}
        for mark in ['in', 'out', 'tropical']:
            for eln in ['early', 'late', 'tropical']:
                condition_key = mark + '~' + eln
                cp_selected_keys = {}
                for cp in tqdm(cross_pix, desc=condition_key):
                    pix_, hi_mean = cross_pix[cp]
                    mark_, eln_ = mark, eln
                    search_keys = self.parition(keys, pix_, mark_, eln_)
                    cp_selected_keys[cp] = search_keys

                selected[condition_key] = cp_selected_keys
        np.save(outf, selected)

    def check_partition(self):
        f = self.this_class_arr + 'RF_partition.npy'
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

    def load_variable(self, partition_keys_dic, condition1, condition2):

        fdir = Prepare().this_class_arr
        print 'loading variables ...'
        Y_dic = dict(np.load(fdir + 'Y.npy').item())
        pre_dic = dict(np.load(fdir + 'PRE.npy').item())
        tmp_dic = dict(np.load(fdir + 'TMP.npy').item())
        swe_dic = dict(np.load(fdir + 'SWE.npy').item())
        cci_dic = dict(np.load(fdir + 'CCI.npy').item())
        NDVI_change_dic = dict(np.load(fdir + 'NDVI_change.npy').item())
        two_month_early_vals_mean_dic = dict(np.load(fdir + 'two_month_early_vals_mean.npy').item())

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
                _list = [pre, tmp, cci, ndvi_change, two_month_early_vals_mean]
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

    def load_variables_dir(self, fdir_dic, partition_keys_dic, condition1, condition2):
        '''
        load all variable in a single directory
        :return:
        '''
        selected_keys = []
        for key in partition_keys_dic:
            if condition1 in key:
                selected_keys += partition_keys_dic[key][condition2]
        pix_dic = {}
        nan = False
        Y = []
        X = []
        for key in selected_keys:
            y = fdir_dic['Y'][key]
            if y > 18:
                continue

            if 'in' == condition1:
                try:
                    variables_names = [
                        'PRE', 'PRE_mean', 'PRE_std','PRE_pre',
                        'TMP', 'TMP_mean', 'TMP_std','TMP_pre',
                        'CCI', 'CCI_mean', 'CCI_std','CCI_pre',
                        # 'SWE', 'SWE_mean', 'SWE_std',
                        'NDVI_change', 'two_month_early_vals_mean',
                        'sand', 'silt', 'clay',
                        'bio','HI'
                    ]
                    variables_vals = []
                    for x in variables_names:
                        val = fdir_dic[x][key]
                        variables_vals.append(val)
                except Exception as e:
                    print e
                    # exit()
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

            elif 'out' == condition1:
                try:
                    variables_names = [
                        'PRE', 'PRE_mean', 'PRE_std','PRE_winter',
                        'TMP', 'TMP_mean', 'TMP_std','TMP_winter',
                        'CCI', 'CCI_mean', 'CCI_std','CCI_winter',
                        'SWE', 'SWE_mean', 'SWE_std','SWE_winter',
                        'NDVI_change', 'two_month_early_vals_mean',
                        'sand', 'silt', 'clay',
                        'bio','HI'
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

            elif 'tropical' == condition1:
                try:
                    variables_names = [
                        'PRE', 'PRE_mean', 'PRE_std', 'PRE_pre',
                        'TMP', 'TMP_mean', 'TMP_std', 'TMP_pre',
                        'CCI', 'CCI_mean', 'CCI_std', 'CCI_pre',
                        # 'SWE', 'SWE_mean', 'SWE_std',
                        'NDVI_change', 'two_month_early_vals_mean',
                        'sand', 'silt', 'clay',
                        'bio','HI'
                    ]
                    variables_vals = []
                    for x in variables_names:
                        val = fdir_dic[x][key]
                        variables_vals.append(val)
                except Exception as e:
                    print e
                    # exit()
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

    def random_forest_train(self, X, Y, selected_pix_spatial, isplot=False, title=''):

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
            out_png_dir = this_root + 'png\\RF_importances\\'
            Tools().mk_dir(out_png_dir)
            # 1 plot spatial
            # plt.figure()
            # plt.imshow(selected_pix_spatial,cmap='gray')

            # 2 plot importance
            plt.subplot(211)
            plt.title(title)
            y_min = min(importances)
            y_max = max(importances)
            offset = (y_max - y_min)
            y_min = y_min - offset * 0.3
            y_max = y_max + offset * 0.3

            plt.ylim(y_min, y_max)
            plt.bar(range(len(importances)), importances, width=0.3)
            ax = plt.subplot(212)
            KDE_plot().plot_scatter(Y_test, y_pred, cmap='jet', s=10, ax=ax, linewidth=0)
            plt.axis('equal')
            plt.savefig(out_png_dir + title + '.png', ppi=300)
            plt.close()
            # plt.show()
        #### plot ####

        return importances, mse, r_model, Y_test, y_pred, r_X

    def kernel_do_random_forest_train(self, params):
        c1, c2, partition_keys_dic, fdir_dic = params
        key = c1 + '-' + c2
        # X, Y, selected_pix_spatial = self.load_variable(partition_keys_dic, c1, c2)
        X, Y, selected_pix_spatial = self.load_variables_dir(fdir_dic, partition_keys_dic, c1, c2)
        ################## debug
        # importances, mse, r, Y_test, y_pred, rX = self.random_forest_train(X, Y, selected_pix_spatial, isplot=True,title=key)
        # print importances, mse, r, Y_test, y_pred, rX
        #         # exit()
        ################## run
        try:
            importances, mse, r, Y_test, y_pred, rX = self.random_forest_train(X, Y, selected_pix_spatial,
                                                                               isplot=False)
            result = key, {'importances': importances, 'mse': mse, 'r': r, 'Y_test': Y_test, 'y_pred': y_pred,
                           'rX': rX}
            return result
        except Exception as e:
            print e,'error'
            return key, []
        pass

    def do_random_forest_train(self):

        out_result_dic = self.this_class_arr + '\\RF_result_dic_arr'
        partition_f = self.this_class_arr+'RF_partition.npy'
        partition_keys_dic = dict(np.load(partition_f).item())
        condition1_list = [
            'in','out','tropical'
        ]
        condition2_list = []
        for i in partition_keys_dic['in~early']:
            condition2_list.append(i)

        fdir = Prepare1().this_class_arr
        fdir_dic = {}
        for f in tqdm(os.listdir(fdir), desc='loading each variable...'):
            if not f.endswith('.npy'):
                continue
            x = f.split('.')[0]
            x_dic = dict(np.load(fdir + f).item())
            fdir_dic[x] = x_dic
        params = []
        for c1 in condition1_list:
            for c2 in condition2_list:
                params.append([c1, c2, partition_keys_dic, fdir_dic])
        ###1### debug
        results = []
        for p in tqdm(params):
            # try:
            result = self.kernel_do_random_forest_train(p)
            results.append(result)
            # except Exception as e:
            #     pass
                # print e
        np.save(out_result_dic, results)

        pass

    def check_result(self):
        result_f = self.this_class_arr+'RF_result_dic_arr.npy'
        dic = np.load(result_f)
        for region,result in dic:
            print region
            print result
            # sleep()

class Plot_RF_train_events_result:
    def __init__(self):
        self.this_class_arr = this_root_PLOT + 'Random_Forest\\arr\\Plot_RF_train_events_result\\'
        self.this_class_tif = this_root_PLOT + 'Random_Forest\\tif\\Plot_RF_train_events_result\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        self.this_root_branch = this_root + 'branch2020\\'
        pass


    def run(self):
        # self.plot_bar()
        # self.plot_scatter()
        # self.check_scatter_size()
        self.plot_box_plot()
        pass


    def plot_box_plot(self):
        f = RF().this_class_arr + 'RF_result_dic_arr.npy'
        arr = np.load(f)
        for condition in ['in','out','tropical']:
        # for condition in ['in']:
            box = {}
            for i in arr:
                region, content_dic = i
                if not condition in region:
                    continue
                if condition == 'in' or condition == 'tropical':
                    variables = [
                        'PRE', 'PRE_mean', 'PRE_std', 'PRE_pre',
                        'TMP', 'TMP_mean', 'TMP_std', 'TMP_pre',
                        'CCI', 'CCI_mean', 'CCI_std', 'CCI_pre',
                        'NDVI_change', 'two_month_early_vals_mean',
                        'sand', 'silt', 'clay',
                        'bio', 'HI'
                    ]
                    climate_variables_index = range(12)
                    traits_variables_index = range(12,14)
                    constant_variables_index = range(14,19)

                elif condition == 'out':
                    variables = [
                        'PRE', 'PRE_mean', 'PRE_std','PRE_winter',
                        'TMP', 'TMP_mean', 'TMP_std','TMP_winter',
                        'CCI', 'CCI_mean', 'CCI_std','CCI_winter',
                        'SWE', 'SWE_mean', 'SWE_std','SWE_winter',
                        'NDVI_change', 'two_month_early_vals_mean',
                        'sand', 'silt', 'clay',
                        'bio','HI'
                    ]
                    climate_variables_index = range(12)
                    traits_variables_index = range(12, 14)
                    constant_variables_index = range(14, 19)
                else:
                    raise IOError
                try:
                    importance = content_dic['importances']
                    climate_variables = []
                    for indx in climate_variables_index:
                        climate_variables.append(importance[indx])
                    traits_variables = []
                    for indx in traits_variables_index:
                        traits_variables.append(importance[indx])
                    constant_variables = []
                    for indx in constant_variables_index:
                        constant_variables.append(importance[indx])
                    # exit()
                    box[region] = [climate_variables,traits_variables,constant_variables]
                except:
                    pass


            all_climate = []
            all_traits = []
            all_constant = []
            for i in box:
                climate = box[i][0]
                traits = box[i][1]
                constant = box[i][2]
                for i in climate:
                    all_climate.append(i)
                for i in traits:
                    all_traits.append(i)
                for i in constant:
                    all_constant.append(i)
            data = [all_climate,all_traits,all_constant]
            plt.boxplot(data,showfliers=False)
            plt.title(condition)
            plt.show()
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


    def check_scatter_size(self):
        f = RF().this_class_arr + 'RF_result_dic_arr.npy'
        arr = np.load(f)
        len_in_out_dic = {}
        importances_hist = []
        for i in arr:
            region, result_dic = i

            try:
                importances = result_dic['importances']
                for im in importances:
                    importances_hist.append(im)
                r = result_dic['r']
                rX = result_dic['rX']
                if np.isnan(r):
                    continue
            except:
                continue
        plt.hist(importances_hist,bins=20)
        plt.show()


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
        # f = RF_train_events().this_class_arr+'RF_result_dic_arr.npy'
        # f = this_root+'branch2020\\PLOT\\arr\\RF_result_dic_arr.npy'
        f = RF().this_class_arr+'RF_result_dic_arr.npy'
        arr = np.load(f)
        len_in_out_dic = {}
        for i in arr:
            region,content_dic = i
            # print region
            # continue
            # exit()
            if 'in' in region:
                len_in = len(content_dic['importances'])
                len_in_out_dic['in'] = len_in
                len_in_out_dic['tropical'] = len_in
            if 'out' in region:
                len_out = len(content_dic['importances'])
                len_in_out_dic['out'] = len_out
        # exit()
        # print len_in_out_dic
        x0list = []
        ylist = []
        size_list = []
        colors_list = []
        rX_colors_list = []

        self.__plot_grid(len_in_out_dic)
        for key,result_dic in arr:
            # print key
            in_out = key.split('-')[0]
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
            ############## reference size
            # scatter_size = self.__importances_to_scatter_size(importances,len_in_out)
            ############## absolute size
            # scatter_size = []
            # for imp in importances:
            #     scatter_size.append(self.__importances_to_scatter_size_abs(imp))
            ############## log size
            # scatter_size = []
            # for imp in importances:
            #     scatter_size.append(self.__importances_to_scatter_size_log(imp))
            # print scatter_size
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
        _in = len_in_out_dic['in']
        _out = len_in_out_dic['out']
        tropical = len_in_out_dic['in']
        X = range(_in+_out+tropical+2+1)
        # plot horizental lines
        Y = range(17)
        plt.figure(figsize=(25,5))
        for x in X:
            plt.plot([x] * 2, [0, 17-1],c='black',linewidth=0.4)
        for y in Y:
            plt.plot([0, _in+_out+tropical+2],[y]*2,c='black',linewidth=0.4)

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


    def __importances_to_scatter_size_abs(self,importance):


        minv = 0
        maxv = 0.4
        num = int((maxv-minv)*10+1)
        size_dic = {}
        for i in range(1, num+1):
            size_dic[i] = i * 30
        importance_range_list = np.linspace(minv,maxv,num)
        for i in range(len(importance_range_list)):
            if importance_range_list[i] <= importance < importance_range_list[i+1]:
                size = size_dic[i+1]
                return size
            elif importance > maxv:
                size = size_dic[num]
                return size

        raise IOError



    def __importances_to_scatter_size_log(self,importance):

        return (math.log(importance + 1,1000) + 1) * 40


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


    def __importances_to_colors(self,inlist):
        # TODO: 热点图 用颜色表示importance 大小表示相关性
        pass


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

    def __rX_to_size(self,rX_list):

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
        if conditions == 'in':
            x0 = 0
        elif conditions == 'out':
            x0 = 0 + len_in_out_dic['in'] + 1
        elif conditions == 'tropical':
            x0 = 0 + len_in_out_dic['in'] + len_in_out_dic['out'] + 2
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



class Prepare1:
    def __init__(self):
        self.this_class_arr = this_root_PLOT + 'Random_Forest\\arr\\Prepare\\'
        self.this_class_tif = this_root_PLOT + 'Random_Forest\\tif\\Prepare\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        self.this_root_branch = this_root+'branch2020\\'
        pass


    def run(self):
        # 1.准备因变量 Y
        start = time.time()
        # self.prepare_Y()
        # # self.check_Y()
        # # 2.准备自变量 X的 delta
        # x = ['TMP', 'PRE', 'CCI', 'SWE']
        # MUTIPROCESS(self.prepare_X,x).run()
        # # 3.准备自变量 X 的标准差
        # x = ['PRE_std', 'TMP_std', 'CCI_std', 'SWE_std']
        # MUTIPROCESS(self.prepare_X_std, x).run()
        # # 4.准备自变量 X 的平均值
        # x = ['PRE_mean','TMP_mean','CCI_mean','SWE_mean']
        # MUTIPROCESS(self.prepare_X_mean, x).run()
        # # 5.准备自变量 NDVI 的平均值和delta
        # self.prepare_NDVI()
        # # 6.准备自变量 soil 的值，soil是常量
        # self.prepare_soil()
        # # 7.准备自变量 bio diversity 的值，常量
        # self.prepare_bio_diversity()
        # # 8.准备自变量冬季，out['PRE_winter', 'CCI_winter', 'SWE_winter','NDVI_winter']
        # x = ['PRE_winter', 'CCI_winter', 'SWE_winter', 'TMP_winter']
        # MUTIPROCESS(self.prepare_winter_variables, x).run()
        # # 9.准备pre-drought 变量，in ['PRE_pre','CCI_pre','TMP_pre']
        # x = ['PRE_pre','CCI_pre','TMP_pre']
        # MUTIPROCESS(self.prepare_pre_drought, x).run()
        # # 10. 准备常量 HI
        # self.prepare_HI()
        # 11.['PRE', 'CCI', 'SWE','NDVI'] 为亏损量，应为负值，需要加负号
        ########## 需再要手动覆盖 ###########
        x = [
            'PRE','PRE_pre','PRE_winter',
            'SWE','SWE_winter',
            'CCI','CCI_pre','CCI_winter',
            'NDVI_change'
             ]
        for i in tqdm(x):
            self.minus_X_i(i)
        # end = time.time()
        # print 'duration: %0.2f seconds'%(end-start)
        # -1. check_x
        # self.check_X()
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

    def minus_X_i(self,X):

        fdir = self.this_class_arr
        minus_fdir = fdir+'\\random_forest_minus\\'
        Tools().mk_dir(minus_fdir)
        Y_dic = dict(np.load(fdir + 'Y.npy').item())
        X_dic = dict(np.load(fdir+'{}.npy'.format(X)).item())
        new_X_dic = {}

        for key in Y_dic:
            try:
                x = X_dic[key]
            except:
                continue
            new_X_dic[key] = -x

        np.save(minus_fdir+'{}'.format(X),new_X_dic)



    def prepare_Y(self):
        # config
        out_dir = self.this_class_arr+'\\'
        Tools().mk_dir(out_dir)
        # 1 drought periods
        print '1. loading recovery time'
        f_recovery_time = this_root+'branch2020\\arr\\Recovery_time1\\recovery_time_composite\\composite.npy'
        recovery_time_dic = dict(np.load(f_recovery_time).item())
        print 'done'
        Y = {}
        flag = 0
        for pix in tqdm(recovery_time_dic):
            vals = recovery_time_dic[pix]
            # continue
            for r_time,mark,recovery_date_range,drought_range,eln in vals:
                if r_time == None:  #r_time 为 TRUE
                    continue
                flag += 1
                recovery_start = recovery_date_range[0]
                recovery_end = recovery_date_range[-1]
                key = pix+'~'+mark+'~'+eln+'~'+'{}.{}'.format(recovery_start,recovery_end)
                # print key
                Y[key] = r_time
        # print flag
        # flag=1192218
        # flag=198075
        np.save(out_dir+'Y',Y)


    def check_Y(self):
        print 'loading Y'
        f = self.this_class_arr+'Y.npy'
        # f = r'D:\project05\new_2020\random_forest\Y.npy'
        dic = dict(np.load(f).item())
        # print len(dic)
        pix_dic = DIC_and_TIF().void_spatial_dic()
        flag = 0
        for key in dic:
            flag += 1
            # print key,dic[key]
            pix, mark, eln, date_range, drought_start, recovery_start = self.__split_keys(key)
            # print pix, mark, eln, date_range, drought_start, recovery_start
            # print dic[key]
            pix_dic[pix].append(1)
            # exit()

        spatial_dic = {}
        for pix in pix_dic:
            val = pix_dic[pix]
            if len(val)>0:
                new_val = np.sum(val)
            else:
                new_val = np.nan
            spatial_dic[pix] = new_val
        print flag
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
            recovery_start = int(split_date_range[0])
            recovery_end = int(split_date_range[1])
            # drought_range = range(drought_start, recovery_start)
            vals = all_dic[pix]
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
        sand_tif = self.this_root_branch+'tif\\HWSD\\T_SAND_resample.tif'
        silt_tif = self.this_root_branch+'tif\\HWSD\\T_SILT_resample.tif'
        clay_tif = self.this_root_branch+'tif\\HWSD\\T_CLAY_resample.tif'

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
        bio_tif = self.this_root_branch + 'tif\\Bio_diversity\\bio_diversity_normalized.tif'
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


    def __index_to_year(self,indx):
        year = indx // 12 + 1982
        return year

    def __index_to_mon(self,indx):
        mon = indx % 12 + 1
        return mon


    def prepare_winter_variables(self,x):
        '''
        cv: 变异系数
        std: 标准差
        :return:
        '''

        growing_date_range_f = this_root+'branch2020\\arr\\Winter1\\growing_season_index.npy'
        growing_date_range_dic = dict(np.load(growing_date_range_f).item())
        # x = ['PRE_std','TMP_std','CCI_std','SWE_std']
        product = x.split('_')[0]
        if product == 'SWE':
            per_pix_dir = this_root + 'data\\GLOBSWE\\per_pix\\SWE_max_408\\'
        else:
            per_pix_dir = this_root + 'data\\{}\\per_pix\\'.format(product)

        if product in ['TMP', 'PRE','NDVI']:
            mean_dir = this_root + 'data\\{}\\mon_mean_tif\\'.format(product)
        elif product == 'CCI':
            mean_dir = this_root + 'data\\CCI\\\monthly_mean\\'
        elif product == 'SWE':
            mean_dir = this_root + 'data\\GLOBSWE\\monthly_SWE_max\\'
        else:
            raise IOError('x error')


        out_dir = self.this_class_arr
        Tools().mk_dir(out_dir)
        Y_dic = dict(np.load(self.this_class_arr + 'Y.npy').item())
        all_dic = {}
        for f in tqdm(os.listdir(per_pix_dir), desc='1/2 loading per_pix_dir ...'):
            dic = dict(np.load(per_pix_dir + f).item())
            for pix in dic:
                all_dic[pix] = dic[pix]

        # 2 加载月平均数据

        if product == 'SWE':
            month_range = [1, 2, 3, 4, 5, 10, 11, 12]
        else:
            month_range = range(1, 13)
        mean_dic = {}
        for m in tqdm(month_range, desc='2/3 loading monthly mean ...'):
            m = '%02d' % m
            arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(mean_dir + m + '.tif')
            arr_dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            mean_dic[m] = arr_dic

        # 3 找干旱事件对应的X的std
        X = {}
        for key in tqdm(Y_dic, desc='2/2 generate X dic ...'):
            split_key = key.split('~')
            pix, mark, eln, date_range = split_key
            # print key
            if mark != 'out':
                continue
            growing_date_range = growing_date_range_dic[pix]
            split_date_range = date_range.split('.')
            start = split_date_range[0]
            end = split_date_range[1]
            start = int(start)
            end = int(end)

            drought_range = range(start, end)
            winter_drought_range = []
            for indx in drought_range:
                mon = self.__index_to_mon(indx)
                if not mon in growing_date_range:
                    winter_drought_range.append(indx)
                    if len(winter_drought_range) > 6:
                        break
            # if len
            # print '\n'
            # print drought_range
            # print winter_drought_range
            # print growing_date_range
            # continue

            # print pix,mark,drought_range
            # exit()
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
                if product == 'TMP' or product == 'CCI':
                    juping_mean = np.mean(selected_val)
                else:
                    juping_mean = np.sum(selected_val)
            else:
                juping_mean = np.nan
            X[key] = juping_mean
        np.save(out_dir + '{}'.format(x), X)
        pass

    def prepare_pre_drought(self,x):
        '''
        干旱前期变量状态
        :param x:
        :return:
        '''
        product = x.split('_')[0]
        if product == 'SWE':
            per_pix_dir = this_root + 'data\\GLOBSWE\\per_pix\\SWE_max_408\\'
        else:
            per_pix_dir = this_root + 'data\\{}\\per_pix\\'.format(product)

        if product in ['TMP', 'PRE']:
            mean_dir = this_root + 'data\\{}\\mon_mean_tif\\'.format(product)
        elif product == 'CCI':
            mean_dir = this_root + 'data\\CCI\\\monthly_mean\\'
        elif product == 'SWE':
            mean_dir = this_root + 'data\\GLOBSWE\\monthly_SWE_max\\'
        else:
            raise IOError('x error')

        out_dir = self.this_class_arr
        Tools().mk_dir(out_dir)
        Y_dic = dict(np.load(self.this_class_arr + 'Y.npy').item())
        all_dic = {}
        for f in tqdm(os.listdir(per_pix_dir), desc='1/2 loading per_pix_dir ...'):
            dic = dict(np.load(per_pix_dir + f).item())
            for pix in dic:
                all_dic[pix] = dic[pix]

        # 2 加载月平均数据

        if product == 'SWE':
            month_range = [1, 2, 3, 4, 5, 10, 11, 12]
        else:
            month_range = range(1, 13)
        mean_dic = {}
        for m in tqdm(month_range, desc='2/3 loading monthly mean ...'):
            m = '%02d' % m
            arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(mean_dir + m + '.tif')
            arr_dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            mean_dic[m] = arr_dic

        # 3 找干旱事件对应的X的std
        X = {}
        for key in tqdm(Y_dic, desc='2/2 generate X dic ...'):
            split_key = key.split('~')
            pix, mark, eln, date_range = split_key
            # print key
            if mark == 'out':
                continue
            split_date_range = date_range.split('.')
            start = split_date_range[0]
            end = split_date_range[1]
            start = int(start)
            drought_range = range(start-4, start-1)
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
                if product == 'TMP' or product == 'CCI':
                    juping_mean = np.mean(selected_val)
                else:
                    juping_mean = np.sum(selected_val)
            else:
                juping_mean = np.nan
            X[key] = juping_mean
        np.save(out_dir + '{}'.format(x), X)
        pass



    def check_X(self):
        x = 'NDVI_change'
        # x = 'PRE'
        print 'loading {}'.format(x)
        f = self.this_class_arr + '{}.npy'.format(x)
        # f = r'D:\project05\new_2020\random_forest\Y.npy'
        dic = dict(np.load(f).item())
        # print len(dic)
        pix_dic = DIC_and_TIF().void_spatial_dic()
        flag = 0
        for key in dic:
            # print key
            val = dic[key]
            # print val
            # sleep()
            flag += 1
            # print key,dic[key]
            pix, mark, eln, date_range, drought_start, recovery_start = self.__split_keys(key)
            # print pix, mark, eln, date_range, drought_start, recovery_start
            # print dic[key]
            pix_dic[pix].append(val)
            # exit()

        spatial_dic = {}
        for pix in pix_dic:
            val = pix_dic[pix]
            if len(val) > 0:
                new_val = np.mean(val)
            else:
                new_val = np.nan
            spatial_dic[pix] = new_val
        print flag
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr, cmap='jet', vmin=0.5, vmax=-0.5)
        # plt.imshow(arr, cmap='jet')
        plt.colorbar()
        plt.show()
        pass



    def prepare_HI(self):

        out_dir = self.this_class_arr + '\\'
        Y_dic = dict(np.load(self.this_class_arr + 'Y.npy').item())
        HI_tif = this_root + 'tif\\HI\\HI.tif'
        HI_arr = to_raster.raster2array(HI_tif)[0]
        HI_arr[HI_arr < -9999] = np.nan
        HI_dic = DIC_and_TIF().spatial_arr_to_dic(HI_arr)
        HI_x = {}
        for key in tqdm(Y_dic, desc='1/2 generate X dic ...'):
            split_key = key.split('~')
            pix, mark, eln, date_range = split_key
            bio = HI_dic[pix]
            if np.isnan(bio):
                continue
            HI_x[key] = bio

        np.save(out_dir + 'HI', HI_x)
        pass



def main():
    # 1 plot events numbers
    # plot_events_spatial()
    # 2.1 plot_recovery_time_mix
    # plot_recovery_time_mix()
    # 2.2 plot_recovery_time_out
    # for condition in ['in','out']:
    #     plot_recovery_time_in_and_out(condition)
    # 3 花点图
    # Water_balance().run()
    # 4.1 ratio of winter
    # plot_ratio_of_winter()
    # 4.2 time series
    # plot_time_series()
    # 5 Tenery plot
    # Ternary_plot().plot_scatter()
    # 6 Threshold
    # Find_Threshold().run()
    # 7 Prepare Random Forest
    # Prepare1().run()
    # 7.1 run Random Forest
    # RF().run()
    # 7.2 Plot Random Forest Results
    Plot_RF_train_events_result().run()
    pass


if __name__ == '__main__':
    main()