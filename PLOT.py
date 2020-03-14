# coding=gbk

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
    Find_Threshold().run()
    pass


if __name__ == '__main__':
    main()