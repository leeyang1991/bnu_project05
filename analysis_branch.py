# coding=gbk

from analysis import *
import recovery_time_branch
this_root_branch = this_root+'branch2020\\'

class Winter1:
    '''
    20200301 更新
    1、生长季选为6个月，前3个月为Early 后3个月为Late

    old ver:
    主要思想：
    1、计算每个NDVI像素每个月的多年平均值
    2、计算值大于3000的月的个数
    3、如果大于3000的个数大于10，则没有冬季，反之则有冬季
    4、选出冬季date range
    '''
    def __init__(self):
        self.this_class_arr = this_root_branch+'arr\\Winter1\\'
        self.this_class_tif = this_root_branch+'tif\\Winter1\\'
        Tools().mk_dir(self.this_class_arr,force=True)
        Tools().mk_dir(self.this_class_tif,force=True)
        pass


    def run(self):
        # self.cal_monthly_mean()
        # self.count_num()
        self.get_grow_season_index()
        self.check_pix()
        pass

    def cal_monthly_mean(self):

        outdir = this_root+'NDVI\\mon_mean_tif\\'
        Tools().mk_dir(outdir)
        fdir = this_root+'NDVI\\clip_tif\\'
        for m in tqdm(range(1,13)):
            arrs_sum = 0.
            for y in range(1982,2016):
                date = '{}{}'.format(y,'%02d'%m)
                tif = fdir+date+'.tif'
                arr,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(tif)
                arr[arr<-9999] = np.nan
                arrs_sum += arr
            mean_arr = arrs_sum/len(range(1982,2016))
            mean_arr = np.array(mean_arr,dtype=float)
            DIC_and_TIF().arr_to_tif(mean_arr,outdir+'%02d.tif'%m)

    def count_num(self):
        # 计算tropical区域
        fdir = this_root + 'NDVI\\mon_mean_tif\\'
        # pix_lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        arrs = []
        month = range(1,13)
        for m in tqdm(month):
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
                    if val>5000:
                        flag += 1.
                if flag == 12:
                    winter_pix.append('%03d.%03d'%(i,j))
                temp.append(flag)
            winter_count.append(temp)

        np.save(this_root+'NDVI\\tropical_pix',winter_pix)


        ##### show #####
        winter_count = np.array(winter_count)
        winter_count = np.ma.masked_where(winter_count<12,winter_count)
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


    def max_6_vals(self,pix,vals):
        '''
        20200301更新
        选取top 6 NDVI月份
        :param vals:
        :return:
        '''
        vals = list(np.array(vals))

        # 1 从小到大排序，获取索引值
        sort = np.argsort(vals)
        sort = sort[::-1]

        # 2 选取排序排名前6的索引值
        max_6_vals = []
        for i in range(6):
            max_6_index = sort[i]
            rank = i
            max_6_vals.append([vals[max_6_index],max_6_index,rank])

        # 3 选取生长季月份
        selected_months = []
        for i in range(len(max_6_vals)):
            selected_months.append(max_6_vals[i][1])
        selected_months.sort()

        # 3.1 生长季跨年
        if 0 in selected_months and 11 in selected_months:
            # 1 计算selected_months与12月份的差值delta，相差几个月
            bias = []
            for i in selected_months:
                if i < 6:
                    delta = i + 1
                    pass
                else:
                    delta = i - 11
                bias.append(delta)
            # 2 计算中间月份，取floor作为mid_month
            mid_month = np.floor(np.mean(bias)) + 11

            # 3 计算 mid month前3个月，后3个月，结果会大于12
            growing_season = [
                mid_month - 3,
                mid_month - 2,
                mid_month - 1,
                mid_month + 0,
                mid_month + 1,
                mid_month + 2,
            ]
            # 4 调整，结果大于12的月份
            new_growing_season = []
            for i in growing_season:
                if i > 11:
                    new_growing_season.append(int(i-12))
                else:
                    new_growing_season.append(int(i))
            growing_season = new_growing_season

        # 3.2 生长在年内
        else:
            mid_month = int(np.mean(selected_months))
            growing_season = [
                mid_month - 3,
                mid_month - 2,
                mid_month - 1,
                mid_month + 0,
                mid_month + 1,
                mid_month + 2,
                ]

        growing_season = np.array(growing_season) + 1
        new_growing_season = []
        for i in growing_season:
            if i == 0:
                i = 12
            new_growing_season.append(i)
        growing_season = new_growing_season
        ################### plot ###################
        # x, y = pix.split('.')
        # print int(x),int(y)
        ########### pix selection ###########
        # if int(x) in range(80, 85) and int(y) in range(580, 585):
        #     print '\n'
        #     print pix
        #     print selected_months
        #     print mid_month
        #     plt.figure()
        #     plt.subplot(211)
        #     plt.plot(vals)
        #     plt.scatter(range(len(vals)),vals)
        #     for i in range(len(max_6_vals)):
        #         plt.text(max_6_vals[i][1], max_6_vals[i][0], str(max_6_vals[i][2]) + '\n' + '%0.2f' % max_6_vals[i][0])
        #     plt.subplot(212)
        #     DIC_and_TIF().plot_back_ground_arr()
        #     void_dic = DIC_and_TIF().void_spatial_dic_nan()
        #     xlist = []
        #     ylist = []
        #     buffer_ = 5
        #     for i in (np.array(range(buffer_)) - buffer_/2):
        #         xlist.append(int(x) + i)
        #         ylist.append(int(y) + i)
        #     # buffer_pix = []
        #     for i in range(len(xlist)):
        #         for j in range(len(ylist)):
        #             buffer_pix_i = '%03d.%03d'%(xlist[i],ylist[j])
        #             # buffer_pix.append(buffer_pix_i)
        #             # print buffer_pix_i
        #             void_dic[buffer_pix_i] = 1
        #     arr = DIC_and_TIF().pix_dic_to_spatial_arr(void_dic)
        #     plt.imshow(arr,'gray')
        #     plt.show()
        #
        #     print '*******'
        ################### plot ###################
        return growing_season

    def get_grow_season_index(self):
        outdir = this_root_branch + 'arr\\Winter1\\'
        Tools().mk_dir(outdir,force=True)
        tropical_pix = np.load(this_root+'data\\NDVI\\tropical_pix.npy')
        fdir = this_root + 'data\\NDVI\\mon_mean_tif\\'
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
                # if i < 250:
                #     continue
                pix = '%03d.%03d' % (i, j)
                # print pix
                if pix in tropical_pix:
                    winter_dic[pix] = np.array(range(1,13))
                    continue
                vals = []
                for arr in arrs:
                    val = arr[i][j]
                    vals.append(val)
                if vals[0] > -10000:
                    std = np.std(vals)
                    if std == 0:
                        continue
                    growing_season = self.max_6_vals(pix,vals)
                    # print growing_season
                    winter_dic[pix] = growing_season
        np.save(outdir+'growing_season_index',winter_dic)

        pass

    def composite_tropical_growingseason(self):
        growing_season_index = dict(np.load(this_root + 'NDVI\\growing_season_index.npy').item())
        tropical_pix = np.load(this_root + 'NDVI\\tropical_pix.npy')
        pix_dic = {}
        for i in growing_season_index:
            pix_dic[i] = growing_season_index[i]
        for pix in tropical_pix:
            # pix_dic[pix] = 2
            pix_dic[pix] = range(1,13)
        np.save(this_root+'NDVI\\composite_growing_season',pix_dic)


    def check_pix(self):
        growing_season_index = dict(np.load(self.this_class_arr+'growing_season_index.npy').item())
        # growing_season_index = dict(np.load(this_root+'data\\NDVI\\growing_season_index.npy').item())
        pix_dic = {}
        for pix in growing_season_index:
            # length = len(growing_season_index[pix])
            pix_dic[pix] = growing_season_index[pix][2]
            # print growing_season_index[pix]

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(pix_dic)

        plt.imshow(arr)
        plt.colorbar()
        plt.show()

        pass


    def check_composite_growing_season(self):

        f = this_root+'NDVI\\composite_growing_season.npy'
        dic = dict(np.load(f).item())
        spatial_dic = {}

        for pix in dic:
            val = len(dic[pix])
            spatial_dic[pix] = val

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        plt.show()


class Pick1:
    '''
    foreest 前后4年 single event
    '''
    def __init__(self):
        self.this_class_arr = this_root_branch + 'arr\\Pick1\\'
        self.this_class_tif = this_root_branch + 'tif\\Pick1\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        pass

    def run(self):
        # self.do_pick()
        self.plot_spatial_events()
        pass


    def do_pick(self):
        param = []
        for i in range(1,13):
            param.append(i)
        MUTIPROCESS(self.pick,param).run()

        pass


    def pick(self, interval):
        # 前n个月和后n个月无极端干旱事件
        normal_n = 24 # 正常植被24个月的间隔
        forest_n = 4 * 12 # 森林前后4年的间隔
        spei_dir = this_root + 'data\\SPEI\\per_pix_smooth\\' + 'SPEI_{:0>2d}\\'.format(interval)
        out_dir = self.this_class_arr + '\\single_events_no_n\\' + 'SPEI_{:0>2d}\\'.format(interval)
        # 加载landcover
        index_landuse_dic = this_root + 'arr\\landcover_dic.npy'
        dic = dict(np.load(index_landuse_dic).item())

        forest_dic = {}
        for key in dic:
            if key in range(1,6):
                pixs = dic[key]
                for pix in pixs:
                    forest_dic[pix] = 1

        Tools().mk_dir(out_dir, force=True)
        # for f in tqdm(os.listdir(spei_dir), 'file...'):
        for f in os.listdir(spei_dir):
            # if not '005' in f:
            #     continue
            spei_dic = dict(np.load(spei_dir + f).item())
            single_event_dic = {}
            for pix in spei_dic:
                spei = spei_dic[pix]
                spei = Tools().interp_1d(spei, -10)
                if len(spei) == 1 or spei[0] == -999999:
                    single_event_dic[pix] = []
                    continue
                params = [spei, pix]
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
                # 如果像素在forest里，n=4*12，否则n=24
                # if pix in forest_dic:
                #     n = forest_n
                # else:
                #     n = normal_n
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

    def kernel_find_drought_period(self, params):
        # 根据不同干旱程度查找干旱时期
        pdsi = params[0]
        key = params[1]
        drought_month = []
        for i, val in enumerate(pdsi):
            if val < -0.5:# SPEI
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
            # SPEI
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


    def plot_spatial_events(self):
        # fdir = self.this_class_arr + '\\single_events\\'
        fdir = self.this_class_arr + '\\single_events_no_n\\'
        arr_sum = 0.
        void_dic = DIC_and_TIF().void_spatial_dic()

        for folder in tqdm(os.listdir(fdir)):
            folder_dir = fdir + folder + '\\'
            # spatial_dic = {}
            for f in os.listdir(folder_dir):
                dic = dict(np.load(folder_dir + f).item())
                for pix in dic:
                    events = dic[pix]
                    flag = 0
                    for event in events:
                        flag += 1
                    if flag == 0:
                        continue
                    # spatial_dic[pix] = flag
                    void_dic[pix].append(flag)
        # arr_sum[arr_sum==0] = np.nan
        spatial_dic = {}
        for pix in void_dic:
            vals_list = void_dic[pix]
            mean = np.mean(vals_list)
            spatial_dic[pix] = mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # DIC_and_TIF().arr_to_tif(arr,outf)
        plt.imshow(arr, 'jet')
        plt.colorbar()
        plt.show()
        pass

class CWD:
    '''
    CWD = P - PET 气候水分亏缺
    '''
    def __init__(self):
        pass

    def run(self):

        pass

    def cal_cwd(self):


        pass



class Water_balance:

    def __init__(self):
        # self.cross_landuse_WB_recovery_time()
        # self.gen_latitude_zone_arr()
        pass
    def run(self):
        recovery_time_tif1 = recovery_time_branch.Recovery_time1().this_class_tif + 'recovery_time\\early.tif'
        recovery_time_tif2 = recovery_time_branch.Recovery_time1().this_class_tif + 'recovery_time\\late.tif'
        plt.figure(figsize=(8,7))

        # self.cross_landuse_WB_recovery_time(recovery_time_tif1)
        # plt.twinx()
        self.cross_landuse_WB_recovery_time(recovery_time_tif2,color_='r')
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
        # Tools().plot_fit_line(a,b,r,X,Y)
        # plot cmap
        # sns.palplot(cmap)
        # plt.xlim(0,1.6)
        # plt.ylim(0,15)
        # plt.ylim(7,13)
        # plt.ylim(0.5,2.3)
        # plt.ylim(-5,100)
        # plt.show()
        # plt.savefig(this_root+'AI\\Ratio\\'+title+'.pdf')


class Water_balance_3d:


    def __init__(self):
        from mpl_toolkits.mplot3d import Axes3D
        # self.cross_landuse_WB_recovery_time()
        # self.gen_latitude_zone_arr()
        pass
    def run(self):
        recovery_time_tif1 = recovery_time_branch.Recovery_time1().this_class_tif + 'recovery_time\\early.tif'
        recovery_time_tif2 = recovery_time_branch.Recovery_time1().this_class_tif + 'recovery_time\\late.tif'
        # self.cross_landuse_WB_recovery_time(recovery_time_tif1)
        # self.plot_error_bar()
        self.cross_landuse_WB_recovery_time(recovery_time_tif2)
        pass

    def plot_error_bar(self):

        N = 21
        x = np.linspace(0, 10, 11)
        y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1, 9.9, 13.9, 15.1, 12.5]

        # fit a linear curve an estimate its y-values and their error.
        a, b = np.polyfit(x, y, deg=1)
        # a, b,r = Tools().linefit(x,y)
        y_est = a * x + b
        y_err = x.std() * np.sqrt(1 / len(x) +
                                  (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

        fig, ax = plt.subplots()
        ax.plot(x, y_est, '-')
        ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
        ax.plot(x, y, 'o', color='tab:brown')


        # sns.regplot(x,y)


        plt.show()

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
        # 三维花点图


        #########################    # 1、加载恢复期 (底图)    #########################
        pass
        #########################    # 1、加载恢复期 (底图)    #########################

        recovery_time_arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(recovery_time_tif)
        # mask NDVI
        recovery_time_arr = NDVI().mask_arr_with_NDVI(recovery_time_arr)
        # recovery_time_arr = ''
        # plt.imshow(recovery_time_arr)
        # plt.show()
        # 2、生成 HI 分级别 dic
        HI_tif = this_root+'tif\\HI\\HI.tif'
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

        # 5、soil arr
        soil_tif = this_root_branch+'tif\\HWSD\\S_CLAY_resample.tif'
        soil_arr,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(soil_tif)
        soil_arr[soil_arr<0] = np.nan
        plt.imshow(soil_arr)
        plt.colorbar()
        plt.show()
        # soil_dic = DIC_and_TIF().spatial_arr_to_dic(soil_arr)

        # 6、交叉像素

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
                    # 挑z轴
                    recovery_picked_val = Tools().pick_vals_from_2darray(recovery_time_arr,intersect_int)
                    recovery_picked_val[recovery_picked_val<0] = np.nan
                    recovery_mean,yerr = Tools().arr_mean_nan(recovery_picked_val)
                    if recovery_mean == None:
                        continue
                    # 挑y轴
                    soil_picked_val = Tools().pick_vals_from_2darray(soil_arr,intersect_int)
                    soil_picked_mean,zerr = Tools().arr_mean_nan(soil_picked_val)

                    # append
                    scatter_dic[key] = [HI_mean,recovery_mean,soil_picked_mean,xerr,yerr,zerr]
            # print scatter_labels
            # marker = markers[markers_flag]
            # plt.scatter(x,y,c=color_dic[],marker=marker)
            # markers_flag += 1
        X = []
        Y = []
        # Z = []
        # sns.set(color_codes=True) # 背景
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        data = []
        for key in scatter_dic:
            lc,lat = key.split('.')
            # print key
            # lat = int(lat)
            # print lc,lat
            marker = markers_dic[lc]
            # color = cls_color_dic[lat]
            color = color_dic[lat]
            x,z,y,xerr,yerr,zerr = scatter_dic[key]
            # zorder : 图层顺序
            # plt.scatter(x,y,s=80,c='#'+color,marker=marker,edgecolors='black',linewidths=1,zorder=99,label=lc)
            # plt.scatter(x,y,s=80,c='#'+color,marker=marker,edgecolors='black',linewidths`=1,zorder=99)
            ax.scatter(x,y,z,s=80,c=color,marker=marker,edgecolors='black',linewidths=1,zorder=99)
            # print x,y,xerr,yerr
            # plt.errorbar(x,y,xerr=xerr/8.,yerr=yerr/8.,c='gray',zorder=0,alpha=0.5)
            data.append([x,y,z])
            X.append(x)
            Y.append(y)
            # Z.append(z)
        # plt.legend()
        # set fit surface
        X = np.array(X)
        Y = np.array(Y)
        Xi, Yi = np.meshgrid(np.arange(min(X), max(X), 0.1), np.arange(min(Y), max(Y), 0.1))
        XX = Xi.flatten()
        YY = Yi.flatten()
        data = np.array(data)

        order = 2  # 1: linear, 2: quadratic
        if order == 1:
            # best-fit linear plane
            A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
            C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

            # evaluate it on grid
            Z = C[0] * Xi + C[1] * Yi + C[2]

            # or expressed using matrix/vector product
            # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

        elif order == 2:
            # best-fit quadratic curve
            A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
            C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

            # evaluate it on a grid
            Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(Xi.shape)
        else:
            raise IOError('error')
        ax.plot_surface(Xi, Yi, Z, rstride=1, cstride=1, alpha=0.2)
        ax.plot_wireframe(Xi, Yi, Z, rstride=10, cstride=1, alpha=0.4)
        # ax.plot(X, Y, zs=0, zdir='z', label='curve in (x,y)')
        # sns.regplot(X, Y,scatter=False, ax=ax)
        print X
        print Y
        ax.set_xlim(0.1, 1.4)
        ax.set_ylim(10, 35)
        ax.set_zlim(0, 11)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.title(title)
        # plt.show()


        ################## plot animation ##################
        from matplotlib import animation
        def animate(i):
            ax.view_init(elev=10., azim=i)
            return fig,

        anim = animation.FuncAnimation(fig, animate, init_func=None,
                                       frames=360, interval=20, blit=True)
        anim.save('basic_animation_late.html', fps=30, extra_args=['-vcodec', 'libx264'])
        ################## plot animation ##################



class Ternary_plot:

    def __init__(self):
        self.this_class_arr = this_root_branch + 'arr\\Ternary_plot\\'
        self.this_class_tif = this_root_branch + 'tif\\Ternary_plot\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        pass



    def plot_scatter(self):
        # 1 load soil and recovery time data
        # recovery_tif = this_root_branch+'tif\\Recovery_time1\\recovery_time\\mix.tif'
        recovery_tif = this_root_branch+'tif\\Recovery_time1\\recovery_time\\late.tif'
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
        tax.scatter(points,c=c_vals,cmap=cmap,colormap=cmap,s=16,vmin=0,marker='h',vmax=9,colorbar=True,alpha=1,linewidth=0)
        tax.boundary()
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
        tif_CLAY = this_root_branch + '\\tif\\HWSD\\T_CLAY_resample.tif'
        tif_SAND = this_root_branch + '\\tif\\HWSD\\T_SAND_resample.tif'
        tif_SILT = this_root_branch + '\\tif\\HWSD\\T_SILT_resample.tif'

        arr_clay = to_raster.raster2array(tif_CLAY)[0]
        arr_sand = to_raster.raster2array(tif_SAND)[0]
        arr_silt = to_raster.raster2array(tif_SILT)[0]

        arr_sand[arr_sand<0]=np.nan
        arr_silt[arr_silt<0]=np.nan
        arr_clay[arr_clay<0]=np.nan

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



        fig, tax = ternary.figure(scale=100, permutation='120')

        flag = 0
        for zone in tqdm(zones):
            flag+=1
            data = {}
            for i in range(len(arr_sand)):
                for j in range(len(arr_sand[0])):
                    sand = arr_sand[i][j]
                    if np.isnan(sand):
                        continue
                    silt = arr_silt[i][j]
                    clay = arr_clay[i][j]
                    if sand + silt + clay == 100:
                        # print zone
                        if self.soil_texture_condition(sand,silt,clay,zone):
                            data[(sand,silt,clay)] = flag
                        # exit()
            if len(data) == 0:
                continue
            tax.heatmap(data, style="triangular")
        tax.boundary(linewidth=1.5)
        tax.gridlines(color="black", multiple=10)

        tax.ticks(linewidth=1, multiple=10, offset=0.03, clockwise=True)
        fontsize = 12
        tax.left_axis_label("Clay", fontsize=fontsize, offset=0.1)
        tax.right_axis_label("Silt", fontsize=fontsize, offset=0.1)
        tax.bottom_axis_label("Sand", fontsize=fontsize, offset=0.1)
        plt.axis('equal')

        tax.show()
        pass


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


def kernel_smooth_SPEI(params):
    fdir,f,outdir_i = params
    dic = dict(np.load(fdir + f).item())
    smooth_dic = {}
    for key in dic:
        vals = dic[key]
        smooth_vals = SMOOTH().forward_window_smooth(vals)
        smooth_dic[key] = smooth_vals
    np.save(outdir_i + f, smooth_dic)
    pass


def smooth_SPEI():
    outdir = this_root+'data\\SPEI\\per_pix_smooth\\'
    Tools().mk_dir(outdir)
    for interval in range(1,13):
        fdir = this_root + 'data\\SPEI\\per_pix\\SPEI_%02d\\' % interval
        outdir_i = outdir + 'SPEI_%02d\\' % interval
        Tools().mk_dir(outdir_i)
        params = []
        for f in os.listdir(fdir):
            params.append([fdir,f,outdir_i])
        MUTIPROCESS(kernel_smooth_SPEI,params).run(desc=outdir_i)


def main():

    # Pick1().run()
    # Winter1().run()
    # smooth_SPEI()
    # Water_balance().run()
    # Water_balance_3d().run()
    Ternary_plot().plot_classify_soil()
    # Ternary_plot().plot_classify_soil()
    pass

if __name__ == '__main__':
    main()