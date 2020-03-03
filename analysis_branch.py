# coding=gbk

from analysis import *
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

    pass


def main():

    # Pick1().run()
    # Winter1().run()
    # smooth_SPEI()

    pass

if __name__ == '__main__':
    main()