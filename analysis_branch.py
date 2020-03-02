# coding=gbk

from analysis import *


class Winter:
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

        pass


    def run(self):
        # self.cal_monthly_mean()
        # self.count_num()
        self.get_grow_season_index()
        # self.composite_tropical_growingseason()
        # self.check_composite_growing_season()
        # self.check_pix()
        # self.growing_season_one_month_in_advance()
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
                    winter_pix.append('%04d.%04d'%(i,j))
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


    def __get_val_and_pop(self,in_list):
        print in_list
        # print max_indx



        pass

    def max_6_vals(self,vals):
        '''
        20200301更新
        选取top 6 NDVI月份
        :param vals:
        :return:
        '''
        vals = np.array(vals)
        # 从小到大排序，获取索引值
        sort = np.argsort(vals)
        sort_dic = {} # {rank:val}
        for i in range(len(vals)):
            val = vals[i]
            month = i
            rank = sort[i]
            sort_dic[rank] = [val]
        print vals
        print sort
        print sort_dic

        plt.plot(vals)
        plt.show()


    def get_grow_season_index(self):
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
                if i < 150:
                    continue
                pix = '%04d.%04d' % (i, j)
                # print pix
                if pix in tropical_pix:
                    winter_dic[pix] = range(1,13)
                    continue
                vals = []
                for arr in arrs:
                    val = arr[i][j]
                    vals.append(val)
                if vals[0] > -10000:
                    std = np.std(vals)
                    if std == 0:
                        continue
                    growing_season = self.max_6_vals(vals)
                    # print growing_season
                    # plt.plot(vals)
                    # plt.grid()
                    # plt.show()
                    winter_dic[pix] = growing_season
        # np.save(this_root+'NDVI\\growing_season_index',winter_dic)

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

    def growing_season_one_month_in_advance(self):
        # 将生长季提前一个月

        growing_season_f = this_root + 'NDVI\\composite_growing_season.npy'
        growing_season_dic = dict(np.load(growing_season_f).item())
        new_growing_season_dic = {}
        for pix in tqdm(growing_season_dic):
            growing_season = growing_season_dic[pix]
            new_growing_season = Tools().growing_season_index_one_month_in_advance(growing_season)
            new_growing_season_dic[pix] = new_growing_season
        np.save(this_root + 'NDVI\\composite_growing_season_one_month_in_advance.npy',new_growing_season_dic)
        pass

    def check_pix(self):
        growing_season_index = dict(np.load(this_root+'NDVI\\growing_season_index.npy').item())
        tropical_pix = np.load(this_root+'NDVI\\tropical_pix.npy')
        pix_dic = {}
        for pix in tropical_pix:
            pix_dic[pix] = 2

        for pix in growing_season_index:
            pix_dic[pix] = 1
            # print growing_season_index[pix]

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(pix_dic)

        plt.imshow(arr)
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


class Pick_new:
    '''
    foreest 前后4年 single event
    '''
    def __init__(self):

        pass

    def run(self):

        pass

    def pick(self, interval, mode='SPEI'):
        # 前n个月和后n个月无极端干旱事件
        n = 24
        spei_dir = this_root + mode + '\\per_pix\\' + 'SPEI_{:0>2d}\\'.format(interval)
        # spei_dir = this_root+'PDSI\\per_pix\\'
        out_dir = this_root + mode + '\\single_events_{}\\'.format(n) + 'SPEI_{:0>2d}\\'.format(interval)
        # out_dir = this_root+'PDSI\\single_events\\'
        Tools().mk_dir(out_dir, force=True)
        for f in tqdm(os.listdir(spei_dir), 'file...'):
            # if not '005' in f:
            #     continue
            spei_dic = dict(np.load(spei_dir + f).item())
            single_event_dic = {}
            for pix in tqdm(spei_dic, f):
                spei = spei_dic[pix]
                spei = Tools().interp_1d(spei, -10)
                if len(spei) == 1 or spei[0] == -999999:
                    single_event_dic[pix] = []
                    continue
                spei = Tools().forward_window_smooth(spei, 3)
                params = [spei, pix, mode]
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
                # single_event_dic[pix] = events_4
                # print events_4
                # # # # # # # # # # # # # # # # # # # # # # #

                # # # # # # # # # # # # # # # # # # # # # # #
                # # 筛选单次事件（前后n个月无干旱事件）
                single_event = []
                for i in range(len(events_4)):
                    if i - 1 < 0:  # 首次事件
                        if events_4[i][0] - n < 0 or events_4[i][-1] + n >= len(spei):  # 触及两边则忽略
                            continue
                        if len(events_4) == 1:
                            single_event.append(events_4[i])
                        elif events_4[i][-1] + n <= events_4[i + 1][0]:
                            single_event.append(events_4[i])
                        continue

                    # 最后一次事件
                    if i + 1 >= len(events_4):
                        if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= len(spei):
                            single_event.append(events_4[i])
                        break

                    # 中间事件
                    if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= events_4[i + 1][0]:
                        single_event.append(events_4[i])
                single_event_dic[pix] = single_event
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



def main():

    # Pick_new().run()
    Winter().run()


if __name__ == '__main__':
    main()