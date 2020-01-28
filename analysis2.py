# coding=gbk
from analysis import *

class Recovery_time1:

    def __init__(self):
        pass

    def run(self):
        params = [3,'pick_post_growing_season_events']
        self.gen_recovery_time(params)
        # self.composite_per_pix(3)
        pass

    def composite_per_pix(self, interval):
        fdir = this_root+'SPEI\\single_events_24\\SPEI_{}\\'.format(interval)
        composite = {}

        for f in tqdm(os.listdir(fdir),desc='loading per pix events'):
            dic = dict(np.load(fdir+f).item())
            for pix in dic:
                val = dic[pix]
                if len(val) > 0:
                    composite[pix] = val

        return composite


    def early_late_non(self,min_ndvi_indx, growing_date_range):
        print min_ndvi_indx
        print growing_date_range

        pass


    def gen_recovery_time(self,params):
        '''
        生成全球恢复期
        :param interval: SPEI_{interval}
        :return:
        '''
        interval, _ = params
        # pix_lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        growing_season_daterange = dict(np.load(this_root + 'NDVI\\global_growing_season.npy').item())
        interval = '%02d' % interval
        # out_dir = this_root + 'arr\\recovery_time\\{}\\SPEI_{}\\'.format(mode, interval)
        # Tools().mk_dir(out_dir, force=True)
        # 1 加载事件
        # interval = '%02d' % interval
        events = self.composite_per_pix(interval)
        # 2 加载NDVI
        ndvi_dir = this_root + 'NDVI\\per_pix_anomaly\\'
        spei_dir = this_root + 'SPEI\\per_pix\\SPEI_{}\\'.format(interval)
        for f in os.listdir(ndvi_dir):
            if not '020' in f:
                continue
            ndvi_dic = dict(np.load(ndvi_dir + f).item())
            # ndvi_dic = Tools().detrend_dic(ndvi_dic)
            spei_dic = dict(np.load(spei_dir + f).item())
            recovery_time_dic = {}
            for pix in ndvi_dic:
                if pix in events and pix in growing_season_daterange:
                    growing_date_range = growing_season_daterange[pix]
                    ndvi = ndvi_dic[pix]
                    spei = spei_dic[pix]
                    event = events[pix]
                    smooth_window = 3
                    # ndvi = Tools().forward_window_smooth(ndvi, smooth_window)
                    spei = Tools().forward_window_smooth(spei, smooth_window)
                    recovery_time_result = []
                    for date_range in event:
                        ndvi = np.array(ndvi)
                        grid = ndvi < -100
                        ndvi[grid] = np.nan
                        ndvi = Tools().interp_nan(ndvi)

                        spei = np.array(spei)
                        grid_1 = spei < -100
                        spei[grid_1] = np.nan
                        spei = Tools().interp_nan(spei)

                        if len(ndvi) < 300:
                            continue
                        # 1 挑出此次干旱事件的NDVI和SPEI值 （画图需要）
                        spei_picked_vals = Tools().pick_vals_from_1darray(spei, date_range)
                        # 2 挑出此次干旱事件SPEI最低的索引
                        min_spei_indx = Tools().pick_min_indx_from_1darray(spei, date_range)
                        # 3 在此次干旱事件SPEI最低索引的周围搜索NDVI的索引和值
                        # 在当前生长季搜索
                        growing_index, growing_vals = Tools().pick_growing_season_vals(ndvi, min_spei_indx,
                                                                                       growing_date_range)
                        # 无法满足筛选条件 continue
                        if len(growing_index) == 0:
                            continue
                        # 4 搜索恢复期
                        # 4.1 获取growing season NDVI的最小值
                        min_ndvi_indx = Tools().pick_min_indx_from_1darray(ndvi, growing_index)
                        # 4.2 判断NDVI 最低点在early late 或是 non Growing Season
                        self.early_late_non(min_ndvi_indx, growing_date_range)
                        continue
                        exit()
                        # 4.3 搜索恢复到正常情况的时间，recovery_time：恢复期； mark：'in', 'out', 'tropical'
                        recovery_time, mark, recovery_date_range = self.search(ndvi, min_ndvi_indx, growing_date_range)
                        # recovery_time, mark = self.search_non_growing_season(ndvi, min_ndvi_indx)
                        recovery_time_result.append([recovery_time, mark, recovery_date_range, date_range])

                        ################# plot ##################
                        # print recovery_time, mark
                        # print growing_date_range
                        # recovery_date_range = range(min_ndvi_indx, min_ndvi_indx + recovery_time + 1)
                        # recovery_ndvi = Tools().pick_vals_from_1darray(ndvi, recovery_date_range)
                        #
                        # tmp_pre_date_range = []
                        # for i in recovery_date_range:
                        #     tmp_pre_date_range.append(i)
                        # for i in date_range:
                        #     tmp_pre_date_range.append(i)
                        # tmp_pre_date_range = list(set(tmp_pre_date_range))
                        # tmp_pre_date_range.sort()
                        # # pre_picked_vals = Tools().pick_vals_from_1darray(pre, tmp_pre_date_range)
                        # # tmp_picked_vals = Tools().pick_vals_from_1darray(tmp, tmp_pre_date_range)
                        # # if len(swe) == 0:
                        # #     continue
                        # # swe_picked_vals = Tools().pick_vals_from_1darray(swe, tmp_pre_date_range)
                        #
                        # plt.figure(figsize=(8, 6))
                        # # plt.plot(tmp_pre_date_range, pre_picked_vals, '--', c='blue', label='precipitation')
                        # # plt.plot(tmp_pre_date_range, tmp_picked_vals, '--', c='cyan', label='temperature')
                        # # plt.plot(tmp_pre_date_range, swe_picked_vals, '--', c='black', linewidth=2, label='SWE',
                        # #          zorder=99)
                        # plt.plot(recovery_date_range, recovery_ndvi, c='g', linewidth=6, label='Recovery Period')
                        # plt.plot(date_range, spei_picked_vals, c='r', linewidth=6,
                        #          label='SPEI_{} Event'.format(interval))
                        # plt.plot(range(len(ndvi)), ndvi, '--', c='g', zorder=99, label='ndvi')
                        # plt.plot(range(len(spei)), spei, '--', c='r', zorder=99, label='SPEI_{}'.format(interval))
                        # plt.legend()
                        #
                        # minx = 9999
                        # maxx = -9999
                        #
                        # for ii in recovery_date_range:
                        #     if ii > maxx:
                        #         maxx = ii
                        #     if ii < minx:
                        #         minx = ii
                        #
                        # for ii in date_range:
                        #     if ii > maxx:
                        #         maxx = ii
                        #     if ii < minx:
                        #         minx = ii
                        # # print date_range[0]-5,recovery_date_range[-1]+5
                        #
                        # xtick = []
                        # for iii in np.arange(len(ndvi)):
                        #     year = 1982 + iii / 12
                        #     mon = iii % 12 + 1
                        #     mon = '%02d' % mon
                        #     xtick.append('{}.{}'.format(year, mon))
                        # # plt.xticks(range(len(xtick))[::3], xtick[::3], rotation=90)
                        # plt.xticks(range(len(xtick)), xtick, rotation=90)
                        # plt.grid()
                        # plt.xlim(minx - 5, maxx + 5)
                        #
                        # lon, lat, address = Tools().pix_to_address(pix)
                        # plt.title('lon:{} lat:{} address:{}'.format(lon, lat, address))
                        # plt.show()
                        #################plot##################

                    recovery_time_dic[pix] = recovery_time_result
                else:
                    recovery_time_dic[pix] = []
            # np.save(out_dir + f, recovery_time_dic)
        pass



    def search(self, ndvi, min_ndvi_indx, growing_date_range):
        # if ndvi[min_ndvi_indx] >= 0:  # 如果在生长季中，NDVI最小值大于0，则恢复期为0个月
        #     return 0,'in'
        for i in range(len(ndvi)):
            if (min_ndvi_indx + i) >= len(ndvi):  # 到头了
                return None, None, None
            search_ = min_ndvi_indx + i
            search_v = ndvi[search_]
            if search_v >= 0:
                recovery_time = i
                end_mon = search_ % 12 + 1
                recovery_date_range = range(min_ndvi_indx,min_ndvi_indx+i+1)
                if len(growing_date_range) <= 10:  # 存在冬季的地区
                    if end_mon in growing_date_range:  # 在当年内恢复
                        if recovery_time <= 5:
                            return recovery_time, 'in',recovery_date_range  # 在生长季恢复
                        else:
                            return recovery_time,'out',recovery_date_range  # 不在生长季恢复
                    else:
                        continue  # 继续搜索
                else:  # 不存在冬季的地区
                    return recovery_time, 'tropical', recovery_date_range



def main():
    Recovery_time1().run()
    pass

if __name__ == '__main__':
    main()