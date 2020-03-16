# coding=utf-8
from analysis import *

this_root_branch = this_root+'branch2020\\'


class Recovery_time1:
    # method 1
    def __init__(self):

        self.this_class_arr = this_root_branch + 'arr\\Recovery_time1\\'
        self.this_class_tif = this_root_branch + 'tif\\Recovery_time1\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)

        self.valid_pix()
        pass

    def run(self):
        # calculate func
        # params = range(1,13)
        # MUTIPROCESS(self.gen_recovery_time,params).run(process=6)
        # self.compose_recovery_time()
        pass


    def run1(self):
        # plot func
        # self.check()
        # self.plot_early_late_pdf()
        # self.recovery_early_late_tif()
        # self.recovery_mix_tif()
        # self.recovery_early_late_in_out()
        self.ratio()

        pass

    def valid_pix(self):
        self.ndvi_valid_pix = NDVI().filter_NDVI_valid_pix()
        self.tropical_pix = np.load(this_root + 'data\\NDVI\\tropical_pix.npy')


    def composite_per_pix(self, interval):
        fdir = this_root_branch+'arr\\Pick1\\single_events\\SPEI_{}\\'.format(interval)
        composite = {}

        # for f in tqdm(os.listdir(fdir),desc='loading per pix events'):
        for f in os.listdir(fdir):
            dic = dict(np.load(fdir+f).item())
            for pix in dic:
                val = dic[pix]
                if len(val) > 0:
                    composite[pix] = val

        return composite


    def early_late_non(self,input_ind, growing_date_range):
        # print input_ind
        # print growing_date_range
        mon = input_ind % 12 + 1
        growing_date_range = list(growing_date_range)
        if len(growing_date_range) < 10:
            ind = growing_date_range.index(mon)
            if ind in [0,1,2]:
                return 'early'
            elif ind in [3,4,5]:
                return 'late'
            else:
                raise IOError('index error')
        else:
            ind = growing_date_range.index(mon)
            mark = 'tropical'
            return mark


    def pick_growing_season_vals(self, arr, spei_min_index, growing_season_range):
        # 获取当前生长季的index
        year = int(spei_min_index / 12)
        select_vals = []
        select_index = []
        for i in range(0, 12):
            s_index = spei_min_index + i
            s_mon = s_index % 12 + 1
            s_year = int(s_index / 12)
            if (s_year == year or s_year == (year + 1)) and s_mon in growing_season_range:
                select_index.append(spei_min_index + i)
                select_vals.append(arr[spei_min_index + i])

            # s_mon = mon + i
            # if s_mon in growing_season_range:
            #     select_index.append(spei_min_index+i)
            #     select_vals.append(arr[spei_min_index+i])
        return select_index, select_vals

    def gen_recovery_time(self,params):
        '''
        生成全球恢复期
        :param interval: SPEI_{interval}
        :return:
        '''
        interval = params
        # pix_lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        growing_season_daterange = dict(np.load(this_root_branch + 'arr\\Winter1\\growing_season_index.npy').item())


        interval = '%02d' % interval
        out_dir = self.this_class_arr + '\\recovery_time\\SPEI_{}\\'.format(interval)
        Tools().mk_dir(out_dir, force=True)
        # 1 加载事件
        # interval = '%02d' % interval
        events = self.composite_per_pix(interval)
        # 2 加载NDVI 与 SPEI
        ndvi_dir = this_root + 'data\\NDVI\\per_pix_anomaly_smooth\\'
        spei_dir = this_root + 'data\\SPEI\\per_pix_smooth\\SPEI_{}\\'.format(interval)
        # for f in tqdm(os.listdir(ndvi_dir)):
        for f in os.listdir(ndvi_dir):
            # if not '015' in f:
            #     continue
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
                        growing_index, growing_vals = self.pick_growing_season_vals(ndvi, min_spei_indx,
                                                                                       growing_date_range)
                        # print 'min_spei_indx',min_spei_indx,1982+min_spei_indx//12,min_spei_indx%12+1,growing_date_range
                        # print growing_index,1982+growing_index[0]//12,growing_index[0]%12+1,growing_vals
                        # print growing_index,1982+growing_index[-1]//12,growing_index[-1]%12+1,growing_vals
                        # 无法满足筛选条件 continue
                        if len(growing_index) == 0:
                            continue
                        # 4 搜索恢复期
                        # 4.1 获取growing season NDVI的最小值
                        min_ndvi_indx = Tools().pick_min_indx_from_1darray(ndvi, growing_index)
                        # 4.2 判断NDVI 最低点在early late 或是 non Growing Season
                        # mark
                        # eln: Early Late Non
                        # print 'min_ndvi_indx',min_ndvi_indx,1982+min_ndvi_indx//12,min_ndvi_indx%12+1
                        eln = self.early_late_non(min_ndvi_indx, growing_date_range)
                        # eln = ''
                        # 4.3 搜索恢复到正常情况的时间，recovery_time：恢复期； mark：'in', 'out', 'tropical'
                        # mark: In Out Tropical
                        recovery_time, mark, recovery_date_range = self.search(ndvi, min_ndvi_indx, growing_date_range)
                        # recovery_time, mark = self.search_non_growing_season(ndvi, min_ndvi_indx)
                        recovery_time_result.append([recovery_time, mark, recovery_date_range, date_range,eln])

                        ################# plot ##################
                        # print recovery_time, mark,eln
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
            np.save(out_dir + f, recovery_time_dic)
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
                recovery_date_range = list(range(min_ndvi_indx,min_ndvi_indx+i+1))
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

    def compose_recovery_time(self):
        '''
        合成SPEI 1 - 24 的recovery time
        :return:
        '''
        fdir = self.this_class_arr + 'recovery_time\\'
        out_dir = self.this_class_arr + '\\recovery_time_composite\\'
        Tools().mk_dir(out_dir)
        void_dic = DIC_and_TIF().void_spatial_dic()
        for folder in tqdm(os.listdir(fdir)):
            for f in os.listdir(fdir + folder):
                dic = dict(np.load(fdir + folder + '\\' + f).item())
                for pix in dic:
                    recovery_events = dic[pix]
                    for event in recovery_events:
                        void_dic[pix].append(event)
        # print '\nsaving...'
        np.save(out_dir + 'composite', void_dic)
        # exit()
        pass


    def check(self):
        print('loading...')
        f = self.this_class_arr+'recovery_time_composite\\composite.npy'
        dic = dict(np.load(f).item())
        for pix in dic:
            print(pix,dic[pix])



    def recovery_mix_tif(self):

        print('loading...')
        out_dir = self.this_class_tif + 'recovery_time\\'
        out_f = out_dir + 'mix.tif'
        Tools().mk_dir(out_dir, force=1)
        f = self.this_class_arr + 'recovery_time_composite\\composite.npy'
        dic = dict(np.load(f).item())
        recovery_time_dic = {}
        for pix in dic:
            events = dic[pix]
            recovery = []
            if len(events) == 0:
                continue
            for event in events:
                recovery_time, mark, recovery_date_range, date_range, eln = event
                if recovery_time == None:
                    continue
                recovery.append(recovery_time)

            if len(recovery) > 0:
                mean_recovery = np.mean(recovery)
            else:
                mean_recovery = np.nan
            recovery_time_dic[pix] = mean_recovery
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(recovery_time_dic)
        arr = NDVI().mask_arr_with_NDVI(arr)
        DIC_and_TIF().arr_to_tif(arr,out_f)

    def recovery_early_late_tif(self):
        print('loading...')
        out_dir = self.this_class_tif+'recovery_time\\'
        Tools().mk_dir(out_dir,force=1)
        f = self.this_class_arr + 'recovery_time_composite\\composite.npy'
        dic = dict(np.load(f).item())
        recovery_time_early_dic = {}
        recovery_time_late_dic = {}
        for pix in dic:
            events = dic[pix]
            recovery_early = []
            recovery_late = []
            if len(events) == 0:
                continue
            for event in events:
                recovery_time, mark, recovery_date_range, date_range, eln = event
                if recovery_time == None:
                    continue
                # print pix, recovery_time, mark, recovery_date_range, date_range,eln
                if eln == 'early' or eln == 'tropical':
                    # recovery_time_early[pix]
                    recovery_early.append(recovery_time)
                if eln == 'late' or eln == 'tropical':
                    recovery_late.append(recovery_time)
                # exit()
            # print recovery_early
            # exit()
            if len(recovery_early) > 0:
                mean_recovery_early = np.mean(recovery_early)
            else:
                mean_recovery_early = np.nan
            if len(recovery_late) > 0:
                mean_recovery_late = np.mean(recovery_late)
            else:
                mean_recovery_late = np.nan
            recovery_time_early_dic[pix] = mean_recovery_early
            recovery_time_late_dic[pix] = mean_recovery_late

        early = DIC_and_TIF().pix_dic_to_spatial_arr(recovery_time_early_dic)
        early = NDVI().mask_arr_with_NDVI(early)
        late = DIC_and_TIF().pix_dic_to_spatial_arr(recovery_time_late_dic)
        late = NDVI().mask_arr_with_NDVI(late)


        DIC_and_TIF().arr_to_tif(early,out_dir+'early.tif')
        DIC_and_TIF().arr_to_tif(late,out_dir+'late.tif')

        # plt.imshow(arr,'jet',vmin=0,vmax=18)
        # plt.colorbar()
        # plt.show()

    def recovery_early_late_in_out(self):
        outdir = self.this_class_tif+'\\recovery_time_in_out\\'
        Tools().mk_dir(outdir)
        f = self.this_class_arr+'recovery_time_composite\\composite.npy'
        dic = dict(np.load(f).item())

        early_in_dic = {}
        early_out_dic = {}
        late_in_dic = {}
        late_out_dic = {}
        tropical_pix = self.tropical_pix
        ndvi_valid_pix = self.ndvi_valid_pix
        for pix in tqdm(dic):
            if pix in tropical_pix:
                continue
            if pix not in ndvi_valid_pix:
                continue
            events = dic[pix]
            early_in = []
            early_out = []
            late_in = []
            late_out = []
            for event in events:
                recovery_time, mark, recovery_date_range, date_range, eln = event
                # print recovery_time, mark, recovery_date_range, date_range, eln
                if mark == 'in' and 'early' in eln:
                    early_in.append(recovery_time)
                if mark == 'out' and 'early' in eln:
                    early_out.append(recovery_time)
                if mark == 'in'  and 'late' in eln:
                    late_in.append(recovery_time)
                if mark == 'out' and 'late' in eln:
                    late_out.append(recovery_time)
            early_in_mean = np.mean(early_in)
            early_out_mean = np.mean(early_out)
            late_in_mean = np.mean(late_in)
            late_out_mean = np.mean(late_out)

            early_in_dic[pix] = early_in_mean
            early_out_dic[pix] = early_out_mean
            late_in_dic[pix] = late_in_mean
            late_out_dic[pix] = late_out_mean

        early_in_arr = DIC_and_TIF().pix_dic_to_spatial_arr(early_in_dic)
        early_out_arr = DIC_and_TIF().pix_dic_to_spatial_arr(early_out_dic)
        late_in_arr = DIC_and_TIF().pix_dic_to_spatial_arr(late_in_dic)
        late_out_arr = DIC_and_TIF().pix_dic_to_spatial_arr(late_out_dic)

        DIC_and_TIF().arr_to_tif(early_in_arr,outdir+'early_in_arr.tif')
        DIC_and_TIF().arr_to_tif(early_out_arr,outdir+'early_out_arr.tif')
        DIC_and_TIF().arr_to_tif(late_in_arr,outdir+'late_in_arr.tif')
        DIC_and_TIF().arr_to_tif(late_out_arr,outdir+'late_out_arr.tif')



    def ratio(self):
        outdir = self.this_class_tif + '\\ratio\\'
        Tools().mk_dir(outdir)
        f = self.this_class_arr + '\\recovery_time_composite\\composite.npy'
        dic = dict(np.load(f).item())

        tropical_pix = self.tropical_pix
        ndvi_valid_pix = self.ndvi_valid_pix

        early_ratio_dic = {}
        late_ratio_dic = {}
        for pix in tqdm(dic):
            if pix in tropical_pix:
                continue
            if pix not in ndvi_valid_pix:
                continue
            events = dic[pix]
            early_flag = 0.
            late_flag = 0.
            flag = 0.

            for event in events:
                recovery_time, mark, recovery_date_range, date_range, eln = event
                if mark != 'out':
                    continue
                flag += 1.

                # print recovery_time, mark, recovery_date_range, date_range, eln
                if 'early' in eln:
                    early_flag += 1.
                if 'late' in eln:
                    late_flag += 1.
            if flag == 0:
                continue
            early_ratio = early_flag/flag
            late_ratio = late_flag/flag
            early_ratio_dic[pix] = early_ratio
            late_ratio_dic[pix] = late_ratio
        # arr = DIC_and_TIF
        DIC_and_TIF().pix_dic_to_tif(early_ratio_dic,outdir+'early_ratio.tif')
        DIC_and_TIF().pix_dic_to_tif(late_ratio_dic,outdir+'late_ratio.tif')



    def plot_early_late_pdf(self):
        fdir = this_root+'new_2020\\tif\\recovery_time\\'
        early = fdir+'early.tif'
        late = fdir+'late.tif'

        early_arr,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(early)
        late_arr,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(late)
        early_pdf = []
        late_pdf = []
        tropical_pix = self.tropical_pix
        for i in tqdm(list(range(len(early_arr)))):
            for j in range(len(early_arr[0])):
                pix = '%03d.%03d'%(i,j)
                # print pix
                if pix in tropical_pix:
                    continue
                early_val = early_arr[i][j]
                late_val = late_arr[i][j]
                if early_val > 0 and early_val < 18:
                    early_pdf.append(early_val)
                if late_val > 0 and late_val < 18:
                    late_pdf.append(late_val)

        plt.hist(early_pdf,bins=20)
        plt.figure()
        plt.hist(late_pdf,bins=20)
        plt.show()




def main():

    Recovery_time1().run1()

    pass



if __name__ == '__main__':
    main()

