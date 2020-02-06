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
        x = ['TMP', 'PRE', 'CCI', 'SWE']
        for i in x:
            self.prepare_X(i)
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
            for r_time,mark,date_range,drought_range,eln in vals:
                if r_time == None:  #r_time 为 TRUE
                    continue
                flag += 1
                start = date_range[0]
                end = start + r_time
                key = pix+'~'+mark+'~'+eln+'~'+'{}.{}'.format(start,end)
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
        Y_dic = dict(np.load(this_root + 'new_2020\\random_forest\\Y.npy').item())
        per_pix_dir = this_root + 'NDVI\\per_pix_anomaly\\'
        # 1 加载所有原始数据
        all_dic = {}
        for f in tqdm(os.listdir(per_pix_dir), desc='1/3 loading per_pix_dir ...'):
            dic = dict(np.load(per_pix_dir + f).item())
            for pix in dic:
                all_dic[pix] = dic[pix]
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

        pass


def main():

    Prepare().run()
    pass


if __name__ == '__main__':
    main()