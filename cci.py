# coding=utf-8
import os
import numpy as np
from tqdm import tqdm
import analysis
# import time
from matplotlib import pyplot as plt
# import clip
# from multiprocessing import Process
import multiprocessing as mp
# import psutil
# import threading
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import interpolate
from netCDF4 import Dataset
from osgeo import gdalconst
import datetime
import osr, ogr
import gdal
import time



this_root = r'D:\project05\\'

def mk_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def read_nc(nc):
    # print(nc)
    # exit()
    ncin = Dataset(nc, 'r')
    # print(ncin)
    lat = ncin['lat']
    lon = ncin['lon']
    longitude_grid_distance = abs((lon[-1] - lon[0]) / (len(lon) - 1))
    latitude_grid_distance = -abs((lat[-1] - lat[0]) / (len(lat) - 1))
    longitude_start = lon[0]
    latitude_start = lat[0]

    start = datetime.datetime(1970, 01, 01)
    time = ncin.variables['time']
    date = start + datetime.timedelta(days=int(time[0]))
    array = ncin['sm'][0][::]
    array = np.array(array)
    return date,array


def compose_month(f_dir,out_dir):
    # f_dir = this_root + 'CCI\\1997\\'
    mk_dir(out_dir)
    f_list = os.listdir(f_dir)

    for m in range(1,13):
        m = '%02d' % m
        output = out_dir+'\\month_compose_'+m+'.npy'
        if os.path.isfile(output):
            continue

        array_stack = []
        for f in f_list:
            # print f
            date_f = f.split('-')[-2]
            mon_f = date_f[4:6]
            # print(date_f)
            # print(mon_f)
            # print(m)
            # exit()
            if not mon_f == m:
                continue
            # print('stacking',f)
            date, array = read_nc(f_dir + f)
            array_stack.append(array)

        month_compose = []
        # for i in tqdm(range(len(array_stack[0])),desc='composing_{}'.format(m)):
        for i in range(len(array_stack[0])):
            # print m,i,'/',len(array_stack[0])
            temp = []
            for j in range(len(array_stack[0][0])):
                month = []
                for day in range(len(array_stack)):
                    month.append(array_stack[day][i][j])

                flag = 0.
                sum = 0.
                for d in range(len(month)):
                    if month[d]>-1:
                        flag += 1
                        sum += month[d]
                if sum>0:
                    temp.append(sum/flag)
                else:
                    temp.append(-999999)
            month_compose.append(temp)
        np.save(out_dir + '\\month_compose_'+m,month_compose)


def rasterize_shp(shp, output ,x_min, y_min , x_size, y_size, x_res, y_res):
    # tif = 'D:\\MODIS\\data_tif\\MOD11A2.006\\2003.01.09.LST_Day_1km.tif'
    # shp = 'D:\\MODIS\\shp\\china_dissolve.shp'
    # output = 'D:\\MODIS\\my.tif'
    # x_min, y_min, x_size, y_size, x_res, y_res = -179.875, -89.875, 1440, 720, 0.25, 0.25
    mb_v = ogr.Open(shp)
    mb_l = mb_v.GetLayer()

    target_ds = gdal.GetDriverByName('GTiff').Create(output, x_size, y_size, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, x_res, 0, y_min, 0, y_res))
    band = target_ds.GetRasterBand(1)
    NoData_value = -999999
    band.SetNoDataValue(NoData_value)
    band.FlushCache()
    gdal.RasterizeLayer(target_ds, [1], mb_l)
    # gdal.RasterizeLayer(target_ds, [1], mb_l, options=["ATTRIBUTE=hedgerow"])

    target_ds = None


def tif_to_array(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
        return
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
    im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    im_proj = dataset.GetProjection()  # 获取投影信息
    array = np.array(im_data)
    # np.save('D:\\MODIS\\conf\\china',array)
    # print(np.shape(array))
    # plt.imshow(array)
    # plt.show()
    print('im_width','im_height','im_geotrans','im_proj')
    return array,im_width,im_height,im_geotrans,im_proj


def gen_mask_array(rasterized_tif,out_arr):
    # rasterized_tif = this_root+'/conf/my.tif'
    array = tif_to_array(rasterized_tif)[0]
    new_array = []
    k=0
    for i in array:
        k+=1
        if k%100 == 0:
            print((k,'/',len(array)))
        # print(k,'/',len(array))
        temp = []
        for j in i:
            if j == 0:
                temp.append(False)
            else:
                temp.append(True)
        new_array.append(temp)
    new_array = np.array(new_array)
    np.save(out_arr,new_array)


def clip(input_arr,clip_array,out_array):
    # tif = 'D:\\MODIS\\data_tif\\MOD11A2.006\\2003.01.01.LST_Day_1km.tif'
    # array = np.load('D:\\MODIS\\conf\\china_T_F.npy')
    input_arr[np.logical_not(clip_array)] = -1
    np.save(out_array,input_arr)



def do_clip():
    fdir = 'D:\\project04\\CCI\\composed\\'
    clipped_dir = 'D:\\project04\\CCI\\midasia_clipped\\'
    mk_dir(clipped_dir)
    flist = os.listdir(fdir)
    for y in flist:
        mk_dir(clipped_dir+y)
        if y != '1982':
            continue
        for f in os.listdir(fdir+y):
            print(fdir+y+'\\'+f)
            arr = np.load(fdir+y+'\\'+f)
            clip_arr = np.load('midasia.npy')[::-1]
            out_arr = clipped_dir+y+'\\'+f
            # print(arr, clip_arr, out_arr)
            # exit()
            clip(arr, clip_arr, out_arr)




def gen_lon_lat_dic(rasterized_tif,clip_array,save_array):
    '''
    生成栅格和经纬度对应的字典
    数据格式:
    dic[str(x.y)] = [lon,lat]
    :return:
    '''
    # rasterized_tif = this_root + '/conf/my.tif'

    # clip_array = np.load(this_root+'\\conf\\china_T_F_cci.npy')
    # save_array = 'lon_lat_dic_cci'

    array, im_width, im_height, im_geotrans, im_proj = tif_to_array(rasterized_tif)
    clip_array = clip_array[::-1]
    print(im_geotrans)
    # print(clip_array)
    # plt.imshow(clip_array)
    # plt.show()
    # print(len(clip_array))
    # exit()
    lon_lat_dic = {}
    for y in range(len(clip_array)):
        print(y)
        for x in range(len(clip_array[y])):
            if clip_array[y][x]:
                lon_start = im_geotrans[0]
                lon_step = im_geotrans[1]
                lon = lon_start + lon_step * x

                lat_start = im_geotrans[3]
                lat_step = im_geotrans[5]
                lat = -lat_start - lat_step * y

                lon_lat_dic[str(x)+'.'+str(y)] = [lon,lat]
    # exit()
    np.save(save_array,lon_lat_dic)




def transform_data_npz(lon_lat_dic,cliped_dir,save_path):
    start = time.time()
    # cliped_dir = this_root + '\\clipped\\MOD11A2.006\\'
    # data_transform_folder = this_root + '\\data_transform\\'
    # data_transform_split_folder = this_root + '\\data_transform\\split\\'
    mk_dir(save_path)
    cliped_list = os.listdir(cliped_dir)
    flag1 = 0
    valid_year = []
    for y in range(2003,2017):
        valid_year.append(str(y))

    flatten_dic = {}

    for year in cliped_list:
        if not year in valid_year:
            continue
        print(year)
        for f in os.listdir(cliped_dir+'\\'+year):
            flag1 += 1

            # start = time.time()
            npy = np.load(cliped_dir+'\\'+year+'\\' + f)
            # plt.imshow(npy)
            # plt.show()
            flag = 0
            # dic_key_index = []
            for pix in lon_lat_dic:
                # print(pix)
                # exit()
                flag += 1.
                x = int(pix.split('.')[0])
                y = int(pix.split('.')[1])
                val = npy[y][x]
                dic_key = str(lon_lat_dic[pix][0])+'_'+str(lon_lat_dic[pix][1])

                if dic_key in flatten_dic:
                    flatten_dic[dic_key].append(val)
                    pass
                else:
                    flatten_dic[dic_key] = [val]
                    pass
            # if flag == 100000:
            #     break
        # np.save(data_transform_split_folder+f,dic_key_index)

    # return flatten_dic
    print 'saving npz...'
    np.savez(save_path,**flatten_dic)
    end = time.time()
    print(save_path, end - start, 's')



def interp_1d(val):
    # 1、插缺失值
    x = []
    val_new = []
    for i in range(len(val)):
        if 0 < val[i] < 1:
            index = i
            x = np.append(x,index)
            val_new = np.append(val_new,val[i])

    if len(x)>3:
        interp = interpolate.interp1d(x, val_new, kind='nearest', fill_value="extrapolate")

        xi = range(len(val))
        yi = interp(xi)


        # 2、利用三倍sigma，去除离群值
        # print(len(yi))
        val_mean = np.mean(yi)
        sigma = np.std(yi)
        n = 3
        yi[(val_mean - n * sigma) > yi] = -999999
        yi[(val_mean + n * sigma) < yi] = 999999
        bottom = val_mean - n * sigma
        top = val_mean + n * sigma
        # plt.scatter(range(len(yi)),yi)
        # print(len(yi),123)
        # plt.scatter(range(len(yi)),yi)
        # plt.plot(yi)
        # plt.show()
        # print(len(yi))

        # 3、插离群值
        xii = []
        val_new_ii = []

        for i in range(len(yi)):
            if -999999 < yi[i] < 999999:
                index = i
                xii = np.append(xii, index)
                val_new_ii = np.append(val_new_ii, yi[i])

        interp_1 = interpolate.interp1d(xii, val_new_ii, kind='nearest', fill_value="extrapolate")

        xiii = range(len(val))
        yiii = interp_1(xiii)


        # for i in range(len(yi)):
        #     if yi[i] == -999999:
        #         val_new_ii = np.append(val_new_ii, bottom)
        #     elif yi[i] == 999999:
        #         val_new_ii = np.append(val_new_ii, top)
        #     else:
        #         val_new_ii = np.append(val_new_ii, yi[i])

        return yiii

    else:
        return []



def loop(val):
    # val = insert_data(val)
    # zero_num = np.count_nonzero(0 < val)
    # total_num = len(val)
    # percent = float(zero_num) / total_num * 100
    # print(zero_num)
    # print(total_num)
    # print(percent)
    # exit()
    # if percent > 50:
    #     array = []
    # else:
    array = interp_1d(val)
    # plt.figure()
    # plt.plot(array)
    # plt.show()
    return array
    # if len(array) > 0:
    #     # plt.plot(range(len(array)), array)
    #     array_month = compose_month_data(array)
    #     return array_month

def time_interp(npz_dir,save_dir):
    # npz_dir = this_root+'data_transform\\split_lst\\'
    npz_list = os.listdir(npz_dir)
    # save_dir = this_root + 'monthly_data_npz\\'
    mk_dir(save_dir)

    all_interp_dic = {}
    flag = 0
    for f in npz_list:
        if os.path.isfile(save_dir + f):
            print(save_dir + f + ' is already existed')
            continue
        flag += 1
        print(flag)
        # if flag == 3:
        #     break
        print(save_dir + f)
        # pool = mp.Pool(processes=1)
        npz = np.load(npz_dir+f)
        per = []
        flag = 0
        start = time.time()
        interp_dic = {}

        # flag1 = 0
        # muti_res = {}
        for i in npz:
            flag += 1
            if flag % 1000 == 0:
                print float(flag)/len(npz)*100,'%'
            # if flag == 1000:
            #     break
            val = npz[i]
            # plt.plot(val)
            # plt.show()
            res = loop(val)
            # res = pool.apply_async(loop,(val,))
            # muti_res[i]=res
            interp_dic[i] = res

        # print('fetching results')
        # for i in muti_res:
        #     array = muti_res[i].get()
        #     interp_dic[i] = array
            # break
        # pool.close()
        # pool.join()
        end = time.time()
        print(end - start, 's')
        print('saving results')
        np.savez(this_root+'CCI\\time_interp',**interp_dic)
        # all_interp_dic[f] = interp_dic
        # return interp_dic,f
    # return all_interp_dic



def save_all_interp_dic(dic,f):
    start = time.time()
    save_dir = this_root + 'monthly_data_npz\\'
    mk_dir(save_dir)
    save_path = save_dir+f
    np.savez(save_path,**dic)
    end = time.time()
    print('save '+save_path+' success, time '+str(end-start)+' s')
    pass


def concurrent_save_all_interp_dic():
    all_interp_dic = time_interp()
    pool = mp.Pool()
    for ind in all_interp_dic:
        dic = all_interp_dic[ind]
        pool.apply_async(save_all_interp_dic, (dic,ind,))
    pool.close()
    pool.join()


def extract_cci_from_sta():
    this_root = 'e:\\MODIS\\'
    npz = np.load(this_root+'CCI\\time_interp.npz')
    sta_pos_dic = np.load(this_root + 'ANN_input_para\\00_PDSI\\sta_pos_dic.npz')
    date_list = []
    for year in range(2003,2017):
        for mon in range(1,13):
            date_list.append(str(year)+'%02d'%mon)

    CCI_sta_extract_dic = {}
    import log_process
    time_init = time.time()
    flag = 0
    for npy in npz:
        time_start = time.time()
        npy_split = npy.split('_')
        lon = float(npy_split[0])
        lat = float(npy_split[1])
        vals = npz[npy]
        if len(vals) < 10:
            continue
        for sta in sta_pos_dic:

            pos = sta_pos_dic[sta]

            lon_sta = pos[1]
            lat_sta = pos[0]
            lon_max = lon_sta+0.25
            lon_min = lon_sta-0.25
            lat_max = lat_sta+0.25
            lat_min = lat_sta-0.25

            if lon_min < lon < lon_max and lat_min < lat < lat_max:
                for date in range(len(date_list)):
                    key = sta+'_'+date_list[date]
                    CCI_sta_extract_dic[key] = vals[date]
        time_end = time.time()
        log_process.process_bar(flag,len(npz),time_init,time_start,time_end)
        flag += 1
    print('saving...')
    np.save(this_root+'CCI\\CCI_sta_extract_dic',CCI_sta_extract_dic)



def check_monthly():


    f = this_root+'CCI\\monthly\\1987\\month_compose_01.npy'
    arr = np.load(f)
    grid = arr < -9999
    arr[grid] = np.nan
    print np.shape(arr)
    plt.imshow(arr)
    plt.colorbar()
    plt.show()



def kernel_main(params):
    fdir,out_dir,year = params
    fdir = fdir + '{}\\'.format(year)
    out_dir = out_dir + '{}\\'.format(year)
    compose_month(fdir, out_dir)
    pass


def npy_to_tif():


    fdir = this_root+'CCI\\0.25\\monthly\\'
    outdir = this_root+'CCI\\0.25\\tif\\'
    mk_dir(outdir)
    for year in tqdm(os.listdir(fdir)):
        for f in os.listdir(fdir+year):
            mon = f.split('.')[0][-2:]
            fname = outdir+year+mon+'.tif'
            arr = np.load(fdir+year+'\\'+f)
            grid = arr < 0
            arr[grid] = -999999
            longitude_start = -179.875
            latitude_start = 89.875
            pixelWidth = 0.25
            pixelHeight = -0.25
            analysis.to_raster.array2raster_polar(fname,longitude_start, latitude_start, pixelWidth, pixelHeight, arr, -999999)
def main():
    # 1 合成月数据
    # fdir = this_root+'CCI\\COMBINED\\'
    # out_dir = this_root+'CCI\\monthly\\'
    # params = []
    # for year in range(1982,2016):
    #     params.append([fdir,out_dir,year])
    # analysis.MUTIPROCESS(kernel_main,params).run()

    # 2 转换为tif
    npy_to_tif()


    pass



if __name__ == '__main__':
    main()
    # check_monthly()