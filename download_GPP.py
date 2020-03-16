# coding=gbk
import ee
import time
from osgeo import ogr
import os
import requests
import sys
import log_process
import zipfile
import os
import shutil
import sys
from analysis import *

this_root = 'd:\\project05\\'
log = log_process.Logger('log2001.log', level='info')

def mk_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def get_download_url(datatype = "LANDSAT/LC08/C01/T1_32DAY_NDVI",
                     band='',
                     start = '2000-01-01',
                     end = '2000-02-01',
                     # region='[[117, 47], [119, 47], [119, 45], [117, 45]]',
                     region='[[-180, 90], [180, 90], [180, -90], [-180, -90]]',
                     ):

    dataset = ee.ImageCollection(datatype).filterDate(start, end)

    Gpp = dataset.select(band)
    # print(Gpp.toList(10))
    # exit()
    image1 = Gpp.mean()
    # fvc = image1.rename('FVC')
    path = image1.getDownloadUrl({
        'scale': 40000,
        'crs': 'EPSG:4326',
        'region': '[[-180, 90], [180, 90], [180, -90], [-180, -90]]'
                                })

    return path


def gen_polygons():
    shapefile = this_root + 'shp\\fishnet.shp'
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    layer = dataSource.GetLayer()
    all_polygons = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        g = geom.GetEnvelope()
        # print(type(g))
        # [[117, 47], [119, 47], [119, 45], [117, 45]]
        # print(g)
        region = [
            [g[0],g[2]],
            [g[1],g[2]],
            [g[1],g[3]],
            [g[0],g[3]]
                  ]
        # print(str(region))
        all_polygons.append(str(region))
    # return
    return all_polygons


# def gen_landsat_8_date():
#     date_list = []
#     for year in range(2013,2018):
#         for mon in range(1,13):
#             if year == 2013:
#                 if mon not in [1,2,3,4]:
#                     date = str(year)+'-'+'%02d'%mon+'-01'
#                 else:
#                     date = None
#             else:
#                 date = str(year) + '-' + '%02d' % mon+'-01'
#             if not date == None:
#                 date_list.append(date)
#     return date_list


# def gen_landsat_5_date():
#     date_list = []
#     for year in range(1985,2012):
#         for mon in range(1,13):
#             if year == 1984:
#                 if mon not in [1,2,3,4]:
#                     date = str(year)+'-'+'%02d'%mon+'-01'
#                 else:
#                     date = None
#             else:
#                 date = str(year) + '-' + '%02d' % mon+'-01'
#             if not date == None:
#                 date_list.append(date)
#     return date_list


def gen_landsat_7_date():
    date_list = []
    for year in range(2000,2017):
        for mon in range(1,13):
            if year == 2016:
                if mon not in [2,3,4,5,6,7,8,9,10,11,12]:
                    date = str(year)+'-'+'%02d'%mon+'-01'
                else:
                    date = None
            else:
                date = str(year) + '-' + '%02d' % mon+'-01'
            if not date == None:
                date_list.append(date)
    return date_list


def gen_download_url_grace(start = '2000-01-01',
                     end = '2000-02-01',
                     region='[[117, 47], [119, 47], [119, 45], [117, 45]]',):
    datatype = 'NASA/GRACE/MASS_GRIDS/LAND'
    dataset = ee.ImageCollection(datatype).filterDate(start, end)
    grace1 = dataset.select('lwe_thickness_csr')
    grace2 = dataset.select('lwe_thickness_gfz')
    grace3 = dataset.select('lwe_thickness_jpl')

    grace = grace1.merge(grace2).merge(grace3)
    image = grace.mean()

    path = image.getDownloadUrl({
        'scale': 4000,
        'crs': 'EPSG:4326',
        'region': region
    })

    return path


    pass


def gen_urls():

    # os.makedirs(this_root + 'url\\')
    start = time.time()
    log.logger.info('Initializing auth...')
    ee.Initialize()
    end = time.time()
    log.logger.info('Account initialized '+'time %0.2f' % (end - start)+' s')

    # time_start = time.time()
    # polygon_list = gen_polygons()
    date_list = gen_landsat_7_date()
    # print date_list
    # exit()
    # for i in polygon_list:
    #     print(i)
    # dataset = "MODIS/006/MCD12Q1"
    # dataset = "MODIS/006/MOD13A1"
    # dataset = "IDAHO_EPSCOR/TERRACLIMATE"
    # dataset = "MODIS/006/MOD16A2"
    dataset = "MODIS/006/MOD17A2H"
    band = 'Gpp'
    # band = 'lwe_thickness_gfz'
    # band = 'lwe_thickness_csr'
    flag = 0
    download_dir = this_root + 'download_data\\'
    mk_dir(download_dir)
    # date_list = ['2010-01-01','2011-01-01']
    for i in range(len(date_list)):
        if i == len(date_list)-1:
            break
        start = date_list[i]
        end = date_list[i+1]

        # for j in range(len(polygon_list)):
        data = dataset.split('/')[1]+'_'+band
        # file_name = data + '_' + start + '_' + end + '_' + '%02d' % (j + 1) + '.zip'
        file_name = data + '_' + start + '_' + end + '.zip'
        path = download_dir + file_name
        if os.path.isfile(path):
            log.logger.info(path+' is already existed')
            continue
        flag += 1
        # print(polygon_list[j])
        # polygon = '[[73, 54], [135, 54], [135, 18], [73, 18]]'
        polygon = '[[73, 54], [135, 54], [135, 18], [73, 18]]'
        url = get_download_url(dataset,band,start,end,polygon)
        # url = gen_download_url_grace(start,end,polygon)

        download_data(url,path)
    return 1



def download_data(url,file_name):


    path = file_name
    if not os.path.isfile(path):
        # success = 0
        attempt = 0
        while 1:
            try:
                with open(path, "wb") as f:
                    log.logger.info("\nDownloading %s" % file_name)
                    response = requests.get(url, stream=True)
                    total_length = 7.*1024.*1024.

                    if total_length is None:  # no content length header
                        f.write(response.content)
                    else:
                        dl = 0
                        total_length = int(total_length)
                        for data in response.iter_content(chunk_size=1024):
                            dl += len(data)
                            f.write(data)
                            done = int(50 * dl / total_length)
                            sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                            sys.stdout.flush()
                success = 1
            except Exception as e:
                attempt += 1
                time.sleep(1)
                log.logger.info(e)
                log.logger.info('try '+str(attempt))
                success = 0
            if success == 1:
                break
            if attempt > 10:
                log.logger.info(str(attempt)+' try')
                break
    else:
        log.logger.info(path+' is already existed')

    # r = requests.get(url, stream=True)
    # with open(path, 'wb') as f:
    #     # total_length = int(r.headers.get('content-length'))
    #     total_length = int(150. * 1024. * 1024.)
    #     for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
    #         if chunk:
    #             f.write(chunk)
    #             f.flush()


def do_download():
    attempts = 0
    while 1:
        a = gen_urls()
        try:
            a = gen_urls()
            attempts += a
            if attempts > 2:
                break
        except Exception as e:
            attempts+=1
            if attempts > 10:
                break
            time.sleep(1)
            log.logger.info(e)
            log.logger.info('try '+str(attempts))


def unzip(zip,move_dst_folder):
    mk_dir(move_dst_folder)
    path_to_zip_file = zip
    tif_name = zip.split('\\')[-1].split('.')[0]
    # print(tif_name)
    # exit()
    # move_dst_folder = this_root+'tif\\'
    if not os.path.isfile(move_dst_folder+tif_name+'.tif'):
        directory_to_extract_to = this_root+'temp\\'
        zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
        zip_ref.extractall(directory_to_extract_to)
        zip_ref.close()

        file_list = os.listdir(directory_to_extract_to)
        for i in file_list:
            if i.endswith('.tif'):
                shutil.move(directory_to_extract_to+'\\'+i,move_dst_folder+tif_name+'.tif')
    else:
        print((move_dst_folder+tif_name+'.tif is existed'))


def check_zip(path):
    ZipFile = zipfile.ZipFile
    BadZipfile = zipfile.BadZipfile
    try:
        with ZipFile(path) as zf:
            pass

    except BadZipfile:
        print(path + " Does not work")
        os.remove(path)
    pass

def do_unzpi(year):
    year = str(year)
    import time
    fdir = this_root+'30m_fvc_annual\\'+year+'\\'
    flist = os.listdir(fdir)
    P = log_process.process_bar
    dest_folder = this_root+'30m_fvc_annual_unzip\\'+year+'\\'
    mk_dir(dest_folder)
    time_init = time.time()
    for f in range(len(flist)):
        start = time.time()
        fzip = fdir+flist[f]
        # print(fzip)
        unzip(fzip,dest_folder)
        end = time.time()
        P(f,len(flist),time_init,start,end,year)


def do_unzpi1():
    import time
    fdir = 'E:\\before2000\\zip\\'
    flist = os.listdir(fdir)
    P = log_process.process_bar
    dest_folder = 'E:\\before2000\\unzip\\'
    mk_dir(dest_folder)
    time_init = time.time()
    for f in range(len(flist)):
        start = time.time()
        fzip = fdir+flist[f]
        # print(fzip)
        if os.path.isfile(dest_folder+fzip.split('\\')[-1].split('.')[0]):
            print((fzip+'is existed'))
            continue
        try:
            unzip(fzip,dest_folder)
        except Exception as e:
            print(e)
        #check_zip(fzip)
        end = time.time()
        P(f,len(flist),time_init,start,end)


def do_unzip2():
    import time
    fdir = this_root+'download_data\\'
    # fdir = 'E:\\before2000\\zip\\'
    flist = os.listdir(fdir)
    P = log_process.process_bar
    dest_folder = 'E:\\GPP\\unzip\\'
    mk_dir(dest_folder)
    time_init = time.time()
    for f in range(len(flist)):
        start = time.time()
        fzip = fdir + flist[f]
        # print(fzip)
        if os.path.isfile(dest_folder + fzip.split('\\')[-1].split('.')[0]):
            print((fzip + 'is existed'))
            continue
        try:
            unzip(fzip, dest_folder)
        except Exception as e:
            print(e)
        # check_zip(fzip)
        end = time.time()
        P(f, len(flist), time_init, start, end)
    pass


def rename():
    fdir = this_root+'GPP\\pre-prosess\\resample\\'

    outdir = this_root+'GPP\\tif\\'
    mk_dir(outdir)
    for f in os.listdir(fdir):
        if f.endswith('.tif'):
            fsplit = f.split('_')
            start = fsplit[2]
            year,mon,day = start.split('-')
            # print f
            print(year,mon,day)
            fname = year+mon+'.tif'
            shutil.copy(fdir+f,outdir+fname)
    pass


def move_one_pix():
    # import analysis
    # 移动1个像素
    fdir = this_root+'GPP\\pre-prosess\\tif\\'
    outdir = this_root+'GPP\\tif\\'
    mk_dir(outdir)
    for f in tqdm(os.listdir(fdir)):
        if f.endswith('.tif'):
            array,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(fdir+f)
            arr = array[:360]
            # arr[arr==0] = np.nan
            DIC_and_TIF().arr_to_tif(arr,outdir+f)
            # print analysis.np.shape(arr)
            # analysis.plt.imshow(arr)
            # analysis.plt.show()
    pass


def main():
    # do_unzip2()
    # rename()
    move_one_pix()
    pass


if __name__ == '__main__':
    main()