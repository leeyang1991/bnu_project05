# coding=gbk
import ee
import time
from osgeo import ogr
import os
import requests
import sys
import log_process

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


def main():
    # f = open(this_root+'url\\LC08_2013-05-01_2013-06-01_08.txt','r')
    # url=f.read()
    # print(url)
    # file_name = 'LC08_2013-05-01_2013-06-01_08.zip'
    # print('downloading..')
    # download_data(url,file_name)

    # url_dir = this_root+'url\\'
    # url_list = os.listdir(url_dir)
    # for fi in url_list:
    #     if 'LC08_2013-05-01_2013-06-01' in fi:
    #         f = open(url_dir+fi,'r')
    #         url = f.read()
    #         file_name = fi.split('.')[0]+'.zip'
    #         download_data(url, file_name)
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

if __name__ == '__main__':
    # import zipfile

    # log.logger.info(123123)
    main()