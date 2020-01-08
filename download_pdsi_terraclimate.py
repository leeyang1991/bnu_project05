# coding=utf-8
import urllib2
import requests
import os
import codecs
import time
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import analysis

year = []
for y in range(1982,2016):
    year.append(str(y))

def download(y):
    outdir = r'D:\project05\PET\download\\'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    url = 'https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_pet_{}.nc'.format(y)
    video_i = requests.get(url)
    ts = video_i.content
    movie = codecs.open(outdir+'pet_{}.nc'.format(y), 'wb')
    movie.write(ts)


analysis.MUTIPROCESS(download,year).run(process=4,process_or_thread='t',text='downloading...')
