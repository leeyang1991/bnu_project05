# coding=utf-8
'''
author: LiYang
Date: 20190801
Location: zzq BeiJing
Desctiption
'''
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal
import time
import to_raster
import ogr, os, osr
from tqdm import tqdm
import log_process
import multiprocessing
import datetime
import lon_lat_to_address
from scipy import stats, linalg
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
import imageio
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl
import analysis
this_root = r'd:\\project05\\'



def func():
    fdir = r'D:\project05\NDVI\per_pix_anomaly\\'
    spei_dir = r'D:\project05\SPEI\per_pix\\'
    for f in os.listdir(fdir):

        if not '015' in f:
            continue
        print(f)
        dic = dict(np.load(fdir+f).item())
        spei_dic = dict(np.load(spei_dir+f).item())
        for key in dic:

            vals = dic[key]
            spei = spei_dic[key]
            if vals[0] == -999999 or spei[0] == -999999:
                continue
            print(key)
            plt.plot(vals)
            plt.plot(spei)
            # plt.ylim(-4,4)
            plt.show()

        print('*******')
    pass


def main():
    func()

if __name__ == '__main__':
    main()