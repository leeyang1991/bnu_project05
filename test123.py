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
# import to_raster
# import ogr, os, osr
from tqdm import tqdm
import log_process
import multiprocessing
import datetime
# import lon_lat_to_address
from scipy import stats, linalg
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
# import imageio
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl
# import analysis
# import copy_reg
import types
import sys

this_root = r'd:\\project05\\'


def foo_():

    for i in tqdm(range(1000)):
        time.sleep(0.01)


def func():
    interval = 50
    a = 'SPEI\\per_pix\\SPEI_{:1>3d}\\'.format(interval)
    print(a)

    pass


def main():
    foo_()
if __name__ == '__main__':
    main()