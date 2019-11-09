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

this_root = r'd:\\project05\\'


def foo_():
    time.sleep(0.3)

def func():
    from tqdm import trange
    for i in tqdm(range(4), desc='1st loop'):
        for j in tqdm(range(5), desc='2nd loop'):
            for k in tqdm(range(50), desc='3nd loop', leave=False):
                time.sleep(0.01)


def main():
    func()
    # MULTIPROCESS().func()
    # MyTask().run()
if __name__ == '__main__':
    main()