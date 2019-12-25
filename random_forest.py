# coding=gbk
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import datetime
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import sklearn
import scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import math
import analysis

this_root = 'D:\\project05\\'

def prepare_data():

    # 1 drought periods
    f_recovery_time = this_root+'arr\\recovery_time\\composite_3_modes\\composite_3_mode_recovery_time.npy'
    recovery_time = dict(np.load(f_recovery_time).item())
    for pix in recovery_time:
        vals = recovery_time[pix]

    # 2 per-pix growing season
    # f = this_root+'NDVI\\global_growing_season.npy'
    # global_growing_season_dic = np.load(f)




    # mode_dic = analysis.RATIO().load_data()

    # for m in mode_dic:
    #     dic = mode_dic[m]
    #     for pix in dic:
    #         print pix,dic[pix]



def main():
    prepare_data()


if __name__ == '__main__':
    main()