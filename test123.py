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
    a = [1,2,3,11,12]

    if 1 in a and 12 in a:
        if list(a) == [1,2,3,4,12]:
            pre = [1,2,12]
        elif list(a) == [1,2,3,11,12]:
            pre = [11,12,1]
        elif list(a) == [1,2,10,11,12]:
            pre = [10,11,12]
        elif list(a) == [1,9,10,11,12]:
            pre = [9,10,11]
        else:
            raise IOError('error')
        print pre
def func():
    print '123'
    pass


class abc:

    def __init__(self):
        self.get_val()
        pass

    def get_val(self):

        self.a = 123

    def get_a(self):

        print self.a

def main():
    foo_()
    # func()


# main()
    # pass
if __name__ == '__main__':
#     # import this
#     a = 123
#     s1 = '\'hello, world!\''
#     print s1
    main()