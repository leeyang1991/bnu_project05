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
    import matplotlib.pyplot as plt
    import numpy as np

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    N = 100
    r0 = 0.6
    x = 0.9 * np.random.rand(N)
    y = 0.9 * np.random.rand(N)
    area = (20 * np.random.rand(N)) ** 2  # 0 to 10 point radii
    c = np.sqrt(area)
    r = np.sqrt(x ** 2 + y ** 2)
    area1 = np.ma.masked_where(r < r0, area)
    area2 = np.ma.masked_where(r >= r0, area)
    plt.scatter(x, y, s=area1, marker='^', c=c)
    plt.scatter(x, y, s=area2, marker='o', c=c)
    # Show the boundary between the regions:
    theta = np.arange(0, np.pi / 2, 0.01)
    plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))

    plt.show()

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
    # foo_()
    func()

    pass
if __name__ == '__main__':
    main()