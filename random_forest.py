# coding=gbk


# import psutil
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

this_root = 'D:\\project05\\'