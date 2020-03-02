# coding=utf-8
'''
author: LiYang
Date: 20190801
Location: zzq BeiJing
Desctiption
'''
from analysis import *
import random
import matplotlib.patches as patches

def foo():
    for y in range(1982,2016):
        url = 'https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_def_{}.nc'.format(y)
        print url
    pass

def func():
    x = range(1,10000)
    y = []
    y2 =[]
    for x_i in tqdm(x):
        y_i = x_i / (34 - np.log(abs(x_i)+1)/np.log(2))
        y.append(y_i)
        y2.append(x_i/20.)
    plt.plot(x,y)
    plt.plot(x,y2)
    # plt.axis("equal")
    plt.show()


class __abc:

    def __init__(self):
        self.get_val()
        pass

    def get_val(self):

        self.a = 123

    def get_a(self):

        print self.a


def test123():

    print this_root

def main():
    foo()



    pass
if __name__ == '__main__':
    main()