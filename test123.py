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
    from matplotlib import rcParams, cycler
    import matplotlib.pyplot as plt
    import numpy as np

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    N = 10
    data = [np.logspace(0, 1, 100) + np.random.randn(100) + ii for ii in range(N)]
    data = np.array(data).T
    print np.shape(data)
    for i in data:
        print i
    exit()
    cmap = plt.cm.coolwarm
    rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))

    fig, ax = plt.subplots()
    lines = ax.plot(data)
    ax.legend(lines)
    plt.show()
    pass

def pencil():
    plt.figure(figsize=(3, 3))
    for _ in tqdm(range(1000)):
        x = []
        y = []
        a = -(random.random())
        b = random.random()
        c = random.random()*5
        for i in np.linspace(-3,3,100):
            yy = a*((i)**2) +b*i + c
            x.append(i)
            y.append(yy)
        plt.plot(x,y,c='black',alpha=0.007)
    plt.show()
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