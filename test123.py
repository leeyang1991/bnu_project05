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
    import numpy as np
    from matplotlib.patches import Circle, Wedge, Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt

    # Fixing random state for reproducibility
    np.random.seed(1)

    fig, ax = plt.subplots()

    resolution = 50  # the number of vertices
    N = 6
    x = np.random.rand(N)
    y = np.random.rand(N)
    radii = 0.1 * np.random.rand(N)
    patches = []
    # for x1, y1, r in zip(x, y, radii):
    #     circle = Circle((x1, y1), r)
    #     patches.append(circle)
    #
    # x = np.random.rand(N)
    # y = np.random.rand(N)
    # radii = 0.1 * np.random.rand(N)
    # theta1 = 360.0 * np.random.rand(N)
    # theta2 = 360.0 * np.random.rand(N)
    # for x1, y1, r, t1, t2 in zip(x, y, radii, theta1, theta2):
    #     wedge = Wedge((x1, y1), r, t1, t2)
    #     patches.append(wedge)
    #
    # # Some limiting conditions on Wedge
    # patches += [
    #     Wedge((.3, .7), .1, 0, 360),  # Full circle
    #     Wedge((.7, .8), .2, 0, 360, width=0.05),  # Full ring
    #     Wedge((.8, .3), .2, 0, 45),  # Full sector
    #     Wedge((.8, .3), .2, 45, 90, width=0.10),  # Ring sector
    # ]

    for i in range(N):
        print np.random.rand(N, 2)
        polygon = Polygon(np.random.rand(N, 2), True)
        patches.append(polygon)

    colors = 100 * np.random.rand(len(patches))
    p = PatchCollection(patches, alpha=0.4)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    fig.colorbar(p, ax=ax)

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