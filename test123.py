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
    selected_months = [0,11,1,2,10,5]
    selected_months.sort()

    if 0 in selected_months and 11 in selected_months:
        bias = []
        for i in selected_months:
            if i < 6:
                delta = i + 1
                pass
            else:
                delta = i - 11
            bias.append(delta)

        mid_month = np.floor(np.mean(bias)) + 11
        print np.mean(bias)
        print bias
        print selected_months
        print mid_month

        growing_season = [
            mid_month - 2,
            mid_month - 1,
            mid_month + 0,
            mid_month + 1,
            mid_month + 2,
            mid_month + 3,
        ]
        new_growing_season = []
        for i in growing_season:
            if i > 11:
                new_growing_season.append(int(i - 12))
            else:
                new_growing_season.append(int(i))
        # new_growing_season.sort()
        print new_growing_season

        # plt.plot(vals)
        # plt.scatter(range(len(vals)), vals)
        # for i in range(len(max_6_vals)):
        #     plt.text(max_6_vals[i][1], max_6_vals[i][0], str(max_6_vals[i][2]) + '\n' + '%0.2f' % max_6_vals[i][0])
        # plt.show()
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