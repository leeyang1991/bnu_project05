# coding=utf-8
'''
author: LiYang
Date: 20190801
Location: zzq BeiJing
Desctiption
'''
from analysis import *


def _foo_(growingseason,ind):

    # a = [1,2,10,11,12]
    # ind = 11
    mon = ind%12+1
    b = growingseason.index(mon)
    # print a
    # print mon
    # print b
    return b

def func(in_list):

    a=[12,1,2,3,4]
    b=[11,12,1,2,3]
    c=[10,11,12,1,2]
    d=[9,10,11,12,1]

    aa = copy.deepcopy(a)
    bb = copy.deepcopy(a)
    cc = copy.deepcopy(a)
    dd = copy.deepcopy(a)

    aa.sort()
    bb.sort()
    cc.sort()
    dd.sort()
    if in_list == aa:
        return a
    elif in_list == bb:
        return b
    elif in_list == cc:
        return c
    elif in_list == dd:
        return d
    pass



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

    a=[1,2,3,4,12]
    b = func(a)
    print a
    print b

if __name__ == '__main__':
    main()