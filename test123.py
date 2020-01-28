# coding=utf-8
'''
author: LiYang
Date: 20190801
Location: zzq BeiJing
Desctiption
'''
from analysis import *

def foo_():

    a = []
    ind = 18
    print a
    print ind%12+1

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


if __name__ == '__main__':
    # import this
#     a = 123
#     s1 = '\'hello, world!\''
#     print s1
    main()