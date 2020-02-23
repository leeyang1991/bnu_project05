# coding=gbk
'''
python 获取管理员权限
启动禁用网卡
'''
import os
import ctypes, sys

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def hello_message():

    print('1: Enable Network card')
    print('0: Disable Network card')

def enable_disable():
    try:
        input_ = input()
        if input_ == 1:
            stat = 'netsh interface set interface 以太网 enable'
        elif input_ == 0:
            stat = 'netsh interface set interface 以太网 disabled'
        else:
            raise IOError('input error...')
        os.system(stat)

    except:
        print('input error...')


def main():
    if is_admin():
        while 1:
            hello_message()
            enable_disable()
    else:
        if sys.version_info[0] == 3:
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)
        else:#in python2.x
            ctypes.windll.shell32.ShellExecuteW(None, u"runas", unicode(sys.executable), unicode(__file__), None, 1)


if __name__ == '__main__':
    main()