# coding=gbk

import requests

def lonlat_to_address(lon,lat):
    ak="mziulWyNDGkBdDnFxWDTvELlMSun8Obt" # 参照自己的应用
    url = 'http://api.map.baidu.com/reverse_geocoding/v3/?ak=mziulWyNDGkBdDnFxWDTvELlMSun8Obt&output=json&coordtype=wgs84ll&location=%s,%s'%(lat,lon)
    content = requests.get(url).text
    dic = eval(content)
    # for key in dic['result']:
    add = dic['result']['formatted_address']
    return add


def main():
    print(lonlat_to_address(117,43))

if __name__ == '__main__':
    main()