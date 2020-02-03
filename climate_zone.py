# coding=gbk
'''
自定义colormap
'''
from analysis import *

# koppen
cls_color_dic = {
    'Af':'8d1c21',
    'Am':'e7161a',
    'As':'f19596',
    'Aw':'f8c8c9',

    'BWk':'f1ee70',
    'BWh':'f4c520',
    'BSk':'c7a655',
    'BSh':'c58a19',

    'Cfa':'113118',
    'Cfb':'114f2a',
    'Cfc':'137539',
    'Csa':'6cb92c',
    'Csb':'9bc82a',
    'Csc':'bfd62e',
    'Cwa':'ad6421',
    'Cwb':'916425',
    'Cwc':'583d1b',

    'Dfa':'2d112f',
    'Dfb':'5a255d',
    'Dfc':'9b3e93',
    'Dfd':'b9177d',
    'Dsa':'bf7cb2',
    'Dsb':'deb3d2',
    'Dsc':'d9c5df',
    'Dsd':'c8c8c9',
    'Dwa':'bdafd5',
    'Dwb':'957cac',
    'Dwc':'7f57a1',
    'Dwd':'603691',

    'EF':'688cc7',
    'ET':'87cfd9',
}


f = this_root+'climate_zone\\Koeppen-Geiger-ASCII.txt'
fr = open(f)
fr.readline()
lines = fr.readlines()
fr.close()
lon_list = []
lat_list = []
val_list = []
for line in lines:
    line = line.split('\n')[0]
    lat,lon,cls = line.split()
    lon_list.append(float(lon))
    lat_list.append(float(lat))
    val_list.append(cls)

vals = list(set(val_list))
vals.sort()
vals_dic = {}
for i in range(len(vals)):
    vals_dic[vals[i]] = i

new_val_list = []
for val in val_list:
    new_val = vals_dic[val]
    new_val_list.append(new_val)



arr = DIC_and_TIF().ascii_to_arr(lon_list,lat_list,new_val_list)

arr_ascii = DIC_and_TIF().ascii_to_arr(lon_list,lat_list,val_list)
# np.save(this_root+'arr\\koppen_spatial_arr_ascii',arr_ascii)
# exit()
DIC_and_TIF().spatial_arr_to_dic(arr_ascii)
# arr to tif
DIC_and_TIF().arr_to_tif(arr,this_root+'climate_zone\\koppen.tif')

palette = []
for i in vals:
    palette.append('#'+cls_color_dic[i])
colors = sns.color_palette(palette)
cmap = mpl.colors.ListedColormap(colors)
# print cmap
plt.imshow(arr,cmap=cmap)
plt.show()