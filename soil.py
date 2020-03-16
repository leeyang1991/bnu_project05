# coding=gbk

from analysis import *
from NC_to_tif import *
this_root_branch = this_root+'branch2020\\'
class HWSD:

    def __init__(self):
        self.this_class_arr = this_root_branch + 'arr\\HWSD\\'
        self.this_class_tif = this_root_branch + 'tif\\HWSD\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        pass

    def run(self):

        fdir = this_root+'data\\HWSD\\data\\'
        # nc = 'T_CLAY.nc4'
        # nc = 'T_SAND.nc4'
        nc = 'T_SILT.nc4'
        f = fdir+nc
        outf = self.this_class_tif + nc.split('.')[0]+'.tif'
        variables_ = nc.split('.')[0]
        self.nc_to_tif(f,outf,variables_)
        pass

    def nc_to_tif(self,nc,outf,variables_):
        ncin = Dataset(nc, 'r')
        # print ncin.variables
        # exit()
        lat = ncin['lat']
        lon = ncin['lon']

        pixelWidth = lon[1]-lon[0]
        pixelHeight = lat[1]-lat[0]
        longitude_start = lon[0]
        latitude_start = lat[0]
        # exit()
        ndv = np.nan
        arr = ncin.variables[variables_]
        for name,variable in list(ncin.variables.items()):
            for var in variable.ncattrs():
                if var == 'missing_value':
                    ndv = variable.getncattr(var)
        if np.isnan(ndv):
            raise IOError('no key missing_value')
        arr = np.array(arr)
        grid = arr == -1
        arr[grid] = -999999

        to_raster.array2raster(outf,longitude_start,latitude_start,pixelWidth,pixelHeight,arr)

        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()



def main():
    HWSD().run()
    pass


if __name__ == '__main__':
    main()