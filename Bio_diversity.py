# coding=gbk

from analysis import *
from scipy.interpolate import Rbf


this_root_branch = this_root+'branch2020\\'

class Bio_diversity:
    def __init__(self):
        self.this_class_arr = this_root_branch + 'arr\\Bio_diversity\\'
        self.this_class_tif = this_root_branch + 'tif\\Bio_diversity\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        pass


    def run(self):

        # self.plot_scatter()
        self.interpolate_scatter()
        pass


    def plot_scatter(self):
        '''
        read from raw csv
        x = longitude
        y = latitude
        val = Native Species Richness
        :return:
        '''
        csv = this_root+'data\\bio_diversity\\ellis_2012_l8_dataset_2012_01_17.dbf.csv'
        data = pd.read_csv(csv)
        x = data['X']
        y = data['Y']
        val = data['N']
        new_val = []
        for i in val:

            if i < 0.2:
                new_val.append(0)
            else:
                new_val.append(1)
        # plt.hist(val)
        # plt.show()
        plt.scatter(x,y,s=1,marker='.',c=val,cmap='jet')
        plt.colorbar()
        plt.axis('equal')
        plt.show()
        pass


    def interpolate_scatter(self):
        '''
        transform [lon1,lon2,...,lonN], [lat1,lat2,...,latN] to spatial array
        :return:
        '''

        csv = this_root + 'data\\bio_diversity\\ellis_2012_l8_dataset_2012_01_17.dbf.csv'
        data = pd.read_csv(csv)
        x = data['X']
        y = data['Y']
        val = data['N']

        # xx = np.linspace(-180,179.5,720)
        xx = np.arange(-180,179.5,0.5)
        # yy = np.linspace(-90,89.5,360)
        yy = np.arange(-90,89.5,0.5)[::-1]

        xi, yi = np.meshgrid(xx, yy)
        #
        # print xi
        # exit()
        function = 'linear'
        # ------------------------------------------------#
        # 'multiquadric': sqrt((r/self.epsilon)**2 + 1)   #
        # 'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)    #
        # 'gaussian': exp(-(r/self.epsilon)**2)           #
        # 'linear': r                                     #
        # 'cubic': r**3                                   #
        # 'quintic': r**5                                 #
        # 'thin_plate': r**2 * log(r)                     #
        # -------------------------------------------q-----#
        print 'interpolating1'
        interp = Rbf(x, y, val, function=function)
        print 'interpolating2'
        zi = interp(xi, yi)
        print 'saving'
        np.save(self.this_class_arr+'bio_diversity_arr_non_clip',zi)
        plt.imshow(zi,'jet')
        plt.colorbar()
        plt.show()


        pass


def main():
    Bio_diversity().run()
    # Bio_diversity().plot_scatter()
    pass


if __name__ == '__main__':
    main()