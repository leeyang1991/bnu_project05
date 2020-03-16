# coding=gbk
'''
Bio-diversity pre-processing
'''
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

        ##### transform scatter to tif #####
        # 1 Rough interpolation with RBF (1 degree resolution)
        # self.interpolate_scatter1()
        # 2 interpolate Rough array to 0.5 degree
        # self.interpolate_scatter2()
        # 3 mask interpolated array
        # self.mask_extended_arr()
        # 4 normalize mask interpolated array
        self.normalize_arr()
        ##### transform scatter to tif #####
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

    def extened_grid(self, zi, x1, y1, zoom):

        # print(x1)
        nx = np.size(x1)
        ny = np.size(y1)
        x2 = np.linspace(x1.min(), x1.max(), nx * zoom)
        y2 = np.linspace(y1.min(), y1.max(), ny * zoom)
        xi, yi = np.meshgrid(x2, y2)

        from mpl_toolkits.basemap import interp
        z2 = interp(zi, x1, y1, xi, yi, checkbounds=True, masked=False, order=1)

        return z2, xi, yi, x2, y2, nx * zoom, ny * zoom


    def interpolate_scatter1(self):
        '''
        step 1
        transform [lon1,lon2,...,lonN], [lat1,lat2,...,latN] to spatial array
        :return:
        '''

        csv = this_root + 'data\\bio_diversity\\ellis_2012_l8_dataset_2012_01_17.dbf.csv'
        data = pd.read_csv(csv)
        x = data['X']
        y = data['Y']
        val = data['N']

        # xx = np.linspace(-180,179.5,720)
        xx = np.arange(-180,179.5,1)
        # yy = np.linspace(-90,89.5,360)
        yy = np.arange(-90,89.5,1)[::-1]

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
        print('interpolating1')
        interp = Rbf(x, y, val, function=function)
        print('interpolating2')
        zi = interp(xi, yi)
        print('saving')
        np.save(self.this_class_arr+'bio_diversity_arr_1_degree_non_clip',zi)
        plt.imshow(zi,'jet')
        plt.colorbar()
        plt.show()


    def interpolate_scatter2(self):
        f = self.this_class_arr+'bio_diversity_arr_1_degree_non_clip.npy'
        extend_f = self.this_class_arr+'bio_diversity_arr_0.5_degree_non_clip'
        arr_1 = np.load(f)
        xx = np.arange(-180, 179.5, 1)
        # yy = np.linspace(-90,89.5,360)
        yy = np.arange(-90, 89.5, 1)
        arr_extend, xii, yii, a, b, c, d = self.extened_grid(arr_1, xx, yy, 2)
        print(np.shape(arr_extend))
        np.save(extend_f,arr_extend)
        plt.imshow(arr_1)
        plt.figure()
        plt.imshow(arr_extend)
        plt.show()

    def mask_extended_arr(self):

        template_tif = this_root+'conf\\tif_template.tif'
        template = to_raster.raster2array(template_tif)[0]
        mask_arr = []
        for i in tqdm(list(range(len(template)))):
            temp = []
            for j in range(len(template[0])):
                val = template[i][j]
                if val < -9999:
                    temp.append(True)
                else:
                    temp.append(False)
            mask_arr.append(temp)
        mask_arr = np.array(mask_arr)

        extended_arr = np.load(self.this_class_arr+'bio_diversity_arr_0.5_degree_non_clip.npy')
        extended_arr[mask_arr] = np.nan
        DIC_and_TIF().arr_to_tif(extended_arr,self.this_class_tif+'bio_diversity.tif')
        # plt.imshow(extended_arr,'jet')
        # plt.colorbar()
        # plt.show()



        pass

    def normalize_arr(self):
        tif = self.this_class_tif+'bio_diversity.tif'
        arr = to_raster.raster2array(tif)[0]
        arr[arr<-9999]=np.nan

        all_vals = []
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                val = arr[i][j]
                if np.isnan(val):
                    continue
                all_vals.append(val)

        xmin = np.min(all_vals)
        xmax = np.max(all_vals)

        normalized_arr = []
        for i in range(len(arr)):
            temp = []
            for j in range(len(arr[0])):
                val = arr[i][j]
                new_val = (val-xmin)/(xmax-xmin)
                temp.append(new_val)
            normalized_arr.append(temp)
        normalized_arr = np.array(normalized_arr)
        DIC_and_TIF().arr_to_tif(normalized_arr,self.this_class_tif+'bio_diversity_normalized.tif')

        plt.imshow(normalized_arr)
        plt.colorbar()
        plt.show()



        pass




def main():
    Bio_diversity().run()
    # Bio_diversity().plot_scatter()
    pass


if __name__ == '__main__':
    main()