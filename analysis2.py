# coding=gbk
from analysis import *

class Pick:

    def __init__(self):
        pass

    def run(self):
        interval = 3
        # self.pick(interval)

        pass


    def pick_events(self, interval):
        interval = '%02d' % interval
        out_dir = this_root + 'SPEI\\pick_non_growing_season_events\\SPEI_{}\\'.format(interval)
        Tools().mk_dir(out_dir, force=True)
        tropical_pix = np.load(this_root + 'NDVI\\tropical_pix.npy')
        growing_season_daterange = dict(np.load(this_root + 'NDVI\\growing_season_index.npy').item())

        SPEI_dir = this_root + 'SPEI\\per_pix\\SPEI_{}\\'.format(interval)
        single_event_dir = this_root + 'SPEI\\single_events_24\\SPEI_{}\\'.format(interval)

        non_growing_season = {}
        for f in tqdm(os.listdir(single_event_dir)):
            dic = dict(np.load(single_event_dir + f).item())
            spei_dic = dict(np.load(SPEI_dir + f).item())
            for pix in dic:
                # exit()
                val = dic[pix]
                if len(val) == 0:
                    continue
                spei = spei_dic[pix]
                smooth_window = 3
                spei = Tools().forward_window_smooth(spei, smooth_window)

                if pix in tropical_pix:
                    growing_daterange = range(1, 13)
                    # print growing_daterange
                elif pix in growing_season_daterange:
                    growing_daterange = growing_season_daterange[pix]
                    growing_daterange = Pick_Single_events1.remove_vals_from_list(growing_daterange)
                else:
                    growing_daterange = []

                selected_date_range = []
                for date_range in val:
                    # picked_vals = self.get_spei_vals(spei,date_range)
                    min_index = Pick_Single_events1.get_min_spei_index(spei, date_range)
                    mon = Pick_Single_events1.index_to_mon(min_index)
                    if mon in growing_daterange:
                        # hemi_dic[pix] = date_range
                        selected_date_range.append(date_range)
                non_growing_season[pix] = selected_date_range

        np.save(out_dir + 'global', non_growing_season)
def main():
    Pick().run()
    pass

if __name__ == '__main__':
    main()