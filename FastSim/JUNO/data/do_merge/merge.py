import os
import h5py
import numpy as np

## merge files with same r #######

in_file_path = '/cefs/higgs/wxfang/sim_ouput/r_grid_1MeV_h5_data_hittime/'

in_files = os.listdir(in_file_path)

output_path = '/cefs/higgs/fangwx/data/r_grid_1MeV_h5_data_hittime_merged/'

r_dict = {}

for i in in_files:
    if '.h5' not in i : continue
    r = i.split('_')[3]
    if r not in r_dict: r_dict[r] = []
    r_dict[r].append(i)
print('get all different r, size=', len(r_dict))
for r in r_dict:
    new_f = h5py.File(output_path+'/%s.h5'%(str(r)), 'w')
    first = True
    tmp_ditc = {}
    for i in r_dict[r]:
        full_name = str(in_file_path+'/'+i)
        d = h5py.File(full_name, 'r')
        list_key = list(d.keys())
        d.close()
        for ikey in list_key:
            if 'HitTimeByPMT' not in ikey: continue
            if ikey not in tmp_ditc : tmp_ditc[ikey] = []
            tmp_ditc[ikey].append(full_name)
    for ikey in tmp_ditc:
        if len(tmp_ditc[ikey]) == 0:continue
        tmp_h = h5py.File(tmp_ditc[ikey][0],'r')
        new_df = tmp_h[ikey][:]
        tmp_h.close()
        for j in range(1, len(tmp_ditc[ikey])):
            tmp_h = h5py.File(tmp_ditc[ikey][j],'r')
            tmp_df = tmp_h[ikey][:]
            tmp_h.close()
            new_df = np.concatenate((new_df, tmp_df), axis=-1)
        new_f.create_dataset('%s'%str(ikey), data=new_df)
    new_f.close()
print('done for merging')
