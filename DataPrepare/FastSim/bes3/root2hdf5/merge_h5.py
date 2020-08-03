import h5py
import numpy as np

outName = 'Hit_Barrel_100k.h5'

h5_list=[]
h5_list.append('/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Hit_Barrel_em_100k.h5')
h5_list.append('/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Hit_Barrel_ep_100k.h5')
if len(h5_list) > 1:
    values = {}
    temp = h5py.File(h5_list[0],'r')
    keys = []
    for ikey in temp.keys():
        print(ikey)
        keys.append(ikey)
        values[ikey] = temp[ikey][:]
    temp.close()
    for ikey in keys:
        for i in range(1, len(h5_list)):
             temp = h5py.File(h5_list[i],'r')
             values[ikey] = np.concatenate((values[ikey], temp[ikey][:]),axis=0)
             temp.close()
    out = h5py.File(outName,'w')
    for ikey in keys:
        out.create_dataset(ikey   , data=values[ikey])
    out.close()
print('done')
