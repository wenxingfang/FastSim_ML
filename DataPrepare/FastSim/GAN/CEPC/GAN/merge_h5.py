import os
import h5py
import numpy as np
def mergeInput(datafile, path, out_path):
    print('reading input:', datafile)
    df1 = None
    df2 = None
    df3 = None
    First = True
    for f in datafile:
        d = h5py.File(str(path+"/"+f), 'r')
        if First:
            df1 = d['Barrel_Hit'][:]
            df2 = d['Barrel_Hit_HCAL'][:]
            df3 = d['MC_info'][:]
            First = False
        else:
            df1 = np.concatenate ((df1, d['Barrel_Hit'][:])     , axis=0)
            df2 = np.concatenate ((df2, d['Barrel_Hit_HCAL'][:]), axis=0)
            df3 = np.concatenate ((df3, d['MC_info'][:])        , axis=0)
        d.close()
    print('df1=',df1.shape)
    hf = h5py.File(str(out_path+'/'+datafile[0]), 'w')
    hf.create_dataset('Barrel_Hit'     , data=df1)
    hf.create_dataset('Barrel_Hit_HCAL', data=df2)
    hf.create_dataset('MC_info'        , data=df3)
    hf.close()




h5_path_out = '/cefs/higgs/wxfang/cepc/pionm/h5_fix_merge/'
h5_path = '/cefs/higgs/wxfang/cepc/pionm/h5_fix/'
h5s = os.listdir(h5_path)
merge_n = 10.0
batch = 0
if (len(h5s)/merge_n)%1 == 0:
    batch = int(len(h5s)/merge_n)
else:
    batch = int(len(h5s)/merge_n) + 1

for i in range(batch): 
    f_list = h5s[int(merge_n*i):int(merge_n*(i+1))]
    mergeInput(f_list,h5_path,h5_path_out)
print('done')
