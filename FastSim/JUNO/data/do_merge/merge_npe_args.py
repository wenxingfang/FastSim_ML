import os
import h5py
import numpy as np
import argparse
## merge files with same r, theta #######
def get_parser():
    parser = argparse.ArgumentParser(
        description='Run MDN training. '
        'Sensible defaults come from https://github.com/taboola/mdn-tensorflow-notebook-example/blob/master/mdn.ipynb',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input_data', action='store', type=str,
                        help='input_data files')
    parser.add_argument('--output_path', action='store', type=str,
                        help='output_path')
    parser.add_argument('--inFormat', action='store', type=int, default=0,
                        help='input format.')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    parse_args = parser.parse_args()
    output_path = parse_args.output_path
    input_data  = parse_args.input_data

#in_files_txt = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/dataset_train.txt' 
#in_files_txt = 'dataset_test.txt' 

#output_path = '/cefs/higgs/fangwx/data/h5_data_npe_merged/'

    in_files = [] 
    in_files.append(input_data)
    
    r_index = 2 
    
    f_in = open(in_files_txt, 'r')
    for line in f_in:
        if '#' in line: continue
        in_files.append(line.strip('\n'))
    f_in.close()
    
    
    
    
    r_t_dict = {}
    
    for i in in_files:
        if '.h5' not in i : continue
        #full_name = str(in_file_path+'/'+i)
        d = h5py.File(i, 'r')
        df = d['nPEByPMT'][:]
        d.close()
        for r_t in range(df.shape[0]):
            str_r_t = str('%.3f_%.1f'%(df[r_t,0], df[r_t,1]))
            if str_r_t not in r_t_dict: r_t_dict[str_r_t] = {}
            if i not in r_t_dict[str_r_t]: r_t_dict[str_r_t][i]=[]
            r_t_dict[str_r_t][i].append(r_t)
    print('get all different r,theta, size=', len(r_t_dict))
    first = True
    new_f = 0
    batch_size = 2000
    df_index = 0
    f_index = 0
    for r_t in r_t_dict:
        print('r_t=',r_t,",\n ", r_t_dict[r_t])    
        if first:
            new_f = h5py.File(output_path+'/%d.h5'%(f_index), 'w')
            first = False
        elif df_index%batch_size==0:
            new_f.close()
            f_index += 1
            new_f = h5py.File(output_path+'/%d.h5'%(f_index), 'w')
       
        tmp_df = 0 
        first_try = True
        for i in r_t_dict[r_t]:
            #full_name = str(in_file_path+'/'+i)
            d = h5py.File(i, 'r')
            df = d['nPEByPMT'][:]
            d.close()
            if first_try:
                tmp_df = df[r_t_dict[r_t][i],2:df.shape[1]]
                tmp_df = tmp_df.reshape((-1,))
                first_try = False
            else:
                tmp_0 = df[r_t_dict[r_t][i],2:df.shape[1]]
                tmp_0 = tmp_0.reshape((-1,))
                tmp_df = np.concatenate((tmp_df, tmp_0), axis=-1)
            
            ''' 
            for j in r_t_dict[r_t][i]:
                str_r_t = str('%.3f_%.1f'%(df[j,0], df[j,1]))
                assert(str_r_t == r_t)
                #if r_t != str_r_t:continue
                if first_try:
                    tmp_df = df[j,2:df.shape[1]]
                    first_try = False
                else:
                    tmp_df = np.concatenate((tmp_df, df[j,2:df.shape[1]]), axis=-1)
            ''' 
        new_f.create_dataset('nPEByPMT_%s'%r_t, data=tmp_df)
        df_index += 1
    
    new_f.close()
    
    
    print('done for merging')
