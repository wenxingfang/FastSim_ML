import h5py
import argparse
import numpy as np
import sys 
import os 

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
    inFormat  = parse_args.inFormat
    
    original_files =[]
    if inFormat == 0:
        f_in = open(input_data, 'r')
        for line in f_in:
            if '#' in line: continue
            original_files.append(line.strip('\n'))
        f_in.close()
    elif inFormat == 1:
        dirs = os.listdir(input_data)
        for i in dirs:
            if '.h5' not in i:continue
            original_files.append(str(input_data+'/'+i))

 
    orig_list = list(range(len(original_files)))
    np.random.shuffle(orig_list)
    print('input file size=',len(orig_list))
    print('input file set size=',len(set(orig_list)))
    #sys.exit()
    for i in range(0, len(orig_list), 2):
        if i >= len(orig_list) or (i+1) >= len(orig_list): continue
        p1_f = original_files[orig_list[i  ]]
        p2_f = original_files[orig_list[i+1]]
        p1_h = h5py.File(p1_f,'r')
        p2_h = h5py.File(p2_f,'r')
        p1_d = p1_h['nPEByPMT'][:]
        p2_d = p2_h['nPEByPMT'][:]
        p1_h.close()
        p2_h.close()
        comb = np.concatenate((p1_d, p2_d), axis=0)
        comb_list = list(range(comb.shape[0]))
        np.random.shuffle(comb_list)
        split_point = int(len(comb_list)/2)
        d1_index = comb_list[0:split_point]
        d2_index = comb_list[split_point:len(comb_list)]
        d1_d = comb[d1_index,:] 
        d2_d = comb[d2_index,:] 
        d1_f = p1_f.split('/')[-1]
        d1_f = output_path + '/' + d1_f
        d2_f = p2_f.split('/')[-1]
        d2_f = output_path + '/' + d2_f
        d1_h = h5py.File(d1_f, 'w')
        d1_h.create_dataset('nPEByPMT', data=d1_d)
        d1_h.close()
        d2_h = h5py.File(d2_f, 'w')
        d2_h.create_dataset('nPEByPMT', data=d2_d)
        d2_h.close()
    
    print('done mixing')
    
