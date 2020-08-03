import os
import random
import math
########## v4 ###########
tmp = '''
#python /junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/root2hdf5_v4.py --input %(m_input)s --output %(m_output)s --X %(m_x)s --Y %(m_y)s --Z %(m_z)s  
#python /junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/root2hdf5_npe_only.py --input %(m_input)s --output %(m_output)s --X %(m_x)s --Y %(m_y)s --Z %(m_z)s  
#python /junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/root2hdf5_npe_only_v1.py --input %(m_input)s --output %(m_output)s --X %(m_x)s --Y %(m_y)s --Z %(m_z)s  
python /junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/root2hdf5_hittime_only.py --input %(m_input)s --output %(m_output)s --X %(m_x)s --Y %(m_y)s --Z %(m_z)s  
'''

in_file_path = '/cefs/higgs/wxfang/sim_ouput/photon_sample_r_17700/'
#in_file_path = '/cefs/higgs/wxfang/sim_ouput/photon_sample_r_linear/'
#in_file_path = '/cefs/higgs/wxfang/sim_ouput/photon_sample_zAxis_noDE/'
#in_file_path = '/cefs/higgs/wxfang/sim_ouput/photon_sample_z_axis/'
#in_file_path = '/cefs/higgs/wxfang/sim_ouput/photon_sample_r_1m_plane_valid/'
#in_file_path = '/cefs/higgs/wxfang/sim_ouput/photon_sample_r_1m_plane/'
#in_file_path = '/cefs/higgs/wxfang/sim_ouput/photon_sample_grid_noDE/'
#in_file_path = '/junofs/users/wxfang/JUNO/sim_output/r_grid_1MeV/'
#in_file_path = '/cefs/higgs/wxfang/sim_ouput/photon_sample/'

output_path = '/cefs/higgs/wxfang/sim_ouput/photon_sample_r_17700_hittime_h5_data/' 
in_files = os.listdir(in_file_path)
batch_size = 1
index_sh = 0
index = 1
f_out = 0
First = True
for i in in_files:
    abs_file_name = '%s%s'%(in_file_path,i)
    #if os.path.getsize(abs_file_name) < 70629425 : continue
    if First:
        print('first')
        f_out= open('job_%d.sh'%index_sh,'w')
        index_sh = index_sh + 1
        f_out.write('source /junofs/users/wxfang/FastSim/setup_conda.sh')
        First = False
    elif index%batch_size==0:
        print('index=',index)
        f_out.write('echo "done"')
        f_out.close()
        f_out= open('job_%d.sh'%index_sh,'w')
        index_sh = index_sh + 1
        f_out.write('source /junofs/users/wxfang/FastSim/setup_conda.sh')
    str_in = abs_file_name.split('/')[-1]
    str_in = str_in.replace('.root','.h5')
    #print(str_in)
    str_x  = str_in.split('_')[4]
    str_y  = str_in.split('_')[5]
    str_z  = str_in.split('_')[6]
    str_z  = str_z.split('.h5')[0]
    #str_output = '/cefs/higgs/wxfang/sim_ouput/photon_sample_h5_data/%s'%(str_in) 
    #str_output = '/cefs/higgs/wxfang/sim_ouput/r_grid_1MeV_h5_data_hittime/%s'%(str_in) 
    str_output = '%s/%s'%(output_path, str_in) 
    f_out.write(tmp%({'m_input':abs_file_name, 'm_output':str_output, 'm_x':str_x, 'm_y':str_y, 'm_z':str_z}))
    index = index + 1
f_out.write('echo "done"')
f_out.close()
os.system('chmod +x *.sh')
print('done')
