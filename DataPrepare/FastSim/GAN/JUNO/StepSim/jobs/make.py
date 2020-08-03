import os
import random
import math

tmp = '''
source /junofs/users/wxfang/FastSim/setup_conda.sh
python /junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/root2hdf5_v4.py --input %(m_input)s --output %(m_output)s --X %(m_x)s --Y %(m_y)s --Z %(m_z)s  
echo "done"
'''

pos_file = open('/junofs/users/wxfang/JUNO/pos.txt','r')
pos_info = pos_file.readlines()
pos_dict = {}
for line in pos_info:
    info = line.split(' ')
    pos_dict[info[0]] = [info[1], info[2], info[3]]
pos_file.close()

in_file_path = '/junofs/users/wxfang/JUNO/sim_output/'
in_files = os.listdir(in_file_path)

index = 1
for i in in_files:
    abs_file_name = '%s%s'%(in_file_path,i)
    if os.path.getsize(abs_file_name) < 70629425 : continue
    f_out= open('jobs_%d.sh'%index,'w')
    str_in = abs_file_name.split('/')[-1]
    str_in = str_in.replace('.root','.h5')
    str_output = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/h5_data/%s'%(str_in) 
    f_out.write(tmp%({'m_input':abs_file_name, 'm_output':str_output, 'm_x':pos_dict[abs_file_name][0], 'm_y':pos_dict[abs_file_name][1], 'm_z':pos_dict[abs_file_name][2]}))
    f_out.close()
    index = index + 1
os.system('chmod +x *.sh')
print('done')
