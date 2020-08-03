import os


tmp = '''
source /junofs/users/wxfang/FastSim/setup_conda.sh
#python /junofs/users/wxfang/FastSim/GAN/CEPC/GAN/root2hdf5.py --input %(input)s --output %(output)s --tag %(tag)s --str_particle '%(particle)s'
python /junofs/users/wxfang/FastSim/GAN/CEPC/GAN/root2hdf5_v1.py --input %(input)s --output %(output)s --tag %(tag)s --str_particle '%(particle)s'
echo "done"
'''
Particle = 'pi-'
root_path = '/cefs/higgs/wxfang/cepc/pionm/root_fix/'
out_path = '/cefs/higgs/wxfang/cepc/pionm/h5_fix/'
root_file = os.listdir(root_path)

index = 0
for i in root_file:
    if ".root" not in i:continue
    file_name = i.replace('.root','.h5')
    str_output = '%s/%s'%(out_path,file_name) 
    out_name = str('job_%d.sh'%index)
    f = open(out_name,'w')
    f.write("#!/bin/bash")
    f.write(tmp%({'input':str(root_path+i), 'output':str_output, 'tag':'%s'%(str(index)), 'particle':Particle}))
    f.close()
    index = index + 1
os.system('chmod +x *.sh')
print('done')
