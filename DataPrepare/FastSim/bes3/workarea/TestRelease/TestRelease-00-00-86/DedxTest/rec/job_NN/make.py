import os

sim_dir = '/junofs/users/wxfang/FastSim/bes3/workarea/TestRelease/TestRelease-00-00-86/DedxTest/sim/output_NN/'

sim_file = {}
files = os.listdir(sim_dir)
for i in files:
    str1 = i.replace('.rtraw','')
    sim_file[str1] = sim_dir+i


template = '/junofs/users/wxfang/FastSim/bes3/workarea/TestRelease/TestRelease-00-00-86/DedxTest/rec/job_template.txt'
f_in = open(template,'r')
lines = f_in.readlines()
out_path = '/junofs/users/wxfang/FastSim/bes3/workarea/TestRelease/TestRelease-00-00-86/DedxTest/rec/job_NN/'
seed = 100
index = 0
for i in sim_file:
    seed = seed + 1
    f_out =open(out_path+'job_%d.txt'%index,'w')
    for line in lines:
        if 'EventCnvSvc.digiRootOutputFile' in line:
            line = line.replace('dummy.rec','/junofs/users/wxfang/FastSim/bes3/workarea/TestRelease/TestRelease-00-00-86/DedxTest/rec/output_NN/%s.rec'%i)
            f_out.write(line)
        elif 'BesRndmGenSvc.RndmSeed' in line:
            line = line.replace('1','%d'%seed)
            f_out.write(line)
        elif 'EventCnvSvc.digiRootInputFile' in line:
            line = line.replace('dummy.rtraw','%s'%sim_file[i])
            f_out.write(line)
        else:
            f_out.write(line)
    f_out.close()
    
    f_out = open(out_path+'job_%d.sh'%index,'w')
    f_out.write("#! /bin/bash")
    temp = '''
source /junofs/users/wxfang/FastSim/bes3/setup.sh
cd /junofs/users/wxfang/FastSim/bes3/workarea/TestRelease/TestRelease-00-00-86/
boss.exe %(steerName)s 
echo "done!"
    '''
    f_out.write(temp % ({'steerName':str(out_path+'job_%d.txt'%index)}))
    f_out.close()
    index += 1
os.system('chmod +x *.sh')

