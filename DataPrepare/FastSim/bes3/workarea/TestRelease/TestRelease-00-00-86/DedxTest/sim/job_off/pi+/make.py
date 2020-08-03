import os

N_file = 100
N_event= 10000
template = '/junofs/users/wxfang/FastSim/bes3/workarea/TestRelease/TestRelease-00-00-86/DedxTest/sim/job_template_pi.txt'
f_in = open(template,'r')
lines = f_in.readlines()
out_path = '/junofs/users/wxfang/FastSim/bes3/workarea/TestRelease/TestRelease-00-00-86/DedxTest/sim/job_off/pi+/'
for i in range(N_file):
    f_out =open(out_path+'job_%d.txt'%i,'w')
    for line in lines:
        if 'RootCnvSvc.digiRootOutputFile' in line:
            #line = line.replace('dummy','/junofs/users/wxfang/FastSim/bes3/workarea/TestRelease/TestRelease-00-00-86/DedxTest/sim/output_off/Single_pi+_off_%d.rtraw'%i)
            line = line.replace('dummy','/junofs/users/wxfang/FastSim/bes3/workarea/TestRelease/TestRelease-00-00-86/DedxTest/sim/output_off/Single_noDecay_pi+_off_%d.rtraw'%i)
            f_out.write(line)
        elif 'BesRndmGenSvc.RndmSeed' in line:
            line = line.replace('= 100','= %d'%(i+100))
            f_out.write(line)
        elif 'ApplicationMgr.EvtMax' in line:
            line = line.replace('= 10','= %d'%N_event)
            f_out.write(line)
        else:
            f_out.write(line)
    f_out.close()

    f_out = open(out_path+'job_%d.sh'%i,'w')
    f_out.write("#! /bin/bash")
    temp = '''
source /junofs/users/wxfang/FastSim/bes3/setup.sh
source /junofs/users/wxfang/FastSim/bes3/workarea/setup_torch.sh
cd /junofs/users/wxfang/FastSim/bes3/workarea/TestRelease/TestRelease-00-00-86/
boss.exe %(steerName)s 
echo "done!"
    '''
    f_out.write(temp % ({'steerName':str(out_path+'job_%d.txt'%i)}))
    f_out.close()

os.system('chmod +x *.sh')

