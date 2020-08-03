import os

rec_dir = '/junofs/users/wxfang/FastSim/bes3/workarea/TestRelease/TestRelease-00-00-86/DedxTest/rec/output_NN/'

rec_file = {}
files = os.listdir(rec_dir)
for i in files:
    #if 'noDecay' not in i: continue
    #if '_pip_' not in i: continue
    if '.rec' not in i: continue
    str1 = i.replace('.rtraw','')
    rec_file[str1] = rec_dir+i


template = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/template.txt'
f_in = open(template,'r')
lines = f_in.readlines()
out_path = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/job_NN/'
seed = 100
index = 0
for i in rec_file:
    str_particle = ''
    if "_e+_" in i or "_e-_" in i or '_ep_' in i or '_em_' in i:
        str_particle = 'e'
    elif "_mu+_" in i or "_mu-_"in i or '_mup_'in i or '_mum_' in i:
        str_particle = 'mu'
    elif "_p+_"in i or "_p-_"in i or '_pp_'in i or '_pm_' in i:
        str_particle = 'p'
    elif "_k+_"in i or "_k-_"in i or '_kp_'in i or '_km_' in i:
        str_particle = 'k'
    elif "_pi+_"in i or "_pi-_"in i or '_pip_'in i or '_pim_' in i:
        str_particle = 'pi'
    else:
        print('not find particle name for %s, exit()'% i)
        os.exit()
    seed = seed + 1
    f_out =open(out_path+'job_%d.txt'%index,'w')
    for line in lines:
        if 'NTupleSvc.Output' in line:
            ii = i.replace('.rec','.root')
            line = line.replace('dummy.root','/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output_NN/%s'%ii)
            f_out.write(line)
        elif 'BesRndmGenSvc.RndmSeed' in line:
            line = line.replace('1','%d'%seed)
            f_out.write(line)
        elif 'EventCnvSvc.digiRootInputFile' in line:
            line = line.replace('dummy.rec','%s'%rec_file[i])
            f_out.write(line)
        elif 'DedxTrackTest.ParticleType' in line:
            line = line.replace('= "p"','="%s"'%(str_particle))
            f_out.write(line)
        else:
            f_out.write(line)
    f_out.close()
    
    f_out = open(out_path+'job_%d.sh'%index,'w')
    f_out.write("#! /bin/bash")
    temp = '''
source /junofs/users/wxfang/FastSim/bes3/setup.sh
cd /junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/
boss.exe %(steerName)s 
echo "done!"
    '''
    f_out.write(temp % ({'steerName':str(out_path+'job_%d.txt'%index)}))
    f_out.close()
    index += 1
os.system('chmod +x *.sh')

