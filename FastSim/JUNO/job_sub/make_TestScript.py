

temp = '''
/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/JUNO/generator/generator_laser_v1.py --latent-size 100 --nb-events 5000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/params/gan/gen_model_1208_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/params/gan/comb_model_1208_%(s_epoch)s.h5" --dis-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/params/gan/dis_model_1208_%(s_epoch)s.h5"   --output "/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/generator/Gen_1208_100_%(s_epoch)s.h5" --datafile-temp ''  --exact-model False --exact-list ''
'''
template = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/job_sub/Testing_template.sh'
out_file = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/job_sub/Testing_batch.sh'

f_in = open(template,'r')
lines = f_in.readlines()
f_in.close()
f_out = open(out_file,'w')

for line in lines:
    f_out.write(line)

epochs = range(50,200)    
for i in epochs:
    f_out.write(temp % ({'s_epoch':str('epoch%d'%i)}))
    f_out.write('\n')
f_out.close()
print('done for %s'%out_file)
