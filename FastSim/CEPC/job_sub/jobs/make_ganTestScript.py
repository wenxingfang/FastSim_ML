##############lsgan########################
#temp = '''
#/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/generator_v7.py --latent-size 512 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-/gen_model_0221_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-/comb_model_0221_%(s_epoch)s.h5"   --exact-model False --exact-list ''  --output "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/pi-/Gen_0221_%(s_epoch)s.h5" --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-/dis_model_0221_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/dataset_pi-_test.txt' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/pi-/" --name_pb "model.pb" --SavedModel False 
#'''
temp = '''
/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/generator_v7_EcalOnly.py --latent-size 512 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/gen_model_0306_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/comb_model_0306_%(s_epoch)s.h5"   --exact-model False --exact-list ''  --output "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/pi-/Gen_fix_0306_%(s_epoch)s.h5" --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/pi-_fix/dis_model_0306_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-_fix/HoEless2/h5/Digi_sim_10_pionm_94.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/pi-/" --name_pb "model.pb" --SavedModel False 
'''
template = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/jobs/Template_Testing_pi-.sh'
out_file = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/jobs/Testing_pi-_batch.sh'

f_in = open(template,'r')
lines = f_in.readlines()
f_in.close()
f_out = open(out_file,'w')

for line in lines:
    f_out.write(line)

epochs = range(50,100)    
for i in epochs:
    f_out.write(temp % ({'s_epoch':str('epoch%d'%i)}))
    f_out.write('\n')
f_out.close()
print('done for %s'%out_file)
