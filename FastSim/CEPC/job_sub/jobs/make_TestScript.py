

temp = '''
/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/generator_v6.py --latent-size 512 --nb-events 5000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/e-/gen_model_1105_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/e-/comb_model_1105_%(s_epoch)s.h5"  --output "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen_em_1105_%(s_epoch)s.h5" --exact-model True --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/exact_input_em.txt'        --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/e-/dis_model_1105_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/e-/em_10.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/" --name_pb "model_em_%(s_epoch)s.pb" --SavedModel False 
'''

#temp = '''
#/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/generator_v6.py --latent-size 512 --nb-events 4200 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/gamma/gen_model_1105_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/gamma/comb_model_1105_%(s_epoch)s.h5" --for-mc-reco False  --output "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen_gamma_1105_%(s_epoch)s.h5" --exact-model True --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/exact_input_gamma.txt'        --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/gan/gamma/dis_model_1105_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/gamma/gamma_ext9.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/" --name_pb "model_gamma_%(s_epoch)s.pb" --SavedModel False 
#'''

template = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/jobs/Template_Testing_em.sh'
out_file = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/jobs/Testing_em_batch.sh'
#template = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/jobs/Template_Testing_gamma.sh'
#out_file = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/job_sub/jobs/Testing_gamma_batch.sh'
f_in = open(template,'r')
lines = f_in.readlines()
f_out = open(out_file,'w')

for line in lines:
    f_out.write(line)

epochs = range(77,99)    
for i in epochs:
    f_out.write(temp % ({'s_epoch':str('epoch%d'%i)}))
    f_out.write('\n')
f_out.close()
print('done')
