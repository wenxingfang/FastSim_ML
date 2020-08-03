

#temp = '''
#/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/generator_v6.py --latent-size 512 --nb-events 10000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_1114_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_1114_%(s_epoch)s.h5"   --exact-model True --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/exact_input_em.txt'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_1114_%(s_epoch)s.h5" --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_1114_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_zcut.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/" --name_pb "model_em_1114_%(s_epoch)s.pb" --SavedModel False 
#'''
#template = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Template_Testing_em.sh'
#out_file = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Testing_em_batch.sh'
#temp = '''
#/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/generator_v6.py --latent-size 512 --nb-events 10000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_ep_1114_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_ep_1114_%(s_epoch)s.h5"   --exact-model True --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/exact_input_ep.txt'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_ep_1114_%(s_epoch)s.h5" --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_ep_1114_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_ep_zcut.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/" --name_pb "model_ep_1114_%(s_epoch)s.pb" --SavedModel False 
#'''
#template = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Template_Testing_ep.sh'
#out_file = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Testing_ep_batch.sh'


#temp = '''
#/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/generator_v6.py --latent-size 512 --nb-events 13000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_sig1125_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_sig1125_%(s_epoch)s.h5"   --exact-model True --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/exact_input_sig_em.txt'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_sig1125test_%(s_epoch)s.h5" --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_sig1125_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_test_zcut.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/" --name_pb "model_em_sig1125_%(s_epoch)s.pb" --SavedModel False 
#'''
#template = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Template_Testing_em.sh'
#out_file = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Testing_sig_em_batch.sh'
######################################
#temp = '''
#/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/generator_v6.py --latent-size 512 --nb-events 5000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_ep_Low_1107_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_ep_Low_1107_%(s_epoch)s.h5"   --exact-model True --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/exact_input_ep_Low.txt'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_ep_Low_1107_%(s_epoch)s.h5" --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_ep_Low_1107_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_ep_zcut_Low.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/" --name_pb "model_ep_Low_1107.pb" --SavedModel False 
#'''
#template = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Template_Testing_ep_Low.sh'
#out_file = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Testing_ep_Low_batch.sh'
######################################
#temp = '''
#/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/generator_v6.py --latent-size 512 --nb-events 5000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_Low_1107_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_Low_1107_%(s_epoch)s.h5"   --exact-model True --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/exact_input_em_Low.txt'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_Low_1107_%(s_epoch)s.h5" --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_Low_1107_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_em_zcut_Low.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/" --name_pb "model_em_Low_1107.pb" --SavedModel False 
#'''
#template = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Template_Testing_em_Low.sh'
#out_file = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Testing_em_Low_batch.sh'
######################################
#temp = '''
#/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/generator_v6.py --latent-size 512 --nb-events 5000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_High_1107_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_High_1107_%(s_epoch)s.h5"   --exact-model True --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/exact_input_em_High.txt'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_High_1107_%(s_epoch)s.h5" --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_High_1107_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_em_zcut_High.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/" --name_pb "model_em_High_1107.pb" --SavedModel False 
#'''
#template = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Template_Testing_em_High.sh'
#out_file = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Testing_em_High_batch.sh'
######################################
#temp = '''
#/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/generator_v6.py --latent-size 512 --nb-events 5000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_ep_High_1107_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_ep_High_1107_%(s_epoch)s.h5"   --exact-model True --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/exact_input_ep_High.txt'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_ep_High_1107_%(s_epoch)s.h5" --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_ep_High_1107_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_ep_zcut_High.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/" --name_pb "model_ep_High_1107.pb" --SavedModel False 
#'''
#template = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Template_Testing_ep_High.sh'
#out_file = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Testing_ep_High_batch.sh'

############ w gan ###################
#template = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Template_Testing_em_wgan.sh'
#temp = '''
#/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/generator_wgan.py --latent-size 512 --nb-events 13000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_1203w_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_1203w_%(s_epoch)s.h5"   --exact-model True --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/exact_input_em.txt'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_wgan_1203w_%(s_epoch)s.h5" --check-dis-real False --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_1203w_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/single_em/mc_Hit_Barrel_em_test_zcut.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/" --name_pb "model_em_wgan_1203w_%(s_epoch)s.pb" --SavedModel False 
#'''
#out_file = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Testing_em_wgan_batch.sh'
############ wgan_gp ###################
#template = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Template_Testing_em_wgan.sh'
#temp = '''
#/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/generator_wgan.py --latent-size 512 --nb-events 13000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_1217gp_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_1217gp_%(s_epoch)s.h5"   --exact-model True --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/exact_input_em.txt'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_wgan_1217gp_%(s_epoch)s.h5" --check-dis-real False --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_1217gp_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_test_zcut.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/" --name_pb "model_em_wgan_1217gp_%(s_epoch)s.pb" --SavedModel False 
#'''
#out_file = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Testing_em_wgan_gp_batch.sh'
############ stb gan ###################
#template = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Template_Testing_em_stb_gan.sh'
#temp = '''
#/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/generator_stb_gan.py --latent-size 512 --nb-events 13000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_stb_gan_ga0_1206_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_stb_gan_ga0_1206_%(s_epoch)s.h5"   --exact-model False --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/exact_input_em.txt'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_stb_gan_ga0_1206_%(s_epoch)s.h5" --check-dis-real False --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_stb_gan_ga0_1206_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/single_em/mc_Hit_Barrel_em_test_zcut.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/" --name_pb "model_em_stb_gan_ga0_1206_%(s_epoch)s.pb" --SavedModel False 
#'''
#out_file = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Testing_em_stb_gan_batch.sh'
##############lsgan########################
#temp = '''
#/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/generator_v6.py --latent-size 512 --nb-events 13000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_ls0127_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_ls0127_%(s_epoch)s.h5"   --exact-model False --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/exact_input_ep_High.txt'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_ls0127_%(s_epoch)s.h5" --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_ls0127_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_test_zcut.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/" --name_pb "model_ep_High_1107.pb" --SavedModel False 
#'''
##############lsgan and MEA########################
temp = '''
/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/generator_v7.py --latent-size 512 --nb-events 13000 --use_MEA_gen_model True --gen_MEA_model_in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_ls0301_EMA_%(s_epoch)s.h5" --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_ls0301_%(s_epoch)s.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_ls0301_%(s_epoch)s.h5"   --exact-model False --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/exact_input_ep_High.txt'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/output/Gen_em_ls0301_%(s_epoch)s.h5" --check-dis-real True --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_ls0301_%(s_epoch)s.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_test_zcut.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/" --name_pb "model_ep_High_1107.pb" --SavedModel False 
'''
template = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Template_Testing_em.sh'
out_file = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/Testing_em_batch.sh'

f_in = open(template,'r')
lines = f_in.readlines()
f_out = open(out_file,'w')

for line in lines:
    f_out.write(line)

epochs = range(20,200)    
for i in epochs:
    f_out.write(temp % ({'s_epoch':str('epoch%d'%i)}))
    f_out.write('\n')
f_out.close()
print('done for %s'%out_file)
