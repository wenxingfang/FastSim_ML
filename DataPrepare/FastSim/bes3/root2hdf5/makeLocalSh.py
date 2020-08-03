#temp='''
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_ep_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_ep_1114_epoch%(N)d.h5 --event 10000 --tag ep_epoch%(N)d
#'''
#temp='''
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_1114_epoch%(N)d.h5 --event 10000 --tag em_epoch%(N)d
#'''
#temp='''
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/single_em/mc_Hit_Barrel_em_test_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_wgan_1203w_epoch%(N)d.h5 --event 13000 --tag em_wgan_epoch%(N)d
#'''
#temp='''
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_test_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_wgan_1217gp_epoch%(N)d.h5 --event 13000 --tag em_wgan_gp_epoch%(N)d
#'''
#temp='''
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_test_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_stb_gan_1206_epoch%(N)d.h5 --event 13000 --tag em_stb_gan_epoch%(N)d
#'''
#temp='''
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_test_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_sig1125test_epoch%(N)d.h5 --event 13000 --tag em_wgan_epoch%(N)d
#'''
#temp='''
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_test_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_ls0301_epoch%(N)d.h5 --event 13000 --tag em_lsgan_epoch%(N)d
#'''
temp='''
python event_compare_args_v1.py  --useMEA False --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_test_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/output/Gen_em_ls0301_epoch%(N)d.h5 --event 13000 --tag epoch%(N)d
'''

#temp='''
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_ep_zcut_Low.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_ep_Low_1107_epoch%(N)d.h5 --event 5000 --tag epoch%(N)d
#'''
#temp='''
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_ep_zcut_High.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_ep_High_1107_epoch%(N)d.h5 --event 5000 --tag epoch%(N)d
#'''
#temp='''
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_em_zcut_High.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_High_1107_epoch%(N)d.h5 --event 5000 --tag em_High_epoch%(N)d
#'''
#temp='''
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_em_zcut_Low.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_Low_1107_epoch%(N)d.h5 --event 5000 --tag em_Low_epoch%(N)d
#'''
#temp='''
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_wgan_1109_epoch%(N)d.h5 --event 5000 --tag em_wgan_epoch%(N)d
#'''

epochs=range(20,200)

#out_file = '/junofs/users/wxfang/FastSim/bes3/root2hdf5/Local_batch_ep.sh'
#out_file = '/junofs/users/wxfang/FastSim/bes3/root2hdf5/Local_batch_em.sh'
#out_file = '/junofs/users/wxfang/FastSim/bes3/root2hdf5/Local_batch_em_High.sh'
#out_file = '/junofs/users/wxfang/FastSim/bes3/root2hdf5/Local_batch_em_Low.sh'
#out_file = '/junofs/users/wxfang/FastSim/bes3/root2hdf5/Local_batch_em_wgan.sh'
#out_file = '/junofs/users/wxfang/FastSim/bes3/root2hdf5/Local_batch_em_wgan_gp.sh'
#out_file = '/junofs/users/wxfang/FastSim/bes3/root2hdf5/Local_batch_em_stb_gan.sh'
out_file = '/junofs/users/wxfang/FastSim/bes3/root2hdf5/Local_batch_em_lsgan.sh'
f_out = open(out_file,'w')


for i in epochs:
    f_out.write(temp % ({'N':i}))
    f_out.write('\n')
f_out.close()
print('done for %s'%out_file)

