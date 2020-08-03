
temp='''
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/dataset_pi-_test.txt --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/pi-/Gen_0221_epoch%(N)d.h5 --event 0 --tag epoch%(N)d
python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-_fix/HoEless2/h5/Digi_sim_10_pionm_94.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/pi-/Gen_fix_0306_epoch%(N)d.h5 --event 0 --tag sp_epoch%(N)d
'''


epochs=range(50,100)

#out_file = '/junofs/users/wxfang/FastSim/GAN/CEPC/GAN/Local_comp_batch.sh'
out_file = '/junofs/users/wxfang/FastSim/GAN/CEPC/GAN/Local_comp_batch_sp.sh'
f_out = open(out_file,'w')


for i in epochs:
    f_out.write(temp % ({'N':i}))
    f_out.write('\n')
f_out.close()
print('done for %s'%out_file)

