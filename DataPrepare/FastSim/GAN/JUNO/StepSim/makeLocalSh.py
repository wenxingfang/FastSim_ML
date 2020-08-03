temp='''
python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_1D/laser_batch8_N5000.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/JUNO/generator/Gen_1208_100_epoch%(N)d.h5 --event 5000 --tag epoch%(N)d
'''

epochs=range(50,200)

out_file = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/Local_batch.sh'
f_out = open(out_file,'w')


for i in epochs:
    f_out.write(temp % ({'N':i}))
    f_out.write('\n')
f_out.close()
print('done for %s'%out_file)

