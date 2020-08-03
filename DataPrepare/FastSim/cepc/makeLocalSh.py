
#temp='''
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/e-/em_10.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen_em_1105_epoch%(N)d.h5 --event 5000 --tag epoch%(N)d
#'''

temp='''
python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/gamma/gamma_ext9.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen_gamma_1105_epoch%(N)d.h5 --event 4000 --tag epoch%(N)d
'''


epochs=range(71,72)

out_file = '/junofs/users/wxfang/FastSim/cepc/Local_batch.sh'
f_out = open(out_file,'w')


for i in epochs:
    f_out.write(temp % ({'N':i}))
    f_out.write('\n')
f_out.close()
print('done')

