import os
inputs = {}
inputs['/junofs/users/liuyan/muon-sim/sim-nn/data-nn/data-cd-200GeV/20190419/data-1/signal-evt*.root']='/junofs/users/wxfang/FastSim/junows/liuyan/dataset/data_04191.h5' 
inputs['/junofs/users/liuyan/muon-sim/sim-nn/data-nn/data-cd-200GeV/20190419/data-2/signal-evt*.root']='/junofs/users/wxfang/FastSim/junows/liuyan/dataset/data_04192.h5'
inputs['/junofs/users/liuyan/muon-sim/sim-nn/data-nn/data-cd-200GeV/20190421/data-1/signal-evt*.root']='/junofs/users/wxfang/FastSim/junows/liuyan/dataset/data_04211.h5'
inputs['/junofs/users/liuyan/muon-sim/sim-nn/data-nn/data-cd-200GeV/20190421/data-2/signal-evt*.root']='/junofs/users/wxfang/FastSim/junows/liuyan/dataset/data_04212.h5'
inputs['/junofs/users/liuyan/muon-sim/sim-nn/data-nn/data-cd-200GeV/20190422/signal-evt*.root'       ]='/junofs/users/wxfang/FastSim/junows/liuyan/dataset/data_0422.h5'        
inputs['/junofs/users/liuyan/muon-sim/sim-nn/data-nn/data-cd-200GeV/20190425/signal-evt*.root'       ]='/junofs/users/wxfang/FastSim/junows/liuyan/dataset/data_0425.h5'        

batch_size = 5000

for i in inputs:
    out_name = inputs[i].split('/')[-1]
    out_name = out_name.replace('.h5','.sh')
    f=open(out_name,'w')
    f.write('. /junofs/users/wxfang/FastSim/setup_conda.sh\n')
    f.write('conda activate root2hdf5\n')
    str1 = "python root2hdf5_v4.py --input '%s' --output '%s' --batch-size %d"%(i, inputs[i], batch_size) 
    f.write(str1)
    f.close()
    os.system('chmod +x %s'%out_name)
print('done')
