import os


tmp = '''
source /junofs/users/wxfang/FastSim/setup_conda.sh
python /junofs/users/wxfang/FastSim/cepc/ForReco/process_v3.py --saveCellIdInfo True  --saveCellIdInfo-input %(input)s --saveCellIdInfo-output %(output)s --saveCellIdInfo-Nmin %(Nmin)d --saveCellIdInfo-Nmax %(Nmax)d
'''

out_put_path = '/junofs/users/wxfang/FastSim/cepc/ForReco/output/cell_info.h5' 

Dicts = {}

Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/data/final_mc_info_cellID.root'] = range(0, 6000, 1000)
Dicts['/junofs/users/wxfang/FastSim/cepc/ForReco/meta/Dg_cellID_all.root'] = range(0, 100000, 1000)



index = 0
for i in Dicts:
    str_inpput = i
    for j in range(len(Dicts[i])-1):
        Nmin = Dicts[i][j]
        Nmax = Dicts[i][j+1]
        str_output = out_put_path.replace('cell_info','cell_info_%d'%index)
        out_name = str('job_%d.sh'%index)
        f = open(out_name,'w')
        f.write(tmp%({'input':i, 'output':str_output, 'Nmin':Nmin, 'Nmax': Nmax}))
        f.close()
        index = index + 1
os.system('chmod +x *.sh')
print('done')
