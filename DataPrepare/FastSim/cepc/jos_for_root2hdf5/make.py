import os


tmp = '''
source /junofs/users/wxfang/FastSim/setup_conda.sh
python /junofs/users/wxfang/FastSim/cepc/root2hdf5_v6.py --input %(input)s --output %(output)s --tag %(tag)s --str_particle '%(particle)s'
echo "done"
'''

Dicts = {}

#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_1.root']    = [ 'gamma_1'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_2.root']    = [ 'gamma_2'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_3.root']    = [ 'gamma_3'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_4.root']    = [ 'gamma_4'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_5.root']    = [ 'gamma_5'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_6.root']    = [ 'gamma_6'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_7.root']    = [ 'gamma_7'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_8.root']    = [ 'gamma_8'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_9.root']    = [ 'gamma_9'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_ext1.root'] = [ 'gamma_ext1'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_ext2.root'] = [ 'gamma_ext2'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_ext3.root'] = [ 'gamma_ext3'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_ext4.root'] = [ 'gamma_ext4'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_ext5.root'] = [ 'gamma_ext5'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_ext6.root'] = [ 'gamma_ext6'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_ext7.root'] = [ 'gamma_ext7'    ,'#gamma']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_ext8.root'] = [ 'gamma_ext8'    ,'#gamma']
Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/gamma/gamma_ext9.root'] = [ 'gamma_ext9'    ,'#gamma']

#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/e-/em_0.root'] =[ 'em_0','e^{-}']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/e-/em_1.root'] =[ 'em_1','e^{-}']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/e-/em_2.root'] =[ 'em_2','e^{-}']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/e-/em_3.root'] =[ 'em_3','e^{-}']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/e-/em_4.root'] =[ 'em_4','e^{-}']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/e-/em_5.root'] =[ 'em_5','e^{-}']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/e-/em_6.root'] =[ 'em_6','e^{-}']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/e-/em_7.root'] =[ 'em_7','e^{-}']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/e-/em_8.root'] =[ 'em_8','e^{-}']
#Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/e-/em_9.root'] =[ 'em_9','e^{-}']
Dicts['/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/merged_data/e-/em_10.root']=['em_10','e^{-}']


index = 0
for i in Dicts:
    file_name = i.split('/')[-1]
    file_name = file_name.replace('.root','.h5')
    str_output = '/junofs/users/wxfang/FastSim/cepc/h5data/%s'%file_name 
    out_name = str('job_%d.sh'%index)
    f = open(out_name,'w')
    #f.write(tmp%({'input':i, 'output':str_output, 'tag':Dicts[i]}))
    f.write(tmp%({'input':i, 'output':str_output, 'tag':Dicts[i][0], 'particle':Dicts[i][1]}))
    f.close()
    index = index + 1
os.system('chmod +x *.sh')
print('done')
