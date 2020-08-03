

temp = '''
/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/reco_test_v4.py --datafile %(m_input)s --model-in '%(m_model)s' --output %(m_output)s
'''
template = './Template_Testing_reco_pi-.txt'
out_file = './Testing_reco_pi-_batch.sh'
input_data = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/dataset_pi-_test.txt'
output_path = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL/'
input_model = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/params/reco/pi-/reco_model_0217.h5'

f_in = open(template,'r')
lines = f_in.readlines()
f_in.close()
f_out = open(out_file,'w')

for line in lines:
    f_out.write(line)

f_data = open(input_data,'r')
datas = f_data.readlines()
f_data.close()
for data in datas:
    data = data.replace('\n','')
    if "#" in data:continue
    str1 = data.split('/')[-1]
    str2 = str1.replace('.h5','_pred.h5') 
    f_out.write(temp % ({'m_input':data, 'm_output':str(output_path+'/'+str2), 'm_model':input_model}))
    f_out.write('\n')

f_out.close()
print('done for %s'%out_file)
