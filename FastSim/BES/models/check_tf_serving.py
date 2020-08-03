#import tensorflow as tf
import numpy as np
import os




if __name__ == '__main__':
    batch_size = 2
    latent_size = 512
    noise = np.random.normal(0, 1, (batch_size, latent_size))
    sampled_mom        = np.random.uniform( 1.7  , 1.8 ,(batch_size, 1))
    sampled_M_dtheta   = np.random.uniform(0.49 , 0.51 ,(batch_size, 1))
    sampled_M_dphi     = np.random.uniform(-5.1 , -5.0 ,(batch_size, 1))/10
    sampled_P_dz       = np.random.uniform(0.59 , 0.61, (batch_size, 1))
    sampled_P_dphi     = np.random.uniform(0.59 , 0.61, (batch_size, 1))
    sampled_info       = np.concatenate((noise, sampled_mom, sampled_M_dtheta, sampled_M_dphi,sampled_P_dz, sampled_P_dphi),axis=-1)
    sampled_info = np.squeeze(sampled_info)
    #print(sampled_info)
    #print('tolist\n,',sampled_info.tolist())
    #tmp = list(sampled_info)
    tmp = []
    for i in range(batch_size):
        tmp.append(list(sampled_info[i]))
    cmd = "curl -d '{"+'"'+"instances"+'"'+": %s}'  -X POST http://10.10.6.229:12345/v1/models/em_Low:predict"%str(tmp)
    #cmd = "curl -d '{"+'"'+"instances"+'"'+": [%s]}'  -X POST http://10.10.6.229:12345/v1/models/em_Low:predict"%str(tmp)
    cmd = cmd.strip('\n')
    print (cmd)
    result = os.system(cmd)
    #result = os.system("curl -d '{"+'"'+"instances"+'"'+": [%s]}'  -X POST http://10.10.6.229:12345/v1/models/em_Low:predict"%str(tmp))
    #result = os.popen(cmd)
    #result = os.popen("curl -d '{"+'"'+"instances"+'"'+": [%s]}'  -X POST http://10.10.6.229:12345/v1/models/em_Low:predict"%str(tmp))
    print (result)
    
#
#    print (result.shape)
