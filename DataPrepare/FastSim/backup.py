import os

source_path = '/junofs/users/wxfang/' 
dedicate_path = 'FastSim'
backup_path = '/afs/ihep.ac.cn/users/w/wxfang/'

backup_type = ["*.py"]

for itype in backup_type:

    os.system('find %s -name "%s" > tmp.txt'%(source_path+'/'+dedicate_path, itype))
    
    with open('tmp.txt') as f:
        for line in f:
            line = line.strip('\n')
            #print('line=',line)
            line1 = line.replace(source_path,backup_path)
            #print('line1=',line1)
            file_name = line1.split('/')[-1]
            path_name = line1.split(file_name)[0]
            if not os.path.exists(path_name):
                os.makedirs(path_name)
            os.system('cp %s %s'%(line, line1))
print('done back up')
