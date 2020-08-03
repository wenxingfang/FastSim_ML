#import matplotlib
#import numpy as np
#import matplotlib.pyplot as plt
def pmt():
    import math
    id_x_y = {}
    f = open('/junofs/users/wxfang/FastSim/junows/pmt_sorted.txt','r')
    f_out = open('/junofs/users/wxfang/FastSim/junows/liuyan/pmt_id_x_y.txt','w')
    num=0
    total=0
    result=[[-1 for col in  range(360)] for row in range(180)]
    for line in f:
        total=total+1
        (pid,z,phi) = line.split()
        z=float(z)
        phi=float(phi)
        z=math.acos(z/19500)    
        x=int(round(z*180/math.pi))
        y=int(round(phi*180/math.pi))
        
        if x==180:
            x=0
        #if result[x][y]!=-1:
        #    num=num+1
        result[x][y]=int(pid)
        id_x_y[int(pid)]=(x,y)
        f_out.write("%d %d %d\n"%(int(pid), x, y))
    f_out.close()
    '''
    for the in range(180):
            for phi in range(360):            
                if result[the][phi]>-0.5:
                    num=num+1
    print(num)
    '''
    return (result,id_x_y)  

if __name__ == '__main__':
    pmt()
