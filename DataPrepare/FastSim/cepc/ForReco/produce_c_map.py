
import ROOT as rt
import h5py 

if True:

    f_map=open('S_M_I_J_K_ID.txt','w')
    filePath = '/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/ID.root'
    treeName='evt'
    chain =rt.TChain(treeName)
    chain.Add(filePath)
    tree = chain
    totalEntries=tree.GetEntries()
    print (totalEntries)
    id_list = []
    for entryNum in range(0, tree.GetEntries()):
        tree.GetEntry(entryNum)
        if entryNum%10000 == 0 : print (entryNum)
        tmp_cellID0   = getattr(tree, "m_Hit_ID0")
        tmp_ID_S      = getattr(tree, "m_ID_S")
        tmp_ID_M      = getattr(tree, "m_ID_M")
        tmp_ID_I      = getattr(tree, "m_ID_I")
        tmp_ID_J      = getattr(tree, "m_ID_J")
        tmp_ID_K      = getattr(tree, "m_ID_K")
        tmp_Hit_x     = getattr(tree, "m_Hit_x")
        tmp_Hit_y     = getattr(tree, "m_Hit_y")
        tmp_Hit_z     = getattr(tree, "m_Hit_z")
        for i in range(len(tmp_cellID0)):
            if tmp_cellID0[i] in id_list:continue
            id_list.append(tmp_cellID0[i])
            f_map.write('%d %d %d %d %d %d \n'%(int(tmp_ID_S[i]),int(tmp_ID_M[i]),int(tmp_ID_I[i]),int(tmp_ID_J[i]),int(tmp_ID_K[i]),tmp_cellID0[i]))
    f_map.close()
    print('done')            
      





if False:

    f_map=open('x_y_z_ID.txt','w')
    hf = h5py.File('cell_ID_merged.h5','r')
    for ikey in hf.keys():
        df = hf[ikey][:]
        for i in range(df.shape[0]):
            ID =     df[i,0]
            if ID==0: continue
            x  = int(df[i,1])
            y  = int(df[i,2])
            z  = int(df[i,3])
            f_map.write('%d %d %d %d \n'%(x,y,z,ID))
    hf.close()
    f_map.close()

if False:

    f_map=open('x_y_z_ID.cpp','w')
    f_map.write('#include <map> \n')
    f_map.write('using namespace std; \n')
    #f_map.write('void MakeMapXYZID( map<int, map<int, map<int, int>  > > &  Map){ \n')
    f_map.write('void MakeMapXYZID( map<int, pair<int, pair<int, int>  > > &  Map){ \n')
    
    hf = h5py.File('cell_ID_merged.h5','r')
    for ikey in hf.keys():
        df = hf[ikey][:]
        for i in range(df.shape[0]):
            ID =     df[i,0]
            if ID==0: continue
            x  = int(df[i,1])
            y  = int(df[i,2])
            z  = int(df[i,3])
            #f_map.write('Map[%d][%d][%d]=%d ; \n'%(x,y,z,ID))
            f_map.write('Map.insert(make_pair(%d,make_pair(%d,make_pair(%d,%d)))); \n'%(x,y,z,ID))
    f_map.write('} \n')
    hf.close()
    f_map.close()


if False:

    f_map=open('x_y_z_ID_hash.cpp','w')
    f_map.write('#include <unordered_map> \n')
    f_map.write('#include <string> \n')
    f_map.write('using namespace std; \n')
    f_map.write('void MakeMapXYZID( unordered_map<string, int> &  Map){ \n')
    
    hf = h5py.File('cell_ID_merged.h5','r')
    for ikey in hf.keys():
        df = hf[ikey][:]
        for i in range(df.shape[0]):
            ID =     df[i,0]
            if ID==0: continue
            x  = int(df[i,1])
            y  = int(df[i,2])
            z  = int(df[i,3])
            f_map.write('Map["X%dY%sZ%d"]=%d ; \n'%(x,y,z,ID))
    f_map.write('} \n')
    hf.close()
    f_map.close()


print('done')



