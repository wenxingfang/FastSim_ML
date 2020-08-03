import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import json
from array import array
rt.gROOT.SetBatch(rt.kTRUE)

saveCellIdInfo = True

saveMcInfo = False

saveForReco = False

def getPhi(x, y):
    if x == 0 and y == 0: return 0
    elif x == 0 and y > 0: return 90
    elif x == 0 and y < 0: return 270
    phi = math.atan(y/x)
    phi = 180*phi/math.pi
    if x < 0 : phi = phi + 180
    elif x > 0 and y < 0 : phi = phi + 360
    return phi

def getID(x, y, z, lookup):
    tmp_ID = 0
    id_z = int(z/10)
    id_z = str(id_z)
    id_phi = int(getPhi(x, y))
    id_phi = str(id_phi)
    if id_z not in lookup:
        print('exception id_z=', id_z)
        return tmp_ID
    min_distance = 999
    for ID in lookup[id_z][id_phi]:
        c_x = float(lookup[id_z][id_phi][ID][0])
        c_y = float(lookup[id_z][id_phi][ID][1])
        c_z = float(lookup[id_z][id_phi][ID][2])
        distance = math.sqrt( math.pow(x-c_x,2) + math.pow(y-c_y,2) + math.pow(z-c_z,2) )
        if  distance < min_distance :
            min_distance = distance
            tmp_ID = ID
    return int(tmp_ID) 


if saveCellIdInfo:
    ######### get cellID0 for hit ###############
    filePath = '/junofs/users/wxfang/FastSim/cepc/ForReco/meta/Dg_cellID_all.root'
    treeName='evt'
    chain =rt.TChain(treeName)
    chain.Add(filePath)
    tree = chain
    totalEntries=tree.GetEntries()
    print (totalEntries)
    dicti_ID  ={}
    dicti_ID_x={}
    dicti_ID_y={}
    dicti_ID_z={}
    N_max = 1000000
    for entryNum in range(0, tree.GetEntries()):
    #    if entryNum >= N_max: break
        tree.GetEntry(entryNum)
        tmp_cellID0   = getattr(tree, "m_Hit_ID0")
        tmp_Hit_x     = getattr(tree, "m_Hit_x")
        tmp_Hit_y     = getattr(tree, "m_Hit_y")
        tmp_Hit_z     = getattr(tree, "m_Hit_z")
        '''
        for the digi hit, the position of hit is the position of the calo cell central
        '''
        for i in range(0, len(tmp_Hit_x)):
            if tmp_cellID0[i] not in dicti_ID:
                dicti_ID[tmp_cellID0[i]]=[tmp_Hit_x[i], tmp_Hit_y[i], tmp_Hit_z[i]]
    print('dicti_ID size=',len(dicti_ID))
    '''
    8*153*2*235*29=16683120
    '''
    table = {}
    for z in range(-235,235,1):
        table[z]={}
        for phi in range(0, 360):
            table[z][phi]={}
    for ID in dicti_ID:
        id_x = dicti_ID[ID][0]
        id_y = dicti_ID[ID][1]
        id_phi = int(getPhi(id_x, id_y))
        id_z = int(dicti_ID[ID][2]/10)
        if id_z not in table:
            print('exception z=', id_z)
            continue
        table[id_z][id_phi][ID]=[id_x, id_y, dicti_ID[ID][2]]
    json_str = json.dumps(table)
    with open('/junofs/users/wxfang/FastSim/cepc/ForReco/cell_ID_v1.json', 'w') as json_file:
        json_file.write(json_str)


############# mc particle info #######
if saveMcInfo:


    outFileName = 'mc_info_for_reco.h5'
    filePath = './meta/mc_e1e1h_e2e2_1000.root'
    treeName='evt'
    chain =rt.TChain(treeName)
    chain.Add(filePath)
    tree = chain
    totalEntries=tree.GetEntries()
    print (totalEntries)
    N_max = 3*totalEntries
    print ('N max:',N_max)
    MC_info    = np.full((N_max, 8 ), 0 ,dtype=np.float32)#init 
    Index = 0
    for entryNum in range(0, tree.GetEntries()):
        tree.GetEntry(entryNum)
        tmp_mc_Px   = getattr(tree, "m_mc_Px")
        tmp_mc_Py   = getattr(tree, "m_mc_Py")
        tmp_mc_Pz   = getattr(tree, "m_mc_Pz")
        tmp_HitFirst_x   = getattr(tree, "m_mc_pHitx")
        tmp_HitFirst_y   = getattr(tree, "m_mc_pHity")
        tmp_HitFirst_z   = getattr(tree, "m_mc_pHitz")
        tmp_HitFirst_vtheta   = getattr(tree, "m_mc_pHit_theta")
        tmp_HitFirst_vphi     = getattr(tree, "m_mc_pHit_phi")
        tmp_phi_rotated  = getattr(tree, "m_mc_pHit_rotated")
        if Index >= N_max: break
        for i in range(len(tmp_mc_Px)):
            MC_info[Index][0] = math.sqrt(tmp_mc_Px[i]*tmp_mc_Px[i] + tmp_mc_Py[i]*tmp_mc_Py[i] + tmp_mc_Pz[i]*tmp_mc_Pz[i])
            MC_info[Index][1] = tmp_HitFirst_vtheta[i]
            MC_info[Index][2] = tmp_HitFirst_vphi  [i]
            MC_info[Index][3] = tmp_HitFirst_z     [i]
            MC_info[Index][4] = tmp_HitFirst_x     [i]
            MC_info[Index][5] = tmp_HitFirst_y     [i]
            MC_info[Index][6] = tmp_phi_rotated    [i]
            MC_info[Index][7] = entryNum
            Index = Index + 1
    if True:
        dele_list = []
        for i in range(MC_info.shape[0]):
            if MC_info[i][7]==0:
                dele_list.append(i) ## remove the empty event 
        MC_info    = np.delete(MC_info   , dele_list, axis = 0)
    print('final size=', MC_info.shape[0])        
    hf = h5py.File(outFileName, 'w')
    hf.create_dataset('MC_info'   , data=MC_info)
    hf.close()
    print ('Done')

################# now save results for reco ########
if saveForReco:

    cell_ID_file = '/junofs/users/wxfang/FastSim/cepc/ForReco/cell_ID.json'
    cell_ID  = json.load(open(cell_ID_file,'r'))

    maxn = 2000
    nhit = array( 'i',      [-1]  )
    hitx = array( 'f', maxn*[-1.] )
    hity = array( 'f', maxn*[-1.] )
    hitz = array( 'f', maxn*[-1.] )
    hite = array( 'f', maxn*[-1.] )
    hitID= array( 'i', maxn*[-1]  )
    f_out = rt.TFile("test.root","recreate")
    tree_out = rt.TTree("evt","tree")
    tree_out.Branch("n_hit",   nhit,     "n_hit/I")
    tree_out.Branch("hit_x",   hitx,     "hit_x[n_hit]/F")
    tree_out.Branch("hit_y",   hity,     "hit_y[n_hit]/F")
    tree_out.Branch("hit_z",   hitz,     "hit_z[n_hit]/F")
    tree_out.Branch("hit_e",   hite,     "hit_e[n_hit]/F")
    tree_out.Branch("hit_id",  hitID,    "hit_id[n_hit]/I")

    cell_y = 10
    cell_z = 10
    e_threshold = 0
    raw_event_entries = 1000
    Depth = [1850, 1857, 1860, 1868, 1871, 1878, 1881, 1889, 1892, 1899, 1902, 1910, 1913, 1920, 1923, 1931, 1934, 1941, 1944, 1952, 1957, 1967, 1972, 1981, 1986, 1996, 2001, 2011, 2016, 2018]
    inputFileName = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen0928_mc_reco.h5' 
    hf = h5py.File(inputFileName, 'r')
    Hits    = hf['Barrel_Hit'][:]
    Hits    = np.squeeze(Hits)
    mc_info = hf['MC_info'][:]
    for i in range(raw_event_entries):
        print('event=',i)
        index = 0
        if i in mc_info[:,7]: 
            #print ('i=',i,',mc info=',mc_info[mc_info[:,7]==i])
            hit = Hits   [mc_info[:,7]==i]
            info= mc_info[mc_info[:,7]==i]
            n_evt = hit.shape[0]
            dim_y = hit.shape[1]
            dim_z = hit.shape[2]
            dim_dep = hit.shape[3]
            
            for ie in range(n_evt):
                if index >= maxn: break 
                for iy in range(dim_y):    
                    if index >= maxn: break 
                    for iz in range(dim_z):    
                        if index >= maxn: break 
                        for idep in range(dim_dep): 
                            if hit[ie,iy,iz,idep] > e_threshold : 
                                if index >= maxn: break 
                                hitz[index]=info[ie,3]+(iz-0.5*dim_z)*cell_z
                                if abs(hitz[index]) >= 2340: continue
                                tmp_hitx=Depth[idep]
                                tmp_hity=info[ie,5]+(iy-0.5*dim_y)*cell_y
                                rotate_phi =  info[ie,6]
                                tmp_phi = getPhi(tmp_hitx, tmp_hity)
                                origin_phi = tmp_phi + rotate_phi
                                origin_x = math.sqrt(tmp_hitx*tmp_hitx + tmp_hity*tmp_hity)*math.cos(origin_phi*math.pi/180)
                                origin_y = math.sqrt(tmp_hitx*tmp_hitx + tmp_hity*tmp_hity)*math.sin(origin_phi*math.pi/180)
                                hite[index]=hit[ie,iy,iz,idep]
                                hitx[index]=origin_x
                                hity[index]=origin_y
                                hitID[index]= getID(hitx[index], hity[index], hitz[index], cell_ID)
                                index = index + 1
            nhit[0]=index
        else:
            nhit[0]=0
        tree_out.Fill()
    f_out.Write()
    f_out.Close()
    '''
    f_out = rt.TFile("test.root","recreate")
    tree_out = rt.TTree("evt","tree")
    maxn = 10
    n = array( 'i', [0] )
    p = array( 'f', maxn*[0.] )
    
    tree_out.Branch("myn",   n,     "myn/I")
    tree_out.Branch("myp",   p,     "myp[myn]/F")
    
    for i in range(10):
        n[0] = i
        for j in range(10):
            p[j]=j+0.1
        tree_out.Fill()
    f_out.Write()
    f_out.Close()

    '''


    '''
    f = rt.TFile( 'test.root', 'recreate' )
    t = rt.TTree( 't1', 'tree with histos' )
     
    maxn = 10
    n = array( 'i', [ 0 ] )
    d = array( 'f', maxn*[ 0. ] )
    t.Branch( 'mynum', n, 'mynum/I' )
    t.Branch( 'myval', d, 'myval[mynum]/F' )
     
    for i in range(25):
       n[0] = min(i,maxn)
       for j in range(n[0]):
          d[j] = i*0.1+j
       t.Fill()
     
    f.Write()
    f.Close()
    '''
