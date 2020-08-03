import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import json
import ast
import argparse
from array import array
rt.gROOT.SetBatch(rt.kTRUE)





def get_parser():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--saveCellIdInfo'       , action='store', type=ast.literal_eval, default=False, help='.')
    parser.add_argument('--saveCellIdInfo-input' , action='store', type=str             , default=''   , help='.')
    parser.add_argument('--saveCellIdInfo-output', action='store', type=str             , default=''   , help='.')
    parser.add_argument('--saveCellIdInfo-Nmin'  , action='store', type=int             , default=0    , help='.')
    parser.add_argument('--saveCellIdInfo-Nmax'  , action='store', type=int             , default=0    , help='.')
    parser.add_argument('--MergeCellInfo'        , action='store', type=ast.literal_eval, default=False, help='.')
    parser.add_argument('--MergeCellInfo-input'  , action='store', type=str             , default=''   , help='.')
    parser.add_argument('--MergeCellInfo-output' , action='store', type=str             , default=''   , help='.')
    parser.add_argument('--saveMcInfo'           , action='store', type=ast.literal_eval, default=False, help='.')
    parser.add_argument('--saveForReco'          , action='store', type=ast.literal_eval, default=False, help='.')

    return parser



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

def getID_v1(x, y, z, lookup_ori):
    lookup = np.copy(lookup_ori)
    lookup[:,1] = lookup[:,1]-x
    lookup[:,2] = lookup[:,2]-y
    lookup[:,3] = lookup[:,3]-z
    dist = lookup[:,1]*lookup[:,1] + lookup[:,2]*lookup[:,2] + lookup[:,3]*lookup[:,3]
    dist_xy = lookup[:,1]*lookup[:,1] + lookup[:,2]*lookup[:,2] 
    min_index = np.argmin(dist)
    min_dist    = math.sqrt(dist   [min_index])
    min_dist_xy = math.sqrt(dist_xy[min_index])
    ID = lookup[min_index,0]
    find_good_hit = 0
    #if math.sqrt(np.min(dist)) < 10 : find_good_hit =1 
    if min_dist < 10 and min_dist_xy <7 : find_good_hit =1 
    return (int(ID), find_good_hit, lookup_ori[min_index,1], lookup_ori[min_index,2], lookup_ori[min_index,3])

if __name__ == '__main__':


    parser = get_parser()
    parse_args = parser.parse_args()
    saveCellIdInfo = parse_args.saveCellIdInfo
    MergeCellInfo  = parse_args.MergeCellInfo
    saveMcInfo     = parse_args.saveMcInfo
    saveForReco    = parse_args.saveForReco
  
    saveCellIdInfo_input  = parse_args.saveCellIdInfo_input
    saveCellIdInfo_output = parse_args.saveCellIdInfo_output
    saveCellIdInfo_Nmin   = parse_args.saveCellIdInfo_Nmin
    saveCellIdInfo_Nmax   = parse_args.saveCellIdInfo_Nmax
    MergeCellInfo_input   = parse_args.MergeCellInfo_input
    MergeCellInfo_output  = parse_args.MergeCellInfo_output
    if saveCellIdInfo:
        np_list=[]
        z_max = 235 #cm
        for z in range(-z_max, z_max+1):
            for phi in range(360):
                np_list.append(np.full((1, 4), 0 ,dtype=np.float32))#init 
        ######### get cellID0 for hit ###############
        #filePath = '/junofs/users/wxfang/FastSim/cepc/ForReco/meta/Dg_cellID_all.root'
        #filePath = '/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/data/Cell_ID_reco_final_orgin_1000.root'
        #filePath = '/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/data/final_mc_info_cellID.root'
        filePath = saveCellIdInfo_input
        #hf = h5py.File('/junofs/users/wxfang/FastSim/cepc/ForReco/cell_ID_orign_1000.h5', 'w')
        hf = h5py.File(saveCellIdInfo_output, 'w')
        treeName='evt'
        chain =rt.TChain(treeName)
        chain.Add(filePath)
        tree = chain
        totalEntries=tree.GetEntries()
        print (totalEntries)
        for entryNum in range(0, tree.GetEntries()):
            if entryNum < saveCellIdInfo_Nmin or entryNum > saveCellIdInfo_Nmax :continue
            tree.GetEntry(entryNum)
            tmp_cellID0   = getattr(tree, "m_Hit_ID0")
            tmp_Hit_x     = getattr(tree, "m_Hit_x")
            tmp_Hit_y     = getattr(tree, "m_Hit_y")
            tmp_Hit_z     = getattr(tree, "m_Hit_z")
            #for the digi hit, the position of hit is the position of the calo cell central
            for i in range(0, len(tmp_Hit_x)):
                id_x = tmp_Hit_x[i]
                id_y = tmp_Hit_y[i]
                id_phi = int(getPhi(id_x, id_y))
                id_z = int(tmp_Hit_z[i]/10)
                if abs(id_z) > z_max: continue
                if tmp_cellID0[i] not in np_list[(id_z+z_max)*360+ id_phi][:,0]:
                    tmp = np.array([[tmp_cellID0[i],id_x,id_y,tmp_Hit_z[i]]])
                    np_list[(id_z+z_max)*360+ id_phi] = np.row_stack((np_list[(id_z+z_max)*360+ id_phi],tmp))
        N_tot =0
        for i in range(len(np_list)):
            N_tot = N_tot + np_list[i].shape[0]-1
            z = int(i/360.0)-z_max 
            #phi = i - 360*z
            phi = i%360
            hf.create_dataset('Hit_info_%d_%d'%(z,phi)   , data=np_list[i])
        hf.close()
        #8*153*2*235*29=16683120
        print('dicti_ID size=',N_tot)
     
    ################# not used ########
    '''
    if doCorrect:
        hf = h5py.File('cell_ID_v3.h5','r')
        out_hf = h5py.File('cell_ID_v3_corr.h5','w')
        for ikey in hf.keys():
            z = int(ikey.split('_')[2])
            phi = ikey.split('_')[-1]
            phi = (float(phi)+360*z)%360
            print('z=',z,',phi=',phi)
            out_hf.create_dataset('Hit_info_%d_%d'%(z,int(phi))   , data=hf[ikey])
        out_hf.close()
        hf.close()
    '''
    
    ''' 
    ################# do merge ########
    if MergeCellInfo:
        print('begin merge')
        f_in = open( MergeCellInfo_input,'r')
        files = f_in.readlines()
        out_hf = h5py.File(MergeCellInfo_output,'w')
        for i in range(len(files)-1):
            f_name      = files[i].replace('\n','')
            f_name_last = files[-1].replace('\n','')
            print('for %s'%(f_name))
            hf1 = h5py.File(f_name , 'r') 
            hf2 = h5py.File(f_name_last, 'r') if i ==0 else out_hf 
            for ikey in hf1.keys():
                df1 = hf1[ikey]
                df2 = hf2[ikey]
                if i==0:
                    df3 = np.row_stack((df1,df2)) 
                    df3 = np.unique(df3, axis=0)
                    out_hf.create_dataset('%s'%(ikey)   , data=df3)
                else:
                    df2 = np.row_stack((df1,df2)) 
                    df2 = np.unique(df2, axis=0)
            hf1.close()
        #    hf2.close()
        out_hf.close()
        print('done merge')
    ''' 

    ################# do merge ########
    if MergeCellInfo:
        print('begin merge')
        f_in = open( MergeCellInfo_input,'r')
        files = f_in.readlines()
        out_hf = h5py.File(MergeCellInfo_output,'w')
        np_dict ={}
        for i in range(len(files)-1):
            f_name      = files[i].replace('\n','')
            f_name_last = files[-1].replace('\n','')
            print('for %s'%(f_name))
            hf1 = h5py.File(f_name , 'r') 
            hf2 = h5py.File(f_name_last, 'r') if i ==0 else np_dict 
            for ikey in hf1.keys():
                df1 = hf1[ikey]
                df2 = hf2[ikey]
                df3 = np.row_stack((df1,df2)) 
                df3 = np.unique(df3, axis=0)
                np_dict[ikey]=df3
            hf1.close()
        #    hf2.close()
        for i in np_dict:
            out_hf.create_dataset('%s'%(i), data=np_dict[i])
        out_hf.close()
        print('done merge')

    ############# mc particle info #######
    if saveMcInfo:
    
        #outFileName = 'mc_info_for_reco.h5'
        outFileName = 'test.h5'
        #filePath = '/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/data/mc_info_and_cell_ID_final_orgin_1000.root'
        filePath = '/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/data/final_mc_info_cellID.root'
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
            print('Evt=',entryNum,'size=',len(tmp_mc_Px))
            if entryNum> 3:break
            if Index >= N_max: 
                print('warning:out of range')
                break
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
                if MC_info[i][0]==0:
                    dele_list.append(i) ## remove the empty event 
            MC_info    = np.delete(MC_info   , dele_list, axis = 0)
        print('final size=', MC_info.shape[0])        
        hf = h5py.File(outFileName, 'w')
        hf.create_dataset('MC_info'   , data=MC_info)
        hf.close()
        print ('Done')
    
    ################# now save results for reco ########
    if saveForReco:
    
        cell_ID_file = h5py.File('/junofs/users/wxfang/FastSim/cepc/ForReco/output/cell_info_all_new.h5','r')
    
        maxn = 2000 # set limit 2000 hits for each event
        nhit = array( 'i',      [-1]  )
        hitx = array( 'f', maxn*[-1.] )
        hity = array( 'f', maxn*[-1.] )
        hitz = array( 'f', maxn*[-1.] )
        hite = array( 'f', maxn*[-1.] )
        hitID= array( 'i', maxn*[-1]  )
        f_out = rt.TFile("/junofs/users/wxfang/FastSim/cepc/ForReco/Hit_info_for_reco_new.root","recreate")
        tree_out = rt.TTree("evt","tree")
        tree_out.Branch("n_hit",   nhit,     "n_hit/I")
        tree_out.Branch("hit_x",   hitx,     "hit_x[n_hit]/F")
        tree_out.Branch("hit_y",   hity,     "hit_y[n_hit]/F")
        tree_out.Branch("hit_z",   hitz,     "hit_z[n_hit]/F")
        tree_out.Branch("hit_e",   hite,     "hit_e[n_hit]/F")
        tree_out.Branch("hit_id",  hitID,    "hit_id[n_hit]/I")
    
        cell_y = 10
        cell_z = 10
        e_threshold = 1e-6
        raw_event_entries = 10000
        Depth = [1850, 1857, 1860, 1868, 1871, 1878, 1881, 1889, 1892, 1899, 1902, 1910, 1913, 1920, 1923, 1931, 1934, 1941, 1944, 1952, 1957, 1967, 1972, 1981, 1986, 1996, 2001, 2011, 2016, 2018]
    
        #inputFileName = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen1007_mc_reco.h5' 
        #inputFileName = '/junofs/users/wxfang/FastSim/cepc/ForReco/Gen1007_mc_reco.h5' 
        inputFileName = '/junofs/users/wxfang/FastSim/cepc/ForReco/Gen1008_mc_reco.h5' 
        hf = h5py.File(inputFileName, 'r')
        Hits    = hf['Barrel_Hit'][:]
        Hits    = np.squeeze(Hits)
        mc_info = hf['MC_info'][:]
        N_total_hit = 0
        N_find_hit = 0
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
                                    N_total_hit = N_total_hit + 1
                                    tmp_hitz=info[ie,3]+(iz-0.5*dim_z)*cell_z
                                    if abs(tmp_hitz) >= 2340: continue
                                    tmp_hitx=Depth[idep]
                                    tmp_hity=info[ie,5]+(iy-0.5*dim_y)*cell_y
                                    rotate_phi =  info[ie,6]
                                    tmp_phi = getPhi(tmp_hitx, tmp_hity)
                                    origin_phi = tmp_phi + rotate_phi 
                                    if origin_phi >= 360 : origin_phi = origin_phi - 360
                                    origin_x = math.sqrt(tmp_hitx*tmp_hitx + tmp_hity*tmp_hity)*math.cos(origin_phi*math.pi/180)
                                    origin_y = math.sqrt(tmp_hitx*tmp_hitx + tmp_hity*tmp_hity)*math.sin(origin_phi*math.pi/180)
                                    key = str('Hit_info_%d_%d'%(int(tmp_hitz/10),origin_phi))
                                    cell_ID = cell_ID_file[key] 
                                    (tmp_id, find_hit, new_x, new_y, new_z) = getID_v1(origin_x, origin_y, tmp_hitz, cell_ID) ##what to do when we have some hit with same ID, should it be merged or ?

                                    #if abs(tmp_hitz) >= 2340 or find_hit==0 : continue
                                    if find_hit==0 : continue
                                    #hitx[index]=origin_x
                                    #hity[index]=origin_y
                                    #hitz[index]=tmp_hitz
                                    hitx[index]=new_x
                                    hity[index]=new_y
                                    hitz[index]=new_z
                                    hite[index]=hit[ie,iy,iz,idep]
                                    hitID[index]= tmp_id
                                    index = index + 1
                                    N_find_hit = N_find_hit + find_hit
                nhit[0]=index
            else:
                nhit[0]=0
            if index >= maxn: print('Warning: hit out of range')
            tree_out.Fill()
        f_out.Write()
        f_out.Close()
        print('Hit find ratio=', N_find_hit/float(N_total_hit) )

'''
    ################# now save results for reco ########
    if saveForReco:
    
        cell_ID_file = h5py.File('/junofs/users/wxfang/FastSim/cepc/ForReco/output/cell_info_all_new.h5','r')
    
        maxn = 2000 # set limit 2000 hits for each event
        nhit = array( 'i',      [-1]  )
        hitx = array( 'f', maxn*[-1.] )
        hity = array( 'f', maxn*[-1.] )
        hitz = array( 'f', maxn*[-1.] )
        hite = array( 'f', maxn*[-1.] )
        hitID= array( 'i', maxn*[-1]  )
        f_out = rt.TFile("Hit_info_for_reco.root","recreate")
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
        raw_event_entries = 10000
        Depth = [1850, 1857, 1860, 1868, 1871, 1878, 1881, 1889, 1892, 1899, 1902, 1910, 1913, 1920, 1923, 1931, 1934, 1941, 1944, 1952, 1957, 1967, 1972, 1981, 1986, 1996, 2001, 2011, 2016, 2018]
    
        #inputFileName = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen1007_mc_reco.h5' 
        inputFileName = '/junofs/users/wxfang/FastSim/cepc/ForReco/Gen1007_mc_reco.h5' 
        hf = h5py.File(inputFileName, 'r')
        Hits    = hf['Barrel_Hit'][:]
        Hits    = np.squeeze(Hits)
        mc_info = hf['MC_info'][:]
        N_total_hit = 0
        N_find_hit = 0
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
                                    if origin_phi >= 360 : origin_phi = origin_phi - 360
                                    origin_x = math.sqrt(tmp_hitx*tmp_hitx + tmp_hity*tmp_hity)*math.cos(origin_phi*math.pi/180)
                                    origin_y = math.sqrt(tmp_hitx*tmp_hitx + tmp_hity*tmp_hity)*math.sin(origin_phi*math.pi/180)
                                    hite[index]=hit[ie,iy,iz,idep]
                                    hitx[index]=origin_x
                                    hity[index]=origin_y
                                    key = str('Hit_info_%d_%d'%(int(hitz[index]/10),origin_phi))
                                    cell_ID = cell_ID_file[key] 
                                    #hitID[index]= getID(hitx[index], hity[index], hitz[index], cell_ID) ##what to do when we have some hit with same ID, should it be merged or ?
                                    (hitID[index], find_hit) = getID_v1(hitx[index], hity[index], hitz[index], cell_ID) ##what to do when we have some hit with same ID, should it be merged or ?
                                    index = index + 1
                                    N_total_hit = N_total_hit + 1
                                    N_find_hit = N_find_hit + find_hit
                nhit[0]=index
            else:
                nhit[0]=0
            if index >= maxn: print('Warning: hit out of range')
            tree_out.Fill()
        f_out.Write()
        f_out.Close()
        print('Hit find ratio=', N_find_hit/float(N_total_hit) )

'''
