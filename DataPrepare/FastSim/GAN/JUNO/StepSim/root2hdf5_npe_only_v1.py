import ROOT as rt
import numpy as np
import h5py
import math
import sys
import argparse
# npe will be save one dataframe for each r,theta
def get_parser():
    parser = argparse.ArgumentParser(
        description='Produce training samples for JUNO study. ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--batch-size', action='store', type=int, default=5000,
                        help='Number of event for each batch.')
    parser.add_argument('--input', action='store', type=str, default='',
                        help='input root file.')
    parser.add_argument('--output', action='store', type=str, default='',
                        help='output hdf5 file.')
    parser.add_argument('--X', action='store', type=float, default=0,
                        help='X of source.')
    parser.add_argument('--Y', action='store', type=float, default=0,
                        help='Y of source.')
    parser.add_argument('--Z', action='store', type=float, default=0,
                        help='Z of source.')

    return parser

def root2hdf5 (Max_event, tree, start_event, out_name, id_dict, s_r):

    ls_dict_npe = {}
    for i in id_dict:
        if id_dict[i] in ls_dict_npe: continue
        ls_dict_npe[id_dict[i]] = []


    for ie in range(start_event, start_event+Max_event):
        tree.GetEntry(ie)
        tmp_dict = {}
        entryNum  = ie - start_event
        pmtID     = getattr(tree, "pmtID")
        for i in range(0, len(pmtID)):
            ID     = pmtID[i]
            if ID not in id_dict:continue
            if ID not in tmp_dict:
                tmp_dict[ID] = 1
            else:
                tmp_dict[ID] = tmp_dict[ID] + 1
        for i in id_dict:
            if i in tmp_dict:
                ls_dict_npe[id_dict[i]].append(tmp_dict[i]) 
            else:
                ls_dict_npe[id_dict[i]].append(0) 
         
    hf = h5py.File(out_name, 'w')
    s_r = round(s_r,3) # keep three digi
    for i in ls_dict_npe:
        hf.create_dataset('nPEByPMT_%s_%s'%(str(s_r), str(i)), data=np.array(ls_dict_npe[i]) )
    hf.close()
    print('saved %s'%out_name)




def get_pmt_theta_phi(file_pos, sep, i_id, i_theta, i_phi):
    id_dict = {}
    theta_list = []
    phi_list = []
    f = open(file_pos,'r')
    lines = f.readlines()
    for line in lines:
        #items = line.split(sep)
        items = line.split()
        #print(items)
        ID    = float(items[i_id])
        ID    = int(ID)
        theta = float(items[i_theta])
        phi   = float(items[i_phi])
        phi   = int(phi) ## otherwise it will be too much
        if theta not in theta_list:
            theta_list.append(theta)
        if phi not in phi_list:
            phi_list.append(phi)
        if ID not in id_dict:
            id_dict[ID]=[theta, phi]
    return (id_dict, theta_list, phi_list)



if __name__ == '__main__':

    large_PMT_pos = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/PMT_pos/PMTPos_Acrylic_with_chimney.csv' 
    (L_id_dict, L_theta_list, L_phi_list) = get_pmt_theta_phi(large_PMT_pos, '', 0, 4, 5)
    print('L theta size=',len(L_theta_list),',L phi size=',len(L_phi_list))
    small_PMT_pos = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/PMT_pos/3inch_pos.csv' 
    (S_id_dict, S_theta_list, S_phi_list) = get_pmt_theta_phi(small_PMT_pos, '', 0, 1, 2)
    print('S theta size=',len(S_theta_list),',S phi size=',len(S_phi_list))
    #all_theta_list = list(set(L_theta_list + S_theta_list))
    #all_phi_list   = list(set(L_phi_list   + S_phi_list  ))
    #all_theta_list = list(set(L_theta_list))
    #all_phi_list   = list(set(L_phi_list  ))
    #nRows = len(all_theta_list)
    #nCols = len(all_phi_list)
    #print('theta size=',nRows,',phi size=',nCols)
    
    #id_list = list(set(list(L_id_dict.keys())+list(S_id_dict.keys())))
    #id_list.sort()
    #print('id size=',len(id_list))

    #all_theta_list.sort()## 0 - 180
    #all_phi_list.sort()  ## 0 - 360
    #L_id_dict.update(S_id_dict)
    #ID_index_theta_phi = {}
    #for i in L_id_dict:
    #    index_theta = all_theta_list.index(L_id_dict[i][0])
    #    index_phi   = all_phi_list  .index(L_id_dict[i][1])
    #    ID_index_theta_phi[i] = [index_theta, index_phi]
    ###########################################################
    parser = get_parser()
    parse_args = parser.parse_args()

    Max_event = parse_args.batch_size
    filePath = parse_args.input
    outFileName= parse_args.output
    S_X = parse_args.X/1000
    S_Y = parse_args.Y/1000
    S_Z = parse_args.Z/1000
    S_R = math.sqrt(S_X*S_X + S_Y*S_Y + S_Z*S_Z) # from mm TO m
    #S_theta = math.acos(S_Z/S_R)*180/math.pi # from 0 to 180
    L_id_dict_rel = {}
    fake_r = 20
    for i in L_id_dict:
        itheta = L_id_dict[i][0]
        iphi   = L_id_dict[i][1]
        iz = fake_r*math.cos(itheta*math.pi/180)
        ix = fake_r*math.sin(itheta*math.pi/180)*math.cos(iphi*math.pi/180)
        iy = fake_r*math.sin(itheta*math.pi/180)*math.sin(iphi*math.pi/180)
        rel_cos_theta =  (S_X*ix + S_Y*iy + S_Z*iz)/(S_R*fake_r)       
        rel_theta = math.acos(rel_cos_theta)*180/math.pi # from 0 to 180
        L_id_dict_rel[i] = round(rel_theta,1) # keep one digi, to reduce a bit the different thetas

    treeName='evt'
    chain =rt.TChain(treeName)
    chain.Add(filePath)
    tree = chain
    totalEntries=tree.GetEntries()
    Max_event = totalEntries
    batch = int(float(totalEntries)/Max_event)
    print ('total=%d, max=%d, batch=%d, last=%d'%(totalEntries, Max_event, batch, totalEntries%Max_event))
    start = 0
    for i in range(batch):
        out_name = outFileName.replace('.h5','_batch%d_N%d.h5'%(i, Max_event))
        root2hdf5 (Max_event, tree, start, out_name, L_id_dict_rel, S_R)
        #start = start + Max_event
        #if i == batch-1:
        #    out_name = outFileName.replace('.h5','_batch%d_N%d.h5'%(i+1,totalEntries%Max_event))
        #    root2hdf5 (totalEntries%Max_event, tree, start, out_name, L_id_dict)
    
        
    print('done')  
