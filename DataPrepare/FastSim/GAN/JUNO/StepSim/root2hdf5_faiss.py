import ROOT as rt
import numpy as np
import h5py
import math
import random
import sys
import os
import copy
import ast
import time
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
    parser.add_argument('--forNpe', action='store', type=ast.literal_eval, default=False,
                        help='forNpe')
    parser.add_argument('--forHitTime', action='store', type=ast.literal_eval, default=False,
                        help='forHitTime')

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
    f = open(file_pos,'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        #items = line.split(sep)
        items = line.split()
        #print(items)
        ID    = int(items[i_id])
        theta = float(items[i_theta])
        phi   = float(items[i_phi])
        if ID not in id_dict:
            id_dict[ID]=[theta, phi]
    return id_dict



def make_data(fpath, file_list, scale, forHitTime, forNpe, outFileName):
    treeName='evt'
    tot_evt = 0
    for ifile in file_list:
        tfile = rt.TFile('%s/%s'%(fpath, ifile),'read')
        tree = tfile.Get(treeName)
        tot_evt += tree.GetEntries()
        tfile.Close()
    tot_item = int(len(L_id_dict)*tot_evt/scale);
    print('tot evt=',tot_evt,',tot_item=',tot_item)
    if forHitTime:
        print('for Hit time')
        #hit_times = np.full((tot_item,6), 0 ,dtype=np.float32) 
        hit_times = np.full((tot_item,6), 0 ,dtype=np.float16)# seems enough 
        N_filled = 0 
        for ifile in file_list:
            print('processed %f %%'%(100.0*file_list.index(ifile)/len(file_list)))
            tfile = rt.TFile('%s/%s'%(fpath, ifile),'read')
            tree = tfile.Get(treeName)
            src_X = float(ifile.split('_')[4])
            src_Y = float(ifile.split('_')[5])
            str_Z = (ifile.split('_')[6])
            src_Z = float(str_Z.split('.root')[0])
            src_R = math.sqrt(src_X*src_X + src_Y*src_Y + src_Z*src_Z)
            src_Rt = math.sqrt(src_X*src_X + src_Y*src_Y)
            src_theta = math.acos(src_Z/src_R)*180/math.pi # from 0 to 180
            src_phi   = math.acos(src_X/src_Rt)*180/math.pi if src_Y > 0 else 360 - math.acos(src_X/src_Rt)*180/math.pi # from 0 to 360
            for i in range(tree.GetEntries()):
                tree.GetEntry(i)
                pmtID     = getattr(tree, "pmtID")
                hitTime   = getattr(tree, "hitTime")
                for j in range(len(pmtID)):
                    if pmtID[j] not in L_id_dict:continue
                    hit_times[N_filled,0] = src_R/17700   #normalize
                    hit_times[N_filled,1] = src_theta/180 #normalize
                    hit_times[N_filled,2] = src_phi/360   #normalize
                    hit_times[N_filled,3] = L_id_dict[pmtID[j]][0]/180   #normalize
                    hit_times[N_filled,4] = L_id_dict[pmtID[j]][1]/360   #normalize
                    hit_times[N_filled,5] = hitTime[j]/100   #normalize
                    N_filled += 1
                    if N_filled >= tot_item: break
                if N_filled >= tot_item: break
            tfile.Close()
            if N_filled >= tot_item: break
            
        print('hit_times size=',hit_times.shape[0])
        if True:
            dele_list = []
            for i in range(hit_times.shape[0]):
                if hit_times[i][5] ==0 :
                    dele_list.append(i) ## remove the event has zero hit time
            hit_times            = np.delete(hit_times           , dele_list, axis = 0)
        print('final hit_times size=',hit_times.shape[0])
        hf = h5py.File(outFileName, 'w')
        hf.create_dataset('data', data=hit_times)
        hf.close()
        print('saved ',outFileName)

    if forNpe:
        print('for NEP')
        npe_dict={}
        for ID in L_id_dict:
            npe_dict[ID]=0
        npes = np.full((tot_item,6), 0 ,dtype=np.float16)
        N_filled = 0 
        for ifile in file_list:
            print('processed %f %%'%(100.0*file_list.index(ifile)/len(file_list)))
            src_X = float(ifile.split('_')[4])
            src_Y = float(ifile.split('_')[5])
            str_Z = (ifile.split('_')[6])
            src_Z = float(str_Z.split('.root')[0])
            src_R = math.sqrt(src_X*src_X + src_Y*src_Y + src_Z*src_Z)
            src_Rt = math.sqrt(src_X*src_X + src_Y*src_Y)
            src_theta = math.acos(src_Z/src_R)*180/math.pi # from 0 to 180
            src_phi   = math.acos(src_X/src_Rt)*180/math.pi if src_Y > 0 else 360 - math.acos(src_X/src_Rt)*180/math.pi # from 0 to 360
            tfile = rt.TFile('%s/%s'%(fpath, ifile),'read')
            tree = tfile.Get(treeName)
            for i in range(tree.GetEntries()):
                tree.GetEntry(i)
                pmtID     = getattr(tree, "pmtID")
                tmp_dict = copy.deepcopy(npe_dict)
                for j in range(len(pmtID)):
                    if pmtID[j] in tmp_dict:
                        tmp_dict[pmtID[j]] += 1
                for ID in tmp_dict:
                    npes[N_filled,0] = src_R/17700   #normalize
                    npes[N_filled,1] = src_theta/180 #normalize
                    npes[N_filled,2] = src_phi/360   #normalize
                    npes[N_filled,3] = L_id_dict[ID][0]/180   #normalize
                    npes[N_filled,4] = L_id_dict[ID][1]/360   #normalize
                    npes[N_filled,5] = tmp_dict [ID] 
                    N_filled += 1
                    if N_filled >= tot_item: break
                if N_filled >= tot_item: break
            tfile.Close()
            if N_filled >= tot_item: break
        hf = h5py.File(outFileName, 'w')
        hf.create_dataset('data', data=npes)
        hf.close()
        print('saved ',outFileName)

if __name__ == '__main__':
    print('Start...')
    parser = get_parser()
    parse_args = parser.parse_args()
    ###########################################################
    large_PMT_pos = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/PMT_pos/J20v1r0-Pre2/PMTPos_Acrylic_with_chimney.csv' 
    L_id_dict = get_pmt_theta_phi(large_PMT_pos, '', 0, 4, 5)
    print('L_id_dict=',len(L_id_dict))
    small_PMT_pos = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/PMT_pos/3inch_pos.csv' 
    S_id_dict = get_pmt_theta_phi(small_PMT_pos, '', 0, 1, 2)
    ###########################################################
    forNpe = parse_args.forNpe
    forHitTime = parse_args.forHitTime
    filesPath = parse_args.input
    files = os.listdir(filesPath)
    outFileName= parse_args.output
    random.seed(1)
    random.shuffle(files) 
    if forHitTime:
        split_list = [int(0.2*len(files)), int(0.6*len(files))]
        #make_data(fpath=filesPath, file_list=files[0:split_list[0]]            , scale=10, forHitTime=True, forNpe=False, outFileName=outFileName.replace('.h5','_test.h5'))
        #time.sleep(60)
        make_data(fpath=filesPath, file_list=files[split_list[0]:split_list[1]], scale=10, forHitTime=True, forNpe=False, outFileName=outFileName.replace('.h5','_train1.h5'))
        time.sleep(60)
        make_data(fpath=filesPath, file_list=files[split_list[1]:-1           ], scale=10, forHitTime=True, forNpe=False, outFileName=outFileName.replace('.h5','_train2.h5'))
        time.sleep(60)
    if forNpe:
        split_list = [int(0.05*len(files)), int(0.1*len(files)), int(0.15*len(files))]
        make_data(fpath=filesPath, file_list=files[0:split_list[0]]            , scale=1, forHitTime=False, forNpe=True, outFileName=outFileName.replace('.h5','_test.h5'))
        time.sleep(60)
        make_data(fpath=filesPath, file_list=files[split_list[0]:split_list[1]], scale=1, forHitTime=False, forNpe=True, outFileName=outFileName.replace('.h5','_train1.h5'))
        time.sleep(60)
        make_data(fpath=filesPath, file_list=files[split_list[1]:split_list[2]], scale=1, forHitTime=False, forNpe=True, outFileName=outFileName.replace('.h5','_train2.h5'))

    '''
    Max_event = parse_args.batch_size
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
    '''
