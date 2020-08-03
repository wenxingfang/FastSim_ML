import ROOT as rt
import numpy as np
import h5py 
import pandas as pd
import sys 
import matplotlib.pyplot as plt
import pmt_col as pmt
import argparse

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

    return parser

def root2hdf5 (Max_event, nRows, nCols, tree, start_event, out_name):
    nLargePMT = 17739
    default_time = -1 
    default_nPE =  -1 
    firstHitTimeByPMT = np.full((Max_event, nRows, nCols), default_time,dtype=np.float32)#init 
    nPEByPMT          = np.full((Max_event, nRows, nCols), default_nPE, dtype=np.int32  )#init 
    infoMuon          = np.full((Max_event, 6), 0, dtype=np.float32)#init 
    print (firstHitTimeByPMT.shape, nPEByPMT.shape, infoMuon.shape)
    
    template_firstHitTimeByPMT_1 = np.full((1, nRows, nCols), default_time, dtype=np.float32)#init 
    template_firstHitTimeByPMT_2 = np.full((1, nRows, nCols), 0           , dtype=np.float32)#init 
    template_nPEByPMT_1          = np.full((1, nRows, nCols), default_nPE , dtype=np.int32  )#init 
    template_nPEByPMT_2          = np.full((1, nRows, nCols), 0           , dtype=np.int32  )#init 
    
    First = True
    for ie in range(start_event, start_event+Max_event):
        tree.GetEntry(ie)
        entryNum = ie - start_event
        tmp_nPE     = getattr(tree, "q")
        tmp_hitTime = getattr(tree, "t")
        tmp_pmtId   = getattr(tree, "pmtid")
        muonMomTheta = getattr(tree, "thep")
        muonMomPhi   = getattr(tree, "phip")
        muonPosTheta = getattr(tree, "thes")
        muonPosPhi   = getattr(tree, "phis")
        muonPosRho   = getattr(tree, "rhos")
        '''
        tmp_nPE      = tree.q
        tmp_hitTime  = tree.t
        tmp_pmtId    = tree.pmtid
        muonMomTheta = tree.thep
        muonMomPhi   = tree.phip
        muonPosTheta = tree.thes
        muonPosPhi   = tree.phis
        muonPosRho   = tree.rhos
        '''
        #print (len(tmp_nPE))
        for i in range(0, len(tmp_nPE)):
            if tmp_pmtId[i] >= nLargePMT: continue # just select large PMT in CD
            x   = id_x_y[int(tmp_pmtId[i])][0]
            y   = id_x_y[int(tmp_pmtId[i])][1]
            firstHitTimeByPMT[entryNum, x, y] = tmp_hitTime[i]
            nPEByPMT         [entryNum, x, y] = tmp_nPE    [i]
            if First:        
                template_firstHitTimeByPMT_1[0, x, y] = 0
                template_firstHitTimeByPMT_2[0, x, y] = 1
                template_nPEByPMT_1         [0, x, y] = 0
                template_nPEByPMT_2         [0, x, y] = 1
        First = False
        infoMuon[entryNum,0] = muonMomTheta
        infoMuon[entryNum,1] = muonMomPhi
        infoMuon[entryNum,2] = muonPosTheta
        infoMuon[entryNum,3] = muonPosPhi
        infoMuon[entryNum,4] = muonPosRho
        infoMuon[entryNum,5] = 200000 # 200 GeV
    
    #print (firstHitTimeByPMT[0, 60, 110], nPEByPMT[0, 60, 110], infoMuon[0,0], infoMuon[0,1], infoMuon[0,2], infoMuon[0,3], infoMuon[0,4], infoMuon[0,5])
    hf = h5py.File(out_name, 'w')
    hf.create_dataset('firstHitTimeByPMT', data=firstHitTimeByPMT)
    hf.create_dataset('nPEByPMT'         , data=nPEByPMT)
    hf.create_dataset('infoMuon'         , data=infoMuon)
    hf.create_dataset('temp1_firstHitTimeByPMT', data=template_firstHitTimeByPMT_1)
    hf.create_dataset('temp2_firstHitTimeByPMT', data=template_firstHitTimeByPMT_2)
    hf.create_dataset('temp1_nPEByPMT', data=template_nPEByPMT_1)
    hf.create_dataset('temp2_nPEByPMT', data=template_nPEByPMT_2)
    hf.close()
    print('saved %s'%out_name)

######################################################
## put the PMT into theta*phi (shape=180*360) array###
######################################################



print ('Begin')
parser = get_parser()
parse_args = parser.parse_args()

x_y_id, id_x_y = pmt.pmt()
nRows = len(x_y_id)
nCols = len(x_y_id[0])
print ('Read root file')
#Max_event = 5000
#filePath = '/junofs/users/liuyan/muon-sim/sim-nn/data-nn/data-cd-200GeV/20190425/signal-evt*.root'
#outFileName='/junofs/users/wxfang/FastSim/junows/liuyan/data_all_20190425.h5'
Max_event = parse_args.batch_size
filePath = parse_args.input
outFileName= parse_args.output
treeName='evt'
chain =rt.TChain(treeName)
chain.Add(filePath)
tree = chain
totalEntries=tree.GetEntries()
batch = int(float(totalEntries)/Max_event)
print ('total=%d, max=%d, batch=%d, last=%d'%(totalEntries, Max_event, batch, totalEntries%Max_event))
start = 0
for i in range(batch):
    out_name = outFileName.replace('.h5','_batch%d_N%d.h5'%(i, Max_event))
    root2hdf5 (Max_event, nRows, nCols, tree, start, out_name)
    start = start + Max_event
    if i == batch-1:
        out_name = outFileName.replace('.h5','_batch%d_N%d.h5'%(i+1,totalEntries%Max_event))
        root2hdf5 (totalEntries%Max_event, nRows, nCols, tree, start, out_name)
    




print ('Done')
