import ROOT as rt
import numpy as np
import h5py 
import pandas as pd
import sys 
import matplotlib.pyplot as plt
print ('Begin')

print ('Read pmt pos file')

inputFileName    = '/junofs/users/wxfang/FastSim/junows/pmt_sorted.txt'
f = open(inputFileName, 'r')
lines = f.readlines()
list_ID= []
list_Z = []
list_Phi = []
ID_z_phi = {}
for line in lines:
    pos0 = line.strip('\n')
    pos1 = pos0.split(' ')
    pos = []
    for i in pos1:
        if i == '': continue
        pos.append(float(i))
    assert len(pos) == 3
    list_ID .append(pos[0])
    list_Z  .append(pos[1])
    list_Phi.append(pos[2])
    ID_z_phi[pos[0]] = (pos[1], pos[2])
info = {}
info['ID']  = list_ID
info['Z' ]  = list_Z
info['Phi'] = list_Phi
df=pd.DataFrame(info)
#print (df.head(10))
count_value = df['Z'].value_counts().values
count_size  = df['Z'].value_counts().size
count_index = df['Z'].value_counts().index
sorted_index = count_index.sort_values(ascending=False)
sorted_index = sorted_index.tolist()
max_z = max(sorted_index)
min_z = min(sorted_index)
print ('z max=%f, z min=%f'%(max_z, min_z))

nCols = count_value.max()
nRows = count_size
Dict_z_phis = {}
for z in sorted_index:
    b = df[df.Z==z]
    phis = b.Phi
    sorted_phis = phis.sort_values(ascending=False)
    sorted_phis = sorted_phis.tolist()
    offset = int((nCols - len(sorted_phis))/2)
    Dict_z_phis[z]=(sorted_phis, offset)

print ('Read root file')

#filePath = '/junofs/users/liuyan/muon-sim/sim-nn/data-nn/data-cd-200GeV/20190425/signal-evt-800*.root'
filePath = '/junofs/users/liuyan/muon-sim/sim-nn/data-nn/data-cd-200GeV/20190425/signal-evt-80010-3.root'
#outFileName='/junofs/users/wxfang/FastSim/junows/data_allTemp.h5'
outFileName='/junofs/users/wxfang/FastSim/junows/test.h5'
treeName='evt'
chain =rt.TChain(treeName)
chain.Add(filePath)
#inFile = rt.TFile.Open(inFileName,"READ")
#tree = inFile.Get(treeName)
tree = chain
totalEntries=tree.GetEntries()
print (totalEntries)

nLargePMT = 17739
default_time = -100 
default_nPE = -1000 
firstHitTimeByPMT = np.full((totalEntries, nRows, nCols), default_time,dtype=np.float32)#init 
nPEByPMT          = np.full((totalEntries, nRows, nCols), default_nPE, dtype=np.int32  )#init 
infoMuon          = np.full((totalEntries, 6), 0)#init 
print (firstHitTimeByPMT.shape, nPEByPMT.shape, infoMuon.shape)

template_firstHitTimeByPMT_1 = np.full((1, nRows, nCols), default_time, dtype=np.float32)#init 
template_firstHitTimeByPMT_2 = np.full((1, nRows, nCols), 0           , dtype=np.float32)#init 
template_nPEByPMT_1          = np.full((1, nRows, nCols), default_nPE , dtype=np.int32  )#init 
template_nPEByPMT_2          = np.full((1, nRows, nCols), 0           , dtype=np.int32  )#init 

First = True
for entryNum in range(0, tree.GetEntries()):
    tree.GetEntry(entryNum)
    tmp_nPE     = getattr(tree, "q")
    tmp_hitTime = getattr(tree, "t")
    tmp_pmtId   = getattr(tree, "pmtid")
    muonMomTheta = getattr(tree, "thep")
    muonMomPhi   = getattr(tree, "phip")
    muonPosTheta = getattr(tree, "thes")
    muonPosPhi   = getattr(tree, "phis")
    muonPosRho   = getattr(tree, "rhos")
    #print (len(tmp_nPE))
    for i in range(0, len(tmp_nPE)):
        if tmp_pmtId[i] >= nLargePMT: continue # just select large PMT in CD
        z   = ID_z_phi[float(tmp_pmtId[i])][0]
        phi = ID_z_phi[float(tmp_pmtId[i])][1]
        index_row = sorted_index.index(z)
        index_col = Dict_z_phis[z][0].index(phi) + Dict_z_phis[z][1]
        #print ("phi:", phi, ",index:", Dict_z_phis[z][0].index(phi), ",index_col:", index_col), 0 to 2pi for phi
        firstHitTimeByPMT[entryNum, index_row, index_col] = tmp_hitTime[i]
        nPEByPMT         [entryNum, index_row, index_col] = tmp_nPE    [i]
        if First:        
            template_firstHitTimeByPMT_1[0, index_row, index_col] = 0
            template_firstHitTimeByPMT_2[0, index_row, index_col] = 1
            template_nPEByPMT_1         [0, index_row, index_col] = 0
            template_nPEByPMT_2         [0, index_row, index_col] = 1
    First = False
    infoMuon[entryNum,0] = muonMomTheta
    infoMuon[entryNum,1] = muonMomPhi
    infoMuon[entryNum,2] = muonPosTheta
    infoMuon[entryNum,3] = muonPosPhi
    infoMuon[entryNum,4] = muonPosRho
    infoMuon[entryNum,5] = 200000 # 200 GeV



print (firstHitTimeByPMT[1, 60, 110], nPEByPMT[1, 60, 110], infoMuon[1,0], infoMuon[1,1], infoMuon[1,2], infoMuon[1,3], infoMuon[1,4], infoMuon[1,5])
hf = h5py.File(outFileName, 'w')
hf.create_dataset('firstHitTimeByPMT', data=firstHitTimeByPMT)
hf.create_dataset('nPEByPMT'         , data=nPEByPMT)
hf.create_dataset('infoMuon'         , data=infoMuon)
hf.create_dataset('temp1_firstHitTimeByPMT', data=template_firstHitTimeByPMT_1)
hf.create_dataset('temp2_firstHitTimeByPMT', data=template_firstHitTimeByPMT_2)
hf.create_dataset('temp1_nPEByPMT', data=template_nPEByPMT_1)
hf.create_dataset('temp2_nPEByPMT', data=template_nPEByPMT_2)
hf.close()
print ('Done')
