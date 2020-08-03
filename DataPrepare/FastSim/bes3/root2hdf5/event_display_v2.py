import ROOT as rt
import numpy as np
import h5py 
#import pandas as pd
import sys 
import gc
rt.gROOT.SetBatch(rt.kTRUE)
##########################################################################
#using Mom dtheta, Mom dphi, Pos dz, Pos dphi, array with 121 cell energy#
##########################################################################

def mc_info(particle, mom, M_dtheta, M_dphi, P_dZ, P_dphi, P_Z):
    lowX=0.1
    lowY=0.8
    info  = rt.TPaveText(lowX, lowY+0.06, lowX+0.30, lowY+0.16, "NDC")
    info.SetBorderSize(   0 )
    info.SetFillStyle(    0 )
    info.SetTextAlign(   12 )
    info.SetTextColor(    1 )
    info.SetTextSize(0.025)
    info.SetTextFont (   42 )
    #info.AddText("%s (Mom=%.1f GeV, #Delta#theta=%.1f^{#circ}, #Delta#phi=%.1f^{#circ}, Z=%.1f cm)"%(particle, mom, dtheta, dphi, Z))
    info.AddText("%s (Mom=%.1f GeV, #Delta#theta^{Mom}=%.1f^{#circ}, #Delta#phi^{Mom}=%.1f^{#circ}, #DeltaZ^{Pos}=%.1f cm, #Delta#phi^{Pos}=%.1f^{#circ}, Z=%.1f cm)"%(particle, mom, M_dtheta, M_dphi, P_dZ, P_dphi, P_Z))
    return info

def layer_info(layer):
    lowX=0.85
    lowY=0.8
    info  = rt.TPaveText(lowX, lowY+0.06, lowX+0.30, lowY+0.16, "NDC")
    info.SetBorderSize(   0 )
    info.SetFillStyle(    0 )
    info.SetTextAlign(   12 )
    info.SetTextColor(    1 )
    info.SetTextSize(0.04)
    info.SetTextFont (   42 )
    info.AddText("%s"%(layer))
    return info

def do_plot(event,hist,out_name,title, str_particle):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    #h_corr.Draw("COLZ")
    #h_corr.LabelsDeflate("X")
    #h_corr.LabelsDeflate("Y")
    #h_corr.LabelsOption("v")
    hist.SetStats(rt.kFALSE)
    if "Barrel_z_phi" in out_name:
        hist.GetYaxis().SetTitle("cell #phi")
        hist.GetXaxis().SetTitle("cell Z")
    hist.SetTitle(title)
    #hist.SetTitleSize(0.1)
    #hist.Draw("COLZ TEXT")
    hist.Draw("COLZ")
    if n3 is not None:
        Info = mc_info(str_particle, n3[event][0], n3[event][1], n3[event][2], n3[event][3], n3[event][4], n3[event][5])
        Info.Draw()
    if 'layer' in out_name:
        str_layer = out_name.split('_')[3] 
        l_info = layer_info(str_layer)
        l_info.Draw()
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

show_real = False
show_fake = True

str_e='e'
str_e1='e'
hf = 0
nB = 0
n3 = 0
event_list=0
plot_path=""
if show_real:
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_em_zcut_Low.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_em_zcut_High.h5', 'r')
    hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_zcut.h5', 'r')
    str_e='e^{-}'
    str_e1='em'
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_ep_zcut_Low.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_ep_zcut_High.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_ep_zcut.h5', 'r')
    #str_e='e^{+}'
    #str_e1='ep'
    nB = hf['Barrel_Hit'][0:100]
    n3 = hf['MC_info'   ][0:100]
    hf.close()
    event_list=range(20)
    plot_path="./plots_event_display/real"
    print (nB.shape, n3.shape)
elif show_fake:
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1020_em_Low.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1020_em_High.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1027_em_Low.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1027v1_em_Low.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1028_em_Low.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1029_em_Low.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1029_em_Low_ep79.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1029_em_Low_ep29.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1029_single_e.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1029_em_Low_add59.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1029_em_Low_add79.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1029_em_Low_add139.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1029_em_Low_add49.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1101_em_add29.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1104_em_epoch115.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1107_em_wgan.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_1112_epoch57.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_1112_epoch59.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_1114_epoch56.h5', 'r')
    hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_em_wgan_1125_epoch199v1.h5', 'r')
    str_e='e^{-}'
    str_e1='em'
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1020_ep_Low.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1020_ep_High.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1104_ep_epoch129.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_ep_1114_epoch48.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_ep_1114_epoch181.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_ep_1114_epoch182.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen_ep_1114_epoch179.h5', 'r')
    #str_e='e^{+}'
    #str_e1='ep'
    nB = hf['Barrel_Hit']
    n3 = hf['MC_info']
    event_list=range(20,40)
    plot_path="./plots_event_display/fake"
    print (nB.shape, n3.shape)
else: 
    print ('wrong config')
    sys.exit()

cellZ=5     # 5 cm
cellPhi = 3 # 3 dergee
for event in event_list:
    if n3 is not None:
        print ('event=%d, Energy=%f, M_dtheta=%f, M_dphi=%f, P_dZ=%f, P_dphi=%f, P_Z=%f'%(event, n3[event,0], n3[event,1], n3[event,2], n3[event,3], n3[event,4], n3[event,5] ))
    nRow = nB[event].shape[0]
    nCol = nB[event].shape[1]
    #nDep = nB[event].shape[2]
    #print ('nRow=',nRow,',nCol=',nCol,',nDep=',nDep , 'Hit depth=', nD[event])
    str1 = "_gen" if show_fake else ""
    ## z-phi plane ## 
    #h_Hit_B_z_phi = rt.TH2F('Hit_B_z_phi_evt%d'%(event)  , '', nCol, 0, nCol, nRow, 0, nRow)
    #x_min = -1*cellZ*0.5*nCol
    #x_max =    cellZ*0.5*nCol
    #y_min = -1*cellPhi*0.5*nRow
    #y_max =    cellPhi*0.5*nRow
    #h_Hit_B_z_phi = rt.TH2F('Hit_B_z_phi_evt%d'%(event)  , '', nCol, x_min, x_max, nRow, y_min, y_max)
    h_Hit_B_z_phi = rt.TH2F('Hit_B_z_phi_evt%d'%(event)  , '', nCol, 0, nCol, nRow, 0, nRow)
    for i in range(0, nRow):
        for j in range(0, nCol):
            #h_Hit_B_z_phi.Fill(x_min+cellZ*(j+0.01),y_min+cellPhi*(i+0.01), sum(nB[event,i,j,:]))
            h_Hit_B_z_phi.Fill(j+0.1, nRow-i-0.1, sum(nB[event,i,j,:]))
    do_plot(event, h_Hit_B_z_phi,'%s_Hit_Barrel_z_phi_evt%d_%s'%(str_e1, event,str1),'', str_e)
