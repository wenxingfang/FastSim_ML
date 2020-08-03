import ROOT as rt
import numpy as np
import h5py 
#import pandas as pd
import sys 
import gc
rt.gROOT.SetBatch(rt.kTRUE)

def mc_info(particle, mom, dtheta, dphi, Z):
    lowX=0.1
    lowY=0.8
    info  = rt.TPaveText(lowX, lowY+0.06, lowX+0.30, lowY+0.16, "NDC")
    info.SetBorderSize(   0 )
    info.SetFillStyle(    0 )
    info.SetTextAlign(   12 )
    info.SetTextColor(    1 )
    info.SetTextSize(0.04)
    info.SetTextFont (   42 )
    #info.AddText("%s (Mom=%.1f GeV, #Delta#theta=%.1f^{#circ}, #Delta#phi=%.1f^{#circ}, Z=%.1f cm)"%(particle, mom, dtheta, dphi, Z))
    info.AddText("%s (Mom=%.1f GeV, #theta=%.1f^{#circ}, #Delta#phi=%.1f^{#circ}, Z=%.1f cm)"%(particle, mom, dtheta, dphi, Z))
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
        hist.GetYaxis().SetTitle("#Delta#phi (degree)")
        hist.GetXaxis().SetTitle("#DeltaZ (cm)")
    hist.SetTitle(title)
    #hist.SetTitleSize(0.1)
    #hist.Draw("COLZ TEXT")
    hist.Draw("COLZ")
    if n3 is not None:
        Info = mc_info(str_particle, n3[event][0], n3[event][1], n3[event][2], n3[event][3])
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
    hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Hit_Barrel_em_100k.h5', 'r')
    str_e='e^{-}'
    str_e1='em'
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Hit_Barrel_ep_100k.h5', 'r')
    #str_e='e^{+}'
    #str_e1='ep'
    nB = hf['Barrel_Hit'][0:100]
    nD = hf['Hit_Depth' ][0:100]
    n3 = hf['MC_info'   ][0:100]
    hf.close()
    event_list=range(10)
    plot_path="./plots_event_display/real"
    print (nB.shape, nD.shape, n3.shape)
elif show_fake:
    hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen0925.h5', 'r')
    nB = hf['Barrel_Hit']
    nD = hf['Hit_Depth' ]
    #nB = np.squeeze(nB)
    #nD = np.squeeze(nD)
    n3 = hf['MC_info']
    event_list=range(10)
    plot_path="./plots_event_display/fake"
    print (nB.shape, nD.shape, n3.shape)
else: 
    print ('wrong config')
    sys.exit()

cellZ=5     # 5 cm
cellPhi = 3 # 3 dergee
for event in event_list:
    if n3 is not None:
        print ('event=%d, Energy=%f, dtheta=%f, dphi=%f, Z=%f'%(event, n3[event,0], n3[event,1], n3[event,2], n3[event,3]))
    nRow = nB[event].shape[0]
    nCol = nB[event].shape[1]
    nDep = nB[event].shape[2]
    print ('nRow=',nRow,',nCol=',nCol,',nDep=',nDep , 'Hit depth=', nD[event])
    str1 = "_gen" if show_fake else ""
    ## z-phi plane ## 
    #h_Hit_B_z_phi = rt.TH2F('Hit_B_z_phi_evt%d'%(event)  , '', nCol, 0, nCol, nRow, 0, nRow)
    x_min = -1*cellZ*0.5*nCol
    x_max =    cellZ*0.5*nCol
    y_min = -1*cellPhi*0.5*nRow
    y_max =    cellPhi*0.5*nRow
    h_Hit_B_z_phi = rt.TH2F('Hit_B_z_phi_evt%d'%(event)  , '', nCol, x_min, x_max, nRow, y_min, y_max)
    for i in range(0, nRow):
        for j in range(0, nCol):
            h_Hit_B_z_phi.Fill(x_min+cellZ*(j+0.01),y_min+cellPhi*(i+0.01), sum(nB[event,i,j,:]))
    do_plot(event, h_Hit_B_z_phi,'%s_Hit_Barrel_z_phi_evt%d_%s'%(str_e1, event,str1),'', str_e)
