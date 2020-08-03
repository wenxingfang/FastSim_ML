import ROOT as rt
import numpy as np
import h5py 
#import pandas as pd
import sys 
import gc
rt.gROOT.SetBatch(rt.kTRUE)

###############
# use cell ID #
###############
def mc_info(particle, mom, dtheta, dphi, dz, dy, Z):
    lowX=0.1
    lowY=0.8
    info  = rt.TPaveText(lowX, lowY+0.06, lowX+0.30, lowY+0.16, "NDC")
    info.SetBorderSize(   0 )
    info.SetFillStyle(    0 )
    info.SetTextAlign(   12 )
    info.SetTextColor(    1 )
    info.SetTextSize(0.025)
    info.SetTextFont (   42 )
    info.AddText("%s (mom=%.1f GeV, #theta_{in}=%.1f, #phi_{in}=%.1f, dz=%.1f cm, dy=%.1f cm, Z=%.1f cm)"%(particle, mom, dtheta, dphi, dz/10, dy/10,  Z/10))
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
    if "Barrel_z_y" in out_name:
        hist.GetYaxis().SetTitle("cell Y")
        hist.GetXaxis().SetTitle("cell Z")
    elif "Barrel_z_dep" in out_name:
        hist.GetYaxis().SetTitle("cell X")
        hist.GetXaxis().SetTitle("cell Z")
    elif "Barrel_y_dep" in out_name:
        hist.GetYaxis().SetTitle("cell X")
        hist.GetXaxis().SetTitle("cell Y")
    hist.SetTitle(title)
    #hist.SetTitleSize(0.1)
    #hist.Draw("COLZ TEXT")
    hist.Draw("COLZ")
    if n3 is not None:
        Info = mc_info(str_particle, n3[event][0], n3[event][1], n3[event][2], n3[event][3], n3[event][4], n3[event][5] )
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

str_p='em'
lat_p='e^{-}'

show_details = False
hf = 0
nB = 0
n3 = 0
event_list=0
plot_path=""
if show_real:
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/gamma/Digi_1_100_gamma_ext_1.h5', 'r')
    hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/gamma/gamma_ext9.h5', 'r')
    str_p='gamma'
    lat_p='#gamma'
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/e-/Digi_1_100_em_ext_9.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/e-/em_10.h5', 'r')
    #str_p='em'
    #lat_p='e^{-}'
    #hf = h5py.File('./test_e.h5', 'r')
    nB = hf['Barrel_Hit'][0:100]
    n3 = hf['MC_info'][0:100]
    hf.close()
    #event_list=[10,20,30,40,50,60,70,80,90]
    #event_list=range(50)
    event_list=range(20)
    plot_path="./plots_event_display/real"
    print (nB.shape, n3.shape)
elif show_fake:
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen_gamma_1105_epoch71.h5', 'r')
    #str_p='gamma'
    #lat_p='#gamma'
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen_em_1105_epoch43.h5', 'r')
    hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen_em_1105_epoch83.h5', 'r')
    str_p='em'
    lat_p='e^{-}'
    nB = hf['Barrel_Hit']
    nB = np.squeeze(nB)
    n3 = hf['MC_info']
    #event_list=range(0,50)
    event_list=range(20)
    plot_path="./plots_event_display/fake"
    print (nB.shape, n3.shape)
else: 
    print ('wrong config')
    sys.exit()


for event in event_list:
    if n3 is not None:
        #print ('event=%d, e- Direction: theta=%f, phi=%f. Energy=%f GeV'%(event, n3[event,0], n3[event,1], n3[event,2]))
        print ('event=%d, Mom=%f, M_dtheta=%f, M_dphi=%f, P_dz=%f, P_dy=%f, Z=%f'%(event, n3[event,0], n3[event,1], n3[event,2], n3[event,3], n3[event,4], n3[event,5]))
    nRow = nB[event].shape[0]
    nCol = nB[event].shape[1]
    nDep = nB[event].shape[2]
    print ('nRow=',nRow,',nCol=',nCol,',nDep=',nDep)
    str1 = "_gen" if show_fake else ""
    ## z-y plane ## 
    h_Hit_B_z_y = rt.TH2F('Hit_B_z_y_evt%d'%(event)  , '', nCol, 0, nCol, nRow, 0, nRow)
    for i in range(0, nRow):
        for j in range(0, nCol):
            h_Hit_B_z_y.Fill(j+0.01, nRow-(i+0.01), sum(nB[event,i,j,:]))
    do_plot(event, h_Hit_B_z_y,'%s_Hit_Barrel_z_y_evt%d_%s'%(str_p,event,str1),'', lat_p)
    ## z-dep or z-x plane ## 
    h_Hit_B_z_dep = rt.TH2F('Hit_B_z_dep_evt%d'%(event)  , '', nCol, 0, nCol, nDep, 0, nDep)
    for i in range(0, nDep):
        for j in range(0, nCol):
            h_Hit_B_z_dep.Fill(j+0.01, i+0.01, sum(nB[event,:,j,i]))
    do_plot(event, h_Hit_B_z_dep,'%s_Hit_Barrel_z_dep_evt%d_%s'%(str_p,event,str1),'', lat_p)
    ## y-dep or y-x plane ##
    h_Hit_B_y_dep = rt.TH2F('Hit_B_y_dep_evt%d'%(event)  , '', nRow, 0, nRow, nDep, 0, nDep)
    for i in range(0, nDep):
        for j in range(0, nRow):
            h_Hit_B_y_dep.Fill(nRow-(j+0.01), i+0.01, sum(nB[event,j,:,i]))
    do_plot(event, h_Hit_B_y_dep,'%s_Hit_Barrel_y_dep_evt%d_%s'%(str_p,event,str1),'', lat_p)
    ## for z-y plane at each layer## 
    if show_details == False : continue
    for z in range(nDep): 
        h_Hit_B = rt.TH2F('Hit_B_evt%d_layer%d'%(event,z+1)  , '', nCol, 0, nCol, nRow, 0, nRow)
        for i in range(0, nRow):
            for j in range(0, nCol):
                h_Hit_B.Fill(j+0.01, i+0.01, nB[event,i,j,z])
        do_plot(event, h_Hit_B,'Hit_Barrel_evt%d_layer%d_%s'%(event,z+1,str1),'', lat_p)
