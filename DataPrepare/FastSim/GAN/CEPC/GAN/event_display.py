import ROOT as rt
import numpy as np
import h5py 
#import pandas as pd
import sys 
import gc
rt.gROOT.SetBatch(rt.kTRUE)

###############
# ECAL : use cell ID #
# HCAL
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
    elif "phi_r" in out_name:
        hist.GetYaxis().SetTitle("bin R")
        hist.GetXaxis().SetTitle("bin #phi")
    elif "z_r" in out_name:
        hist.GetYaxis().SetTitle("bin R")
        hist.GetXaxis().SetTitle("bin Z")
    elif "z_phi" in out_name:
        hist.GetYaxis().SetTitle("bin #phi")
        hist.GetXaxis().SetTitle("bin Z")
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
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/gamma/gamma_ext9.h5', 'r')
    #str_p='gamma'
    #lat_p='#gamma'
    #hf = h5py.File('/cefs/higgs/wxfang/cepc/pionm/h5/Digi_sim_2_20_pionm_1.h5', 'r')
    hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-_fix/h5/Digi_sim_10_pionm_90.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/HCAL/pi-/Digi_sim_2_20_pionm_89.h5', 'r')
    str_p='pi-'
    lat_p='#pi^{-}'
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/e-/Digi_1_100_em_ext_9.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/e-/em_10.h5', 'r')
    #str_p='em'
    #lat_p='e^{-}'
    #hf = h5py.File('./test_e.h5', 'r')
    nB = hf['Barrel_Hit'][:]
    nB_H = hf['Barrel_Hit_HCAL'][:]
    n3 = hf['MC_info'][:]
    hf.close()
    #event_list=[10,20,30,40,50,60,70,80,90]
    event_list=range(20)
    #event_list=range(323,326)
    plot_path="./plots_event_display/real"
    print (nB.shape, nB_H.shape, n3.shape)
elif show_fake:
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen_gamma_1105_epoch71.h5', 'r')
    #str_p='gamma'
    #lat_p='#gamma'
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen_em_1105_epoch43.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen_em_1105_epoch83.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/pi-/Gen_fix_0227_epoch90.h5', 'r')
    hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/pi-/Gen_fix_0228_epoch99.h5', 'r')
    #str_p='em'
    #lat_p='e^{-}'
    str_p='pi-'
    lat_p='#pi^{-}'
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
    if False:
        HoE_cut = 1.0
        #HoE_cut = 10.0
        HoE= np.sum(nB_H[event,:,:,:])/np.sum(nB[event,:,:,:])
        if HoE > HoE_cut: continue
        #if HoE < HoE_cut: continue
    
    if n3 is not None:
        print ('event=%d, Mom=%f, M_dtheta=%f, M_dphi=%f, P_dz=%f, P_dy=%f, Z=%f'%(event, n3[event,0], n3[event,1], n3[event,2], n3[event,3], n3[event,4], n3[event,5]))
        #print ('event=%d, Mom=%f, M_dtheta=%f, M_dphi=%f, P_dz=%f, P_dy=%f, Z=%f, HoE=%f'%(event, n3[event,0], n3[event,1], n3[event,2], n3[event,3], n3[event,4], n3[event,5], HoE))
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
    ######################### HCAL ####################################################3#####
    continue
    n_z   = nB_H[event].shape[0]
    n_phi = nB_H[event].shape[1]
    n_r   = nB_H[event].shape[2]
    print ('n_z=',n_z,',n_phi=',n_phi,',n_r=',n_r)
    ##### for HCAL z_r #######
    h_Hit_BH_z_r = rt.TH2F('Hit_BH_z_r_evt%d'%(event)  , '', n_z, 0, n_z, n_r, 0, n_r)
    for i in range(0, n_r):
        for j in range(0, n_z):
            h_Hit_BH_z_r.Fill(j+0.01, i+0.01, sum(nB_H[event,j,:,i]))
    do_plot(event, h_Hit_BH_z_r,'%s_Hit_HBarrel_z_r_evt%d_%s'%(str_p,event,str1),'', lat_p)
    #
    h_Hit_BH_phi_r = rt.TH2F('Hit_BH_phi_r_evt%d'%(event)  , '', n_phi, 0, n_phi, n_r, 0, n_r)
    for i in range(0, n_r):
        for j in range(0, n_phi):
            h_Hit_BH_phi_r.Fill(j+0.01, i+0.01, sum(nB_H[event,:,j,i]))
    do_plot(event, h_Hit_BH_phi_r,'%s_Hit_HBarrel_phi_r_evt%d_%s'%(str_p,event,str1),'', lat_p)
    #
    h_Hit_BH_z_phi = rt.TH2F('Hit_BH_z_phi_evt%d'%(event)  , '', n_z, 0, n_z, n_phi, 0, n_phi)
    for i in range(0, n_phi):
        for j in range(0, n_z):
            h_Hit_BH_z_phi.Fill(j+0.01, i+0.01, sum(nB_H[event,j,i,:]))
    do_plot(event, h_Hit_BH_z_phi,'%s_Hit_HBarrel_z_phi_evt%d_%s'%(str_p,event,str1),'', lat_p)


    ## for z-y plane at each layer## 
    if show_details == False : continue
    for z in range(nDep): 
        h_Hit_B = rt.TH2F('Hit_B_evt%d_layer%d'%(event,z+1)  , '', nCol, 0, nCol, nRow, 0, nRow)
        for i in range(0, nRow):
            for j in range(0, nCol):
                h_Hit_B.Fill(j+0.01, i+0.01, nB[event,i,j,z])
        do_plot(event, h_Hit_B,'Hit_Barrel_evt%d_layer%d_%s'%(event,z+1,str1),'', lat_p)
