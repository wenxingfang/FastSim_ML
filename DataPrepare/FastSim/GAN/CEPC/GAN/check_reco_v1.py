import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math

rt.gROOT.SetBatch(rt.kTRUE)

def plot_gr(gr,out_name,title, isTH2):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    if isTH2: gr.SetStats(rt.kFALSE)
    x1 = 0
    x2 = 100
    y1 = x1
    y2 = x2
    if 'dtheta' in out_name:
        #gr.GetXaxis().SetTitle("True #Delta#theta (degree)")
        #gr.GetYaxis().SetTitle("Predicted #Delta#theta (degree)")
        gr.GetXaxis().SetTitle("True #theta_{ext} (degree)") ## since the Z axis for the plane is the same as detector Z axis
        gr.GetYaxis().SetTitle("Predicted #theta_{ext} (degree)")
        x1 = 40
        x2 = 140
        y1 = x1
        y2 = x2
    elif 'dphi' in out_name:
        #if isTH2==False: gr.GetXaxis().SetLimits(-50,10)
        #gr.GetXaxis().SetTitle("True #Delta#phi (degree)")
        #gr.GetYaxis().SetTitle("Predicted #Delta#phi (degree)")
        gr.GetXaxis().SetTitle("True #phi_{ext} (degree)")## since the normal director for the plane is the same as detector X axis
        gr.GetYaxis().SetTitle("Predicted #phi_{ext} (degree)")
        x1 = -15
        x2 = 22
        y1 = x1
        y2 = x2
    elif 'dz' in out_name:
        gr.GetXaxis().SetTitle("True dz (mm)")## since the normal director for the plane is the same as detector X axis
        gr.GetYaxis().SetTitle("Predicted dz (mm)")
        x1 = -10
        x2 = 10
        y1 = x1
        y2 = x2
    elif 'dy' in out_name:
        gr.GetXaxis().SetTitle("True dy (mm)")## since the normal director for the plane is the same as detector X axis
        gr.GetYaxis().SetTitle("Predicted dy (mm)")
        x1 = -10
        x2 = 10
        y1 = x1
        y2 = x2
    elif 'mom' in out_name:
        gr.GetXaxis().SetTitle("True Momentum (GeV)")
        gr.GetYaxis().SetTitle("Predicted momentum (GeV)")
        x1 = 0
        x2 = 22
        y1 = x1
        y2 = x2
    elif 'Z' in out_name:
        gr.GetXaxis().SetTitle("True Z (cm)")
        gr.GetYaxis().SetTitle("Predicted Z (cm)")
        x1 = -210
        x2 =  210
        y1 = x1
        y2 = x2
    elif 'Y' in out_name:
        gr.GetXaxis().SetTitle("True Y (cm)")
        gr.GetYaxis().SetTitle("Predicted Y (cm)")
        x1 = -70
        x2 =  70
        y1 = x1
        y2 = x2
    #gr.SetTitle(title)
    if isTH2==False:
        gr.Draw("ap")
    else:
        gr.Draw("COLZ")
    
    line = rt.TLine(x1, y1, x2, y2)
    line.SetLineColor(rt.kRed)
    line.SetLineWidth(2)
    line.Draw('same')
    
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()





plot_path='./reco_plots'

#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/Reco_gamma_result_1102.h5','r')
#print(d.keys())
#real = d['input_info'][:]
#reco = d['reco_info'][:]

file_list = []
#file_list.append('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL/Digi_sim_2_20_pionm_90_pred.h5')
#file_list.append('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL/Digi_sim_2_20_pionm_91_pred.h5')
#file_list.append('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL/Digi_sim_2_20_pionm_92_pred.h5')
#file_list.append('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL/Digi_sim_2_20_pionm_93_pred.h5')
#file_list.append('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL/Digi_sim_2_20_pionm_94_pred.h5')
file_list.append('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL/Digi_sim_2_20_pionm_95_pred.h5')
file_list.append('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL/Digi_sim_2_20_pionm_96_pred.h5')
file_list.append('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL/Digi_sim_2_20_pionm_97_pred.h5')
file_list.append('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL/Digi_sim_2_20_pionm_98_pred.h5')
file_list.append('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL/Digi_sim_2_20_pionm_99_pred.h5')
file_list.append('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/HCAL/Digi_sim_2_20_pionm_9_pred.h5')


real = 0
reco = 0
First = True
for ifile in file_list:
    d = h5py.File(ifile,'r')
    if First:
        #real = d['input_info'][:]
        real = d['mc_info'][:]
        reco = d['reco_info'][:]
        First = False
    else:
        #real = np.concatenate((real, d['input_info'][:]),axis=0) 
        real = np.concatenate((real, d['mc_info'][:]),axis=0) 
        reco = np.concatenate((reco, d['reco_info' ][:]),axis=0) 
print(real.shape)
    

h_mom     = rt.TH2F('h_mom'   , '', 22,0, 22, 22, 0, 22)
h_dtheta  = rt.TH2F('h_dtheta', '', 130, 30, 160, 130,  30, 160)
h_dphi    = rt.TH2F('h_dphi'  , '', 50 ,-20,  30 , 50 , -20, 30)
h_dz      = rt.TH2F('h_dz'    , '', 20, -10, 10, 20,  -10, 10)
h_dy      = rt.TH2F('h_dy'    , '', 20 ,-10, 10 ,20 , -10, 10)
h_z       = rt.TH2F('h_z'     , '', 500,-250,250, 500,-250, 250)
h_y       = rt.TH2F('h_y'     , '', 120,-60,60, 120,-60, 60)
gr_dtheta =  rt.TGraph()
gr_dphi   =  rt.TGraph()
gr_dz     =  rt.TGraph()
gr_dy     =  rt.TGraph()
gr_mom    =  rt.TGraph()
gr_z      =  rt.TGraph()
gr_y      =  rt.TGraph()
gr_dtheta.SetMarkerColor(rt.kBlack)
gr_dtheta.SetMarkerStyle(8)
gr_dphi.SetMarkerColor(rt.kBlack)
gr_dphi.SetMarkerStyle(8)
gr_dz.SetMarkerColor(rt.kBlack)
gr_dz.SetMarkerStyle(8)
gr_dy.SetMarkerColor(rt.kBlack)
gr_dy.SetMarkerStyle(8)
gr_mom.SetMarkerColor(rt.kBlack)
gr_mom.SetMarkerStyle(8)
gr_z.SetMarkerColor(rt.kBlack)
gr_z.SetMarkerStyle(8)
gr_y.SetMarkerColor(rt.kBlack)
gr_y.SetMarkerStyle(8)
for i in range(real.shape[0]):
    gr_mom   .SetPoint(i, real[i][0], reco[i][0])
    gr_dtheta.SetPoint(i, real[i][1], reco[i][1])
    gr_dphi  .SetPoint(i, real[i][2], reco[i][2])
    h_mom    .Fill(real[i][0], reco[i][0])
    h_dtheta .Fill(real[i][1], reco[i][1])
    h_dphi   .Fill(real[i][2], reco[i][2])

    gr_dz    .SetPoint(i, real[i][3]   , real[i][3])
    gr_dy    .SetPoint(i, real[i][4]   , real[i][4])
    gr_z     .SetPoint(i, real[i][5]/10, real[i][5]/10)
    gr_y     .SetPoint(i, real[i][6]/10, real[i][6]/10)
    h_dz     .Fill(real[i][3]   , real[i][3])
    h_dy     .Fill(real[i][4]   , real[i][4])
    h_z      .Fill(real[i][5]/10, real[i][5]/10)
    h_y      .Fill(real[i][6]/10, real[i][6]/10)

plot_gr(gr_mom   , "gr_mom",""   , False)
plot_gr(gr_dtheta, "gr_dtheta","", False)
plot_gr(gr_dphi  , "gr_dphi",""  , False)
plot_gr(gr_dz  , "gr_dz",""  , False)
plot_gr(gr_dy  , "gr_dy",""  , False)
plot_gr(gr_z     , "gr_Z",""     , False)
plot_gr(gr_y     , "gr_Y",""     , False)
plot_gr(h_mom   , "h_mom",""   , True)
plot_gr(h_dtheta, "h_dtheta","", True)
plot_gr(h_dphi  , "h_dphi",""  , True)
plot_gr(h_dz  , "h_dz",""  , True)
plot_gr(h_dy  , "h_dy",""  , True)
plot_gr(h_z     , "h_Z",""     , True)
plot_gr(h_y     , "h_Y",""     , True)
print('done')
