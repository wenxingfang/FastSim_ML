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
    canvas.SetGridy()
    canvas.SetGridx()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    if isTH2: gr.SetStats(rt.kFALSE)
    x1 = 0
    x2 = 100
    y1 = x1
    y2 = x2
    if 'M_dtheta' in out_name:
        gr.GetXaxis().SetTitle("True #Delta#theta^{Mom} (degree)")
        gr.GetYaxis().SetTitle("Predicted #Delta#theta^{Mom} (degree)")
        x1 = 0
        x2 = 4
        if for_em==False :
            x1 = -4
            x2 = 0
        y1 = x1
        y2 = x2
    elif 'M_dphi' in out_name:
        #if isTH2==False: gr.GetXaxis().SetLimits(-50,10)
        gr.GetXaxis().SetTitle("True #Delta#phi^{Mom} (degree)")
        gr.GetYaxis().SetTitle("Predicted #Delta#phi^{Mom} (degree)")
        x1 = -15
        x2 = -2
        if for_em==False :
            x1 = 4
            x2 = 20
        y1 = x1
        y2 = x2
    elif 'P_dphi' in out_name:
        gr.GetXaxis().SetTitle("True #Delta#phi^{Pos} (degree)")
        gr.GetYaxis().SetTitle("Predicted #Delta#phi^{Pos} (degree)")
        x1 = -2
        x2 = 2
        y1 = x1
        y2 = x2
    elif 'P_dz' in out_name:
        gr.GetXaxis().SetTitle("True #DeltaZ^{Pos} (cm)")
        gr.GetYaxis().SetTitle("Predicted #DeltaZ^{Pos} (cm)")
        x1 = -2
        x2 = 2
        y1 = x1
        y2 = x2
    elif 'mom' in out_name:
        gr.GetXaxis().SetTitle("True Momentum (GeV)")
        gr.GetYaxis().SetTitle("Predicted momentum (GeV)")
        x1 = 0.6
        x2 = 1.9
        y1 = x1
        y2 = x2
    elif '_z' in out_name:
        gr.GetXaxis().SetTitle("True Z (cm)")
        gr.GetYaxis().SetTitle("Predicted Z (cm)")
        x1 = -140
        x2 =  140
        if for_HighZ and for_em :
            x1 = -140
            x2 = -110
        elif for_HighZ and for_em==False :
            x1 = 110
            x2 = 145
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
for_em = True
for_HighZ = False
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/mc_reco_result_em_Low_1008.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/mc_reco_result_em_High_1008.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/mc_reco_result_ep_High_1008.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/mc_reco_result_ep_Low_1007.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/fake_reco_result_em_Low_1025.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_Low_1025.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/fale_reco_result_em_Low_1027.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_Low_1027new.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_Low_1028.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_single_e_1029.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_1030.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_1030v1.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_1030v2.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_1031.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_1031v1.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_ep_1030.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_ep_1101.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_ep_1101v1.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_ep_1102.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_1102.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_Low_1107.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/mc_reco_result_em_High_1107.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/mc_reco_result_ep_High_1107.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/mc_reco_result_ep_Low_1107.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_1112.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_1114.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_ep_1114.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_sig_em_1116.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_1119.h5','r')
d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_sig_em_1119.h5','r')
print(d.keys())
real = d['input_info'][:]
reco = d['reco_info'][:]
print(real.shape)
h_mom     = rt.TH2F('h_mom'   , '', 200,0.2, 2.2, 200, 0.2, 2.2)
h_M_dtheta  = 0
if for_em : h_M_dtheta  = rt.TH2F('h_M_dtheta', '', 60, -2, 5, 60,  -2, 5)
else      : h_M_dtheta  = rt.TH2F('h_M_dtheta', '', 60, -5, 2, 60,  -5, 2)
h_M_dphi = 0
if for_em: h_M_dphi    = rt.TH2F('h_M_dphi'  , '', 200 ,-20, 0, 200 , -20, 0)
else      :h_M_dphi    = rt.TH2F('h_M_dphi'  , '', 200 ,0, 20, 200 , 0, 20)
h_P_dz      = rt.TH2F('h_P_dz'    , '', 60, -3, 3  , 60,  -3, 3)
h_P_dphi    = rt.TH2F('h_P_dphi'  , '', 60 ,-3, 3, 60 , -3, 3)
if for_HighZ and for_em :
    h_P_z       = rt.TH2F('h_P_z'     , '', 30,-140,-110, 30,-140, -110)
elif for_HighZ and for_em==False :
    h_P_z       = rt.TH2F('h_P_z'     , '', 35,110,145, 35,110, 145)
else:
    h_P_z       = rt.TH2F('h_P_z'     , '', 300,-150,150, 300,-150, 150)
gr_mom      =  rt.TGraph()
gr_M_dtheta =  rt.TGraph()
gr_M_dphi   =  rt.TGraph()
gr_P_dz     =  rt.TGraph()
gr_P_dphi   =  rt.TGraph()
gr_P_z      =  rt.TGraph()
gr_mom     .SetMarkerColor(rt.kBlack)
gr_mom     .SetMarkerStyle(8)
gr_M_dtheta.SetMarkerColor(rt.kBlack)
gr_M_dtheta.SetMarkerStyle(8)
gr_M_dphi  .SetMarkerColor(rt.kBlack)
gr_M_dphi  .SetMarkerStyle(8)
gr_P_dz    .SetMarkerColor(rt.kBlack)
gr_P_dz    .SetMarkerStyle(8)
gr_P_dphi  .SetMarkerColor(rt.kBlack)
gr_P_dphi  .SetMarkerStyle(8)
gr_P_z     .SetMarkerColor(rt.kBlack)
gr_P_z     .SetMarkerStyle(8)
for i in range(real.shape[0]):
#    if reco[i][0]<1.65 or real[i][0]>1.6 : continue
    gr_mom   .SetPoint(i, real[i][0], reco[i][0])
    gr_M_dtheta.SetPoint(i, real[i][1], reco[i][1])
    gr_M_dphi  .SetPoint(i, real[i][2], reco[i][2])
#    gr_P_dz    .SetPoint(i, real[i][3], reco[i][3])
#    gr_P_dphi  .SetPoint(i, real[i][4], reco[i][4])
#    gr_P_z     .SetPoint(i, real[i][5], reco[i][5])
    h_mom      .Fill(real[i][0], reco[i][0])
    h_M_dtheta .Fill(real[i][1], reco[i][1])
    h_M_dphi   .Fill(real[i][2], reco[i][2])
#    h_P_dz     .Fill(real[i][3], reco[i][3])
#    h_P_dphi   .Fill(real[i][4], reco[i][4])
#    h_P_z      .Fill(real[i][5], reco[i][5])
plot_gr(gr_mom     , "gr_mom",""   , False)
plot_gr(gr_M_dtheta, "gr_M_dtheta","", False)
plot_gr(gr_M_dphi  , "gr_M_dphi",""  , False)
plot_gr(gr_P_dz    , "gr_P_dz",""  , False)
plot_gr(gr_P_dphi  , "gr_P_dphi","" , False)
plot_gr(gr_P_z     , "gr_z",""     , False)
plot_gr(h_mom      , "h_mom",""   , True)
plot_gr(h_M_dtheta , "h_M_dtheta","", True)
plot_gr(h_M_dphi   , "h_M_dphi",""  , True)
plot_gr(h_P_dz     , "h_P_dz",""   , True)
plot_gr(h_P_dphi   , "h_P_dphi",""  , True)
plot_gr(h_P_z      , "h_z",""     , True)
print('done')
