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
        x2 = 17
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
        x1 = 10
        x2 = 100
        y1 = x1
        y2 = x2
    elif 'Z' in out_name:
        gr.GetXaxis().SetTitle("True Z (cm)")
        gr.GetYaxis().SetTitle("Predicted Z (cm)")
        x1 = -200
        x2 =  200
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

#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/reco_result_1002.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/Reco_gamma_result_1012.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/Reco_em_result_1013.h5','r')
#d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/Reco_em_result_1102.h5','r')
d = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/reco/Reco_gamma_result_1102.h5','r')
print(d.keys())
real = d['input_info'][:]
reco = d['reco_info'][:]
print(real.shape)
h_mom     = rt.TH2F('h_mom'   , '', 110,0, 110, 110, 0, 110)
h_dtheta  = rt.TH2F('h_dtheta', '', 130, 30, 160, 130,  30, 160)
h_dphi    = rt.TH2F('h_dphi'  , '', 40 ,-20,  20 , 40 , -20, 20)
h_dz      = rt.TH2F('h_dz'    , '', 20, -10, 10, 20,  -10, 10)
h_dy      = rt.TH2F('h_dy'    , '', 20 ,-10, 10 ,20 , -10, 10)
h_z       = rt.TH2F('h_z'     , '', 500,-250,250, 500,-250, 250)
gr_dtheta =  rt.TGraph()
gr_dphi   =  rt.TGraph()
gr_dz     =  rt.TGraph()
gr_dy     =  rt.TGraph()
gr_mom    =  rt.TGraph()
gr_z      =  rt.TGraph()
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
for i in range(real.shape[0]):
    gr_mom   .SetPoint(i, real[i][0], reco[i][0])
    gr_dtheta.SetPoint(i, real[i][1], reco[i][1])
    gr_dphi  .SetPoint(i, real[i][2], reco[i][2])
#    gr_dz    .SetPoint(i, real[i][3], reco[i][3])
#    gr_dy    .SetPoint(i, real[i][4], reco[i][4])
#    gr_z     .SetPoint(i, real[i][5]/10, reco[i][5]/10)
    h_mom    .Fill(real[i][0], reco[i][0])
    h_dtheta .Fill(real[i][1], reco[i][1])
    h_dphi   .Fill(real[i][2], reco[i][2])
#    h_dz     .Fill(real[i][3], reco[i][3])
#    h_dy     .Fill(real[i][4], reco[i][4])
#    h_z      .Fill(real[i][5]/10, reco[i][5]/10)
plot_gr(gr_mom   , "gr_mom",""   , False)
plot_gr(gr_dtheta, "gr_dtheta","", False)
plot_gr(gr_dphi  , "gr_dphi",""  , False)
plot_gr(gr_dz  , "gr_dz",""  , False)
plot_gr(gr_dy  , "gr_dy",""  , False)
plot_gr(gr_z     , "gr_Z",""     , False)
plot_gr(h_mom   , "h_mom",""   , True)
plot_gr(h_dtheta, "h_dtheta","", True)
plot_gr(h_dphi  , "h_dphi",""  , True)
plot_gr(h_dz  , "h_dz",""  , True)
plot_gr(h_dy  , "h_dy",""  , True)
plot_gr(h_z     , "h_Z",""     , True)
print('done')
