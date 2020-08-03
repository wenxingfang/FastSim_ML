import ROOT as rt
import numpy as np
import h5py 
#import pandas as pd
import sys 
import gc
import math
rt.gROOT.SetBatch(rt.kTRUE)

#######################################################
### Show the PMT arranged in theta(180)*phi(360) grid##
#######################################################

def muno_info(pos_theta, pos_phi, pos_r, theta_mom, phi_mom, energy):
    lowX=0.1
    lowY=0.8
    info  = rt.TPaveText(lowX, lowY+0.06, lowX+0.30, lowY+0.16, "NDC")
    info.SetBorderSize(   0 )
    info.SetFillStyle(    0 )
    info.SetTextAlign(   12 )
    info.SetTextColor(    1 )
    info.SetTextSize(0.03)
    info.SetTextFont (   42 )
    info.AddText("#mu(pos_{#theta}^{%.2f^{#circ}}, pos_{#phi}^{%.2f^{#circ}}, pos_{r}^{%.1f m}, mom_{#theta}^{%.2f^{#circ}}, mom_{#phi}^{%.2f^{#circ}}, E=%d GeV)"%(180*pos_theta/math.pi, 180*pos_phi/math.pi, pos_r/1000, 180*theta_mom/math.pi, 180*phi_mom/math.pi, energy/1000))
    return info


def do_plot(event,hist,out_name,title):
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
    hist.GetXaxis().SetTitle("bin #phi")
    hist.GetYaxis().SetTitle("bin #theta")
    hist.SetTitle(title)
    #hist.SetTitleSize(0.1)
    #hist.Draw("COLZ TEXT")
    hist.Draw("COLZ")
    if n3 is not None:
        pass
        #Info = muno_info(n3[event][2], n3[event][3], n3[event][4], n3[event][0], n3[event][1], n3[event][5])
        #Info.Draw()
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

show_real = True
show_fake = False

use_shifted = False

hf = 0
n1 = 0
n2 = 0
n3 = 0
event_list=0
plot_path=""
if show_real:
    hf = h5py.File('/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/h5_data/laser_batch0_N5000.h5', 'r')
    n1_name = 'firstHitTimeByPMT'
    n2_name = 'nPEByPMT'
    n3_name = 'infoMC'
    if use_shifted:
        hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/generator/Gen0914.h5', 'r')
        n1_name = 'real_firstHitTimeByPMT_shifted'
        n2_name = 'real_nPEByPMT_shifted'
        n3_name = 'real_infoMuon'
    n1 = hf[n1_name]
    n2 = hf[n2_name]
    n3 = hf[n3_name]
    n1 = np.squeeze(n1)
    n2 = np.squeeze(n2)
    #event_list=[0,1,2]
    event_list=range(10)
    plot_path="./plots_event_display/real"
    print (n1.shape, n2.shape, n3.shape)
elif show_fake:
    hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/generator/Gen0914.h5', 'r')
    n1_name = 'firstHitTimeByPMT'
    n2_name = 'nPEByPMT'
    if use_shifted:
        n1_name = 'firstHitTimeByPMT_shifted'
        n2_name = 'nPEByPMT_shifted'
    n1 = hf[n1_name]
    n2 = hf[n2_name]
    n1 = np.squeeze(n1)
    n2 = np.squeeze(n2)
    n3 = hf['infoMuon']
    n3 = np.insert(n3, 4, 17700, axis=1) 
    n3 = np.insert(n3, 5, 200000, axis=1) 
    event_list=range(0,10)
    plot_path="./plots_event_display/fake"
    print (n1.shape, n2.shape)
else: 
    print ('wrong config')
    sys.exit()


for event in event_list:
    if n3 is not None:
        print ('event=%d, ptheta=%f, pphi=%f, rtheta=%f, rphi=%f'%(event, n3[event][0], n3[event][1], n3[event][2], n3[event][3]))
    nX = n1[event].shape[1]
    nY = n1[event].shape[0]
    
    h_Hit = rt.TH2F('Hit_evt%d'%event, '', nX, 0, nX, nY, 0, nY)
    h_nPE = rt.TH2F('nPE_evt%d'%event, '', nX, 0, nX, nY, 0, nY)
    for i in range(0, nX):
        for j in range(0, nY):
            h_Hit.Fill(i+0.01, j+0.01, n1[event][j][i])
            h_nPE.Fill(i+0.01, j+0.01, n2[event][j][i])
    str1 = "_gen" if show_fake else ""
    do_plot(event, h_Hit,'Hit_evt%d%s'%(event,str1),'First hit time of PMTs')
    do_plot(event, h_nPE,'nPE_evt%d%s'%(event,str1),'nPE of PMTs')
