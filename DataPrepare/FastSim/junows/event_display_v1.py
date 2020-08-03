import ROOT as rt
import numpy as np
import h5py 
#import pandas as pd
import sys 
import gc
rt.gROOT.SetBatch(rt.kTRUE)

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
    info.AddText("#mu(pos_{#theta}^{%.2f}, pos_{#phi}^{%.2f}, pos_{r}^{%.1f}, mom_{#theta}^{%.2f}, mom_{#phi}^{%.2f}, E=%d MeV)"%(pos_theta, pos_phi, pos_r, theta_mom, phi_mom, energy))
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
    hist.GetXaxis().SetTitle("#phi(AU, 0 #rightarrow 2#pi)")
    hist.GetYaxis().SetTitle("Z(AU) (-19.5 #rightarrow 19.5 m)")
    hist.SetTitle(title)
    #hist.SetTitleSize(0.1)
    #hist.Draw("COLZ TEXT")
    hist.Draw("COLZ")
    if n3 != None:
        Info = muno_info(n3[event][2], n3[event][3], n3[event][4], n3[event][0], n3[event][1], n3[event][5])
        Info.Draw()
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

show_real = False
show_fake = True

hf = 0
n1 = 0
n2 = 0
n3 = 0
event_list=0
plot_path=""
if show_real:
    hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/data/data_all0.h5', 'r')
    n1 = hf['firstHitTimeByPMT']
    n2 = hf['nPEByPMT']
    n3 = hf['infoMuon']
    event_list=[1,10,100,1000,10000]
    plot_path="./plots_event_display/real"
    print (n1.shape, n2.shape, n3.shape)
elif show_fake:
    hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/generator/gen0822_v1.h5', 'r')
    n1 = hf['firstHitTimeByPMT']
    n2 = hf['nPEByPMT']
    n1 = np.squeeze(n1)
    n2 = np.squeeze(n2)
    n3 = None
    event_list=range(0,10)
    plot_path="./plots_event_display/fake"
    print (n1.shape, n2.shape)
else: 
    print ('wrong config')
    sys.exit()


for event in event_list:
    if n3 != None:
        print ('event=%d, muon Direction: theta=%f, phi=%f. Position: theta=%f, phi=%f, r=%f. Energy=%f MeV'%(event, n3[event][0], n3[event][1], n3[event][2], n3[event][3], n3[event][4], n3[event][5]))
    nX = n1[event].shape[1]
    nY = n1[event].shape[0]
    
    h_Hit = rt.TH2F('Hit_evt%d'%event, '', nX, 0, nX, nY, 0, nY)
    h_nPE = rt.TH2F('nPE_evt%d'%event, '', nX, 0, nX, nY, 0, nY)
    for i in range(0, nX):
        i0 = nX-1 - i
        for j in range(0, nY):
            j0 = nY-1 - j
            h_Hit.Fill(i0+0.01, j0+0.01, n1[event][j][i])
            h_nPE.Fill(i0+0.01, j0+0.01, n2[event][j][i])
    str1 = "_gen" if show_fake else ""
    do_plot(event, h_Hit,'Hit_evt%d%s'%(event,str1),'First hit time of PMTs')
    do_plot(event, h_nPE,'nPE_evt%d%s'%(event,str1),'nPE of PMTs')
