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


def do_plot(hist,out_name,title):
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
    hist.GetXaxis().SetTitle("#phi(AU)")
    hist.GetYaxis().SetTitle("Z(AU)")
    hist.SetTitle(title)
    #hist.SetTitleSize(0.1)
    #hist.Draw("COLZ TEXT")
    hist.Draw("COLZ")
    Info = muno_info(n3[event][2], n3[event][3], n3[event][4], n3[event][0], n3[event][1], n3[event][5])
    Info.Draw()
    canvas.SaveAs("./plots_event_display/%s.png"%(out_name))
    del canvas
    gc.collect()



hf = h5py.File('data.h5', 'r')
n1 = hf.get('firstHitTimeByPMT')
n2 = hf.get('nPEByPMT')
n3 = hf.get('infoMuon')
n1 = np.array(n1)
n2 = np.array(n2)
n3 = np.array(n3)

event = 2
print ('event=%d, muon Direction: theta=%f, phi=%f. Position: theta=%f, phi=%f, r=%f. Energy=%f MeV'%(event, n3[event][0], n3[event][1], n3[event][2], n3[event][3], n3[event][4], n3[event][5]))
nX = n1[event].shape[1]
nY = n1[event].shape[0]

h_Hit = rt.TH2F('Hit', '', nX, 0, nX, nY, 0, nY)
h_nPE = rt.TH2F('nPE', '', nX, 0, nX, nY, 0, nY)
for i in range(0, nX):
    i0 = nX-1 - i
    for j in range(0, nY):
        j0 = nY-1 - j
        h_Hit.Fill(i0+0.01, j0+0.01, n1[event][j][i])
        h_nPE.Fill(i0+0.01, j0+0.01, n2[event][j][i])
do_plot(h_Hit,'h_Hit_evt%d'%event,'First hit time of PMTs')
do_plot(h_nPE,'h_nPE_evt%d'%event,'nPE of PMTs')
