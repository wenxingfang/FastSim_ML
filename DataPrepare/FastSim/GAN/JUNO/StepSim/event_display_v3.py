import ROOT as rt
import numpy as np
import h5py 
#import pandas as pd
import sys 
import gc
import math
rt.gROOT.SetBatch(rt.kTRUE)

#######################################################
### Show the PMT according ID in 218*218 grid##
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
    #hist.GetXaxis().SetTitle("bin #phi")
    #hist.GetYaxis().SetTitle("bin #theta")
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

show_real = False
show_fake = True

use_shifted = False

#N_size = 17612
N_size = 100


hf = 0
n1 = 0
n2 = 0
n3 = 0
event_list=0
plot_path=""
if show_real:
    hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_1D/laser_batch8_N5000.h5', 'r')
    n1_name = 'firstHitTimeByPMT'
    n2_name = 'nPEByPMT'
    n3_name = 'infoMC'
    n1 = hf[n1_name]
    n2 = hf[n2_name]
    #n3 = hf[n3_name]
    n3 = None
    n1 = np.squeeze(n1)
    n2 = np.squeeze(n2)
    #event_list=[0,1,2]
    event_list=range(10)
    plot_path="./plots_event_display/real"
    print (n1.shape, n2.shape)
elif show_fake:
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/generator/Gen_1203_epoch5.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/generator/Gen_1203_epoch17.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/generator/Gen_1204_100_epoch199.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/generator/Gen_1204v1_100_epoch199.h5', 'r')
    #hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/generator/Gen_1205_100_epoch199.h5', 'r')
    hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/generator/Gen_1205_100_epoch117.h5', 'r')
    #n1_name = 'firstHitTimeByPMT'
    n2_name = 'nPEByPMT'
    #n1 = hf[n1_name]
    n2 = hf[n2_name]
    #n1 = np.squeeze(n1)
    n2 = np.squeeze(n2)
    n3 = None 
    event_list=range(0,10)
    plot_path="./plots_event_display/fake"
    print (n2.shape)
else: 
    print ('wrong config')
    sys.exit()


for event in event_list:
    if n3 is not None:
        print ('event=%d, ptheta=%f, pphi=%f, rtheta=%f, rphi=%f'%(event, n3[event][0], n3[event][1], n3[event][2], n3[event][3]))
    #nX = n1[event].shape[1]
    #N_size = n2[event].shape[0]
    bin_size = int(math.sqrt(N_size))+1 if math.sqrt(N_size)%1 !=0 else int(math.sqrt(N_size))
    print('bin size=',bin_size)
    #h_Hit = rt.TH2F('Hit_evt%d'%event, '', nX, 0, nX, nY, 0, nY)
    h_nPE = rt.TH2F('nPE_evt%d'%event, '', bin_size, 0, bin_size, bin_size, 0, bin_size)
    for i in range(0, bin_size):
        for j in range(0, bin_size):
            #h_Hit.Fill(i+0.01, j+0.01, n1[event][j][i])
            if (i*bin_size+j) >= n2[event].shape[0]:break
            h_nPE.Fill(j+0.01, bin_size-i-0.01, round(n2[event][i*bin_size+j]))
    str1 = "_gen" if show_fake else ""
    #do_plot(event, h_Hit,'Hit_evt%d%s'%(event,str1),'First hit time of PMTs')
    do_plot(event, h_nPE,'nPE_evt%d%s'%(event,str1),'nPE of PMTs')
