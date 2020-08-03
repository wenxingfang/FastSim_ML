import ROOT as rt
import numpy as np
import h5py 
#import pandas as pd
import sys 
import gc
rt.gROOT.SetBatch(rt.kTRUE)

class Obj:
    def __init__(self, name, fileName, is_real, evt_start, evt_end):
        self.name = name
        self.is_real = is_real
        self.fileName = fileName
        hf = h5py.File(self.fileName, 'r')
        self.nB   = hf['Barrel_Hit'][evt_start:evt_end]
        self.info = hf['MC_info'   ][evt_start:evt_end]
        self.nEvt = self.nB.shape[0]
        self.nRow = self.nB.shape[1]
        self.nCol = self.nB.shape[2]
        self.nDep = self.nB.shape[3]
        hf.close()
        
    def produce_z_sp(self):## produce showershape in z direction
        str1 = "" if self.is_real else "_gen"
        H_z_sp = rt.TH1F('H_z_sp_%s'%(str1)  , '', self.nCol, 0, self.nCol)
        for i in range(self.nEvt):
            for j in range(0, self.nCol):
                H_z_sp.Fill(j+0.01, np.sum(self.nB[i,:,j,:]))
        return H_z_sp

    def produce_y_sp(self):## produce showershape in y direction
        str1 = "" if self.is_real else "_gen"
        H_y_sp = rt.TH1F('H_y_sp_%s'%(str1)  , '', self.nRow, 0, self.nRow)
        for i in range(self.nEvt):
            for j in range(0, self.nRow):
                H_y_sp.Fill(j+0.01, np.sum(self.nB[i,j,:,:]))
        return H_y_sp
         
    def produce_dep_sp(self):## produce showershape in dep direction
        str1 = "" if self.is_real else "_gen"
        H_dep_sp = rt.TH1F('H_dep_sp_%s'%(str1)  , '', self.nDep, 0, self.nDep)
        for i in range(self.nEvt):
            for j in range(0, self.nDep):
                H_dep_sp.Fill(j+0.01, np.sum(self.nB[i,:,:,j]))
        return H_dep_sp

    def produce_cell_energy(self):## 
        str1 = "" if self.is_real else "_gen"
        H_cell_E = rt.TH1F('H_cell_E_%s'%(str1)  , '', 1000, 1, 10e3)
        for i in range(self.nEvt):
            for j in range(0, self.nRow):
                for k in range(0, self.nCol):
                    for z in range(0, self.nDep):
                        H_cell_E.Fill(self.nB[i,j,k,z]*1000)# to MeV
        return H_cell_E

    def produce_prob(self, data, label, evt_start, evt_end):## produce discriminator prob
        str1 = "" if self.is_real else "_gen"
        hf = h5py.File(data, 'r')
        da = hf[label][evt_start:evt_end]
        H_prob = rt.TH1F('H_prob_%s'%(str1)  , '', 100, 0, 1)
        for i in range(da.shape[0]):
            H_prob.Fill(da[i]) 
        return H_prob
        


def mc_info(particle, theta_mom, phi_mom, energy):
    lowX=0.1
    lowY=0.8
    info  = rt.TPaveText(lowX, lowY+0.06, lowX+0.30, lowY+0.16, "NDC")
    info.SetBorderSize(   0 )
    info.SetFillStyle(    0 )
    info.SetTextAlign(   12 )
    info.SetTextColor(    1 )
    info.SetTextSize(0.04)
    info.SetTextFont (   42 )
    info.AddText("%s (#theta=%.1f, #phi=%.1f, E=%.1f GeV)"%(particle, theta_mom, phi_mom, energy))
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
        Info = mc_info(str_particle, n3[event][0], n3[event][1], n3[event][2])
        Info.Draw()
    if 'layer' in out_name:
        str_layer = out_name.split('_')[3] 
        l_info = layer_info(str_layer)
        l_info.Draw()
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

def do_plot_v1(h_real,h_fake,out_name,title, str_particle):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)

    h_real.Scale(1/h_real.GetSumOfWeights())
    h_fake.Scale(1/h_fake.GetSumOfWeights())
    nbin =h_real.GetNbinsX()
    x_min=h_real.GetBinLowEdge(1)
    x_max=h_real.GetBinLowEdge(nbin)+h_real.GetBinWidth(nbin)
    y_min=0
    y_max=1.5*h_real.GetBinContent(h_real.GetMaximumBin())
    if "logx" in out_name:
        canvas.SetLogx()
    if "logy" in out_name:
        canvas.SetLogy()
        y_min = 1e-3
    if "cell_energy" in out_name:
        y_min = 1e-4
        y_max = 1
    elif "prob" in out_name:
        x_min=0.4
        x_max=0.6
    dummy = rt.TH2D("dummy","",nbin,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_Y_title=""
    if "z_showershape" in out_name:
        dummy_Y_title = "Normalized energy"
        dummy_X_title = "cell Z"
    elif "y_showershape" in out_name:
        dummy_Y_title = "Normalized energy"
        dummy_X_title = "cell Y"
    elif "dep_showershape" in out_name:
        dummy_Y_title = "Normalized energy"
        dummy_X_title = "cell X"
    elif "cell_energy" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "cell energy deposition (MeV)"
    elif "prob" in out_name:
        dummy_Y_title = "Normalized Count"
        dummy_X_title = "Real/Fake"
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    dummy.GetXaxis().SetTitle(dummy_X_title)
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.5)
    dummy.GetXaxis().SetTitleOffset(1.0)
    if "cell_energy" not in out_name:
        dummy.GetXaxis().SetMoreLogLabels()
        dummy.GetXaxis().SetNoExponent()  
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw()
    h_real.SetLineColor(rt.kRed)
    h_fake.SetLineColor(rt.kBlue)
    h_real.SetMarkerColor(rt.kRed)
    h_fake.SetMarkerColor(rt.kBlue)
    h_real.SetMarkerStyle(20)
    h_fake.SetMarkerStyle(21)
    h_real.Draw("same:pe")
    h_fake.Draw("same:histe")
    dummy.Draw("AXISSAME")
    legend = rt.TLegend(0.75,0.7,0.85,0.85)
    legend.AddEntry(h_real,'G4','lep')
    legend.AddEntry(h_fake,"GAN",'lep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    #legend.SetFillStyle(0)
    #legend.SetTextFont(42)
    legend.Draw()
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

N_event = 10000

plot_path='plot_comparision'
real = Obj('real', '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/mc_info/mc_info_and_cell_ID_final_orgin.h5', True, 0, N_event)
fake = Obj('real', '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen1008_mc_reco.h5', False, 0, N_event)
real_h_z_ps = real.produce_z_sp()
fake_h_z_ps = fake.produce_z_sp()
do_plot_v1(real_h_z_ps, fake_h_z_ps,'z_showershape','', 'e-')
do_plot_v1(real_h_z_ps, fake_h_z_ps,'z_showershape_logy','', 'e-')
real_h_y_ps = real.produce_y_sp()
fake_h_y_ps = fake.produce_y_sp()
do_plot_v1(real_h_y_ps, fake_h_y_ps,'y_showershape','', 'e-')
do_plot_v1(real_h_y_ps, fake_h_y_ps,'y_showershape_logy','', 'e-')
real_h_dep_ps = real.produce_dep_sp()
fake_h_dep_ps = fake.produce_dep_sp()
do_plot_v1(real_h_dep_ps, fake_h_dep_ps,'dep_showershape','', 'e-')
do_plot_v1(real_h_dep_ps, fake_h_dep_ps,'dep_showershape_logy','', 'e-')
real_h_cell_E = real.produce_cell_energy()
fake_h_cell_E = fake.produce_cell_energy()
do_plot_v1(real_h_cell_E, fake_h_cell_E,'cell_energy_logxlogy','', 'e-')
real_h_prob = real.produce_prob('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen1008_mc_reco.h5', 'Disc_real', 0, N_event)
fake_h_prob = fake.produce_prob('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen1008_mc_reco.h5', 'Disc_fake', 0, N_event)
do_plot_v1(real_h_prob, fake_h_prob,'prob','', 'e-')

'''
show_real = False
show_fake = True
show_details = False
hf = 0
nB = 0
n3 = 0
event_list=0
plot_path=""
if show_real:
    hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/Hit_Barrel_e.h5', 'r')
    nB = hf['Barrel_Hit'][10000:10100]
    n3 = hf['MC_info'][10000:10100]
    hf.close()
    #event_list=[10,20,30,40,50,60,70,80,90]
    #event_list=range(50)
    event_list=range(10)
    plot_path="./plots_event_display/real"
    print (nB.shape, n3.shape)
elif show_fake:
    hf = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen0904v2.h5', 'r')
    nB = hf['Barrel_Hit']
    nB = np.squeeze(nB)
    n3 = hf['MC_info']
    #event_list=range(0,50)
    event_list=range(10)
    plot_path="./plots_event_display/fake"
    print (nB.shape, n3.shape)
else: 
    print ('wrong config')
    sys.exit()


for event in event_list:
    if n3 is not None:
        print ('event=%d, e- Direction: theta=%f, phi=%f. Energy=%f GeV'%(event, n3[event,0], n3[event,1], n3[event,2]))
    nRow = nB[event].shape[0]
    nCol = nB[event].shape[1]
    nDep = nB[event].shape[2]
    print ('nRow=',nRow,',nCol=',nCol,',nDep=',nDep)
    str1 = "_gen" if show_fake else ""
    ## z-y plane ## 
    h_Hit_B_z_y = rt.TH2F('Hit_B_z_y_evt%d'%(event)  , '', nCol, 0, nCol, nRow, 0, nRow)
    for i in range(0, nRow):
        for j in range(0, nCol):
            h_Hit_B_z_y.Fill(j+0.01, i+0.01, sum(nB[event,i,j,:]))
    do_plot(event, h_Hit_B_z_y,'Hit_Barrel_z_y_evt%d_%s'%(event,str1),'', 'e-')
    ## z-dep or z-x plane ## 
    h_Hit_B_z_dep = rt.TH2F('Hit_B_z_dep_evt%d'%(event)  , '', nCol, 0, nCol, nDep, 0, nDep)
    for i in range(0, nDep):
        for j in range(0, nCol):
            h_Hit_B_z_dep.Fill(j+0.01, i+0.01, sum(nB[event,:,j,i]))
    do_plot(event, h_Hit_B_z_dep,'Hit_Barrel_z_dep_evt%d_%s'%(event,str1),'', 'e-')
    ## y-dep or y-x plane ##
    h_Hit_B_y_dep = rt.TH2F('Hit_B_y_dep_evt%d'%(event)  , '', nRow, 0, nRow, nDep, 0, nDep)
    for i in range(0, nDep):
        for j in range(0, nRow):
            h_Hit_B_y_dep.Fill(j+0.01, i+0.01, sum(nB[event,j,:,i]))
    do_plot(event, h_Hit_B_y_dep,'Hit_Barrel_y_dep_evt%d_%s'%(event,str1),'', 'e-')
    ## for z-y plane at each layer## 
    if show_details == False : continue
    for z in range(nDep): 
        h_Hit_B = rt.TH2F('Hit_B_evt%d_layer%d'%(event,z+1)  , '', nCol, 0, nCol, nRow, 0, nRow)
        for i in range(0, nRow):
            for j in range(0, nCol):
                h_Hit_B.Fill(j+0.01, i+0.01, nB[event,i,j,z])
        do_plot(event, h_Hit_B,'Hit_Barrel_evt%d_layer%d_%s'%(event,z+1,str1),'', 'e-')

'''
