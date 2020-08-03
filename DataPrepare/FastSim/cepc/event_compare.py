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

    def produce_cell_sum_energy(self):## 
        str1 = "" if self.is_real else "_gen"
        H_cell_sum_E = rt.TH1F('H_cell_sum_E_%s'%(str1)  , '', 100, 0, 100)
        hit_sum = np.sum(self.nB, axis=(1,2,3), keepdims=False)
        for i in range(self.nEvt):
            H_cell_sum_E.Fill(hit_sum[i])#  GeV
        return H_cell_sum_E

    def produce_e3x3_energy(self):## 
        str1 = "" if self.is_real else "_gen"
        H_e3x3_E = rt.TH1F('H_e3x3_E_%s'%(str1)  , '', 100, 0, 100)
        for i in range(self.nEvt):
            result = self.nB[i,14:17,14:17,:]
            H_e3x3_E.Fill(np.sum(result))#  GeV
        return H_e3x3_E

    def produce_e5x5_energy(self):## 
        str1 = "" if self.is_real else "_gen"
        H_e5x5_E = rt.TH1F('H_e5x5_E_%s'%(str1)  , '', 100, 0, 100)
        for i in range(self.nEvt):
            result = self.nB[i,13:18,13:18,:]
            H_e5x5_E.Fill(np.sum(result))#  GeV
        return H_e5x5_E

    def produce_e3x3_ratio(self):## 
        str1 = "" if self.is_real else "_gen"
        H_e3x3_ratio = rt.TH1F('H_e3x3_ratio_%s'%(str1)  , '', 100, 0, 1)
        for i in range(self.nEvt):
            result = self.nB[i,14:17,14:17,:]
            H_e3x3_ratio.Fill(np.sum(result)/self.info[i,0])
        return H_e3x3_ratio

    def produce_e5x5_ratio(self):## 
        str1 = "" if self.is_real else "_gen"
        H_e5x5_ratio = rt.TH1F('H_e5x5_ratio_%s'%(str1)  , '', 100, 0, 1)
        for i in range(self.nEvt):
            result = self.nB[i,13:18,13:18,:]
            H_e5x5_ratio.Fill(np.sum(result)/self.info[i,0])
        return H_e5x5_ratio

    def produce_ennergy_diff(self):## 
        str1 = "" if self.is_real else "_gen"
        H_diff_sum_E = rt.TH1F('H_diff_sum_E_%s'%(str1)  , '', 100, -50, 50)
        hit_sum = np.sum(self.nB, axis=(1,2,3), keepdims=False)
        for i in range(self.nEvt):
            H_diff_sum_E.Fill(hit_sum[i]-self.info[i,0])#  GeV
        return H_diff_sum_E

    def produce_ennergy_ratio(self):## 
        str1 = "" if self.is_real else "_gen"
        H_ratio_sum_E = rt.TH1F('H_ratio_sum_E_%s'%(str1)  , '', 100, 0, 2)
        hit_sum = np.sum(self.nB, axis=(1,2,3), keepdims=False)
        for i in range(self.nEvt):
            H_ratio_sum_E.Fill(hit_sum[i]/self.info[i,0])
        return H_ratio_sum_E

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
    elif "cell_sum_energy" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "#sum hit energy (GeV)"
    elif "diff_sum_energy" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "E_{#sumhit}-E_{true} (GeV)"
    elif "ratio_energy" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "E_{#sumhit}/E_{true}"
    elif "ratio_e3x3" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "E_{3x3}/E_{true}"
    elif "ratio_e5x5" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "E_{5x5}/E_{true}"
    elif "e3x3_energy" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "E_{3x3} (GeV)"
    elif "e5x5_energy" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "E_{5x5} (GeV)"
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

plot_path='plot_comparision'
N_event = 5000

str_particle = 'e^{-}'
real_file = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/e-/em_10.h5'
fake_file = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen_em_1104_epoch15.h5'

#str_particle = '#gamma'
#real_file = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/e-/Digi_1_100_em_ext_9.h5'
#fake_file = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen1025_em.h5'

real = Obj('real', real_file, True , 0, N_event)
fake = Obj('real', fake_file, False, 0, N_event)
real_h_z_ps = real.produce_z_sp()
fake_h_z_ps = fake.produce_z_sp()
do_plot_v1(real_h_z_ps, fake_h_z_ps,'z_showershape','', str_particle)
do_plot_v1(real_h_z_ps, fake_h_z_ps,'z_showershape_logy','', str_particle)
real_h_y_ps = real.produce_y_sp()
fake_h_y_ps = fake.produce_y_sp()
do_plot_v1(real_h_y_ps, fake_h_y_ps,'y_showershape','', str_particle)
do_plot_v1(real_h_y_ps, fake_h_y_ps,'y_showershape_logy','', str_particle)
real_h_dep_ps = real.produce_dep_sp()
fake_h_dep_ps = fake.produce_dep_sp()
do_plot_v1(real_h_dep_ps, fake_h_dep_ps,'dep_showershape','', str_particle)
do_plot_v1(real_h_dep_ps, fake_h_dep_ps,'dep_showershape_logy','', str_particle)
#real_h_cell_E = real.produce_cell_energy()
#fake_h_cell_E = fake.produce_cell_energy()
#do_plot_v1(real_h_cell_E, fake_h_cell_E,'cell_energy_logxlogy','', str_particle)
real_h_cell_sum_E = real.produce_cell_sum_energy()
fake_h_cell_sum_E = fake.produce_cell_sum_energy()
do_plot_v1(real_h_cell_sum_E, fake_h_cell_sum_E,'cell_sum_energy','', str_particle)
real_h_diff_sum_E = real.produce_ennergy_diff()
fake_h_diff_sum_E = fake.produce_ennergy_diff()
do_plot_v1(real_h_diff_sum_E, fake_h_diff_sum_E,'diff_sum_energy','', str_particle)

real_ratio = real.produce_ennergy_ratio()
fake_ratio = fake.produce_ennergy_ratio()
do_plot_v1(real_ratio, fake_ratio,'ratio_energy','', str_particle)

real_e3x3_ratio = real.produce_e3x3_ratio()
fake_e3x3_ratio = fake.produce_e3x3_ratio()
do_plot_v1(real_e3x3_ratio, fake_e3x3_ratio,'ratio_e3x3','', str_particle)
real_e5x5_ratio = real.produce_e5x5_ratio()
fake_e5x5_ratio = fake.produce_e5x5_ratio()
do_plot_v1(real_e5x5_ratio, fake_e5x5_ratio,'ratio_e5x5','', str_particle)

real_e3x3 = real.produce_e3x3_energy()
fake_e3x3 = fake.produce_e3x3_energy()
do_plot_v1(real_e3x3, fake_e3x3,'e3x3_energy','', str_particle)
real_e5x5 = real.produce_e5x5_energy()
fake_e5x5 = fake.produce_e5x5_energy()
do_plot_v1(real_e5x5, fake_e5x5,'e5x5_energy','', str_particle)

real_h_prob = real.produce_prob(fake_file, 'Disc_real', 0, N_event)
fake_h_prob = fake.produce_prob(fake_file, 'Disc_fake', 0, N_event)
do_plot_v1(real_h_prob, fake_h_prob,'prob','', str_particle)

