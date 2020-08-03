import ROOT as rt
import numpy as np
import h5py 
#import pandas as pd
import sys 
import math 
import gc
import ast
import argparse
from scipy.stats import anderson, ks_2samp
rt.gROOT.SetBatch(rt.kTRUE)
##
# add MEA
##
def add_info(s_name, s_content):
    lowX=0.1
    lowY=0.8
    info  = rt.TPaveText(lowX, lowY+0.06, lowX+0.30, lowY+0.16, "NDC")
    info.SetBorderSize(   0 )
    info.SetFillStyle(    0 )
    info.SetTextAlign(   12 )
    info.SetTextColor(    1 )
    info.SetTextSize(0.025)
    info.SetTextFont (   42 )
    info.AddText("%s%s"%(str(s_name), str(s_content)))
    return info


class Obj2:
    def __init__(self, name, fileName1, fileName2, evt_start, evt_end, use_mea):
        self.name = name
        self.fileName1 = fileName1
        self.fileName2 = fileName2
        hf1 = h5py.File(self.fileName1, 'r')
        hf2 = h5py.File(self.fileName2, 'r')
        self.nB1   = hf1['Barrel_Hit'][evt_start:evt_end]
        self.info1 = hf1['MC_info'   ][evt_start:evt_end]
        self.nB2   = hf2['Barrel_Hit'][evt_start:evt_end] if use_mea==False else hf2['Barrel_Hit_MEA'][evt_start:evt_end]
        self.info2 = hf2['MC_info'   ][evt_start:evt_end]

        if forHighZ or forLowZ:
            dele_list = []
            for i in range(self.info1.shape[0]):
                if forLowZ  and abs(self.info1[i,5])>100:
                    dele_list.append(i) 
                if forHighZ and abs(self.info1[i,5])<100:
                    dele_list.append(i) 
            self.info1   = np.delete(self.info1   , dele_list, axis = 0)
            self.nB1     = np.delete(self.nB1     , dele_list, axis = 0)
            self.info2   = np.delete(self.info2   , dele_list, axis = 0)
            self.nB2     = np.delete(self.nB2     , dele_list, axis = 0)

        self.nEvt = self.nB1.shape[0]
        self.nRow = self.nB1.shape[1]
        self.nCol = self.nB1.shape[2]
        self.nDep = self.nB1.shape[3]
        hf1.close()
        hf2.close()

    def produce_e5x5_energy(self):## 
        H_e5x5_diff = rt.TH1F('H_e5x5_diff', '', 200, -1, 1)
        for i in range(self.nEvt):
            result1 = self.nB1[i,3:8,3:8,:]
            result2 = self.nB2[i,3:8,3:8,:]
            H_e5x5_diff.Fill(np.sum(result1)-np.sum(result2))#  GeV
        return H_e5x5_diff

    def produce_e5x5_res(self):## 
        H_e5x5_res = rt.TH1F('H_e5x5_res', '', 200, -1, 1)
        tmp_list=[]
        for i in range(self.nEvt):
            result1 = self.nB1[i,3:8,3:8,:]
            result2 = self.nB2[i,3:8,3:8,:]
            H_e5x5_res.Fill((np.sum(result1)-np.sum(result2))/np.sum(result1))#  GeV
            tmp_list.append((np.sum(result1)-np.sum(result2))/np.sum(result1))
        return (H_e5x5_res,tmp_list)


    def produce_e3x3_energy(self):## 
        H_e3x3_diff = rt.TH1F('H_e3x3_diff', '', 200, -1, 1)
        for i in range(self.nEvt):
            result1 = self.nB1[i,4:7,4:7,:]
            result2 = self.nB2[i,4:7,4:7,:]
            H_e3x3_diff.Fill(np.sum(result1)-np.sum(result2))#  GeV
        return H_e3x3_diff

    def produce_e3x3_res(self):## 
        H_e3x3_res = rt.TH1F('H_e3x3_res', '', 200, -1, 1)
        tmp_list=[]
        for i in range(self.nEvt):
            result1 = self.nB1[i,4:7,4:7,:]
            result2 = self.nB2[i,4:7,4:7,:]
            H_e3x3_res.Fill((np.sum(result1)-np.sum(result2))/np.sum(result1))#  GeV
            tmp_list.append((np.sum(result1)-np.sum(result2))/np.sum(result1))
        return (H_e3x3_res,tmp_list)

class Obj:
    def __init__(self, name, fileName, is_real, evt_start, evt_end, use_mea):
        self.name = name
        self.is_real = is_real
        self.fileName = fileName
        hf = h5py.File(self.fileName, 'r')
        self.nB   = hf['Barrel_Hit'][evt_start:evt_end] if use_mea==False else hf['Barrel_Hit_MEA'][evt_start:evt_end] 
        self.info = hf['MC_info'   ][evt_start:evt_end]

        if forHighZ or forLowZ:
            dele_list = []
            for i in range(self.info.shape[0]):
                if forLowZ  and abs(self.info[i,5])>125:
                    dele_list.append(i) 
                if forHighZ and abs(self.info[i,5])<125:
                    dele_list.append(i) 
            self.info   = np.delete(self.info   , dele_list, axis = 0)
            self.nB     = np.delete(self.nB     , dele_list, axis = 0)

        self.nEvt = self.nB.shape[0]
        self.nRow = self.nB.shape[1]
        self.nCol = self.nB.shape[2]
        self.nDep = self.nB.shape[3]
        hf.close()
        
    def produce_z_sp(self):## produce showershape in z direction
        str1 = "" if self.is_real else "_gen"
        H_z_sp = rt.TH1F('H_z_sp_%s'%(str1)  , '', self.nCol, 0, self.nCol)
        for i in range(self.nEvt):# use event loop, in order to get the correct bin stat. error
            for j in range(0, self.nCol):
                H_z_sp.Fill(j+0.01, np.sum(self.nB[i,:,j,:]))
        #for j in range(0, self.nCol):
        #    H_z_sp.Fill(j+0.01, np.sum(self.nB[:,:,j,:]))
        return H_z_sp

    def produce_y_sp(self):## produce showershape in y direction
        str1 = "" if self.is_real else "_gen"
        H_y_sp = rt.TH1F('H_y_sp_%s'%(str1)  , '', self.nRow, 0, self.nRow)
        for i in range(self.nEvt):
            for j in range(0, self.nRow):
                H_y_sp.Fill(j+0.01, np.sum(self.nB[i,j,:,:]))
        #for j in range(0, self.nRow):
        #    H_y_sp.Fill(j+0.01, np.sum(self.nB[:,j,:,:]))
        return H_y_sp
         
    def produce_dep_sp(self):## produce showershape in dep direction
        str1 = "" if self.is_real else "_gen"
        H_dep_sp = rt.TH1F('H_dep_sp_%s'%(str1)  , '', 20, 90, 110)
        for i in range(self.nEvt):
            H_dep_sp.Fill(self.nD[i:0])
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
        H_cell_sum_E = rt.TH1F('H_cell_sum_E_%s'%(str1)  , '', 200, 1, 2)
        hit_sum = np.sum(self.nB, axis=(1,2,3), keepdims=False)
        for i in range(self.nEvt):
            H_cell_sum_E.Fill(hit_sum[i])#  GeV
        return H_cell_sum_E

    def produce_ennergy_diff(self):## 
        str1 = "" if self.is_real else "_gen"
        H_diff_sum_E = rt.TH1F('H_diff_sum_E_%s'%(str1)  , '', 200, -1, 1)
        hit_sum = np.sum(self.nB, axis=(1,2,3), keepdims=False)
        for i in range(self.nEvt):
            H_diff_sum_E.Fill(hit_sum[i]-self.info[i,0])#  GeV
        return H_diff_sum_E

    def produce_ennergy_ratio(self):## 
        str1 = "" if self.is_real else "_gen"
        H_ratio_sum_E = rt.TH1F('H_ratio_sum_E_%s'%(str1)  , '', 100, 0.2, 1.2)
        hit_sum = np.sum(self.nB, axis=(1,2,3), keepdims=False)
        for i in range(self.nEvt):
            H_ratio_sum_E.Fill(hit_sum[i]/self.info[i,0])
        return H_ratio_sum_E

    def produce_e3x3_energy(self):## 
        str1 = "" if self.is_real else "_gen"
        H_e3x3_E = rt.TH1F('H_e3x3_E_%s'%(str1)  , '', 100, 1, 2)
        for i in range(self.nEvt):
            result = self.nB[i,4:7,4:7,:]
            H_e3x3_E.Fill(np.sum(result))#  GeV
        return H_e3x3_E

    def produce_e5x5_energy(self):## 
        tmp_list=[]
        str1 = "" if self.is_real else "_gen"
        H_e5x5_E = rt.TH1F('H_e5x5_E_%s'%(str1)  , '', 100, 1, 2)
        for i in range(self.nEvt):
            result = self.nB[i,3:8,3:8,:]
            H_e5x5_E.Fill  (np.sum(result))#  GeV
            tmp_list.append(np.sum(result))
        return (H_e5x5_E,tmp_list)

    def produce_e3x3_ratio(self):## 
        str1 = "" if self.is_real else "_gen"
        H_e3x3_ratio = rt.TH1F('H_e3x3_ratio_%s'%(str1)  , '', 100, 0.2, 1.2)
        for i in range(self.nEvt):
            result = self.nB[i,4:7,4:7,:]
            H_e3x3_ratio.Fill(np.sum(result)/self.info[i,0])
        return H_e3x3_ratio

    def produce_e5x5_ratio(self):## 
        str1 = "" if self.is_real else "_gen"
        H_e5x5_ratio = rt.TH1F('H_e5x5_ratio_%s'%(str1)  , '', 100, 0.2, 1.2)
        for i in range(self.nEvt):
            result = self.nB[i,3:8,3:8,:]
            H_e5x5_ratio.Fill(np.sum(result)/self.info[i,0])
        return H_e5x5_ratio

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


def do_plot_v(h,out_name,tag , str_particle, isNorm):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.12)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    str_norm = "Events"
    nbin =h.GetNbinsX()
    x_min=h.GetBinLowEdge(1)
    x_max=h.GetBinLowEdge(nbin)+h.GetBinWidth(nbin)
    y_min=0
    y_max=1.5*h.GetBinContent(h.GetMaximumBin())
    if "logx" in out_name:
        canvas.SetLogx()
    if "logy" in out_name:
        canvas.SetLogy()
        y_min = 1e-3
    if "cell_energy" in out_name:
        y_min = 1e-4
        y_max = 1
        if do_norm == False:
            y_min = 1e-1
            y_max = 1e6
    elif "prob" in out_name:
        x_min=0.3
        x_max=0.8
    dummy = rt.TH2D("dummy","",nbin,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_Y_title=""
    dummy_X_title=""
    if "z_showershape" in out_name:
        dummy_Y_title = "Energy (GeV)"
        dummy_X_title = "cell Z"
    elif "e3x3_diff" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "real_{e3x3}-fake_{e3x3}"
    elif "e5x5_diff" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "real_{e5x5}-fake_{e5x5}"
    elif "e3x3_res" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "(real_{e3x3}-fake_{e3x3})/real_{e3x3}"
    elif "e5x5_res" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "(real_{e5x5}-fake_{e5x5})/real_{e5x5}"
    elif "phi_showershape" in out_name:
        dummy_Y_title = "Energy (GeV)"
        dummy_X_title = "cell #phi"
    elif "dep_showershape" in out_name:
        dummy_Y_title = "Energy (GeV)"
        dummy_X_title = "cell X"
    elif "cell_energy" in out_name:
        dummy_Y_title = "Hits"
        dummy_X_title = "Energy deposit per Hit (MeV)"
    elif "cell_sum_energy" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "#sum hit energy (GeV)"
    elif "diff_sum_energy" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{#sumhit}-E_{true} (GeV)"
    elif "ratio_energy" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{#sumhit}/E_{true}"
    elif "ratio_e3x3" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{3x3}/E_{true}"
    elif "ratio_e5x5" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{5x5}/E_{true}"
    elif "e3x3_energy" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{3x3} (GeV)"
    elif "e5x5_energy" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{5x5} (GeV)"
    elif "prob" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "Real/Fake"
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    dummy.GetXaxis().SetTitle(dummy_X_title)
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    if "cell_energy" not in out_name:
        dummy.GetXaxis().SetMoreLogLabels()
        dummy.GetXaxis().SetNoExponent()  
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw()
    h.SetStats(rt.kTRUE)
    h.SetLineWidth(2)
    h.SetLineColor(rt.kRed)
    h.SetMarkerColor(rt.kRed)
    h.SetMarkerStyle(20)
    h.Draw("sames:pe")
    dummy.Draw("AXISSAME")
    legend = rt.TLegend(0.6,0.7,0.85,0.85)
    #legend.AddEntry(h_real,'Data','lep')
    legend.AddEntry(h,'real-fake','lep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    #legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    #legend.Draw()
    norm_label = add_info(s_name='isNorm=', s_content=isNorm)
    norm_label.Draw()
    canvas.SaveAs("%s/%s_%s.png"%(plot_path, out_name, tag))
    del canvas
    gc.collect()


def do_plot_v1(h_real,h_fake,out_name,tag , str_particle):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    str_norm = "Events"
    do_norm = False
#    if out_name != 'cell_sum_energy':
    if do_norm:
        h_real.Scale(1/h_real.GetSumOfWeights())
        h_fake.Scale(1/h_fake.GetSumOfWeights())
        str_norm = "Normalized"
    nbin =h_real.GetNbinsX()
    x_min=h_real.GetBinLowEdge(1)
    x_max=h_real.GetBinLowEdge(nbin)+h_real.GetBinWidth(nbin)
    y_min=0
    y_max=1.5*h_real.GetBinContent(h_real.GetMaximumBin())
    if "logx" in out_name:
        canvas.SetLogx()
    if "logy" in out_name:
        canvas.SetLogy()
        y_min = 1e-1
    if "cell_energy" in out_name:
        y_min = 1e-4
        y_max = 1
        if do_norm == False:
            y_min = 1e-1
            y_max = 1e6
    elif "prob" in out_name:
        x_min=0.3
        x_max=0.8
    dummy = rt.TH2D("dummy","",nbin,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_Y_title=""
    dummy_X_title=""
    if "z_showershape" in out_name:
        dummy_Y_title = "Energy (GeV)"
        dummy_X_title = "cell Z"
    elif "phi_showershape" in out_name:
        dummy_Y_title = "Energy (GeV)"
        dummy_X_title = "cell #phi"
    elif "dep_showershape" in out_name:
        dummy_Y_title = "Energy (GeV)"
        dummy_X_title = "cell X"
    elif "cell_energy" in out_name:
        dummy_Y_title = "Hits"
        dummy_X_title = "Energy deposit per Hit (MeV)"
    elif "cell_sum_energy" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "#sum hit energy (GeV)"
    elif "diff_sum_energy" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{#sumhit}-E_{true} (GeV)"
    elif "ratio_energy" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{#sumhit}/E_{true}"
    elif "ratio_e3x3" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{3x3}/E_{true}"
    elif "ratio_e5x5" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{5x5}/E_{true}"
    elif "e3x3_energy" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{3x3} (GeV)"
    elif "e5x5_energy" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{5x5} (GeV)"
    elif "prob" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "Real/Fake"
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    dummy.GetXaxis().SetTitle(dummy_X_title)
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    if "cell_energy" not in out_name:
        dummy.GetXaxis().SetMoreLogLabels()
        dummy.GetXaxis().SetNoExponent()  
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw()
    h_real.SetLineWidth(2)
    h_fake.SetLineWidth(2)
    h_real.SetLineColor(rt.kRed)
    h_fake.SetLineColor(rt.kBlue)
    h_real.SetMarkerColor(rt.kRed)
    h_fake.SetMarkerColor(rt.kBlue)
    h_real.SetMarkerStyle(20)
    h_fake.SetMarkerStyle(21)
    h_real.Draw("same:pe")
    h_fake.Draw("same:histe")
    dummy.Draw("AXISSAME")
    legend = rt.TLegend(0.6,0.7,0.85,0.85)
    #legend.AddEntry(h_real,'Data','lep')
    legend.AddEntry(h_real,'G4','lep')
    legend.AddEntry(h_fake,"GAN",'lep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    #legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()
    canvas.SaveAs("%s/%s_%s.png"%(plot_path, out_name, tag))
    del canvas
    gc.collect()


def get_parser():
    parser = argparse.ArgumentParser(
        description='Run',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--event', action='store', type=int, default=0,
                        help='Number of epochs to train for.')

    parser.add_argument('--real_file'    , action='store', type=str, default='',  help='')
    parser.add_argument('--fake_file'    , action='store', type=str, default='',  help='')
    parser.add_argument('--tag'          , action='store', type=str, default='',  help='')
    parser.add_argument('--forHighZ'     , action='store', type=ast.literal_eval, default=False,  help='')
    parser.add_argument('--forLowZ'      , action='store', type=ast.literal_eval, default=False,  help='')
    parser.add_argument('--useMEA'       , action='store', type=ast.literal_eval, default=False,  help='')


    return parser

if __name__ == '__main__':
    parser = get_parser()
    parse_args = parser.parse_args()
    data_real  = parse_args.real_file
    data_fake  = parse_args.fake_file
    N_event    = parse_args.event
    tag        = parse_args.tag
    forHighZ   = parse_args.forHighZ
    forLowZ    = parse_args.forLowZ
    useMEA     = parse_args.useMEA

    plot_path='plot_comparision'
    #N_event = 5000
    ##N_event = 1400
    #data_real ='/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_em_zcut_Low.h5'
    #data_fake ='/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1029_em_Low_add139.h5' 
 
    real_fake = Obj2('real-fake', data_real, data_fake, 0, N_event, useMEA)
    real_fake_e3x3 = real_fake.produce_e3x3_energy()
    real_fake_e5x5 = real_fake.produce_e5x5_energy()
    real_fake_e3x3_res, e3x3_res_list = real_fake.produce_e3x3_res()
    real_fake_e5x5_res, e5x5_res_list = real_fake.produce_e5x5_res()
    ## at 5% significance_level, significance_level=array([ 15. ,  10. ,   5. ,   2.5,   1. ]
    is_e3x3_norm = 1 if anderson(e3x3_res_list).critical_values[2] > anderson(e3x3_res_list).statistic else 0
    is_e5x5_norm = 1 if anderson(e5x5_res_list).critical_values[2] > anderson(e5x5_res_list).statistic else 0
    do_plot_v(real_fake_e3x3, 'reak_fake_e3x3_diff',tag, 'e-', 0)
    do_plot_v(real_fake_e5x5, 'reak_fake_e5x5_diff',tag, 'e-', 0)
    do_plot_v(real_fake_e3x3_res, 'reak_fake_e3x3_res',tag, 'e-', is_e3x3_norm)
    do_plot_v(real_fake_e5x5_res, 'reak_fake_e5x5_res',tag, 'e-', is_e5x5_norm)

    #############################################################
    real = Obj('real', data_real, True , 0, N_event, False)
    fake = Obj('fake', data_fake, False, 0, N_event, useMEA)
    real_h_z_ps = real.produce_z_sp()
    fake_h_z_ps = fake.produce_z_sp()
    do_plot_v1(real_h_z_ps, fake_h_z_ps,'z_showershape'     ,tag, 'e-')
    do_plot_v1(real_h_z_ps, fake_h_z_ps,'z_showershape_logy',tag, 'e-')
    real_h_y_ps = real.produce_y_sp()
    fake_h_y_ps = fake.produce_y_sp()
    do_plot_v1(real_h_y_ps, fake_h_y_ps,'phi_showershape'     ,tag, 'e-')
    do_plot_v1(real_h_y_ps, fake_h_y_ps,'phi_showershape_logy',tag, 'e-')
    real_h_cell_E = real.produce_cell_energy()
    fake_h_cell_E = fake.produce_cell_energy()
    do_plot_v1(real_h_cell_E, fake_h_cell_E,'cell_energy_logxlogy',tag, 'e-')
    
    real_h_cell_sum_E = real.produce_cell_sum_energy()
    fake_h_cell_sum_E = fake.produce_cell_sum_energy()
    do_plot_v1(real_h_cell_sum_E, fake_h_cell_sum_E,'cell_sum_energy',tag, 'e-')
    
    real_h_diff_sum_E = real.produce_ennergy_diff()
    fake_h_diff_sum_E = fake.produce_ennergy_diff()
    do_plot_v1(real_h_diff_sum_E, fake_h_diff_sum_E,'diff_sum_energy',tag, 'e-')
    
    real_ratio = real.produce_ennergy_ratio()
    fake_ratio = fake.produce_ennergy_ratio()
    do_plot_v1(real_ratio, fake_ratio,'ratio_energy',tag, 'e-')
    
    real_e3x3_ratio = real.produce_e3x3_ratio()
    fake_e3x3_ratio = fake.produce_e3x3_ratio()
    do_plot_v1(real_e3x3_ratio, fake_e3x3_ratio,'ratio_e3x3',tag, 'e-')
    real_e5x5_ratio = real.produce_e5x5_ratio()
    fake_e5x5_ratio = fake.produce_e5x5_ratio()
    do_plot_v1(real_e5x5_ratio, fake_e5x5_ratio,'ratio_e5x5',tag, 'e-')
    
    real_e3x3 = real.produce_e3x3_energy()
    fake_e3x3 = fake.produce_e3x3_energy()
    do_plot_v1(real_e3x3, fake_e3x3,'e3x3_energy',tag, 'e-')
    real_e5x5, real_e5x5_list = real.produce_e5x5_energy()
    fake_e5x5, fake_e5x5_list = fake.produce_e5x5_energy()
    ks_result_e5x5     = ks_2samp(real_e5x5_list , fake_e5x5_list).statistic
    alpha = 0.05
    c_alpha = 1.36
    ks_size = N_event
    D_aplha = c_alpha*math.sqrt((ks_size+ks_size)/(ks_size*ks_size))
    pass_test = 1 if ks_result_e5x5 < D_aplha else 0
    print('tag=',tag,',pass_test=',pass_test,',ks_e5x5=',ks_result_e5x5)
    do_plot_v1(real_e5x5, fake_e5x5,'e5x5_energy',tag, 'e-')


    if 'wgan' in tag or 'stb' in tag:sys.exit() 
    real_h_prob = real.produce_prob(data_fake, 'Disc_real', 0, N_event)
    fake_h_prob = fake.produce_prob(data_fake, 'Disc_fake', 0, N_event)
    do_plot_v1(real_h_prob, fake_h_prob,'prob',tag, 'e-')

