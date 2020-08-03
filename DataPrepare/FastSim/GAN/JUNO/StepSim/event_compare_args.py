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

def add_info(s_content):
    lowX=0.1
    lowY=0.8
    info  = rt.TPaveText(lowX, lowY+0.06, lowX+0.30, lowY+0.16, "NDC")
    info.SetBorderSize(   0 )
    info.SetFillStyle(    0 )
    info.SetTextAlign(   12 )
    info.SetTextColor(    1 )
    info.SetTextSize(0.025)
    info.SetTextFont (   42 )
    info.AddText("%s"%(str(s_content)))
    return info


class Obj2:
    def __init__(self, name, fileName1, fileName2, evt_start, evt_end):
        self.name = name
        self.fileName1 = fileName1
        self.fileName2 = fileName2
        hf1 = h5py.File(self.fileName1, 'r')
        hf2 = h5py.File(self.fileName2, 'r')
        self.nB1   = hf1['Barrel_Hit'][evt_start:evt_end]
        self.info1 = hf1['MC_info'   ][evt_start:evt_end]
        self.nB2   = hf2['Barrel_Hit'][evt_start:evt_end]
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
    def __init__(self, name, fileName, is_real, evt_start, evt_end):
        self.name = name
        self.is_real = is_real
        self.fileName = fileName
        hf = h5py.File(self.fileName, 'r')
        self.nPE   = np.round(hf['nPEByPMT'][evt_start:evt_end])
        self.nEvt = self.nPE.shape[0]
        self.nCol = self.nPE.shape[1]
        hf.close()
        
    def produce_npe(self):
        str1 = "" if self.is_real else "_gen"
        H_npe = rt.TH1F('H_npe_%s'%(str1)  , '', 20, 0, 20)
        for i in range(self.nEvt):# use event loop, in order to get the correct bin stat. error
            H_npe.Fill(np.sum(self.nPE[i,0:100])+0.01)
        return H_npe


    def produce_npe_pmt(self, pmt_i):
        str1 = "" if self.is_real else "_gen"
        H_npe = rt.TH1F('H_npe_pmt_%d_%s'%(pmt_i, str1)  , '', 5, 0, 5)
        for i in range(self.nEvt):# use event loop, in order to get the correct bin stat. error
            H_npe.Fill(self.nPE[i,pmt_i]+0.01)
        return H_npe



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
    norm_label = add_info(str('pmt %d'%pmt_i))
    norm_label.Draw()
    canvas.SaveAs("%s/%s_%s.png"%(plot_path, out_name, tag))
    del canvas
    gc.collect()


def do_plot_v1(h_real,h_fake,out_name,tag , pmt_i):
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
    elif "prob" in out_name:
        x_min=0.3
        x_max=0.8
    dummy = rt.TH2D("dummy","",nbin,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_Y_title=""
    dummy_X_title=""
    if "n_pe" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "N PE"
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
    pmt_label = add_info(str(pmt_i))
    pmt_label.Draw()
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


    return parser

if __name__ == '__main__':
    parser = get_parser()
    parse_args = parser.parse_args()
    data_real  = parse_args.real_file
    data_fake  = parse_args.fake_file
    N_event    = parse_args.event
    tag        = parse_args.tag

    plot_path='plot_comparision'
    #N_event = 5000
    ##N_event = 1400
    #data_real ='/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_em_zcut_Low.h5'
    #data_fake ='/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1029_em_Low_add139.h5' 
 
    #real_fake = Obj2('real-fake', data_real, data_fake, 0, N_event)
    #real_fake_e3x3 = real_fake.produce_e3x3_energy()
    #real_fake_e5x5 = real_fake.produce_e5x5_energy()
    #real_fake_e3x3_res, e3x3_res_list = real_fake.produce_e3x3_res()
    #real_fake_e5x5_res, e5x5_res_list = real_fake.produce_e5x5_res()
    ### at 5% significance_level, significance_level=array([ 15. ,  10. ,   5. ,   2.5,   1. ]
    #is_e3x3_norm = 1 if anderson(e3x3_res_list).critical_values[2] > anderson(e3x3_res_list).statistic else 0
    #is_e5x5_norm = 1 if anderson(e5x5_res_list).critical_values[2] > anderson(e5x5_res_list).statistic else 0
    #do_plot_v(real_fake_e3x3, 'reak_fake_e3x3_diff',tag, 'e-', 0)
    #do_plot_v(real_fake_e5x5, 'reak_fake_e5x5_diff',tag, 'e-', 0)
    #do_plot_v(real_fake_e3x3_res, 'reak_fake_e3x3_res',tag, 'e-', is_e3x3_norm)
    #do_plot_v(real_fake_e5x5_res, 'reak_fake_e5x5_res',tag, 'e-', is_e5x5_norm)

    #############################################################
    real = Obj('real', data_real, True , 0, N_event)
    fake = Obj('real', data_fake, False, 0, N_event)
    real_h_npe = real.produce_npe()
    fake_h_npe = fake.produce_npe()
    print('real_h_npe=',real_h_npe.GetSumOfWeights())
    print('fale_h_npe=',fake_h_npe.GetSumOfWeights())
    do_plot_v1(real_h_npe, fake_h_npe,'n_pe'     ,tag, '')
    do_plot_v1(real_h_npe, fake_h_npe,'n_pe_logy',tag, '')
    '''
    for i in range(100):
        real_h_npe_pmt = real.produce_npe_pmt(i)
        fake_h_npe_pmt = fake.produce_npe_pmt(i)
        do_plot_v1(real_h_npe_pmt, fake_h_npe_pmt,'n_pe_pmt_%d'%i     ,tag, str('pmt %d'%i))
        do_plot_v1(real_h_npe_pmt, fake_h_npe_pmt,'n_pe_pmt_%d_logy'%i,tag, str('pmt %d'%i))
    '''
