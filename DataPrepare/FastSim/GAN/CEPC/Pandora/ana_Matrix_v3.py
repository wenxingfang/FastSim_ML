import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import json
import argparse
from array import array
rt.gROOT.SetBatch(rt.kTRUE)
rt.TGaxis.SetMaxDigits(3);
## analysis relation ##
def get_parser():
    parser = argparse.ArgumentParser(
        description='root to hdf5',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', action='store', type=str,
                        help='input root file')
    parser.add_argument('--output', action='store', type=str, default='',
                        help='output root file')
    parser.add_argument('--tag', action='store', type=str,
                        help='tag name for plots')
    parser.add_argument('--str_particle', action='store', type=str,
                        help='e^{-}')
    parser.add_argument('--E_res_low' , action='store', type=float, default = -0.5, help='')
    parser.add_argument('--E_res_high', action='store', type=float, default =  0.5, help='')
    parser.add_argument('--Theta_res_low' , action='store', type=float, default = -0.5, help='')
    parser.add_argument('--Theta_res_high', action='store', type=float, default =  0.5, help='')
    parser.add_argument('--Phi_res_low' , action='store', type=float, default = -0.5, help='')
    parser.add_argument('--Phi_res_high', action='store', type=float, default =  0.5, help='')


    return parser



def getTheta(r, z):
    if z == 0: return 90
    phi = math.atan(r/z)
    phi = 180*phi/math.pi
    if phi < 0 : phi = phi + 180
    return phi

def findLed(X, Y, Z, vPDG, pdg ):
    max_ = -1
    index = -1
    for i in range(len(X)):
        if vPDG[i] != pdg:continue
        if (X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i]) > max_ :
            max_ = (X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i])
            index = i
    return index

def findLed0(X, Y, Z):
    max_ = -1
    index = -1
    for i in range(len(X)):
        if (X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i]) > max_ :
            max_ = (X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i])
            index = i
    return index



def findLed_abs(X, Y, Z, vPDG, pdg ):
    max_ = -1
    index = -1
    for i in range(len(X)):
        if abs(vPDG[i]) != pdg:continue
        if (X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i]) > max_ :
            max_ = (X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i])
            index = i
    return index


def findLedv1(E, Type, type_, y, z ):
    max_ = -1
    index = -1
    for i in range(len(E)):
        if Type[i] != type_:continue
        if abs(y[i]) > 600 or abs(z[i]) > 2000 :continue
        if E[i] > max_ :
            max_ = E[i]
            index = i
    return index


def findSub(X, Y, Z, led, vPDG, pdg):
    max_ = -1
    index = -1
    for i in range(len(X)):
        if i == led: continue
        if vPDG[i] != pdg:continue
        if (X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i]) > max_ :
            max_ = (X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i])
            index = i
    return index


def findSubv1(E, led, Type, type_, y, z ):
    max_ = -1
    index = -1
    for i in range(len(E)):
        if i == led: continue
        if Type[i] != type_:continue
        if abs(y[i]) > 600 or abs(z[i]) > 2000 :continue
        if E[i] > max_ :
            max_ = E[i]
            index = i
    return index


def getPhi(x, y):
    if x == 0 and y == 0: return 0
    elif x == 0 and y > 0: return 90
    elif x == 0 and y < 0: return 270
    phi = math.atan(y/x)
    phi = 180*phi/math.pi
    if x < 0 : phi = phi + 180
    elif x > 0 and y < 0 : phi = phi + 360
    return phi

def getID(x, y, z, lookup):
    tmp_ID = 0
    id_z = int(z/10)
    id_z = str(id_z)
    id_phi = int(getPhi(x, y))
    id_phi = str(id_phi)
    if id_z not in lookup:
        print('exception id_z=', id_z)
        return tmp_ID
    min_distance = 999
    for ID in lookup[id_z][id_phi]:
        c_x = float(lookup[id_z][id_phi][ID][0])
        c_y = float(lookup[id_z][id_phi][ID][1])
        c_z = float(lookup[id_z][id_phi][ID][2])
        distance = math.sqrt( math.pow(x-c_x,2) + math.pow(y-c_y,2) + math.pow(z-c_z,2) )
        if  distance < min_distance :
            min_distance = distance
            tmp_ID = ID
    return int(tmp_ID) 

def do_plot_2D(hist,out_name,title, region_label):
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
    #hist.SetStats(rt.kFALSE)
    ystr = ''
    if 'GeV/c^{2}' in title['X']:
        ystr = 'GeV/c^{2}'
    elif 'GeV/c' in title['X']:
        ystr = 'GeV/c'
    elif 'GeV' in title['X']:
        ystr = 'GeV'
    hist.GetYaxis().SetTitle(str(title['Y']+' / %.1f %s'%(hist.GetBinWidth(1), ystr)))
    hist.GetXaxis().SetTitle(title['X'])
    hist.GetXaxis().SetTitleOffset(1.2)
    hist.Draw("COLZ")

    label = rt.TLatex(0.32, 0.89, region_label)
    label.SetTextAlign(32)
    label.SetTextSize(0.03)
    label.SetNDC(rt.kTRUE)
    label.Draw() 
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

def do_plot(hist,out_name,title, region_label):
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
    ystr = ''
    if 'GeV/c^{2}' in title['X']:
        ystr = 'GeV/c^{2}'
    elif 'GeV/c' in title['X']:
        ystr = 'GeV/c'
    elif 'GeV' in title['X']:
        ystr = 'GeV'
    hist.GetYaxis().SetTitle(str(title['Y']+' / %.1f %s'%(hist.GetBinWidth(1), ystr)))
    hist.GetXaxis().SetTitle(title['X'])
    hist.GetXaxis().SetTitleOffset(1.2)
    hist.SetLineColor(rt.kBlue)
    hist.SetMarkerColor(rt.kBlue)
    hist.SetMarkerStyle(20)
    hist.Draw("histep")

    label = rt.TLatex(0.32, 0.89, region_label)
    label.SetTextAlign(32)
    label.SetTextSize(0.03)
    label.SetNDC(rt.kTRUE)
    label.Draw() 
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()


def do_eff_plot(h_num, h_deo,out_name,title, region_label):
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
    #hist.SetStats(rt.kFALSE)
    if 'logx' in out_name:
        canvas.SetLogx()
    if 'logy' in out_name:
        canvas.SetLogy()
    ystr = ''
    if 'GeV/c^{2}' in title['X']:
        ystr = 'GeV/c^{2}'
    elif 'GeV/c' in title['X']:
        ystr = 'GeV/c'
    elif 'GeV' in title['X']:
        ystr = 'GeV'
    h_num.Sumw2()
    h_deo.Sumw2()
    graph_eff = rt.TGraphAsymmErrors()
    graph_eff.Divide(h_num, h_deo,"cl=0.683 b(1,1) mode")
    #graph_eff.GetYaxis().SetTitle(str(title['Y']+' / %.1f %s'%(hist.GetBinWidth(1), ystr)))
    graph_eff.GetYaxis().SetTitle('Efficiency')
    graph_eff.GetXaxis().SetTitle(title['X'])
    graph_eff.GetXaxis().SetTitleOffset(1.2)
    graph_eff.SetLineColor(rt.kBlue)
    graph_eff.SetMarkerColor(rt.kBlue)
    graph_eff.SetMarkerStyle(20)
    if 'any' in out_name:
        graph_eff.SetMaximum(1.05)
        graph_eff.SetMinimum(0.95)
    graph_eff.Draw('ap')
    #####################
    f_reco_eff_out = rt.TFile(root_out, 'UPDATE')
    f_reco_eff_out.cd()
    graph_eff.Write(out_name)
    f_reco_eff_out.Close()
    ######################
    label = rt.TLatex(0.25 , 0.89, region_label)
    label.SetTextAlign(32)
    label.SetTextSize(0.03)
    label.SetNDC(rt.kTRUE)
    label.Draw() 
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()



def do_plot_h3(h_real, h_real1, h_fake, out_name, title, str_particle):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)

#    h_real.Scale(1/h_real.GetSumOfWeights())
#    h_fake.Scale(1/h_fake.GetSumOfWeights())
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
    ystr = ''
    if 'GeV/c^{2}' in title['X']:
        ystr = 'GeV/c^{2}'
    elif 'GeV/c' in title['X']:
        ystr = 'GeV/c'
    elif 'GeV' in title['X']:
        ystr = 'GeV'
    dummy.GetYaxis().SetTitle(str(title['Y']+' / %.1f %s'%(h_real.GetBinWidth(1), ystr)))
    dummy.GetXaxis().SetTitle(title['X'])
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.5)
    dummy.GetXaxis().SetTitleOffset(1.5)
    if "cell_energy" not in out_name:
        dummy.GetXaxis().SetMoreLogLabels()
        dummy.GetXaxis().SetNoExponent()  
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw()
    h_real .SetLineColor(rt.kRed)
    h_real1.SetLineColor(rt.kGreen)
    h_fake .SetLineColor(rt.kBlue)
    h_real .SetMarkerColor(rt.kRed)
    h_real1.SetMarkerColor(rt.kGreen)
    h_fake .SetMarkerColor(rt.kBlue)
    h_real .SetMarkerStyle(20)
    h_real1.SetMarkerStyle(22)
    h_fake .SetMarkerStyle(21)
    h_real.Draw("same:pe")
    h_real.Draw("same:pe")
    h_fake.Draw("same:histe")
    dummy.Draw("AXISSAME")
    legend = rt.TLegend(0.75,0.7,0.85,0.85)
    legend.AddEntry(h_real ,'G4','lep')
    legend.AddEntry(h_real1,'G4 (w/o EB Hits)','lep')
    legend.AddEntry(h_fake ,"GAN",'lep')
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

def do_plot_h2(h_real, h_fake, out_name, title, plot_label):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.12)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)

#    h_real.Scale(1/h_real.GetSumOfWeights())
#    h_fake.Scale(1/h_fake.GetSumOfWeights())
    nbin =h_real.GetNbinsX()
    x_min=h_real.GetBinLowEdge(1)
    x_max=h_real.GetBinLowEdge(nbin)+h_real.GetBinWidth(nbin)
    y_min=0
    y_max=1.5*h_real.GetBinContent(h_real.GetMaximumBin()) if h_real.GetBinContent(h_real.GetMaximumBin()) > h_fake.GetBinContent(h_fake.GetMaximumBin()) else 1.5*h_fake.GetBinContent(h_fake.GetMaximumBin()) 
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
    ystr = ''
    if 'GeV/c^{2}' in title['Y']:
        ystr = 'GeV/c^{2}'
    elif 'GeV/c' in title['Y']:
        ystr = 'GeV/c'
    elif 'GeV' in title['Y']:
        ystr = 'GeV'
    dummy.GetYaxis().SetTitle(str(title['Y']+' / %.1f %s'%(h_real.GetBinWidth(1), ystr)))
    dummy.GetXaxis().SetTitle(title['X'])
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    if 'cm_x' in out_name or 'cm_y' in out_name or 'cm_z' in out_name:
        dummy.GetXaxis().SetLabelSize(0.03)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    if "cell_energy" not in out_name:
        dummy.GetXaxis().SetMoreLogLabels()
        dummy.GetXaxis().SetNoExponent()  
    dummy.GetXaxis().SetMoreLogLabels()
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw()
    h_real .SetLineWidth(2)
    h_fake .SetLineWidth(2)
    h_real .SetLineColor(rt.kRed)
    h_fake .SetLineColor(rt.kBlue)
    h_real .SetMarkerColor(rt.kRed)
    h_fake .SetMarkerColor(rt.kBlue)
    h_real .SetMarkerStyle(20)
    h_fake .SetMarkerStyle(21)
    h_real.Draw("same:pe")
    h_real.Draw("same:pe")
    h_fake.Draw("same:histe")
    dummy.Draw("AXISSAME")
    legend = rt.TLegend(0.6,0.7,0.85,0.85)
    legend.AddEntry(h_real ,'G4','lep')
    legend.AddEntry(h_fake ,"GAN",'lep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    #legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()

    label = rt.TLatex(0.38 , 0.82, plot_label)
    label.SetTextAlign(32)
    label.SetTextSize(0.035)
    label.SetNDC(rt.kTRUE)
    label.Draw() 
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

###############
class Obj:
    def __init__(self, name, fileName, max_evt):
        self.name = name
        self.max_evt= max_evt
        self.file_name = fileName
        str_ext = self.name
        self.Led_PID = 22
        self.Sub_PID = 22
        self.doAll = False
        phi_bins = 100
        reco_eff_E_bins = list(range(0,110,1))
        #reco_eff_E_bins = [10]
        #reco_eff_E_bins.append(5)
        #reco_eff_E_bins.append(1)
        #reco_eff_E_bins.append(0.1)
        reco_eff_E_bins.sort()
        self.binsArray_reco_eff_E = array('d', reco_eff_E_bins)
        self.h_mc_theta_all   = rt.TH1F('%s_h_mc_theta_all'     %(str_ext) ,'',210,-10,200)
        self.h_mc_phi_all     = rt.TH1F('%s_h_mc_phi_all'       %(str_ext) ,'',phi_bins, -100 ,100)
        self.h_mc_E_all       = rt.TH1F('%s_h_mc_E_all'         %(str_ext) ,'',15,0,150)
        self.h_mc_theta       = rt.TH1F('%s_h_mc_theta'     %(str_ext) ,'',210,-10,200)
        self.h_mc_phi         = rt.TH1F('%s_h_mc_phi'       %(str_ext) ,'',phi_bins, -100 ,100)
        self.h_mc_px          = rt.TH1F('%s_h_mc_px'        %(str_ext) ,'',20,-100,100)
        self.h_mc_py          = rt.TH1F('%s_h_mc_py'        %(str_ext) ,'',20,-100,100)
        self.h_mc_pz          = rt.TH1F('%s_h_mc_pz'        %(str_ext) ,'',20,-100,100)
        self.h_mc_E           = rt.TH1F('%s_h_mc_E'         %(str_ext) ,'',15,0,150)
        self.h_reco_mc_E_den     = rt.TH1F('%s_h_reco_mc_E_den'%(str_ext)     ,'',len(reco_eff_E_bins)-1, self.binsArray_reco_eff_E)
        self.h_reco_mc_E_num     = rt.TH1F('%s_h_reco_mc_E_num'%(str_ext)     ,'',len(reco_eff_E_bins)-1, self.binsArray_reco_eff_E)
        self.h_reco_mc_E_num_any = rt.TH1F('%s_h_reco_mc_E_num_any'%(str_ext) ,'',len(reco_eff_E_bins)-1, self.binsArray_reco_eff_E)
        self.h_reco_mc_E_num_Barrel  = rt.TH1F('%s_h_reco_mc_E_num_Barrel'%(str_ext)  ,'',len(reco_eff_E_bins)-1, self.binsArray_reco_eff_E)
        self.h_reco_mc_E_den_Barrel  = rt.TH1F('%s_h_reco_mc_E_den_Barrel'%(str_ext)  ,'',len(reco_eff_E_bins)-1, self.binsArray_reco_eff_E)
        self.h_reco_mc_E_num_Endcap  = rt.TH1F('%s_h_reco_mc_E_num_Endcap'%(str_ext)  ,'',len(reco_eff_E_bins)-1, self.binsArray_reco_eff_E)
        self.h_reco_mc_E_den_Endcap  = rt.TH1F('%s_h_reco_mc_E_den_Endcap'%(str_ext)  ,'',len(reco_eff_E_bins)-1, self.binsArray_reco_eff_E)
        self.h_reco_mc_E_num_Gap  = rt.TH1F('%s_h_reco_mc_E_num_Gap'%(str_ext)  ,'',len(reco_eff_E_bins)-1, self.binsArray_reco_eff_E)
        self.h_reco_mc_E_den_Gap  = rt.TH1F('%s_h_reco_mc_E_den_Gap'%(str_ext)  ,'',len(reco_eff_E_bins)-1, self.binsArray_reco_eff_E)
        #self.h_reco_mc_E_num_ep  = rt.TH1F('%s_h_reco_mc_E_num_ep'%(str_ext)  ,'',len(reco_eff_E_bins)-1, self.binsArray_reco_eff_E)
        #self.h_reco_mc_E_den  = rt.TH1F('%s_h_reco_mc_E_den'%(str_ext) ,'',15,0,150)
        #self.h_reco_mc_E_num  = rt.TH1F('%s_h_reco_mc_E_num'%(str_ext) ,'',15,0,150)
        self.h_reco_mc_phi_den= rt.TH1F('%s_h_reco_mc_phi_den'%(str_ext) ,'',phi_bins,-100,100)
        self.h_reco_mc_phi_num= rt.TH1F('%s_h_reco_mc_phi_num'%(str_ext) ,'',phi_bins,-100,100)
        self.h_rec_theta      = rt.TH1F('%s_h_rec_theta'    %(str_ext) ,'',210,-10,200)
        self.h_rec_phi        = rt.TH1F('%s_h_rec_phi'      %(str_ext) ,'',phi_bins, -100 ,100)

        self.h_N_reco              = rt.TH1F('%s_h_N_reco'     %(str_ext) ,'',100,0,10)
        self.h_reco_E              = rt.TH1F('%s_h_reco_E'            %(str_ext) ,'',150,0,150)
        self.h_reco_px             = rt.TH1F('%s_h_reco_px'           %(str_ext) ,'',20,-100,100)
        self.h_reco_py             = rt.TH1F('%s_h_reco_py'           %(str_ext) ,'',20,-100,100)
        self.h_reco_pz             = rt.TH1F('%s_h_reco_pz'           %(str_ext) ,'',20,-100,100)
        self.h_reco_mc_weight      = rt.TH1F('%s_h_reco_mc_weight'    %(str_ext) ,'',100, 0, 1)
        self.h_reco_led_E          = rt.TH1F('%s_h_reco_led_E'        %(str_ext) ,'',150,0,150)
        self.h_reco_led_px         = rt.TH1F('%s_h_reco_led_px'       %(str_ext) ,'',20,-100,100)
        self.h_reco_led_py         = rt.TH1F('%s_h_reco_led_py'       %(str_ext) ,'',20,-100,100)
        self.h_reco_led_pz         = rt.TH1F('%s_h_reco_led_pz'       %(str_ext) ,'',20,-100,100)
        self.h_reco_led_E_res      = rt.TH1F('%s_h_reco_led_E_res'    %(str_ext) ,'',100,E_res_low, E_res_high)
        self.h_reco_led_theta_res  = rt.TH1F('%s_h_reco_led_theta_res'%(str_ext) ,'',40, Theta_res_low, Theta_res_high)
        self.h_reco_led_phi_res    = rt.TH1F('%s_h_reco_led_phi_res'  %(str_ext) ,'',40, Phi_res_low  , Phi_res_high)
        self.h_reco_led_mc_weight  = rt.TH1F('%s_h_reco_led_mc_weight' %(str_ext) ,'',100, 0, 1)
        self.h_reco_sub_E          = rt.TH1F('%s_h_reco_sub_E'        %(str_ext) ,'',150,0,150)
        self.h_reco_sub_px         = rt.TH1F('%s_h_reco_sub_px'       %(str_ext) ,'',20,-100,100)
        self.h_reco_sub_py         = rt.TH1F('%s_h_reco_sub_py'       %(str_ext) ,'',20,-100,100)
        self.h_reco_sub_pz         = rt.TH1F('%s_h_reco_sub_pz'       %(str_ext) ,'',20,-100,100)
        self.h_reco_sub_E_res      = rt.TH1F('%s_h_reco_sub_E_res'    %(str_ext) ,'',100,E_res_low, E_res_high)
        self.h_reco_sub_theta_res  = rt.TH1F('%s_h_reco_sub_theta_res'%(str_ext) ,'',40, Theta_res_low, Theta_res_high)
        self.h_reco_sub_phi_res    = rt.TH1F('%s_h_reco_sub_phi_res'  %(str_ext) ,'',40, Phi_res_low  , Phi_res_high)
        self.h_reco_sub_mc_weight  = rt.TH1F('%s_h_reco_sub_mc_weight' %(str_ext) ,'',100, 0, 1)

        self.h_hit_x_y    = rt.TH2F('h_hit_x_y','',400,1800,2200, 400, -200, 200)
        self.h_hit_x_z    = rt.TH2F('h_hit_x_z','',400,1800,2200, 400, -200, 200)
        self.h_hit_y_z    = rt.TH2F('h_hit_y_z','',400,-200,200 , 400, -200, 200)
        self.h_hit_E      = rt.TH1F('h_hit_E'  ,'',100, 0, 1)

    def setPID(self, p1, p2):
        self.Led_PID = p1
        self.Sub_PID = p2
    def setDoAll(self, doAll):
        self.doAll = doAll
    def fill(self):
        P4_mc = rt.TLorentzVector()
        P4_rec = rt.TLorentzVector()
        P4_rec_Led = rt.TLorentzVector()
        P4_rec_Sub = rt.TLorentzVector()
        P4_cms = rt.TLorentzVector(0,0,0,240)   
        FileName = self.file_name
        treeName='evt'
        chain =rt.TChain(treeName)
        chain.Add(FileName)
        tree = chain
        totalEntries=tree.GetEntries()
        const0 = 180/math.pi
        max_evt = self.max_evt if self.max_evt < totalEntries else totalEntries
        print ('tot evt=',totalEntries,',max evt=',max_evt)
        fout = rt.TFile(out_root,'RECREATE')
        #E_bin = list(range(0,110,10))
        E_bin = list(range(0,100,10))
        E_bin.append(95)
        E_bin.append(100)
        Theta_bin = list(range(0,190,10))
        Phi_bin = list(range(-90,100,10))
        E_bin_hist_n_gamma   = []
        E_bin_hist_pid     = []
        E_bin_hist_E         = []
        E_bin_hist_Theta     = []
        E_bin_hist_Phi       = []
        Theta_bin_hist_E     = []
        Theta_bin_hist_Theta = []
        Theta_bin_hist_Phi   = []
        Phi_bin_hist_E       = []
        Phi_bin_hist_Theta   = []
        Phi_bin_hist_Phi     = []
        N_bin_E_res   = 40  
        Min_bin_E_res = -0.2  
        Max_bin_E_res =  0.2  
        N_bin_Theta_res   = 40  
        Min_bin_Theta_res = -0.2  
        Max_bin_Theta_res =  0.2  
        N_bin_Phi_res   = 40  
        Min_bin_Phi_res = -0.2  
        Max_bin_Phi_res =  0.2  
        for i in range(len(self.binsArray_reco_eff_E)-1):
            E_bin_hist_n_gamma  .append( rt.TH1F('E_%.1f_%.1f_n_gamma'  %(self.binsArray_reco_eff_E[i], self.binsArray_reco_eff_E[i+1]), '', 10  , 0 , 10  ) )
            #E_bin_hist_pid    .append( rt.TH1F('E_%.1f_%.1f_pid'    %(self.binsArray_reco_eff_E[i], self.binsArray_reco_eff_E[i+1]), '', 6000  , -3000 , 3000  ) )
            E_bin_hist_pid    .append( rt.TH1F('E_%.1f_%.1f_pid'    %(self.binsArray_reco_eff_E[i], self.binsArray_reco_eff_E[i+1]), '', 100  , 0 , 100  ) )
        for i in range(len(E_bin)-1):
            E_bin_hist_E    .append( rt.TH1F('E_%d_%d_E'    %(E_bin[i], E_bin[i+1]), '', N_bin_E_res    , Min_bin_E_res    , Max_bin_E_res    ) )
            E_bin_hist_Theta.append( rt.TH1F('E_%d_%d_Theta'%(E_bin[i], E_bin[i+1]), '', N_bin_Theta_res, Min_bin_Theta_res, Max_bin_Theta_res) )
            E_bin_hist_Phi  .append( rt.TH1F('E_%d_%d_Phi'  %(E_bin[i], E_bin[i+1]), '', N_bin_Phi_res  , Min_bin_Phi_res  , Max_bin_Phi_res  ) )
        for i in range(len(Theta_bin)-1):
            Theta_bin_hist_E    .append( rt.TH1F('Theta_%d_%d_E'    %(Theta_bin[i], Theta_bin[i+1]), '', N_bin_E_res    , Min_bin_E_res    , Max_bin_E_res    ) )
            Theta_bin_hist_Theta.append( rt.TH1F('Theta_%d_%d_Theta'%(Theta_bin[i], Theta_bin[i+1]), '', N_bin_Theta_res, Min_bin_Theta_res, Max_bin_Theta_res) )
            Theta_bin_hist_Phi  .append( rt.TH1F('Theta_%d_%d_Phi'  %(Theta_bin[i], Theta_bin[i+1]), '', N_bin_Phi_res  , Min_bin_Phi_res  , Max_bin_Phi_res  ) )
        for i in range(len(Phi_bin)-1):
            Phi_bin_hist_E    .append( rt.TH1F('Phi_%d_%d_E'    %(Phi_bin[i], Phi_bin[i+1]), '', N_bin_E_res    , Min_bin_E_res    , Max_bin_E_res    ) )
            Phi_bin_hist_Theta.append( rt.TH1F('Phi_%d_%d_Theta'%(Phi_bin[i], Phi_bin[i+1]), '', N_bin_Theta_res, Min_bin_Theta_res, Max_bin_Theta_res) )
            Phi_bin_hist_Phi  .append( rt.TH1F('Phi_%d_%d_Phi'  %(Phi_bin[i], Phi_bin[i+1]), '', N_bin_Phi_res  , Min_bin_Phi_res  , Max_bin_Phi_res  ) )
        Nskip = 0
        for entryNum in range(0, max_evt):
            tree.GetEntry(entryNum)
            if entryNum%10000 == 0: print('processed %d evt'%entryNum)
            #if entryNum%1 == 0: print('processed %d evt'%entryNum)
            tmp_mc_p_size = getattr(tree, "m_mc_p_size")
            tmp_mc_pid    = getattr(tree, "m_mc_pid"   )
            tmp_mc_mass   = getattr(tree, "m_mc_mass"  )
            tmp_mc_px     = getattr(tree, "m_mc_px"    )
            tmp_mc_py     = getattr(tree, "m_mc_py"    )
            tmp_mc_pz     = getattr(tree, "m_mc_pz"    )
            tmp_mc_charge = getattr(tree, "m_mc_charge")
            tmp_mc_id     = getattr(tree, "m_mc_id")
             
            tmp_hasConversion = 0
            if removeGammaConversion:
                tmp_hasConversion = getattr(tree, "m_hasConversion")

            tmp_pReco_mc_id       = getattr(tree, "m_pReco_mc_id")
            tmp_pReco_mc_weight   = getattr(tree, "m_pReco_mc_weight")
            tmp_PID   = getattr(tree, "m_pReco_PID")
            tmp_mass  = getattr(tree, "m_pReco_mass")
            tmp_charge= getattr(tree, "m_pReco_charge")
            tmp_E     = getattr(tree, "m_pReco_energy")
            tmp_px    = getattr(tree, "m_pReco_px")
            tmp_py    = getattr(tree, "m_pReco_py")
            tmp_pz    = getattr(tree, "m_pReco_pz")
            skip = False
            for i in range(len(tmp_PID)):
                if tmp_PID[i] != 22:
                    skip = True
                    break
            if skip: 
                Nskip += 1
                continue
            self.h_N_reco.Fill(len(tmp_PID))
            if len(tmp_PID)==1 :
                self.h_reco_E .Fill(tmp_E[0] )
                self.h_reco_px.Fill(tmp_px[0])
                self.h_reco_py.Fill(tmp_py[0])
                self.h_reco_pz.Fill(tmp_pz[0])
                for imc in range(len(tmp_pReco_mc_id[0])):
                    self.h_reco_mc_weight.Fill(tmp_pReco_mc_weight[0][imc])
            
            if len(tmp_PID)==2 :
                led_index = -1
                sub_index = -1
                if(tmp_E[0] > tmp_E[1]):
                    P4_rec_Led.SetPxPyPzE(tmp_px[0],tmp_py[0],tmp_pz[0],tmp_E[0])
                    P4_rec_Sub.SetPxPyPzE(tmp_px[1],tmp_py[1],tmp_pz[1],tmp_E[1])
                    led_index = 0
                    sub_index = 1
                else:
                    P4_rec_Led.SetPxPyPzE(tmp_px[1],tmp_py[1],tmp_pz[1],tmp_E[1])
                    P4_rec_Sub.SetPxPyPzE(tmp_px[0],tmp_py[0],tmp_pz[0],tmp_E[0])
                    led_index = 1
                    sub_index = 0
                    
                    self.h_reco_led_E .Fill(P4_rec_Led.E() )
                    self.h_reco_led_px.Fill(P4_rec_Led.Px())
                    self.h_reco_led_py.Fill(P4_rec_Led.Py())
                    self.h_reco_led_pz.Fill(P4_rec_Led.Pz())
                    self.h_reco_sub_E .Fill(P4_rec_Sub.E() )
                    self.h_reco_sub_px.Fill(P4_rec_Sub.Px())
                    self.h_reco_sub_py.Fill(P4_rec_Sub.Py())
                    self.h_reco_sub_pz.Fill(P4_rec_Sub.Pz())
                    max_weight = -1
                    led_mc_id = -1
                    sub_mc_id = -1
                    for imc in range(len(tmp_pReco_mc_id[led_index])):
                        if(tmp_pReco_mc_weight[led_index][imc] > max_weight):
                            led_mc_id  = tmp_pReco_mc_id    [led_index][imc] 
                            max_weight = tmp_pReco_mc_weight[led_index][imc]
                        self.h_reco_led_mc_weight.Fill(tmp_pReco_mc_weight[led_index][imc]) 
                    max_weight = -1
                    for imc in range(len(tmp_pReco_mc_id[sub_index])):
                        if(tmp_pReco_mc_weight[sub_index][imc] > max_weight):
                            sub_mc_id  = tmp_pReco_mc_id    [sub_index][imc] 
                            max_weight = tmp_pReco_mc_weight[sub_index][imc]
                        self.h_reco_sub_mc_weight.Fill(tmp_pReco_mc_weight[sub_index][imc]) 
                    for imc in range(len(tmp_mc_id)):
                        mc_E = math.sqrt( tmp_mc_px[imc]*tmp_mc_px[imc] + tmp_mc_py[imc]*tmp_mc_py[imc] + tmp_mc_pz[imc]*tmp_mc_pz[imc] + tmp_mc_mass[imc]*tmp_mc_mass[imc])
                        P4_mc.SetPxPyPzE( tmp_mc_px[imc], tmp_mc_py[imc], tmp_mc_pz[imc], mc_E);
                        if tmp_mc_id[imc] == led_mc_id:
                            self.h_reco_led_E_res     .Fill( (P4_rec_Led.E()    -P4_mc.E())/ P4_mc.E())
                            self.h_reco_led_theta_res .Fill( (P4_rec_Led.Theta()-P4_mc.Theta())*const0 ) 
                            self.h_reco_led_phi_res   .Fill( (P4_rec_Led.Phi()  -P4_mc.Phi()  )*const0 )
                        if tmp_mc_id[imc] == sub_mc_id:
                            self.h_reco_sub_E_res     .Fill( (P4_rec_Sub.E()    -P4_mc.E())/ P4_mc.E())
                            self.h_reco_sub_theta_res .Fill( (P4_rec_Sub.Theta()-P4_mc.Theta())*const0 ) 
                            self.h_reco_sub_phi_res   .Fill( (P4_rec_Sub.Phi()  -P4_mc.Phi()  )*const0 )
        print("Nskip =%d, max_evt=%d, skip ratio=%f"%(Nskip, max_evt, float(Nskip)/float(max_evt)))                    
                    


        
############ BEGIN ##############
if __name__ == '__main__':

    print('Hello Ana')
    parser = get_parser()
    parse_args = parser.parse_args()
    
    str_e = parse_args.str_particle
    E_res_low      = parse_args.E_res_low
    E_res_high     = parse_args.E_res_high
    Theta_res_low  = parse_args.Theta_res_low
    Theta_res_high = parse_args.Theta_res_high
    Phi_res_low    = parse_args.Phi_res_low
    Phi_res_high   = parse_args.Phi_res_high
    plot_path = './plots'
    removeGammaConversion = False
    #str_re = '_noGaCov' if removeGammaConversion == True else ''
    out_root ='Matrix_dummy.root' 
    root_out = parse_args.output
    tag = parse_args.tag
    #obj_pan        = Obj('MatrixPfo','/cefs/higgs/wxfang/cepc/Pandora/Ana/gamma/Ana_gamma_Matrix_1GeV_0602_scale.root', 1000000)
    obj_pan        = Obj('MatrixPfo', parse_args.input, 1000000)
    str_led_pa = "#gamma"
    
    print('Hello 1')
    obj_pan.fill()
    print('Hello 2')
    
    ''' 
    do_plot(obj_pan.h_mc_theta_all ,"Pan_mc_theta_all"        ,{'X':"#theta_{mc all}^{%s} degree"%str_led_pa      ,'Y':'Events'} ,'')
    do_plot(obj_pan.h_mc_phi_all   ,"Pan_mc_phi_all"          ,{'X':"#phi_{mc all}^{%s} degree"%str_led_pa        ,'Y':'Events'} ,'')
    do_plot(obj_pan.h_mc_E_all     ,"Pan_mc_E_all"            ,{'X':"E_{mc all}^{%s} GeV"%str_led_pa              ,'Y':'Events'} ,'')
    do_plot(obj_pan.h_mc_theta     ,"Pan_mc_theta"            ,{'X':"#theta_{mc}^{%s} degree"%str_led_pa           ,'Y':'Events'} ,'')
    do_plot(obj_pan.h_mc_phi       ,"Pan_mc_phi"              ,{'X':"#phi_{mc}^{%s} degree"%str_led_pa             ,'Y':'Events'} ,'')
    do_plot(obj_pan.h_mc_px        ,"Pan_mc_Px"               ,{'X':"Px_{mc}^{%s} GeV/c"%str_led_pa                ,'Y':'Events'} ,'')
    do_plot(obj_pan.h_mc_py        ,"Pan_mc_Py"               ,{'X':"Py_{mc}^{%s} GeV/c"%str_led_pa                ,'Y':'Events'} ,'')
    do_plot(obj_pan.h_mc_pz        ,"Pan_mc_Pz"               ,{'X':"Pz_{mc}^{%s} GeV/c"%str_led_pa                ,'Y':'Events'} ,'')
    do_plot(obj_pan.h_mc_E         ,"Pan_mc_E"                ,{'X':"E_{mc}^{%s} GeV"%str_led_pa                   ,'Y':'Events'} ,'')
    do_plot(obj_pan.h_reco_mc_E_num,"Pan_h_reco_mc_E_num"     ,{'X':"E_{mc}^{%s} GeV"%str_led_pa               ,'Y':'Events'} ,'10<#theta_{mc}<170 (num)')
    do_plot(obj_pan.h_reco_mc_E_den,"Pan_h_reco_mc_E_den"     ,{'X':"E_{mc}^{%s} GeV"%str_led_pa               ,'Y':'Events'} ,'10<#theta_{mc}<170 (den)')
    do_plot(obj_pan.h_reco_mc_phi_num,"Pan_h_reco_mc_phi_num" ,{'X':"#phi_{mc}^{%s} degree"%str_led_pa     ,'Y':'Events'} ,'10<#theta_{mc}<170 (num)')
    do_plot(obj_pan.h_reco_mc_phi_den,"Pan_h_reco_mc_phi_den" ,{'X':"#phi_{mc}^{%s} degree"%str_led_pa     ,'Y':'Events'} ,'10<#theta_{mc}<170 (den)')
    ''' 
    do_plot(obj_pan.h_reco_E             ,"Pan_reco_E_%s"%tag                ,{'X':"E_{pfo}^{%s} GeV"%str_led_pa                  ,'Y':'Events'}, '' )
    do_plot(obj_pan.h_reco_px            ,"Pan_reco_Px_%s"%tag               ,{'X':"Px_{pfo}^{%s} GeV/c"%str_led_pa               ,'Y':'Events'}, '' )
    do_plot(obj_pan.h_reco_py            ,"Pan_reco_Py_%s"%tag               ,{'X':"Py_{pfo}^{%s} GeV/c"%str_led_pa               ,'Y':'Events'}, '' )
    do_plot(obj_pan.h_reco_pz            ,"Pan_reco_Pz_%s"%tag               ,{'X':"Pz_{pfo}^{%s} GeV/c"%str_led_pa               ,'Y':'Events'}, '' )
    do_plot(obj_pan.h_reco_mc_weight     ,"Pan_reco_mc_weight_%s"%tag        ,{'X':"mc weight"                                    ,'Y':'Entries'},'' )
    do_plot(obj_pan.h_reco_led_E         ,"Pan_reco_led_E_%s"%tag            ,{'X':"led E_{pfo}^{%s} GeV"%str_led_pa                  ,'Y':'Events'}, '' )
    do_plot(obj_pan.h_reco_led_px        ,"Pan_reco_led_Px_%s"%tag           ,{'X':"led Px_{pfo}^{%s} GeV/c"%str_led_pa               ,'Y':'Events'}, '' )
    do_plot(obj_pan.h_reco_led_py        ,"Pan_reco_led_Py_%s"%tag           ,{'X':"led Py_{pfo}^{%s} GeV/c"%str_led_pa               ,'Y':'Events'}, '' )
    do_plot(obj_pan.h_reco_led_pz        ,"Pan_reco_led_Pz_%s"%tag           ,{'X':"led Pz_{pfo}^{%s} GeV/c"%str_led_pa               ,'Y':'Events'}, '' )
    do_plot(obj_pan.h_reco_led_E_res     ,"Pan_reco_led_E_res_%s"%tag        ,{'X':"led (E_{pfo}^{%s}-E_{mc}^{%s})/E_{mc}^{%s}"%(str_led_pa,str_led_pa,str_led_pa)  ,'Y':'Events'},'' )
    do_plot(obj_pan.h_reco_led_theta_res ,"Pan_reco_led_theta_res_%s"%tag    ,{'X':"led #theta_{pfo}^{%s}-#theta_{mc}^{%s}"%(str_led_pa,str_led_pa)             ,'Y':'Events'},'' )
    do_plot(obj_pan.h_reco_led_phi_res   ,"Pan_reco_led_phi_res_%s"%tag      ,{'X':"led #phi_{pfo}^{%s}-#phi_{mc}^{%s}"%(str_led_pa,str_led_pa)                 ,'Y':'Events'},'' )
    do_plot(obj_pan.h_reco_led_mc_weight ,"Pan_reco_led_mc_weight_%s"%tag    ,{'X':"led mc weight"                                    ,'Y':'Entries'},'' )

    do_plot(obj_pan.h_reco_sub_E         ,"Pan_reco_sub_E_%s"%tag            ,{'X':"sub E_{pfo}^{%s} GeV"%str_led_pa                  ,'Y':'Events'}, '' )
    do_plot(obj_pan.h_reco_sub_px        ,"Pan_reco_sub_Px_%s"%tag           ,{'X':"sub Px_{pfo}^{%s} GeV/c"%str_led_pa               ,'Y':'Events'}, '' )
    do_plot(obj_pan.h_reco_sub_py        ,"Pan_reco_sub_Py_%s"%tag           ,{'X':"sub Py_{pfo}^{%s} GeV/c"%str_led_pa               ,'Y':'Events'}, '' )
    do_plot(obj_pan.h_reco_sub_pz        ,"Pan_reco_sub_Pz_%s"%tag           ,{'X':"sub Pz_{pfo}^{%s} GeV/c"%str_led_pa               ,'Y':'Events'}, '' )
    do_plot(obj_pan.h_reco_sub_E_res     ,"Pan_reco_sub_E_res_%s"%tag        ,{'X':"sub (E_{pfo}^{%s}-E_{mc}^{%s})/E_{mc}^{%s}"%(str_led_pa,str_led_pa,str_led_pa)  ,'Y':'Events'},'' )
    do_plot(obj_pan.h_reco_sub_theta_res ,"Pan_reco_sub_theta_res_%s"%tag    ,{'X':"sub #theta_{pfo}^{%s}-#theta_{mc}^{%s}"%(str_led_pa,str_led_pa)             ,'Y':'Events'},'' )
    do_plot(obj_pan.h_reco_sub_phi_res   ,"Pan_reco_sub_phi_res_%s"%tag      ,{'X':"sub #phi_{pfo}^{%s}-#phi_{mc}^{%s}"%(str_led_pa,str_led_pa)                 ,'Y':'Events'},'' )
    do_plot(obj_pan.h_reco_sub_mc_weight ,"Pan_reco_sub_mc_weight_%s"%tag    ,{'X':"sub mc weight"                                    ,'Y':'Entries'},'' )
    do_plot(obj_pan.h_N_reco ,"Pan_N_reco_%s"%tag    ,{'X':"N reco"                                    ,'Y':'Entries'},'' )
    #print('saved mc E all=',obj_pan.h_mc_E_all.GetSumOfWeights(),',rec E =',obj_pan.h_rec_E.GetSumOfWeights())
    print('done')
