import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import json
from array import array
rt.gROOT.SetBatch(rt.kTRUE)
rt.TGaxis.SetMaxDigits(3);

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

def do_plot_g2(h1, h2, out_name, title, plot_label, scale):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.12)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)
    if scale: h2.Scale(h1.GetSumOfWeights()/h2.GetSumOfWeights())
#    h1.Scale(1/h1.GetSumOfWeights())
#    h2.Scale(1/h2.GetSumOfWeights())
    x_min=h1.GetXaxis().GetXmin()
    x_max=h1.GetXaxis().GetXmax()
    y_min=0
    y_max=1.5*h1.GetBinContent(h1.GetMaximumBin())
    print('x min=',x_min,',x max=',x_max,',y min=',y_min,',y max=',y_max)
    if 'eff_E' in out_name:
        y_min=0.9
    elif 'eff_phi' in out_name:
        y_min=0.9
    elif 'eff_theta' in out_name:
        y_min=0.9
    elif 'E_mean' in out_name:
        y_min=-0.04
        y_max= 0.01
    elif 'E_std' in out_name:
        y_min= 0.01
        y_max= 0.08
    elif 'Phi_mean' in out_name:
        y_min= 0.0
        y_max= 0.008
    elif 'Phi_std' in out_name:
        y_min= 0.01
        y_max= 0.04
    elif 'Theta_mean' in out_name:
        y_min= -0.01
        y_max= 0.015
    elif 'Theta_std' in out_name:
        y_min= 0.006
        y_max= 0.02
    if "logx" in out_name:
        canvas.SetLogx()
    if "logy" in out_name:
        canvas.SetLogy()
        y_min = 1e-3
    dummy = rt.TH2D("dummy","",1,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy.GetYaxis().SetTitle(title['Y'])
    dummy.GetXaxis().SetTitle(title['X'])
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.8)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetMoreLogLabels()
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw()
    h1 .SetLineWidth(2)
    h2 .SetLineWidth(2)
    h1 .SetLineColor(rt.kRed)
    h2 .SetLineColor(rt.kBlue)
    h1 .SetMarkerColor(rt.kRed)
    h2 .SetMarkerColor(rt.kBlue)
    h1 .SetMarkerStyle(20)
    h2 .SetMarkerStyle(21)
    h1.Draw("same:pe")
    h2.Draw("same:pe")
    dummy.Draw("AXISSAME")
    x_l = 0.6
    x_dl = 0.25
    y_h = 0.85
    y_dh = 0.15
    if 'gr_Ebin_E_mean' in out_name:
        x_l = 0.2
        y_h = 0.85
    legend = rt.TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    legend.AddEntry(h1 ,'G4','lep')
    legend.AddEntry(h2 ,"FS",'lep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    #legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()

    label = rt.TLatex(0.26 , 0.9, plot_label)
    label.SetTextAlign(32)
    label.SetTextSize(0.035)
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
    f_reco_eff_out = rt.TFile(reco_root_out, 'UPDATE')
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
    def __init__(self, name, fileName, max_evt, outfile):
        self.name = name
        self.max_evt= max_evt
        self.file_name = fileName
        str_ext = self.name
        self.Led_PID = 22
        self.Sub_PID = 22
        self.doAll = False
        self.out_root = outfile
        phi_bins = 90
        reco_eff_E_bins = list(range(10,120,10))
        reco_eff_E_bins.append(5)
        reco_eff_E_bins.append(1)
        reco_eff_E_bins.append(0.1)
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
        self.h_rec_theta      = rt.TH1F('%s_h_rec_theta'    %(str_ext) ,'',120,30,150)
        self.h_rec_phi        = rt.TH1F('%s_h_rec_phi'      %(str_ext) ,'',phi_bins, -90 ,90)
        self.h_rec_px         = rt.TH1F('%s_h_rec_px'       %(str_ext) ,'',20,-100,100)
        self.h_rec_py         = rt.TH1F('%s_h_rec_py'       %(str_ext) ,'',20,-100,100)
        self.h_rec_pz         = rt.TH1F('%s_h_rec_pz'       %(str_ext) ,'',20,-100,100)
        self.h_rec_E          = rt.TH1F('%s_h_rec_E'        %(str_ext) ,'',110,0,110)
        self.h_rec_E_res      = rt.TH1F('%s_h_rec_E_res'    %(str_ext) ,'',40,-0.2,0.2)
        self.h_rec_theta_res  = rt.TH1F('%s_h_rec_theta_res'%(str_ext) ,'',40,-0.2,0.2)
        self.h_rec_phi_res    = rt.TH1F('%s_h_rec_phi_res'  %(str_ext) ,'',40,-0.2,0.2)

    def setPID(self, p1, p2):
        self.Led_PID = p1
        self.Sub_PID = p2
    def setDoAll(self, doAll):
        self.doAll = doAll
    def fill(self):
        P4_mc = rt.TLorentzVector()
        P4_rec= rt.TLorentzVector()
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
        fout = rt.TFile(self.out_root,'RECREATE')
        #E_bin = list(range(0,110,10))
        E_bin = list(range(0,100,10))
        E_bin.append(95)
        E_bin.append(100)
        #Theta_bin = list(range(0,190,10))
        Theta_bin = list(range(0,190,5))
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
             
            tmp_hasConversion = 0
            if removeGammaConversion:
                tmp_hasConversion = getattr(tree, "m_hasConversion")

            tmp_PID   = getattr(tree, "m_pReco_PID")
            tmp_mass  = getattr(tree, "m_pReco_mass")
            tmp_charge= getattr(tree, "m_pReco_charge")
            tmp_E     = getattr(tree, "m_pReco_energy")
            tmp_px    = getattr(tree, "m_pReco_px")
            tmp_py    = getattr(tree, "m_pReco_py")
            tmp_pz    = getattr(tree, "m_pReco_pz")

            mc_index = -1
            for i in range(len(tmp_mc_p_size)):
                if tmp_mc_p_size[i]==0 and tmp_mc_pid[i]==22:
                    mc_index = i
                    break
            if mc_index == -1 : continue
            if removeGammaConversion:
                if tmp_hasConversion: continue
                
            mc_E = math.sqrt( tmp_mc_px[mc_index]*tmp_mc_px[mc_index] + tmp_mc_py[mc_index]*tmp_mc_py[mc_index] + tmp_mc_pz[mc_index]*tmp_mc_pz[mc_index] + tmp_mc_mass[mc_index]*tmp_mc_mass[mc_index])
            P4_mc.SetPxPyPzE( tmp_mc_px[mc_index], tmp_mc_py[mc_index], tmp_mc_pz[mc_index], mc_E);
            mc_theta = P4_mc.Theta()*const0 
            mc_phi   = P4_mc.Phi  ()*const0 
            #if mc_theta < 10 or mc_theta > 170:continue#remove very forward region
            if mc_theta < 10 or mc_theta > 170 or mc_E < 0.1:continue#remove very forward region
            if mc_theta > 85 and mc_theta < 95:continue#remove very center region
            #if mc_theta < 50 or mc_theta > 130 or mc_E < 0.1:continue#remove very forward region
            self.h_mc_theta_all.Fill(mc_theta)
            self.h_mc_phi_all  .Fill(mc_phi)
            self.h_mc_E_all    .Fill(P4_mc.E())
            self.h_reco_mc_E_den.Fill(P4_mc.E())
            self.h_reco_mc_phi_den.Fill(mc_phi)
            #if 50 < mc_theta and mc_theta < 130: self.h_reco_mc_E_den_Barrel.Fill(P4_mc.E())
            #elif (30 < mc_theta and mc_theta < 50) or (130 < mc_theta and mc_theta < 150) : self.h_reco_mc_E_den_Gap.Fill(P4_mc.E())
            #else: self.h_reco_mc_E_den_Endcap.Fill(P4_mc.E())
            ###########check ###########
            '''
            if len(tmp_px)>0: self.h_reco_mc_E_num_any.Fill(P4_mc.E())
            led_reco = findLed_abs(tmp_px, tmp_py, tmp_pz, tmp_PID, 11)
            if led_reco != -1: self.h_reco_mc_E_num_em.Fill(P4_mc.E())
            '''
            '''
            n_neutron = 0
            E_neutron = 0
            for i in range(len(tmp_px)):
                if abs(tmp_PID[i]) == 2112: n_neutron += 1
            if n_neutron > 0:
                for i in range(len(tmp_mc_pid)):
                    if abs(tmp_mc_pid[i]) == 2112: 
                        tmp_mc_E = math.sqrt( tmp_mc_px[i]*tmp_mc_px[i] + tmp_mc_py[i]*tmp_mc_py[i] + tmp_mc_pz[i]*tmp_mc_pz[i] + tmp_mc_mass[i]*tmp_mc_mass[i])
                        if tmp_mc_E > E_neutron: E_neutron = tmp_mc_E
            if E_neutron > 0 :
                for i in range(len(self.binsArray_reco_eff_E)-1):
                    if self.binsArray_reco_eff_E[i] < P4_mc.E() and self.binsArray_reco_eff_E[i+1] > P4_mc.E():
                        #for k in range(len(tmp_px)):
                        #    E_bin_hist_pid  [i]    .Fill( tmp_PID[k] )
                        #for k in range(len(tmp_mc_pid)):
                        #    E_bin_hist_pid  [i]    .Fill( tmp_mc_pid[k] )
                        E_bin_hist_pid  [i]    .Fill( E_neutron )
            '''
            ###############
            led_reco = findLed(tmp_px, tmp_py, tmp_pz, tmp_PID, self.Led_PID)
            if led_reco == -1 :continue
            P4_rec.SetPxPyPzE(tmp_px[led_reco],tmp_py[led_reco],tmp_pz[led_reco],tmp_E[led_reco]);
           
            self.h_mc_theta      .Fill(mc_theta) 
            self.h_mc_phi        .Fill(mc_phi  ) 
            self.h_mc_px         .Fill(P4_mc.Px()) 
            self.h_mc_py         .Fill(P4_mc.Py()) 
            self.h_mc_pz         .Fill(P4_mc.Pz()) 
            self.h_mc_E          .Fill(P4_mc.E() ) 

            self.h_reco_mc_E_num.Fill(P4_mc.E())
            #if 50 < mc_theta and mc_theta < 130: self.h_reco_mc_E_num_Barrel.Fill(P4_mc.E())
            #elif (30 < mc_theta and mc_theta < 50) or (130 < mc_theta and mc_theta < 150) : self.h_reco_mc_E_num_Gap.Fill(P4_mc.E())
            #else: self.h_reco_mc_E_num_Endcap.Fill(P4_mc.E())
            self.h_reco_mc_phi_num.Fill(mc_phi)

            
            rec_theta = P4_rec.Theta()*const0 
            rec_phi   = P4_rec.Phi  ()*const0 
            #print('mc theta0=',P4_mc.Theta(),',theta1=',mc_theta, ',rec theta0=',P4_rec.Theta(),',theta1=',rec_theta )
            self.h_rec_theta      .Fill(rec_theta) 
            self.h_rec_phi        .Fill(rec_phi  ) 
            self.h_rec_px         .Fill(P4_rec.Px()) 
            self.h_rec_py         .Fill(P4_rec.Py()) 
            self.h_rec_pz         .Fill(P4_rec.Pz()) 
            self.h_rec_E          .Fill(P4_rec.E() ) 
            self.h_rec_E_res      .Fill( (P4_rec.E()-mc_E)/mc_E )
            self.h_rec_theta_res  .Fill( (P4_rec.Theta()-P4_mc.Theta())*const0 ) 
            self.h_rec_phi_res    .Fill( (P4_rec.Phi()  -P4_mc.Phi()  )*const0 )

            for i in range(len(E_bin)-1):
                if E_bin[i] < P4_rec.E() and E_bin[i+1] > P4_rec.E():
                    E_bin_hist_E[i]    .Fill( (P4_rec.E()-mc_E)/mc_E )
                    E_bin_hist_Theta[i].Fill( rec_theta - mc_theta   )
                    E_bin_hist_Phi[i]  .Fill( rec_phi   - mc_phi     )
                    break
            for i in range(len(Phi_bin)-1):
                if Phi_bin[i] < rec_phi and Phi_bin[i+1] > rec_phi:
                    Phi_bin_hist_E[i]    .Fill( (P4_rec.E()-mc_E)/mc_E)
                    Phi_bin_hist_Phi[i]  .Fill( rec_phi   - mc_phi    )
                    Phi_bin_hist_Theta[i].Fill( rec_theta - mc_theta  )
                    break
            for i in range(len(Theta_bin)-1):
                if Theta_bin[i] < rec_theta and Theta_bin[i+1] > rec_theta:
                    Theta_bin_hist_E[i]    .Fill( (P4_rec.E()-mc_E)/mc_E )
                    Theta_bin_hist_Phi[i]  .Fill( rec_phi   - mc_phi     )
                    Theta_bin_hist_Theta[i].Fill( rec_theta - mc_theta   )
                    break
        fout.cd()
        for hist in E_bin_hist_E: 
            hist.Write()
        for hist in E_bin_hist_Phi: 
            hist.Write()
        for hist in E_bin_hist_Theta: 
            hist.Write()
        for hist in Theta_bin_hist_E: 
            hist.Write()
        for hist in Theta_bin_hist_Phi: 
            hist.Write()
        for hist in Theta_bin_hist_Theta: 
            hist.Write()
        for hist in Phi_bin_hist_E: 
            hist.Write()
        for hist in Phi_bin_hist_Phi: 
            hist.Write()
        for hist in Phi_bin_hist_Theta: 
            hist.Write()
        #for hist in E_bin_hist_n_gamma: 
        #    hist.Write()
        #for hist in E_bin_hist_pid: 
        #    hist.Write()
        fout.Close()
        
############ BEGIN ##############
print('Hello Ana')
plot_path = './pandora_plots'
removeGammaConversion = False
str_re = '_noGaCov' if removeGammaConversion == True else ''
For_low_E = False 
For_high_E = False 
str_e = ''
if For_low_E:
    str_e = '_lowE'
elif For_high_E:
    str_e = '_highE'
########## for barrel #################
out ='G4%s_out.root'%(str_re)
reco_root_out = 'G4_reco_eff.root'
obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/G4/*.root', 1000000, out)

#out ='FS%s%s_out.root'%(str_re,str_e)
#reco_root_out = 'FS_reco_eff%s.root'%(str_e)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp_gap/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp_gap_e/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp_gap_e_cellcenter/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp_faiss/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp_faiss_0518/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp_faiss_0518_cCenter/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp_faiss_0525_cCenter/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp_faiss_ExtendX_0526/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp_Force_dmin_10/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp_Force_L2/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp_Force_Y/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp_Force_Y_addNoFind/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp_Force_Y_addNoFind_0624/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp_Force_Escale_0626/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/Pandora/AnaCaloFS/gamma/FS_sp_Force_All_0628/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/FS/EnScale_apply_digi/*.root', 1000000, out)
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/FS/EnScale_apply_v1_digi/*.root', 1000000, out)
str_led_pa = "#gamma"

########## for endcap #################
#out ='G4_Endcap_out.root'
#reco_root_out = 'G4_Endcap_reco_eff.root'
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/FS/Endcap_G4_digi/*.root', 1000000, out)

#out ='FS_Endcap_out.root'
#reco_root_out = 'FS_Endcap_reco_eff.root'
#obj_pan        = Obj('pan','/cefs/higgs/wxfang/cepc/FS/Endcap_noFound_digi/*.root', 1000000, out)
##################
obj_pan.fill()

do_plot(obj_pan.h_mc_theta_all ,"Pan_mc_theta_all",{'X':"#theta_{mc all}^{%s} degree"%str_led_pa      ,'Y':'Events'} ,'')
do_plot(obj_pan.h_mc_phi_all   ,"Pan_mc_phi_all"  ,{'X':"#phi_{mc all}^{%s} degree"%str_led_pa        ,'Y':'Events'} ,'')
do_plot(obj_pan.h_mc_E_all     ,"Pan_mc_E_all"    ,{'X':"E_{mc all}^{%s} GeV"%str_led_pa              ,'Y':'Events'} ,'')
do_plot(obj_pan.h_mc_theta     ,"Pan_mc_theta"   ,{'X':"#theta_{mc}^{%s} degree"%str_led_pa           ,'Y':'Events'} ,'')
do_plot(obj_pan.h_mc_phi       ,"Pan_mc_phi"     ,{'X':"#phi_{mc}^{%s} degree"%str_led_pa             ,'Y':'Events'} ,'')
do_plot(obj_pan.h_mc_px        ,"Pan_mc_Px"      ,{'X':"Px_{mc}^{%s} GeV/c"%str_led_pa                ,'Y':'Events'} ,'')
do_plot(obj_pan.h_mc_py        ,"Pan_mc_Py"      ,{'X':"Py_{mc}^{%s} GeV/c"%str_led_pa                ,'Y':'Events'} ,'')
do_plot(obj_pan.h_mc_pz        ,"Pan_mc_Pz"      ,{'X':"Pz_{mc}^{%s} GeV/c"%str_led_pa                ,'Y':'Events'} ,'')
do_plot(obj_pan.h_mc_E         ,"Pan_mc_E"       ,{'X':"E_{mc}^{%s} GeV"%str_led_pa                   ,'Y':'Events'} ,'')
do_plot(obj_pan.h_reco_mc_E_num,"Pan_h_reco_mc_E_num",{'X':"E_{mc}^{%s} GeV"%str_led_pa               ,'Y':'Events'} ,'10<#theta_{mc}<170 (num)')
do_plot(obj_pan.h_reco_mc_E_den,"Pan_h_reco_mc_E_den",{'X':"E_{mc}^{%s} GeV"%str_led_pa               ,'Y':'Events'} ,'10<#theta_{mc}<170 (den)')
do_plot(obj_pan.h_reco_mc_phi_num,"Pan_h_reco_mc_phi_num",{'X':"#phi_{mc}^{%s} degree"%str_led_pa     ,'Y':'Events'} ,'10<#theta_{mc}<170 (num)')
do_plot(obj_pan.h_reco_mc_phi_den,"Pan_h_reco_mc_phi_den",{'X':"#phi_{mc}^{%s} degree"%str_led_pa     ,'Y':'Events'} ,'10<#theta_{mc}<170 (den)')

do_plot(obj_pan.h_rec_theta     ,"Pan_reco_theta"   ,{'X':"#theta_{pfo}^{%s} degree"%str_led_pa          ,'Y':'Events'}, '' )
do_plot(obj_pan.h_rec_phi       ,"Pan_reco_phi"     ,{'X':"#phi_{pfo}^{%s} degree"%str_led_pa            ,'Y':'Events'}, '' )
do_plot(obj_pan.h_rec_px        ,"Pan_reco_Px"      ,{'X':"Px_{pfo}^{%s} GeV/c"%str_led_pa               ,'Y':'Events'}, '' )
do_plot(obj_pan.h_rec_py        ,"Pan_reco_Py"      ,{'X':"Py_{pfo}^{%s} GeV/c"%str_led_pa               ,'Y':'Events'}, '' )
do_plot(obj_pan.h_rec_pz        ,"Pan_reco_Pz"      ,{'X':"Pz_{pfo}^{%s} GeV/c"%str_led_pa               ,'Y':'Events'}, '' )
do_plot(obj_pan.h_rec_E         ,"Pan_reco_E"       ,{'X':"E_{pfo}^{%s} GeV"%str_led_pa                  ,'Y':'Events'}, '' )
do_plot(obj_pan.h_rec_E_res     ,"Pan_reco_E_res"   ,{'X':"(E_{pfo}^{%s}-E_{mc}^{%s})/E_{mc}^{%s}"%(str_led_pa,str_led_pa,str_led_pa)  ,'Y':'Events'},'' )
do_plot(obj_pan.h_rec_theta_res ,"Pan_reco_theta_res"   ,{'X':"#theta_{pfo}^{%s}-#theta_{mc}^{%s}"%(str_led_pa,str_led_pa)             ,'Y':'Events'},'' )
do_plot(obj_pan.h_rec_phi_res   ,"Pan_reco_phi_res"     ,{'X':"#phi_{pfo}^{%s}-#phi_{mc}^{%s}"%(str_led_pa,str_led_pa)                 ,'Y':'Events'},'' )

do_eff_plot(obj_pan.h_mc_theta         , obj_pan.h_mc_theta_all     , 'Pan_reco_eff_theta', {'X':"#theta_{mc}^{%s} (degree)"%(str_led_pa)  ,'Y':'Efficiency'}, '10<#theta_{mc}<170, E_{mc} > 0.1 GeV')
do_eff_plot(obj_pan.h_reco_mc_E_num    , obj_pan.h_reco_mc_E_den    , 'Pan_reco_eff_E'    , {'X':"E_{mc}^{%s} (GeV)"%(str_led_pa)       ,'Y':'Efficiency'}   , '10<#theta_{mc}<170, E_{mc} > 0.1 GeV')
do_eff_plot(obj_pan.h_reco_mc_phi_num  , obj_pan.h_reco_mc_phi_den  , 'Pan_reco_eff_phi'  , {'X':"#phi_{mc}^{%s} (degree)"%(str_led_pa)    ,'Y':'Efficiency'}, '10<#theta_{mc}<170, E_{mc} > 0.1 GeV')
'''
#do_eff_plot(obj_pan.h_reco_mc_E_num_Endcap, obj_pan.h_reco_mc_E_den_Endcap    , 'Pan_reco_eff_E_Endcap'    , {'X':"E_{mc}^{%s} (GeV)"%(str_led_pa)       ,'Y':'Efficiency'}   , '10<#theta_{mc}<30 || 150<#theta_{mc}<170')
#do_eff_plot(obj_pan.h_reco_mc_E_num_Gap, obj_pan.h_reco_mc_E_den_Gap    , 'Pan_reco_eff_E_Gap'    , {'X':"E_{mc}^{%s} (GeV)"%(str_led_pa)       ,'Y':'Efficiency'}   , '30<#theta_{mc}<50 || 130<#theta_{mc}<150')
#do_eff_plot(obj_pan.h_reco_mc_E_num_Barrel, obj_pan.h_reco_mc_E_den_Barrel    , 'Pan_reco_eff_E_Barrel'    , {'X':"E_{mc}^{%s} (GeV)"%(str_led_pa)       ,'Y':'Efficiency'}   , '50<#theta_{mc}<130')

#do_eff_plot(obj_pan.h_reco_mc_E_num_any    , obj_pan.h_reco_mc_E_den    , 'Pan_reco_eff_E_any_logx'    , {'X':"E_{mc}^{%s} (GeV)"%(str_led_pa)       ,'Y':'Efficiency'}   , '10<#theta_{mc}<170')
#do_eff_plot(obj_pan.h_reco_mc_E_num_em     , obj_pan.h_reco_mc_E_den    , 'Pan_reco_eff_E_ele'    , {'X':"E_{mc}^{%s} (GeV)"%(str_led_pa)       ,'Y':'Efficiency'}   , '10<#theta_{mc}<170')
#print('saved mc E all=',obj_pan.h_mc_E_all.GetSumOfWeights(),',rec E =',obj_pan.h_rec_E.GetSumOfWeights())
'''
print('done')
