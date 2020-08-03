import h5py
import sys
import math
import numpy as np
import gc
import random
import ROOT as rt
rt.gROOT.SetBatch(rt.kTRUE)
from scipy.stats import anderson, ks_2samp
from sklearn.utils import shuffle
############ #################
## pdf(time or npe | r theta)#
##############################
def get_graph_ratio(g1, g2):
    g_ratio=g1.Clone("g_ratio_%s_%s"%(g1.GetName(),g2.GetName()))
    for ibin in range(0, g_ratio.GetN()):
         ratio=0
         if float(g2.GetY()[ibin]) !=0:
             ratio=float(g1.GetY()[ibin]/g2.GetY()[ibin])
         g_ratio.SetPoint(ibin,g_ratio.GetX()[ibin],ratio)
         #g_ratio.SetPointEYlow(ibin ,0)
         #g_ratio.SetPointEYhigh(ibin,0)
    return g_ratio

def get_hist_ratio(h1, h2):
    h_ratio=h1.Clone("h_ratio_%s_%s"%(h1.GetName(),h2.GetName()))
    for i in range(1, h1.GetNbinsX()+1):
        v1 = float(h1.GetBinContent(i))
        e1 = float(h1.GetBinError(i))
        v2 = float(h2.GetBinContent(i))
        e2 = float(h2.GetBinError(i))
        value = 0
        err = 0
        if v2 != 0: 
            value = v1/v2
            err = math.sqrt(v1*v1*e2*e2+v2*v2*e1*e1)/(v2*v2)
        h_ratio.SetBinContent(i, value)
        h_ratio.SetBinError  (i, err  )
    return h_ratio
def chi2_ndf(h_real, h_fake, doNormalized):
    if doNormalized:
        h_real.Scale(1/h_real.GetSumOfWeights())
        h_fake.Scale(1/h_fake.GetSumOfWeights())
    NDF = 0
    chi2 = 0
    for i in range(1, h_real.GetNbinsX()+1):
        if h_real.GetBinContent(i)==0: continue
        if (math.pow(h_real.GetBinError(i),2)+math.pow(h_fake.GetBinError(i),2)) == 0 : continue
        chi2 = chi2 + math.pow(h_real.GetBinContent(i)-h_fake.GetBinContent(i),2)/(math.pow(h_real.GetBinError(i),2)+math.pow(h_fake.GetBinError(i),2))
        NDF = NDF + 1 
    return chi2/NDF if doNormalized == False else chi2/(NDF-1)

def add_info(s_content):
    lowX=0.15
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


def do_plot(tag, h_real,h_fake,out_name, do_fit, c, binx):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    if True:
        h_real.Scale(1/h_real.GetSumOfWeights())
        h_fake.Scale(1/h_fake.GetSumOfWeights())
    nbin =h_real.GetNbinsX()
    x_min=binx[0]
    x_max=binx[1]
    y_min=0
    y_max=1.5*h_real.GetBinContent(h_real.GetMaximumBin())
    if "logx" in out_name:
        canvas.SetLogx()
    if "logy" in out_name:
        canvas.SetLogy()
        y_min = 1e-3
    dummy = rt.TH2D("dummy","",nbin,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_Y_title=""
    dummy_X_title=""
    if "n_pe" in out_name:
        #dummy_Y_title = "Events"
        dummy_Y_title = "Probability"
        dummy_X_title = "N PE"
    elif "tot_pe" in out_name:
        dummy_Y_title = "Probability"
        dummy_X_title = "Tot PE"
    elif "hit_time" in out_name:
        dummy_Y_title = "Probability"
        #dummy_Y_title = "Events"
        dummy_X_title = "Hit Time (ns)"
    dummy_Y_title = "Normalized"
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    dummy.GetXaxis().SetTitle(dummy_X_title)
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
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
    if do_fit:
        #f1 = rt.TF1("f1","(x>=[0])*[1]*[2]*exp(-[2]*(x-[0])) + (x>[3])*exp(-[4]*(x-[3]))", 0, 500)
        f1 = rt.TF1("f1","(x>=[0])*[1]*[2]*exp(-[2]*(x-[0])) + (x>=[3])*(1-[1])*[4]*exp(-[4]*(x-[3]))", 0, 500)
        f1.SetParameters(0, 100.0) # you MUST set non-zero initial values for parameters
        f1.SetParameters(1, 0.5) # you MUST set non-zero initial values for parameters
        f1.SetParameters(2, 0.1) # you MUST set non-zero initial values for parameters
        f1.SetParameters(3, 100.0) # you MUST set non-zero initial values for parameters
        f1.SetParameters(4, 0.1) # you MUST set non-zero initial values for parameters
        h_real.Fit("f1", "R") #"R" = fit between "xmin" and "xmax" of the "f1"

    legend = rt.TLegend(0.6,0.7,0.85,0.85)
    legend.AddEntry(h_real,'G4','lep')
    legend.AddEntry(h_fake,"NN",'lep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    #legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()
    label_theta = add_info(str('(r=%s,theta=%s, #chi^{2}/NDF=%f)'%(tag.split('_')[0], tag.split('_')[-1],c)))
    if 'tot_pe' in out_name :
        label_theta = add_info(str('(r=%s, x=%s, y=%s, z=%s, #chi^{2}/NDF=%f)'%(tag.split('_')[2], tag.split('_')[3], tag.split('_')[4], tag.split('_')[5] ,c)))
    label_theta.Draw()
    canvas.SaveAs("%s/%s_%s.png"%(plot_path, out_name, tag))
    del canvas
    gc.collect()

def do_plot_v1(tag, h1,h2,out_name, xbin, out_path,leg_name, doNormalize, x_label_size):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    if doNormalize:
        h1.Scale(1/h1.GetSumOfWeights())
        h2.Scale(1/h2.GetSumOfWeights())
    nbin =h1.GetNbinsX()
    x_min=xbin[0]
    x_max=xbin[1]
    y_min=0
    y_max=1.5*h1.GetBinContent(h1.GetMaximumBin())
    if "logx" in out_name:
        canvas.SetLogx()
    if "logy" in out_name:
        canvas.SetLogy()
        y_min = 1e-3
    dummy = rt.TH2D("dummy","",nbin,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_Y_title=""
    dummy_X_title=""
    if "n_pe" in out_name:
        #dummy_Y_title = "Events"
        dummy_Y_title = "Probability"
        dummy_X_title = "total PE"
    elif "tot_pe" in out_name:
        dummy_Y_title = "Events"
        #dummy_Y_title = "Probability"
        dummy_X_title = "total PE"
    elif "hit_time" in out_name:
        #dummy_Y_title = "Probability"
        dummy_Y_title = "Entries"
        dummy_X_title = "Hit Time (ns)"
    if doNormalize:
        dummy_Y_title = "Normalized"
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    dummy.GetXaxis().SetTitle(dummy_X_title)
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(x_label_size)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.GetYaxis().SetNdivisions(305)
    dummy.Draw()
    h1.SetLineWidth(2)
    h1.SetLineColor(rt.kRed)
    h1.SetMarkerColor(rt.kRed)
    h1.SetMarkerStyle(20)
    h1.Draw("same:pe")
    h2.SetLineWidth(2)
    h2.SetLineColor(rt.kBlue)
    h2.SetMarkerColor(rt.kBlue)
    h2.SetMarkerStyle(21)
    h2.Draw("same:histe")
    dummy.Draw("AXISSAME")

    legend = rt.TLegend(0.6,0.7,0.85,0.85)
    legend.AddEntry(h1,leg_name[0],'lep')
    legend.AddEntry(h2,leg_name[1],'lep')
    legend.SetBorderSize(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()
    r_info = add_info('(z = %s mm)'%tag)
    r_info.Draw()
    canvas.SaveAs("%s/%s_%s.png"%(out_path, out_name, tag))
    del canvas
    gc.collect()

def do_plot_v1_1(tag, h1,h2,out_name, xbin, out_path,leg_name, doNormalize, x_label_size):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    if doNormalize:
        #h1.Scale(1/h1.GetSumOfWeights())
        h2.Scale(h1.GetSumOfWeights()/h2.GetSumOfWeights())
    nbin =h1.GetNbinsX()
    x_min=xbin[0]
    x_max=xbin[1]
    y_min=0
    y_max=1.5*h1.GetBinContent(h1.GetMaximumBin())
    if "logx" in out_name:
        canvas.SetLogx()
    if "logy" in out_name:
        canvas.SetLogy()
        y_min = 1e-3
    dummy = rt.TH2D("dummy","",nbin,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_Y_title=""
    dummy_X_title=""
    if "n_pe" in out_name:
        dummy_Y_title = "Entries"
        dummy_X_title = "total PE"
    elif "tot_pe" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "total PE"
    elif "hit_time" in out_name:
        dummy_Y_title = "Entries"
        dummy_X_title = "Hit Time (ns)"
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    dummy.GetXaxis().SetTitle(dummy_X_title)
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(x_label_size)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.GetYaxis().SetNdivisions(305)
    dummy.Draw()
    h1.SetLineWidth(2)
    h1.SetLineColor(rt.kRed)
    h1.SetMarkerColor(rt.kRed)
    h1.SetMarkerStyle(20)
    h1.Draw("same:pe")
    h2.SetLineWidth(2)
    h2.SetLineColor(rt.kBlue)
    h2.SetMarkerColor(rt.kBlue)
    h2.SetMarkerStyle(21)
    h2.Draw("same:histe")
    dummy.Draw("AXISSAME")

    legend = rt.TLegend(0.6,0.7,0.85,0.85)
    legend.AddEntry(h1,leg_name[0],'lep')
    legend.AddEntry(h2,leg_name[1],'lep')
    legend.SetBorderSize(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()
    r_info = add_info('(z = %s mm)'%tag)
    r_info.Draw()
    canvas.SaveAs("%s/%s_%s.png"%(out_path, out_name, tag))
    del canvas
    gc.collect()

def do_plot_v2(tag, info,  h1,h2,out_name, xbin, out_path,leg_name, doNormalize, y_label_size, ratio_name):

    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.SetGridy()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    canvas.cd()
    size = 0.312
    pad1 = rt.TPad('pad1', '', 0.0, size, 1.0, 1.0, 0)
    pad2 = rt.TPad('pad2', '', 0.0, 0.0, 1.0, size, 0)
    pad2.Draw() 
    pad1.Draw()
    pad1.SetTickx(1)
    pad1.SetTicky(1)
    pad2.SetTickx(1)
    pad2.SetTicky(1)
    pad2.SetGridy()
    pad1.SetBottomMargin(0.0)
    pad1.SetRightMargin(0.05)
    pad1.SetLeftMargin(0.13)
    pad2.SetTopMargin(0.02)
    pad2.SetBottomMargin(0.5)
    pad2.SetRightMargin(0.05)
    pad2.SetLeftMargin(0.13)
    pad1.cd()

    if doNormalize:
        h1.Scale(1/h1.GetSumOfWeights())
        h2.Scale(1/h2.GetSumOfWeights())
    nbin =h1.GetNbinsX()
    x_min=xbin[0]
    x_max=xbin[1]
    y_min=0
    y_max=1.5*h1.GetBinContent(h1.GetMaximumBin())
    if "logx" in out_name:
        pad1.SetLogx()
        pad2.SetLogx()
    if "logy" in out_name:
        pad1.SetLogy()
        y_min = 1e-1
    dummy = rt.TH2D("dummy","",nbin,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_Y_title=""
    dummy_X_title=""
    if "n_pe" in out_name:
        dummy_Y_title = "Entries"
        dummy_X_title = "nPE"
    elif "tot_pe" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "total PE"
    elif "hit_time" in out_name:
        dummy_Y_title = "Entries"
        dummy_X_title = "Hit Time (ns)"
    elif "First_Hit_time" in out_name:
        dummy_Y_title = "Entries"
        dummy_X_title = "First Hit Time (ns)"
    if doNormalize:
        dummy_Y_title = "Normalized"
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    dummy.GetXaxis().SetTitle("")
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.)
    dummy.GetYaxis().SetLabelSize(y_label_size)
    dummy.GetXaxis().SetLabelSize(0.)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.GetYaxis().SetNdivisions(305)
    dummy.Draw()
    h1.SetLineWidth(2)
    h1.SetLineColor(rt.kRed)
    h1.SetMarkerColor(rt.kRed)
    h1.SetMarkerStyle(20)
    h1.Draw("same:pe")
    h2.SetLineWidth(2)
    h2.SetLineColor(rt.kBlue)
    h2.SetMarkerColor(rt.kBlue)
    h2.SetMarkerStyle(21)
    h2.Draw("same:histe")
    dummy.Draw("AXISSAME")

    legend = rt.TLegend(0.6,0.7,0.85,0.85)
    legend.AddEntry(h1,leg_name[0],'lep')
    legend.AddEntry(h2,leg_name[1],'lep')
    legend.SetBorderSize(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()
    #r_info = add_info('(z = %s mm)'%tag)
    r_info = add_info(str(info))
    r_info.Draw()

    pad2.cd()
    fontScale = pad1.GetHNDC()/pad2.GetHNDC()
    ratio_y_min = 0.5
    ratio_y_max = 1.5
    dummy_ratio = rt.TH2D("dummy_ratio","",1,x_min,x_max,1,ratio_y_min,ratio_y_max)
    dummy_ratio.SetStats(rt.kFALSE)
    dummy_ratio.GetYaxis().SetTitle('%s / %s'%(ratio_name[0], ratio_name[1]))
    dummy_ratio.GetYaxis().CenterTitle()
    dummy_ratio.GetXaxis().SetTitle(dummy_X_title)
    dummy_ratio.GetXaxis().SetTitleSize(0.05*fontScale)
    dummy_ratio.GetXaxis().SetLabelSize(0.05*fontScale)
    dummy_ratio.GetYaxis().SetNdivisions(305)
    dummy_ratio.GetYaxis().SetTitleSize(0.04*fontScale)
    dummy_ratio.GetYaxis().SetLabelSize(0.04*fontScale)
    dummy_ratio.GetYaxis().SetTitleOffset(0.7)
    dummy_ratio.Draw()
    #h_ratio=h1.Clone('ratio_%s'%h1.GetName())
    #h_ratio.Divide(h2)
   
    h_ratio = get_hist_ratio(h1, h2)
    h_ratio.Draw("same:pe")
    canvas.SaveAs("%s/%s_%s_ratio.png"%(out_path, out_name, tag))
    del canvas
    gc.collect()

def do_plot_1(tag, h_real,out_name):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
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
    dummy = rt.TH2D("dummy","",nbin,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_Y_title=""
    dummy_X_title=""
    if "pmt_theta" in out_name:
        dummy_Y_title = "Entries / %.1f"%h_real.GetBinWidth(0)
        dummy_X_title = "PMT #theta"
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    dummy.GetXaxis().SetTitle(dummy_X_title)
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw()
    h_real.SetLineWidth(2)
    h_real.SetLineColor(rt.kRed)
    h_real.SetMarkerColor(rt.kRed)
    h_real.SetMarkerStyle(20)
    h_real.Draw("same:pe")
    dummy.Draw("AXISSAME")
    canvas.SaveAs("%s/%s_%s.png"%(plot_path, out_name, tag))
    del canvas
    gc.collect()


def do_plot_gr(gr, title,out_name):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    x_min=0
    x_max=20
    y_min=0
    y_max=180
    z_min=0
    z_max=10
    dummy = rt.TH3D("dummy","",1,x_min,x_max,1,y_min,y_max, 1,z_min, z_max)
    dummy.SetStats(rt.kFALSE)
    dummy_X_title="r"
    dummy_Y_title="#theta"
    dummy_Z_title="#chi^{2}/NDF"
    dummy.SetTitle(title)
    dummy.GetZaxis().SetTitle(dummy_Z_title)
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    dummy.GetXaxis().SetTitle(dummy_X_title)
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetZaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.03)
    dummy.GetXaxis().SetLabelSize(0.03)
    dummy.GetZaxis().SetLabelSize(0.03)
    dummy.GetZaxis().SetTitleOffset(1.5)
    dummy.GetYaxis().SetTitleOffset(1.5)
    dummy.GetXaxis().SetTitleOffset(1.5)
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.GetZaxis().SetTitleFont(42)
    dummy.GetZaxis().SetLabelFont(42)
    dummy.Draw()
    gr.SetMarkerStyle(20)
    gr.Draw("same:pcol")
    dummy.Draw("AXISSAME")
    canvas.SaveAs("%s/%s.png"%(plot_path, out_name))
    del canvas
    gc.collect()

def do_plot_gr_1D(gr, title,out_name, x_bins, y_bins, xy_title, M_size, output_path, fit_line):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetGridy()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    x_min=x_bins[0]
    x_max=x_bins[1]
    y_min=y_bins[0]
    y_max=y_bins[1]
    dummy = rt.TH2D("dummy","",1,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_X_title=xy_title[0]
    dummy_Y_title=xy_title[1]
    dummy.SetTitle(title)
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    dummy.GetXaxis().SetTitle(dummy_X_title)
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.03)
    dummy.GetXaxis().SetLabelSize(0.03)
    dummy.GetYaxis().SetTitleOffset(1.1)
    dummy.GetXaxis().SetTitleOffset(1.1)
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw()
    gr.SetMarkerStyle(20)
    gr.SetMarkerSize(M_size)
    gr.Draw("same:p")
    dummy.Draw("AXISSAME")
    if fit_line:
        f_x_min = 0
        f_x_max = 5
        f1 = rt.TF1("f1","[0] + [1]*x", f_x_min, f_x_max)
        f1.SetParameters(0, 0.1) # you MUST set non-zero initial values for parameters
        f1.SetParameters(1, 1.1) # you MUST set non-zero initial values for parameters
        gr.Fit("f1", "WR") #"R" = fit between "xmin" and "xmax" of the "f1"

    canvas.SaveAs("%s/%s.png"%(output_path, out_name))
    del canvas
    gc.collect()


def do_plot_2gr_v1(gr1, gr2,  title, out_name, x_bins, y_bins, xy_title, M_size, output_path, fit_line, legend_name, ratio_name):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.SetGridy()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    canvas.cd()
    size = 0.312
    pad1 = rt.TPad('pad1', '', 0.0, size, 1.0, 1.0, 0)
    pad2 = rt.TPad('pad2', '', 0.0, 0.0, 1.0, size, 0)
    pad2.Draw() 
    pad1.Draw()
    pad1.SetTickx(1)
    pad1.SetTicky(1)
    pad2.SetTickx(1)
    pad2.SetTicky(1)
    pad2.SetGridy()
    pad1.SetBottomMargin(0.0)
    pad1.SetRightMargin(0.05)
    pad1.SetLeftMargin(0.13)
    pad2.SetTopMargin(0.02)
    pad2.SetBottomMargin(0.5)
    pad2.SetRightMargin(0.05)
    pad2.SetLeftMargin(0.13)
    pad1.cd()

    x_min=x_bins[0]
    x_max=x_bins[1]
    y_min=y_bins[0]
    y_max=y_bins[1]
    dummy = rt.TH2D("dummy","",1,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_X_title=xy_title[0]
    dummy_Y_title=xy_title[1]
    dummy.SetTitle(title)
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    #dummy.GetXaxis().SetTitle(dummy_X_title)
    dummy.GetXaxis().SetTitle('')
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.0 )
    dummy.GetYaxis().SetTitleOffset(1.4)
    dummy.GetXaxis().SetTitleOffset(1.1)
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw()
    gr1.SetMarkerStyle(20)
    gr1.SetMarkerSize(M_size)
    gr2.SetMarkerStyle(21)
    gr2.SetMarkerSize(M_size)
    gr1.SetMarkerColor(1)
    gr2.SetMarkerColor(2)
    gr1.Draw("same:p")
    gr2.Draw("same:p")
    dummy.Draw("AXISSAME")
    if fit_line:
        f_x_min = 0
        f_x_max = 5
        f1 = rt.TF1("f1","[0] + [1]*x", f_x_min, f_x_max)
        f1.SetParameters(0, 0.1) # you MUST set non-zero initial values for parameters
        f1.SetParameters(1, 1.1) # you MUST set non-zero initial values for parameters
        gr1.Fit("f1", "WR") #"R" = fit between "xmin" and "xmax" of the "f1"
    legend = rt.TLegend(0.2,0.7,0.45,0.85)
    legend.AddEntry(gr1, legend_name[0],'p')
    legend.AddEntry(gr2, legend_name[1],'p')
    legend.SetBorderSize(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()

    pad2.cd()
    fontScale = pad1.GetHNDC()/pad2.GetHNDC()
    ratio_y_min = 0.95
    ratio_y_max = 1.05
    dummy_ratio = rt.TH2D("dummy_ratio","",1,x_min,x_max,1,ratio_y_min,ratio_y_max)
    dummy_ratio.SetStats(rt.kFALSE)
    dummy_ratio.GetYaxis().SetTitle('%s / %s'%(ratio_name[0], ratio_name[1]))
    dummy_ratio.GetYaxis().CenterTitle()
    dummy_ratio.GetXaxis().SetTitle(dummy_X_title)
    dummy_ratio.GetXaxis().SetTitleSize(0.05*fontScale)
    dummy_ratio.GetXaxis().SetLabelSize(0.05*fontScale)
    dummy_ratio.GetYaxis().SetNdivisions(305)
    dummy_ratio.GetYaxis().SetTitleSize(0.04*fontScale)
    dummy_ratio.GetYaxis().SetLabelSize(0.04*fontScale)
    dummy_ratio.GetYaxis().SetTitleOffset(0.7)
    dummy_ratio.Draw()
    g_ratio=rt.TGraph()
    if gr1.GetN()==gr2.GetN():
        g_ratio = get_graph_ratio(gr1, gr2)
    g_ratio.Draw("same:p")
    canvas.SaveAs("%s/%s.png"%(output_path, out_name))
    del canvas
    gc.collect()


def do_plot_2gr(gr1, gr2,  title, out_name, x_bins, y_bins, xy_title, M_size, output_path, fit_line):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetGridy()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    x_min=x_bins[0]
    x_max=x_bins[1]
    y_min=y_bins[0]
    y_max=y_bins[1]
    dummy = rt.TH2D("dummy","",1,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_X_title=xy_title[0]
    dummy_Y_title=xy_title[1]
    dummy.SetTitle(title)
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    dummy.GetXaxis().SetTitle(dummy_X_title)
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.03)
    dummy.GetXaxis().SetLabelSize(0.03)
    dummy.GetYaxis().SetTitleOffset(1.4)
    dummy.GetXaxis().SetTitleOffset(1.1)
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw()
    gr1.SetMarkerStyle(20)
    gr1.SetMarkerSize(M_size)
    gr2.SetMarkerStyle(21)
    gr2.SetMarkerSize(M_size)
    gr1.SetMarkerColor(1)
    gr2.SetMarkerColor(2)
    gr1.Draw("same:p")
    gr2.Draw("same:p")
    dummy.Draw("AXISSAME")
    if fit_line:
        f_x_min = 0
        f_x_max = 5
        f1 = rt.TF1("f1","[0] + [1]*x", f_x_min, f_x_max)
        f1.SetParameters(0, 0.1) # you MUST set non-zero initial values for parameters
        f1.SetParameters(1, 1.1) # you MUST set non-zero initial values for parameters
        gr1.Fit("f1", "WR") #"R" = fit between "xmin" and "xmax" of the "f1"
    legend = rt.TLegend(0.2,0.7,0.45,0.85)
    legend.AddEntry(gr1,'full sim','p')
    legend.AddEntry(gr2,"fast sim",'p')
    legend.SetBorderSize(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()

    canvas.SaveAs("%s/%s.png"%(output_path, out_name))
    del canvas
    gc.collect()

def make_hist_0(tag, hd):
    dict_hist = {}
    tag = tag*180 # back to real theta
    for i in range(tag.shape[0]):
        #dict_hist[tag[i]] = rt.TH1F('v0_%s'%str(tag[i]),'',5,0,5)
        dict_hist[tag[i]] = rt.TH1F('v0_%s'%str(tag[i]),'',10,-5,5)
        for j in range(hd.shape[0]):
            #dict_hist[tag[i]].Fill(hd[j,i])
            dict_hist[tag[i]].Fill(round(hd[j,i]))
    return dict_hist

def make_hist_1(tag, hd):
    dict_hist = {}
    for i in range(tag.shape[1]):
        if tag[0,i] in dict_hist: continue
        #dict_hist[tag[0,i]] = rt.TH1F('v1_%s'%str(tag[0,i]),'',5,0,5)
        dict_hist[tag[0,i]] = rt.TH1F('v1_%s'%str(tag[0,i]),'',10,-5,5)
    for i in range(tag.shape[1]):
        for j in range(hd.shape[0]):
            dict_hist[tag[0,i]].Fill(hd[j,i])
    return dict_hist


def make_hist_time_0(tag, hd):
    dict_hist = {}
    tag = tag*180 # back to real theta
    for i in range(tag.shape[0]):
        dict_hist[tag[i]] = rt.TH1F('time_v0_%s'%str(tag[i]),'',51,-10,500)
        for j in range(hd.shape[0]):
            dict_hist[tag[i]].Fill(hd[j,i])
            #dict_hist[tag[i]].Fill(round(hd[j,i]))
    return dict_hist


def make_hist_time_1(tag, hd):
    dict_hist = {}
    for i in range(tag.shape[1]):
        if tag[0,i] in dict_hist: continue
        dict_hist[tag[0,i]] = rt.TH1F('time_v1_%s'%str(tag[0,i]),'',51,-10,500)
    for i in range(tag.shape[1]):
        for j in range(hd.shape[0]):
            if hd[j,i] <= 0: continue # remove 0 
            dict_hist[tag[0,i]].Fill(hd[j,i])
    return dict_hist

def read_hist(file_name, cat, bins, tag, doRound, scale, refine_bining):

    binning = []
    for i in range(0,100):
        binning.append(i)
    for i in range(100,200,2):
        binning.append(i)
    for i in range(200,400,5):
        binning.append(i)
    for i in range(400,1000,10):
        binning.append(i)
    binning = np.array(binning, dtype=np.float32)


    dict_hist = {}
    for ifile in file_name:
        hd = h5py.File(ifile, 'r')
        for i in hd.keys():
            if cat not in i: continue
            h = hd[i][:]/scale
            r_theta = i.split(cat)[-1]
            if r_theta not in dict_hist:
                dict_hist[r_theta] = rt.TH1F('%s_%s'%(str(tag),str(i)),'',bins[0],bins[1],bins[2]) if refine_bining == False else rt.TH1F('%s_%s'%(str(tag),str(i)),'',len(binning)-1,binning)
            for j in range(h.shape[0]):
                dict_hist[r_theta].Fill(h[j] if doRound == False else round(h[j]))
        hd.close()
    return dict_hist


def read_hist_v1(file_name, bins, tag, doRound):
    dict_hist = {}
    for ifile in file_name:
        hd  = h5py.File(ifile, 'r')
        npe = hd['nPEByPMT'][:]
        print('mpt size=',npe.shape[0])
        for i in range(npe.shape[0]):
            if i%1000 ==0 :print('i mpt =',i)
            r_theta = str('%s_%s'%(npe[i,0], npe[i,1]))
            if r_theta not in dict_hist:
                dict_hist[r_theta] = rt.TH1F('%s_%s'%(str(tag),str(r_theta)),'',bins[0],bins[1],bins[2])
            for j in range(2, len(npe[i,:])):
                dict_hist[r_theta].Fill(round(npe[i,j]))
        hd.close()
    return dict_hist

def read_hist_totPE(file_name, bins, tag, split_str):
    dict_hist = {}
    for ifile in file_name:
        hd  = h5py.File(ifile, 'r')
        npe = np.round(hd['nPEByPMT'][:])
        str_r_x_y_z = ifile.split('/')[-1]
        str_r_x_y_z = str_r_x_y_z.split(split_str)[0]
        if str_r_x_y_z not in dict_hist:
            dict_hist[str_r_x_y_z] = rt.TH1F('%s_%s'%(str(tag),str(str_r_x_y_z)),'',bins[0],bins[1],bins[2])
        for j in range(2, npe.shape[1]):
            dict_hist[str_r_x_y_z].Fill(np.sum(npe[:,j]))
        hd.close()
    return dict_hist

def read_hist_PMT_theta(file_name, bins, tag):
    hist = rt.TH1F('%s_pmt_theta'%str(tag),'',bins[0],bins[1],bins[2])
    for ifile in file_name:
        hd  = h5py.File(ifile, 'r')
        npe = hd['nPEByPMT'][:]
        for i in range(npe.shape[0]):
            hist.Fill(npe[i,1])
        hd.close()
    return hist


def read_graph_r_theta_vs_meanNPE(file_name):
    dict_r_graph = {}
    dict_theta_graph = {}
    for ifile in file_name:
        hd  = h5py.File(ifile, 'r')
        npe = hd['nPEByPMT'][:]
        for i in range(npe.shape[0]):
            r     = npe[i,0]
            theta = npe[i,1]
            if str(r) not in dict_r_graph:
                dict_r_graph[str(r)] = rt.TGraph()
            if str(theta) not in dict_theta_graph:
                dict_theta_graph[str(theta)] = rt.TGraph()
            r_potins     = dict_r_graph[str(r)].GetN()
            theta_potins = dict_theta_graph[str(theta)].GetN()
            dict_r_graph[str(r)]        .SetPoint(r_potins    ,theta,np.mean(npe[i,2:npe.shape[1]]))
            dict_theta_graph[str(theta)].SetPoint(theta_potins,r    ,np.mean(npe[i,2:npe.shape[1]]))
        hd.close()
    return (dict_r_graph, dict_theta_graph)


def read_graph_visE_vs_meanNPE(files):
    Dict = {}
    for ifile in files:
        hd  = h5py.File(ifile, 'r')
        npe = hd['nPEByPMT'][:]
        n_photon = files[ifile]
        vis_en = float(n_photon)/10129
        Dict[vis_en]={}
        for i in range(npe.shape[0]):
            theta = npe[i,1]
            if theta not in Dict[vis_en]:
                Dict[vis_en][theta] = [np.sum(npe[i,2:npe.shape[1]]), npe.shape[1]-2]
            else:
                Dict[vis_en][theta] = [np.sum(npe[i,2:npe.shape[1]])+Dict[vis_en][theta][0], npe.shape[1]-2+Dict[vis_en][theta][1]]
        hd.close()
    Dict_theta = {}
    for itheta in range(0,180):
        Dict_theta[itheta] = rt.TGraph()
        for ie in Dict:
            if itheta not in Dict[ie]:continue
            potins     = Dict_theta[itheta].GetN()
            Dict_theta[itheta].SetPoint(potins, ie, float(Dict[ie][itheta][0])/Dict[ie][itheta][1])
    return Dict_theta 


def get_npe_info(tag, input_files, binning):
    dicts = {}
    for ifile in input_files: 
        dicts[input_files[ifile]] = [0, 0]
        dicts[input_files[ifile]][1] = rt.TH1F('%s_%s'%(str(tag),str(input_files[ifile])),'',binning[0],binning[1],binning[2])
        chain =rt.TChain('evt')
        chain.Add(ifile)
        tree = chain
        totalEntries=tree.GetEntries()
        #print(ifile,',totalEntries=',totalEntries)
        for ie in range(totalEntries):
            tree.GetEntry(ie)
            nPhotons  = getattr(tree, "nPhotons")
            pmtID     = getattr(tree, "pmtID")
            nphoton = 0
            for ip in range(nPhotons):
                if pmtID[ip] <= 17612 : nphoton +=1 ## only large pmt
                #if pmtID[ip] <= 8803 : nphoton +=1 ## only large pmt at positive Z
                #if pmtID[ip] <= 17612 and pmtID[ip] > 8803 : nphoton +=1 ## only large pmt at negative Z
            dicts[input_files[ifile]][0] = dicts[input_files[ifile]][0] + float(nphoton)/totalEntries
            dicts[input_files[ifile]][1].Fill(nphoton)
    return dicts

def get_npe_info_v1(tag, input_files, binning):
    dicts = {}
    for ifile in input_files: 
        dicts[input_files[ifile]] = [0, 0, 0]
        dicts[input_files[ifile]][1] = rt.TH1D('%s_%s'%(str(tag),str(input_files[ifile])),'',binning[0],binning[1],binning[2])
        dicts[input_files[ifile]][2] = rt.TH1D('npe_%s_%s'%(str(tag),str(input_files[ifile])),'',10,0,10)
        chain =rt.TChain('evt')
        chain.Add(ifile)
        tree = chain
        totalEntries=tree.GetEntries()
        #print(ifile,',totalEntries=',totalEntries)
        for ie in range(totalEntries):
            tree.GetEntry(ie)
            nPhotons  = getattr(tree, "nPhotons")
            pmtID     = getattr(tree, "pmtID")
            nphoton = 0
            tmp_dict = {}
            for ip in range(nPhotons):
                if pmtID[ip] > 17612 : continue
                nphoton +=1 ## only large pmt
                if pmtID[ip] not in tmp_dict: tmp_dict[pmtID[ip]] = 1
                else: tmp_dict[pmtID[ip]] += 1
                #if pmtID[ip] <= 8803 : nphoton +=1 ## only large pmt at positive Z
                #if pmtID[ip] <= 17612 and pmtID[ip] > 8803 : nphoton +=1 ## only large pmt at negative Z
            dicts[input_files[ifile]][0] = dicts[input_files[ifile]][0] + float(nphoton)/totalEntries
            dicts[input_files[ifile]][1].Fill(nphoton)
            for i in range(17612):
                if i in tmp_dict: dicts[input_files[ifile]][2].Fill(tmp_dict[i])
                else            : dicts[input_files[ifile]][2].Fill(0)
    return dicts


def get_hittime_info(tag, input_files, refine_binning):
    binning = []
    for i in range(0,100):
        binning.append(i)
    for i in range(100,200,2):
        binning.append(i)
    for i in range(200,400,5):
        binning.append(i)
    for i in range(400,1000,10):
        binning.append(i)
    binning = np.array(binning, dtype=np.float32)

    dicts = {}
    for ifile in input_files: 
        dicts[input_files[ifile]] = rt.TH1F('%s_%s'%(str(tag),str(input_files[ifile])),'',1000,0,1000) if refine_binning==False else rt.TH1F('%s_%s'%(str(tag),str(input_files[ifile])),'',len(binning)-1,binning)
        chain =rt.TChain('evt')
        chain.Add(ifile)
        tree = chain
        totalEntries=tree.GetEntries()
        for ie in range(totalEntries):
            tree.GetEntry(ie)
            nPhotons  = getattr(tree, "nPhotons")
            pmtID     = getattr(tree, "pmtID")
            hitTime   = getattr(tree, "hitTime")
            for ip in range(nPhotons):
                if pmtID[ip] > 17612 : continue ## only large pmt
                dicts[input_files[ifile]].Fill(hitTime[ip])
    return dicts

def get_first_hittime_info(tag, input_files, refine_binning):
    binning = []
    for i in range(0,100):
        binning.append(i)
    for i in range(100,200,2):
        binning.append(i)
    for i in range(200,400,5):
        binning.append(i)
    for i in range(400,1000,10):
        binning.append(i)
    binning = np.array(binning, dtype=np.float32)

    dicts = {}
    for ifile in input_files: 
        dicts[input_files[ifile]] = rt.TH1F('%s_%s'%(str(tag),str(input_files[ifile])),'',1000,0,1000) if refine_binning==False else rt.TH1F('%s_%s'%(str(tag),str(input_files[ifile])),'',len(binning)-1,binning)
        chain =rt.TChain('evt')
        chain.Add(ifile)
        tree = chain
        totalEntries=tree.GetEntries()
        for ie in range(totalEntries):
            tree.GetEntry(ie)
            nPhotons  = getattr(tree, "nPhotons")
            pmtID     = getattr(tree, "pmtID")
            hitTime   = getattr(tree, "hitTime")
            hittime_dict = {}
            for ip in range(nPhotons):
                if pmtID[ip] > 17612 :continue ## only large pmt
                if pmtID[ip] not in hittime_dict:
                    hittime_dict[pmtID[ip]] = hitTime[ip]
                else:
                    hittime_dict[pmtID[ip]] = hitTime[ip] if hitTime[ip] < hittime_dict[pmtID[ip]] else hittime_dict[pmtID[ip]]
            for ID in hittime_dict:
                dicts[input_files[ifile]].Fill(hittime_dict[ID])
    return dicts

########### BEGIN ###############

plot_path = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/r_theta_comp_plots/'
#plot_path = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/tmp/'

For_PMT_Theta = False
For_NPE  = False
For_Time = False
For_r_theta_vs_npeMean   = False
For_visE_vs_npeMean = False
For_real_vs_fast = True 

real_hists = 0
fake_hists = 0
real_hists_totPE = 0

real_file = []


#real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_78_761.054753_256.432825_-131.772148_-704.331345_batch0_N5000.h5')
#real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_113_3255.504263_-2033.526479_462.117123_2499.905165_batch0_N5000.h5')
#real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_371_6006.914466_-332.456396_47.417349_5997.519966_batch0_N5000.h5')
#real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_127_9028.697657_4754.172634_6867.591995_-3428.032107_batch0_N5000.h5')
#real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_107_13232.013021_-8068.007748_-7583.660247_7244.412801_batch0_N5000.h5')
#real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_115_17325.036879_-14375.764716_2691.512325_-9287.090669_batch0_N5000.h5')
###########
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_109_1972.354650_-170.609789_-1324.798385_1451.201022_batch0_N5000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_108_4143.010671_-27.302429_-27.039244_4142.832470_batch0_N5000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_36_5130.729838_-1661.480186_1528.432601_-4607.359987_batch0_N5000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_114_5720.626101_2069.080878_-5293.362866_651.749089_batch0_N5000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_11_6400.581282_4833.706354_-4015.608449_-1215.570820_batch0_N5000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_100_7265.516575_-673.529875_-1760.499121_-7016.746501_batch0_N5000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_140_8235.163246_-391.835173_-2657.752845_7784.646986_batch0_N5000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_104_10193.326270_3103.778049_-253.003481_9706.000799_batch0_N5000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_1_11042.389770_4397.912803_-8872.284301_-4886.236393_batch0_N5000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_105_12609.132920_-6316.114803_1771.981477_-10768.333595_batch0_N5000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_102_14525.704016_10407.992577_-9981.839210_-1741.451597_batch0_N5000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_136_15057.768948_-2826.795242_-6848.761397_-13108.779565_batch0_N5000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_111_15703.688716_-2435.069757_445.717084_-15507.340548_batch0_N5000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_166_16122.719647_3297.384443_4536.220869_-15115.953323_batch0_N5000.h5')

######## r1m plane ##################
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_1_10129_100.000000_89.007998_0.000000_45.580438_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_10_10129_150.000000_-112.719085_0.000000_-98.966701_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_15_10129_200.000000_185.557330_0.000000_-74.622231_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_20_10129_250.000000_-143.929086_0.000000_-204.412373_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_25_10129_300.000000_-242.895823_0.000000_-176.072767_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_30_10129_350.000000_125.901184_0.000000_326.571419_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_35_10129_400.000000_-163.288033_0.000000_365.153417_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_40_10129_450.000000_425.003352_0.000000_147.892361_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_45_10129_500.000000_497.049350_0.000000_-54.239686_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_50_10129_550.000000_-546.214740_0.000000_-64.416282_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_55_10129_600.000000_523.519063_0.000000_293.134424_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_60_10129_650.000000_-105.977586_0.000000_641.302387_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_65_10129_700.000000_-80.507505_0.000000_-695.354975_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_70_10129_750.000000_-745.634357_0.000000_80.804734_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_75_10129_800.000000_-402.135808_0.000000_-691.582817_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_80_10129_850.000000_-590.998470_0.000000_-610.918005_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_85_10129_900.000000_585.740519_0.000000_683.306698_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_90_10129_950.000000_315.878011_0.000000_-895.947031_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_95_10129_1000.000000_734.672485_0.000000_-678.421948_batch0_N1000.h5')
#real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_1m_plane_hittime_h5_data_valid/user_99_10129_1050.000000_-653.760630_0.000000_821.642890_batch0_N1000.h5')


if For_real_vs_fast:
    real_input = {}

    '''
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_18_0.000000_0.000000_0.000000_0.000000.root']  =         0.000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_19_1000.000000_0.000000_0.000000_1000.000000.root']  =   1000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_20_2000.000000_0.000000_0.000000_2000.000000.root']  =   2000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_21_3000.000000_0.000000_0.000000_3000.000000.root']  =   3000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_22_4000.000000_0.000000_0.000000_4000.000000.root']  =   4000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_23_5000.000000_0.000000_0.000000_5000.000000.root']  =   5000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_24_6000.000000_0.000000_0.000000_6000.000000.root']  =   6000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_25_7000.000000_0.000000_0.000000_7000.000000.root']  =   7000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_26_8000.000000_0.000000_0.000000_8000.000000.root']  =   8000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_27_9000.000000_0.000000_0.000000_9000.000000.root']  =   9000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_28_10000.000000_0.000000_0.000000_10000.000000.root']  = 10000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_29_11000.000000_0.000000_0.000000_11000.000000.root']  = 11000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_30_12000.000000_0.000000_0.000000_12000.000000.root']  = 12000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_31_13000.000000_0.000000_0.000000_13000.000000.root']  = 13000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_32_14000.000000_0.000000_0.000000_14000.000000.root']  = 14000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_33_15000.000000_0.000000_0.000000_15000.000000.root']  = 15000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_34_16000.000000_0.000000_0.000000_16000.000000.root']  = 16000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_35_17000.000000_0.000000_0.000000_17000.000000.root']  = 17000
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_15_-3000.000000_0.000000_0.000000_-3000.000000.root']  = -3000
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_16_-2000.000000_0.000000_0.000000_-2000.000000.root']  = -2000
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_17_-1000.000000_0.000000_0.000000_-1000.000000.root']  = -1000
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_10_-8000.000000_0.000000_0.000000_-8000.000000.root']  = -8000
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_1_-17000.000000_0.000000_0.000000_-17000.000000.root']  =-17000 
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_11_-7000.000000_0.000000_0.000000_-7000.000000.root']  = -7000
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_12_-6000.000000_0.000000_0.000000_-6000.000000.root']  = -6000
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_13_-5000.000000_0.000000_0.000000_-5000.000000.root']  = -5000
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_14_-4000.000000_0.000000_0.000000_-4000.000000.root']  = -4000
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_3_-15000.000000_0.000000_0.000000_-15000.000000.root']  =-15000 
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_2_-16000.000000_0.000000_0.000000_-16000.000000.root']  =-16000 
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_4_-14000.000000_0.000000_0.000000_-14000.000000.root']  =-14000 
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_5_-13000.000000_0.000000_0.000000_-13000.000000.root']  =-13000 
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_6_-12000.000000_0.000000_0.000000_-12000.000000.root']  =-12000 
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_7_-11000.000000_0.000000_0.000000_-11000.000000.root']  =-11000 
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_8_-10000.000000_0.000000_0.000000_-10000.000000.root']  =-10000 
    #real_input['/cefs/higgs/wxfang/sim_ouput/official_Co60/user_9_-9000.000000_0.000000_0.000000_-9000.000000.root']  = -9000 
    '''


    '''
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_1_0.000000_0.000000_0.000000_0.000000.root'           ]=0
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_10_9000.000000_0.000000_0.000000_9000.000000.root'    ]=9000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_11_10000.000000_0.000000_0.000000_10000.000000.root'  ]=10000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_12_11000.000000_0.000000_0.000000_11000.000000.root'  ]=11000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_13_12000.000000_0.000000_0.000000_12000.000000.root'  ]=12000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_14_13000.000000_0.000000_0.000000_13000.000000.root'  ]=13000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_15_14000.000000_0.000000_0.000000_14000.000000.root'  ]=14000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_16_15000.000000_0.000000_0.000000_15000.000000.root'  ]=15000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_17_16000.000000_0.000000_0.000000_16000.000000.root'  ]=16000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_18_17000.000000_0.000000_0.000000_17000.000000.root'  ]=17000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_2_1000.000000_0.000000_0.000000_1000.000000.root'     ]=1000
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_3_2000.000000_0.000000_0.000000_2000.000000.root'     ]=2000
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_4_3000.000000_0.000000_0.000000_3000.000000.root'     ]=3000
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_5_4000.000000_0.000000_0.000000_4000.000000.root'     ]=4000
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_6_5000.000000_0.000000_0.000000_5000.000000.root'     ]=5000
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_7_6000.000000_0.000000_0.000000_6000.000000.root'     ]=6000
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_8_7000.000000_0.000000_0.000000_7000.000000.root'     ]=7000
    real_input['/cefs/higgs/wxfang/sim_ouput/official/user_9_8000.000000_0.000000_0.000000_8000.000000.root'     ]=8000
    '''

    '''
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_1_0.000000_0.000000_0.000000_0.000000.root'           ]=0
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_10_9000.000000_0.000000_0.000000_9000.000000.root'    ]=9000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_11_10000.000000_0.000000_0.000000_10000.000000.root'  ]=10000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_12_11000.000000_0.000000_0.000000_11000.000000.root'  ]=11000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_13_12000.000000_0.000000_0.000000_12000.000000.root'  ]=12000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_14_13000.000000_0.000000_0.000000_13000.000000.root'  ]=13000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_15_14000.000000_0.000000_0.000000_14000.000000.root'  ]=14000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_16_15000.000000_0.000000_0.000000_15000.000000.root'  ]=15000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_17_16000.000000_0.000000_0.000000_16000.000000.root'  ]=16000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_18_17000.000000_0.000000_0.000000_17000.000000.root'  ]=17000.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_2_1000.000000_0.000000_0.000000_1000.000000.root'     ]=1000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_3_2000.000000_0.000000_0.000000_2000.000000.root'     ]=2000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_4_3000.000000_0.000000_0.000000_3000.000000.root'     ]=3000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_5_4000.000000_0.000000_0.000000_4000.000000.root'     ]=4000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_6_5000.000000_0.000000_0.000000_5000.000000.root'     ]=5000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_7_6000.000000_0.000000_0.000000_6000.000000.root'     ]=6000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_8_7000.000000_0.000000_0.000000_7000.000000.root'     ]=7000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_ep/user_9_8000.000000_0.000000_0.000000_8000.000000.root'     ]=8000
    '''

    '''
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_1_0.000000_0.000000_0.000000_0.000000.root']         =0.000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_10_450.000000_0.000000_0.000000_450.000000.root']    =450.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_11_500.000000_0.000000_0.000000_500.000000.root']    =500.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_12_550.000000_0.000000_0.000000_550.000000.root']    =550.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_13_600.000000_0.000000_0.000000_600.000000.root']    =600.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_14_650.000000_0.000000_0.000000_650.000000.root']    =650.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_15_700.000000_0.000000_0.000000_700.000000.root']    =700.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_16_750.000000_0.000000_0.000000_750.000000.root']    =750.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_17_800.000000_0.000000_0.000000_800.000000.root']    =800.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_18_850.000000_0.000000_0.000000_850.000000.root']    =850.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_19_900.000000_0.000000_0.000000_900.000000.root']    =900.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_20_950.000000_0.000000_0.000000_950.000000.root']    =950.
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_21_1000.000000_0.000000_0.000000_1000.000000.root']  =1000
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_22_1050.000000_0.000000_0.000000_1050.000000.root']  =1050
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_2_50.000000_0.000000_0.000000_50.000000.root']       =50.00
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_3_100.000000_0.000000_0.000000_100.000000.root']     =100.0
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_4_150.000000_0.000000_0.000000_150.000000.root']     =150.0
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_5_200.000000_0.000000_0.000000_200.000000.root']     =200.0
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_6_250.000000_0.000000_0.000000_250.000000.root']     =250.0
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_7_300.000000_0.000000_0.000000_300.000000.root']     =300.0
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_8_350.000000_0.000000_0.000000_350.000000.root']     =350.0
    real_input['/cefs/higgs/wxfang/sim_ouput/official_r1m/user_9_400.000000_0.000000_0.000000_400.000000.root']     =400.0
    '''

    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_1_0.000000_0.000000_0.000000_0.000000.root'           ]=0.0000
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_10_9000.000000_0.000000_0.000000_9000.000000.root'    ]=9000.
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_11_10000.000000_0.000000_0.000000_10000.000000.root'  ]=10000
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_12_11000.000000_0.000000_0.000000_11000.000000.root'  ]=11000
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_13_12000.000000_0.000000_0.000000_12000.000000.root'  ]=12000
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_14_13000.000000_0.000000_0.000000_13000.000000.root'  ]=13000
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_15_14000.000000_0.000000_0.000000_14000.000000.root'  ]=14000
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_16_15000.000000_0.000000_0.000000_15000.000000.root'  ]=15000
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_17_16000.000000_0.000000_0.000000_16000.000000.root'  ]=16000
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_18_17000.000000_0.000000_0.000000_17000.000000.root'  ]=17000
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_2_1000.000000_0.000000_0.000000_1000.000000.root'     ]=1000.0
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_3_2000.000000_0.000000_0.000000_2000.000000.root'     ]=2000.0
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_4_3000.000000_0.000000_0.000000_3000.000000.root'     ]=3000.0
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_5_4000.000000_0.000000_0.000000_4000.000000.root'     ]=4000.0
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_6_5000.000000_0.000000_0.000000_5000.000000.root'     ]=5000.0
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_7_6000.000000_0.000000_0.000000_6000.000000.root'     ]=6000.0
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_8_7000.000000_0.000000_0.000000_7000.000000.root'     ]=7000.0
    real_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_9_8000.000000_0.000000_0.000000_8000.000000.root'     ]=8000.0






    fast1_input = {}

    '''
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_10_-8000.000000_0.000000_0.000000_-8000.000000.root'] = -8000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_1_-17000.000000_0.000000_0.000000_-17000.000000.root'] =-17000. 
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_11_-7000.000000_0.000000_0.000000_-7000.000000.root'] = -7000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_12_-6000.000000_0.000000_0.000000_-6000.000000.root'] = -6000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_13_-5000.000000_0.000000_0.000000_-5000.000000.root'] = -5000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_14_-4000.000000_0.000000_0.000000_-4000.000000.root'] = -4000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_15_-3000.000000_0.000000_0.000000_-3000.000000.root'] = -3000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_16_-2000.000000_0.000000_0.000000_-2000.000000.root'] = -2000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_17_-1000.000000_0.000000_0.000000_-1000.000000.root'] = -1000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_18_0.000000_0.000000_0.000000_0.000000.root'] =         0.0000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_19_1000.000000_0.000000_0.000000_1000.000000.root'] =   1000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_20_2000.000000_0.000000_0.000000_2000.000000.root'] =   2000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_21_3000.000000_0.000000_0.000000_3000.000000.root'] =   3000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_2_-16000.000000_0.000000_0.000000_-16000.000000.root'] =-16000. 
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_22_4000.000000_0.000000_0.000000_4000.000000.root'] =   4000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_23_5000.000000_0.000000_0.000000_5000.000000.root'] =   5000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_24_6000.000000_0.000000_0.000000_6000.000000.root'] =   6000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_25_7000.000000_0.000000_0.000000_7000.000000.root'] =   7000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_26_8000.000000_0.000000_0.000000_8000.000000.root'] =   8000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_27_9000.000000_0.000000_0.000000_9000.000000.root'] =   9000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_28_10000.000000_0.000000_0.000000_10000.000000.root'] = 10000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_29_11000.000000_0.000000_0.000000_11000.000000.root'] = 11000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_30_12000.000000_0.000000_0.000000_12000.000000.root'] = 12000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_31_13000.000000_0.000000_0.000000_13000.000000.root'] = 13000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_3_-15000.000000_0.000000_0.000000_-15000.000000.root'] =-15000. 
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_32_14000.000000_0.000000_0.000000_14000.000000.root'] = 14000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_33_15000.000000_0.000000_0.000000_15000.000000.root'] = 15000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_34_16000.000000_0.000000_0.000000_16000.000000.root'] = 16000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_35_17000.000000_0.000000_0.000000_17000.000000.root'] = 17000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_4_-14000.000000_0.000000_0.000000_-14000.000000.root'] =-14000. 
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_5_-13000.000000_0.000000_0.000000_-13000.000000.root'] =-13000. 
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_6_-12000.000000_0.000000_0.000000_-12000.000000.root'] =-12000. 
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_7_-11000.000000_0.000000_0.000000_-11000.000000.root'] =-11000. 
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_8_-10000.000000_0.000000_0.000000_-10000.000000.root'] =-10000. 
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_Co60_scale/user_9_-9000.000000_0.000000_0.000000_-9000.000000.root'] =  -9000.0
    '''



    '''
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_1_0.000000_0.000000_0.000000_0.000000.root'           ]=0.0000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_10_9000.000000_0.000000_0.000000_9000.000000.root'    ]=9000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_11_10000.000000_0.000000_0.000000_10000.000000.root'  ]=10000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_12_11000.000000_0.000000_0.000000_11000.000000.root'  ]=11000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_13_12000.000000_0.000000_0.000000_12000.000000.root'  ]=12000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_14_13000.000000_0.000000_0.000000_13000.000000.root'  ]=13000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_15_14000.000000_0.000000_0.000000_14000.000000.root'  ]=14000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_16_15000.000000_0.000000_0.000000_15000.000000.root'  ]=15000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_17_16000.000000_0.000000_0.000000_16000.000000.root'  ]=16000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_18_17000.000000_0.000000_0.000000_17000.000000.root'  ]=17000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_2_1000.000000_0.000000_0.000000_1000.000000.root'     ]=1000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_3_2000.000000_0.000000_0.000000_2000.000000.root'     ]=2000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_4_3000.000000_0.000000_0.000000_3000.000000.root'     ]=3000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_5_4000.000000_0.000000_0.000000_4000.000000.root'     ]=4000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_6_5000.000000_0.000000_0.000000_5000.000000.root'     ]=5000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_7_6000.000000_0.000000_0.000000_6000.000000.root'     ]=6000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_8_7000.000000_0.000000_0.000000_7000.000000.root'     ]=7000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_gamma_scale/user_9_8000.000000_0.000000_0.000000_8000.000000.root'     ]=8000.0
    '''

    '''
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_1_0.000000_0.000000_0.000000_0.000000.root'           ]=0.0000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_10_9000.000000_0.000000_0.000000_9000.000000.root'    ]=9000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_11_10000.000000_0.000000_0.000000_10000.000000.root'  ]=10000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_12_11000.000000_0.000000_0.000000_11000.000000.root'  ]=11000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_13_12000.000000_0.000000_0.000000_12000.000000.root'  ]=12000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_14_13000.000000_0.000000_0.000000_13000.000000.root'  ]=13000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_15_14000.000000_0.000000_0.000000_14000.000000.root'  ]=14000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_16_15000.000000_0.000000_0.000000_15000.000000.root'  ]=15000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_17_16000.000000_0.000000_0.000000_16000.000000.root'  ]=16000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_18_17000.000000_0.000000_0.000000_17000.000000.root'  ]=17000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_2_1000.000000_0.000000_0.000000_1000.000000.root'     ]=1000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_3_2000.000000_0.000000_0.000000_2000.000000.root'     ]=2000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_4_3000.000000_0.000000_0.000000_3000.000000.root'     ]=3000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_5_4000.000000_0.000000_0.000000_4000.000000.root'     ]=4000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_6_5000.000000_0.000000_0.000000_5000.000000.root'     ]=5000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_7_6000.000000_0.000000_0.000000_6000.000000.root'     ]=6000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_8_7000.000000_0.000000_0.000000_7000.000000.root'     ]=7000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r_linear_ep/user_9_8000.000000_0.000000_0.000000_8000.000000.root'     ]=8000.0
    '''
    '''
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_1_0.000000_0.000000_0.000000_0.000000.root']         =0.0000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_10_450.000000_0.000000_0.000000_450.000000.root']    =450.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_11_500.000000_0.000000_0.000000_500.000000.root']    =500.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_12_550.000000_0.000000_0.000000_550.000000.root']    =550.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_13_600.000000_0.000000_0.000000_600.000000.root']    =600.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_14_650.000000_0.000000_0.000000_650.000000.root']    =650.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_15_700.000000_0.000000_0.000000_700.000000.root']    =700.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_16_750.000000_0.000000_0.000000_750.000000.root']    =750.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_17_800.000000_0.000000_0.000000_800.000000.root']    =800.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_18_850.000000_0.000000_0.000000_850.000000.root']    =850.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_19_900.000000_0.000000_0.000000_900.000000.root']    =900.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_20_950.000000_0.000000_0.000000_950.000000.root']    =950.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_21_1000.000000_0.000000_0.000000_1000.000000.root']  =1000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_22_1050.000000_0.000000_0.000000_1050.000000.root']  =1050.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_2_50.000000_0.000000_0.000000_50.000000.root']       =50.000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_3_100.000000_0.000000_0.000000_100.000000.root']     =100.00
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_4_150.000000_0.000000_0.000000_150.000000.root']     =150.00
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_5_200.000000_0.000000_0.000000_200.000000.root']     =200.00
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_6_250.000000_0.000000_0.000000_250.000000.root']     =250.00
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_7_300.000000_0.000000_0.000000_300.000000.root']     =300.00
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_8_350.000000_0.000000_0.000000_350.000000.root']     =350.00
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_r1m_check/user_9_400.000000_0.000000_0.000000_400.000000.root']     =400.00
    '''

    ''' 
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/0.000000.root'           ]=0.0000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/9000.000000.root'    ]=9000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/10000.000000.root'  ]=10000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/11000.000000.root'  ]=11000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/12000.000000.root'  ]=12000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/13000.000000.root'  ]=13000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/14000.000000.root'  ]=14000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/15000.000000.root'  ]=15000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/16000.000000.root'  ]=16000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/17000.000000.root'  ]=17000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/1000.000000.root'     ]=1000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/2000.000000.root'     ]=2000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/3000.000000.root'     ]=3000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/4000.000000.root'     ]=4000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/5000.000000.root'     ]=5000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/6000.000000.root'     ]=6000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/7000.000000.root'     ]=7000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_hadd/8000.000000.root'     ]=8000.0

    ''' 

    ''' 
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/0.000000.root'           ]=0.0000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/9000.000000.root'    ]=9000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/10000.000000.root'  ]=10000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/11000.000000.root'  ]=11000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/12000.000000.root'  ]=12000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/13000.000000.root'  ]=13000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/14000.000000.root'  ]=14000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/15000.000000.root'  ]=15000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/16000.000000.root'  ]=16000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/17000.000000.root'  ]=17000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/1000.000000.root'     ]=1000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/2000.000000.root'     ]=2000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/3000.000000.root'     ]=3000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/4000.000000.root'     ]=4000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/5000.000000.root'     ]=5000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/6000.000000.root'     ]=6000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/7000.000000.root'     ]=7000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_hadd/8000.000000.root'     ]=8000.0
    ''' 

    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/0.000000.root'           ]=0.0000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/9000.000000.root'    ]=9000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/10000.000000.root'  ]=10000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/11000.000000.root'  ]=11000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/12000.000000.root'  ]=12000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/13000.000000.root'  ]=13000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/14000.000000.root'  ]=14000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/15000.000000.root'  ]=15000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/16000.000000.root'  ]=16000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/17000.000000.root'  ]=17000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/1000.000000.root'     ]=1000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/2000.000000.root'     ]=2000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/3000.000000.root'     ]=3000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/4000.000000.root'     ]=4000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/5000.000000.root'     ]=5000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/6000.000000.root'     ]=6000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/7000.000000.root'     ]=7000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_gamma_torch_merge_hadd/8000.000000.root'     ]=8000.0
    
    ''' 
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/0.000000.root'           ]=0.0000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/9000.000000.root'    ]=9000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/10000.000000.root'  ]=10000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/11000.000000.root'  ]=11000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/12000.000000.root'  ]=12000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/13000.000000.root'  ]=13000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/14000.000000.root'  ]=14000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/15000.000000.root'  ]=15000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/16000.000000.root'  ]=16000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/17000.000000.root'  ]=17000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/1000.000000.root'     ]=1000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/2000.000000.root'     ]=2000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/3000.000000.root'     ]=3000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/4000.000000.root'     ]=4000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/5000.000000.root'     ]=5000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/6000.000000.root'     ]=6000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/7000.000000.root'     ]=7000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeDL_Co60_hadd/8000.000000.root'     ]=8000.0
    ''' 

    ''' 
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/0.000000.root'           ]=0.0000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/9000.000000.root'    ]=9000.
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/10000.000000.root'  ]=10000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/11000.000000.root'  ]=11000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/12000.000000.root'  ]=12000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/13000.000000.root'  ]=13000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/14000.000000.root'  ]=14000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/15000.000000.root'  ]=15000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/16000.000000.root'  ]=16000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/17000.000000.root'  ]=17000
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/1000.000000.root'     ]=1000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/2000.000000.root'     ]=2000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/3000.000000.root'     ]=3000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/4000.000000.root'     ]=4000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/5000.000000.root'     ]=5000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/6000.000000.root'     ]=6000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/7000.000000.root'     ]=7000.0
    fast1_input['/cefs/higgs/wxfang/sim_ouput/wx_NpeMean_gamma_check_hadd/8000.000000.root'     ]=8000.0
    ''' 




    out_path = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/full_vs_fast_plots/'
    #out_path = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/full_vs_fast_plots_Co60/'
    #npe_full_sim  = get_npe_info_v1(tag='npe_full_sim' , input_files=real_input ,binning=[200,2000,4000])
    #npe_fast_sim  = get_npe_info_v1(tag='npe_fast1_sim', input_files=fast1_input,binning=[200,2000,4000])
    npe_full_sim  = get_npe_info_v1(tag='npe_full_sim' , input_files=real_input ,binning=[50,1000,1500])
    npe_fast_sim  = get_npe_info_v1(tag='npe_fast1_sim', input_files=fast1_input,binning=[50,1000,1500])
    hittime_full_sim   = get_hittime_info      (tag='hittime_full_sim'  , input_files=real_input , refine_binning=True)
    hittime_fast_sim   = get_hittime_info      (tag='hittime_fast1_sim' , input_files=fast1_input, refine_binning=True)
    Fhittime_full_sim  = get_first_hittime_info(tag='Fhittime_full_sim' , input_files=real_input , refine_binning=True)
    Fhittime_fast_sim  = get_first_hittime_info(tag='Fhittime_fast1_sim', input_files=fast1_input, refine_binning=True)
    gr_full = rt.TGraph()
    gr_fast = rt.TGraph()
    index = 0
    xbin = [1010,1500]
    #xbin = [2010,4000]
    #xbin0 = [-18,18]
    xbin0 = [0,18]
    for ir in npe_full_sim:
        if ir not in npe_fast_sim: continue
        gr_full.SetPoint(index , float(ir)/1000, npe_full_sim[ir][0])
        gr_fast.SetPoint(index , float(ir)/1000, npe_fast_sim[ir][0])
        index += 1
        #do_plot_v1(str(ir),npe_full_sim[ir][1], npe_fast_sim[ir][1], 'tot_pe', xbin, out_path,['full sim', 'fast sim'], False, 0.03)
        do_plot_v1_1(str(ir),npe_full_sim[ir][1], npe_fast_sim[ir][1], 'tot_pe', xbin, out_path,['full sim', 'fast sim'], True, 0.03)
        do_plot_v2(str(ir), str('z=%s mm'%ir),npe_full_sim[ir][2], npe_fast_sim[ir][2], 'n_pe_logy', [0, 10], out_path,['full sim', 'fast sim'], False, 0.04, ['full', 'fast'])
        do_plot_v2(str(ir), str('z=%s mm'%ir),hittime_full_sim[ir], hittime_fast_sim[ir], 'hit_time', [0,500], out_path,['full sim', 'fast sim'], True, 0.04, ['full', 'fast'])
        do_plot_v2(str(ir), str('z=%s mm'%ir),Fhittime_full_sim[ir], Fhittime_fast_sim[ir], 'First_Hit_time', [0,500], out_path,['full sim', 'fast sim'], True, 0.04, ['full', 'fast'])
    do_plot_2gr_v1(gr_full,gr_fast,'','meanNpeVsR_', xbin0, xbin, ['z (m)', 'total PE'], 1 , out_path, False, ['full sim', 'fast sim'], ['full', 'fast'])


if For_PMT_Theta:
    #hist_PMT_theta = read_hist_PMT_theta(real_file, [1800,0,180], 'real')
    hist_PMT_theta = read_hist_PMT_theta(real_file, [180,0,180], 'real')
    do_plot_1('real', hist_PMT_theta, 'pmt_theta')
    
if For_r_theta_vs_npeMean:
    out_path = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/r_theta_vs_NPE_plots/'
    dict_r_gr,dict_theta_gr = read_graph_r_theta_vs_meanNPE(real_file)
    #for r in dict_r_gr: 
    #    #do_plot_gr_1D(dict_r_gr[r], '','r=%s_thetaVsNpe'%str(r), [0,180], [0,0.5], ['#theta', 'mean NPE'], 0.5 ,out_path, False)
    #    do_plot_gr_1D(dict_r_gr[r], '','r=%s_thetaVsNpe'%str(r), [45,55], [0,0.5], ['#theta', 'mean NPE'], 0.5 , out_path, False)
    for theta in dict_theta_gr: 
        if float(theta)%1 !=0 : continue
        do_plot_gr_1D(dict_theta_gr[theta], '','theta=%s_rVsNpe'%str(theta), [0,20], [0,5], ['r', 'mean NPE'], 0.5 , out_path, False)

if For_visE_vs_npeMean:
    out_path = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/VisE_vs_NPE_plots/'
    input_file = {}
    r = 1000
    #input_file['/besfs/groups/higgs/users/fangwx/public/juno/r_photon_skimed_h5/user_1000_2532_1000.000000_623.237470_594.202696_508.427195_skimed_batch0_N5000.h5']         = 2532  
    #input_file['/besfs/groups/higgs/users/fangwx/public/juno/r_photon_skimed_h5/user_1003_5064_1000.000000_-298.807284_-776.444610_-554.840494_skimed_batch0_N5000.h5']      = 5064 
    #input_file['/besfs/groups/higgs/users/fangwx/public/juno/r_photon_skimed_h5/user_1006_10129_1000.000000_410.241040_-313.262063_-856.486526_skimed_batch0_N5000.h5']      = 10129 
    #input_file['/besfs/groups/higgs/users/fangwx/public/juno/r_photon_skimed_h5/user_1009_20258_1000.000000_-826.410690_144.820843_-544.125257_skimed_batch0_N5000.h5']      = 20258 
    #input_file['/besfs/groups/higgs/users/fangwx/public/juno/r_photon_skimed_h5/user_1012_40516_1000.000000_-393.854846_-870.766311_-294.354195_skimed_batch0_N5000.h5']     = 40516
    #r = 10000
    #input_file['/besfs/groups/higgs/users/fangwx/public/juno/r_photon_skimed_h5/user_1001_2532_10000.000000_-880.745647_2150.477848_9726.239362_skimed_batch0_N5000.h5']     = 2532 
    #input_file['/besfs/groups/higgs/users/fangwx/public/juno/r_photon_skimed_h5/user_1004_5064_10000.000000_-572.363291_2132.538520_9753.188183_skimed_batch0_N5000.h5']     = 5064 
    #input_file['/besfs/groups/higgs/users/fangwx/public/juno/r_photon_skimed_h5/user_1007_10129_10000.000000_-1967.785093_420.813206_-9795.444761_skimed_batch0_N5000.h5']   = 10129 
    #input_file['/besfs/groups/higgs/users/fangwx/public/juno/r_photon_skimed_h5/user_1010_20258_10000.000000_1690.790572_8259.669144_-5377.647522_skimed_batch0_N5000.h5']   = 20258 
    #input_file['/besfs/groups/higgs/users/fangwx/public/juno/r_photon_skimed_h5/user_1013_40516_10000.000000_9665.018700_2216.327923_1294.335376_skimed_batch0_N5000.h5']    = 40516
    r = 16000
    input_file['/besfs/groups/higgs/users/fangwx/public/juno/r_photon_skimed_h5/user_1002_2532_16000.000000_-1637.782475_14304.282007_-6978.910002_skimed_batch0_N5000.h5']  = 2532 
    input_file['/besfs/groups/higgs/users/fangwx/public/juno/r_photon_skimed_h5/user_1005_5064_16000.000000_13281.256789_-4957.475885_7418.332088_skimed_batch0_N5000.h5']   = 5064 
    input_file['/besfs/groups/higgs/users/fangwx/public/juno/r_photon_skimed_h5/user_1008_10129_16000.000000_5615.978437_-3527.182566_14560.898645_skimed_batch0_N5000.h5']  = 10129 
    input_file['/besfs/groups/higgs/users/fangwx/public/juno/r_photon_skimed_h5/user_1011_20258_16000.000000_-12516.937362_8270.165903_-5561.531715_skimed_batch0_N5000.h5'] = 20258 

    dict_theta_gr = read_graph_visE_vs_meanNPE(input_file)
    for theta in dict_theta_gr: 
        do_plot_gr_1D(dict_theta_gr[theta], '','r=%s_theta=%s_visEVsNpe'%(str(r),str(theta)), [0,5], [0,2], ['vis E (MeV)', 'mean NPE'], 0.5 , out_path, True)


if For_NPE:
    real_file = []
    '''
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_10_10129_2750.000000_0.000000_0.000000_2750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_1_10129_500.000000_0.000000_0.000000_500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_12_10129_3250.000000_0.000000_0.000000_3250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_13_10129_3500.000000_0.000000_0.000000_3500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_14_10129_3750.000000_0.000000_0.000000_3750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_16_10129_4250.000000_0.000000_0.000000_4250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_17_10129_4500.000000_0.000000_0.000000_4500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_18_10129_4750.000000_0.000000_0.000000_4750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_20_10129_5250.000000_0.000000_0.000000_5250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_2_10129_750.000000_0.000000_0.000000_750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_21_10129_5500.000000_0.000000_0.000000_5500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_22_10129_5750.000000_0.000000_0.000000_5750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_24_10129_6250.000000_0.000000_0.000000_6250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_25_10129_6500.000000_0.000000_0.000000_6500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_26_10129_6750.000000_0.000000_0.000000_6750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_28_10129_7250.000000_0.000000_0.000000_7250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_29_10129_7500.000000_0.000000_0.000000_7500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_30_10129_7750.000000_0.000000_0.000000_7750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_32_10129_8250.000000_0.000000_0.000000_8250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_33_10129_8500.000000_0.000000_0.000000_8500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_34_10129_8750.000000_0.000000_0.000000_8750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_36_10129_9250.000000_0.000000_0.000000_9250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_37_10129_9500.000000_0.000000_0.000000_9500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_38_10129_9750.000000_0.000000_0.000000_9750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_40_10129_10250.000000_0.000000_0.000000_10250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_4_10129_1250.000000_0.000000_0.000000_1250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_41_10129_10500.000000_0.000000_0.000000_10500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_42_10129_10750.000000_0.000000_0.000000_10750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_44_10129_11250.000000_0.000000_0.000000_11250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_45_10129_11500.000000_0.000000_0.000000_11500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_46_10129_11750.000000_0.000000_0.000000_11750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_48_10129_12250.000000_0.000000_0.000000_12250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_49_10129_12500.000000_0.000000_0.000000_12500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_50_10129_12750.000000_0.000000_0.000000_12750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_5_10129_1500.000000_0.000000_0.000000_1500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_52_10129_13250.000000_0.000000_0.000000_13250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_53_10129_13500.000000_0.000000_0.000000_13500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_54_10129_13750.000000_0.000000_0.000000_13750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_6_10129_1750.000000_0.000000_0.000000_1750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_8_10129_2250.000000_0.000000_0.000000_2250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_9_10129_2500.000000_0.000000_0.000000_2500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_56_10129_14250.000000_0.000000_0.000000_14250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_57_10129_14500.000000_0.000000_0.000000_14500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_58_10129_14750.000000_0.000000_0.000000_14750.000000_batch0_N5000.h5')
    '''
    '''
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_71_10129_1.000000_0.000000_0.000000_1.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_3_10129_1000.000000_0.000000_0.000000_1000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_7_10129_2000.000000_0.000000_0.000000_2000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_11_10129_3000.000000_0.000000_0.000000_3000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_15_10129_4000.000000_0.000000_0.000000_4000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_19_10129_5000.000000_0.000000_0.000000_5000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_23_10129_6000.000000_0.000000_0.000000_6000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_27_10129_7000.000000_0.000000_0.000000_7000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_31_10129_8000.000000_0.000000_0.000000_8000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_35_10129_9000.000000_0.000000_0.000000_9000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_39_10129_10000.000000_0.000000_0.000000_10000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_43_10129_11000.000000_0.000000_0.000000_11000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_47_10129_12000.000000_0.000000_0.000000_12000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data//user_51_10129_13000.000000_0.000000_0.000000_13000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_55_10129_14000.000000_0.000000_0.000000_14000.000000_batch0_N5000.h5')

    '''
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_59_10129_15000.000000_0.000000_0.000000_15000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_63_10129_16000.000000_0.000000_0.000000_16000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_67_10129_17000.000000_0.000000_0.000000_17000.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_60_10129_15250.000000_0.000000_0.000000_15250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_61_10129_15500.000000_0.000000_0.000000_15500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_62_10129_15750.000000_0.000000_0.000000_15750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_64_10129_16250.000000_0.000000_0.000000_16250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_65_10129_16500.000000_0.000000_0.000000_16500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_66_10129_16750.000000_0.000000_0.000000_16750.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_68_10129_17250.000000_0.000000_0.000000_17250.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_69_10129_17500.000000_0.000000_0.000000_17500.000000_batch0_N5000.h5')
    real_file.append('/hpcfs/cepc/higgs/wxfang/data/public/photon_sample_r_linera_npe_valid_h5_data/user_70_10129_17750.000000_0.000000_0.000000_17750.000000_batch0_N5000.h5')


    fake_file = []
    '''
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_10_10129_2750.000000_0.000000_0.000000_2750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_1_10129_500.000000_0.000000_0.000000_500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_11_10129_3000.000000_0.000000_0.000000_3000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_12_10129_3250.000000_0.000000_0.000000_3250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_13_10129_3500.000000_0.000000_0.000000_3500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_14_10129_3750.000000_0.000000_0.000000_3750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_15_10129_4000.000000_0.000000_0.000000_4000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_16_10129_4250.000000_0.000000_0.000000_4250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_17_10129_4500.000000_0.000000_0.000000_4500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_18_10129_4750.000000_0.000000_0.000000_4750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_19_10129_5000.000000_0.000000_0.000000_5000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_20_10129_5250.000000_0.000000_0.000000_5250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_2_10129_750.000000_0.000000_0.000000_750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_21_10129_5500.000000_0.000000_0.000000_5500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_22_10129_5750.000000_0.000000_0.000000_5750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_23_10129_6000.000000_0.000000_0.000000_6000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_24_10129_6250.000000_0.000000_0.000000_6250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_25_10129_6500.000000_0.000000_0.000000_6500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_26_10129_6750.000000_0.000000_0.000000_6750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_27_10129_7000.000000_0.000000_0.000000_7000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_28_10129_7250.000000_0.000000_0.000000_7250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_29_10129_7500.000000_0.000000_0.000000_7500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_30_10129_7750.000000_0.000000_0.000000_7750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_3_10129_1000.000000_0.000000_0.000000_1000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_31_10129_8000.000000_0.000000_0.000000_8000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_32_10129_8250.000000_0.000000_0.000000_8250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_33_10129_8500.000000_0.000000_0.000000_8500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_34_10129_8750.000000_0.000000_0.000000_8750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_35_10129_9000.000000_0.000000_0.000000_9000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_36_10129_9250.000000_0.000000_0.000000_9250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_37_10129_9500.000000_0.000000_0.000000_9500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_38_10129_9750.000000_0.000000_0.000000_9750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_39_10129_10000.000000_0.000000_0.000000_10000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_40_10129_10250.000000_0.000000_0.000000_10250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_4_10129_1250.000000_0.000000_0.000000_1250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_41_10129_10500.000000_0.000000_0.000000_10500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_42_10129_10750.000000_0.000000_0.000000_10750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_43_10129_11000.000000_0.000000_0.000000_11000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_44_10129_11250.000000_0.000000_0.000000_11250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_45_10129_11500.000000_0.000000_0.000000_11500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_46_10129_11750.000000_0.000000_0.000000_11750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_47_10129_12000.000000_0.000000_0.000000_12000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_48_10129_12250.000000_0.000000_0.000000_12250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_49_10129_12500.000000_0.000000_0.000000_12500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_50_10129_12750.000000_0.000000_0.000000_12750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_5_10129_1500.000000_0.000000_0.000000_1500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_51_10129_13000.000000_0.000000_0.000000_13000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_52_10129_13250.000000_0.000000_0.000000_13250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_53_10129_13500.000000_0.000000_0.000000_13500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_54_10129_13750.000000_0.000000_0.000000_13750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_6_10129_1750.000000_0.000000_0.000000_1750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_7_10129_2000.000000_0.000000_0.000000_2000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_71_10129_1.000000_0.000000_0.000000_1.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_8_10129_2250.000000_0.000000_0.000000_2250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_9_10129_2500.000000_0.000000_0.000000_2500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_55_10129_14000.000000_0.000000_0.000000_14000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_56_10129_14250.000000_0.000000_0.000000_14250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_57_10129_14500.000000_0.000000_0.000000_14500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_low_0106_50/user_58_10129_14750.000000_0.000000_0.000000_14750.000000_batch0_N5000_pred.h5')
    '''

    '''
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_low_MM//user_71_10129_1.000000_0.000000_0.000000_1.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_low_MM//user_3_10129_1000.000000_0.000000_0.000000_1000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_low_MM//user_7_10129_2000.000000_0.000000_0.000000_2000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_low_MM//user_11_10129_3000.000000_0.000000_0.000000_3000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_low_MM//user_15_10129_4000.000000_0.000000_0.000000_4000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_low_MM//user_19_10129_5000.000000_0.000000_0.000000_5000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_low_MM//user_23_10129_6000.000000_0.000000_0.000000_6000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_low_MM//user_27_10129_7000.000000_0.000000_0.000000_7000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_low_MM//user_31_10129_8000.000000_0.000000_0.000000_8000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_low_MM//user_35_10129_9000.000000_0.000000_0.000000_9000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_low_MM//user_39_10129_10000.000000_0.000000_0.000000_10000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_low_MM//user_43_10129_11000.000000_0.000000_0.000000_11000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_low_MM//user_47_10129_12000.000000_0.000000_0.000000_12000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_low_MM//user_51_10129_13000.000000_0.000000_0.000000_13000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_low_MM//user_55_10129_14000.000000_0.000000_0.000000_14000.000000_batch0_N5000_pred.h5')

    '''
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_high_MM//user_59_10129_15000.000000_0.000000_0.000000_15000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_high_MM//user_63_10129_16000.000000_0.000000_0.000000_16000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_high_MM//user_67_10129_17000.000000_0.000000_0.000000_17000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_high_MM//user_60_10129_15250.000000_0.000000_0.000000_15250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_high_MM//user_61_10129_15500.000000_0.000000_0.000000_15500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_high_MM//user_62_10129_15750.000000_0.000000_0.000000_15750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_high_MM//user_64_10129_16250.000000_0.000000_0.000000_16250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_high_MM//user_65_10129_16500.000000_0.000000_0.000000_16500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_high_MM//user_66_10129_16750.000000_0.000000_0.000000_16750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_high_MM//user_68_10129_17250.000000_0.000000_0.000000_17250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_high_MM//user_69_10129_17500.000000_0.000000_0.000000_17500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/predFiles_linear_high_MM//user_70_10129_17750.000000_0.000000_0.000000_17750.000000_batch0_N5000_pred.h5')

    '''
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_high_r1516_0104/user_59_10129_15000.000000_0.000000_0.000000_15000.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_high_r1516_0104/user_60_10129_15250.000000_0.000000_0.000000_15250.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_high_r1516_0104/user_61_10129_15500.000000_0.000000_0.000000_15500.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_high_r1516_0104/user_62_10129_15750.000000_0.000000_0.000000_15750.000000_batch0_N5000_pred.h5')
    fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_linear_high_r1516_0104/user_63_10129_16000.000000_0.000000_0.000000_16000.000000_batch0_N5000_pred.h5')
    '''


    fake_hists = {}
    real_hists = {}
    #fake_hists = read_hist_v1(fake_file, [10, -5, 5], 'fake', True)
    #real_hists = read_hist_v1(real_file, [10, -5, 5], 'real', False)
    fake_hists_totPE = read_hist_totPE(fake_file, [160, -10, 1500], 'fake','_pred.h5')
    real_hists_totPE = read_hist_totPE(real_file, [160, -10, 1500], 'real','.h5')

if For_Time:
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1202.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1204mini.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1204mini_v1.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1204mini_v2.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1204mini_v3.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1205.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_npe_1205.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_npe_1205mae.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1206ks.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_npe_1206ks.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_npe_1206ksNoise.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_npe_1207fix.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1207fix.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_npe_1207v1.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1208.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1208_ep40.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1208_ep100.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1208_ne13.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1208_ne13ep100.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1209_ne12ep40_gauss.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1209_ne12ep40_gauss_norm1.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1210_elu.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1210_elu_ep100.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1209_full.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1209_part.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1210_part_relu.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1210_part_elu.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1216_part_elu.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1216_part_elu_b512.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1216_full_elu_b10000.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1216_part_sh_SGDM_lr1e-3.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1216_part_sh_RMSP_lr1.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1217_part_sh_Adam_lr6e-3.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1217_part_sh_Adam_rel_mae_lr6e-3.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1217_part_sh_Adam_rel_mae_lr6e-3_b1e4.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1218_part_sh_Adam_rel_mae.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1223_full_sh_Adam_rel_mae.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1227_full_large_sh_Adam_rel_mae.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1229_full_rel_mae_w_early.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1230_full_rel_mae_we0.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_r1m.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_0102_linear_early.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_linear_high_0102_early.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_linear_low_0102_early.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_linear_high_0104_sh.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_linear_low_0106.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_linear_high_0107_r15.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_linear_high_0108_r15.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_linear_high_0111_r15.h5']
    #fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/produced_r_theta_time_linear_low_MM.h5']
    fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/pyTorch/produced_r_theta_time_linear_high_0618_r15.h5']
    real_file = []
    #real_txt = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/plane_r1m_hittime_valid_full.txt' 
    #real_txt = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/photon_sample_r_linera_hittime_valid.txt' 
    real_txt = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/photon_sample_r_linera_hittime_high_valid.txt' 
    #real_txt = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/photon_sample_r_linera_hittime_low_valid.txt' 
    with open(real_txt,'r') as f:
        for line in f:
            if "#" in line:continue
            line = line.replace('\n','')
            real_file.append(line)
    fake_hists = read_hist(fake_file, 'predHitTimeByPMT_', [50, 0, 500], 'fake', False, 1, False)
    real_hists = read_hist(real_file, 'HitTimeByPMT_'    , [50, 0, 500], 'real', False, 1, False)

if For_NPE or For_Time:
    print('start for ploting\n')
    count = 0
    r_theta_list = list(fake_hists.keys())
    r_theta_list = shuffle(r_theta_list, random_state=1)
    for i in r_theta_list:
        if i not in real_hists: continue
        print('r_theta=',i)
        if float(i.split('_')[0])>17.7: continue ### out of CD
        c = 0
        try:
            c = chi2_ndf(real_hists[i], fake_hists[i], True)
        except:
            print('bad i')
        else:
            pass
        if For_NPE:
            do_plot (i, real_hists[i], fake_hists[i], 'n_pe_', False, c, [0, 5])
            do_plot (i, real_hists[i], fake_hists[i], 'n_pe_logy', False, c, [0, 5])
        if For_Time:
            do_plot (i, real_hists[i], fake_hists[i], 'hit_time_'    , False, c, [0, 500])
            do_plot (i, real_hists[i], fake_hists[i], 'hit_time_logy', False, c, [0, 500])
            do_plot_v2(str(i),str('r=%s,#theta=%s,#chi^{2}/NDF=%.3f'%(i.split('_')[0], i.split('_')[1], c)),real_hists[i], fake_hists[i], 'hit_time_ratio', [0,500], plot_path,['G4', 'NN'], True, 0.04, ['G4', 'NN'])
        count = count + 1
        if count > 100: break
    
    if For_NPE:
        gr_mean_real = rt.TGraph()
        gr_mean_fake = rt.TGraph()
        index = 0
        for i in fake_hists_totPE:
            if i not in real_hists_totPE: continue
            c = chi2_ndf(real_hists_totPE[i], fake_hists_totPE[i], True)
            do_plot (i, real_hists_totPE[i], fake_hists_totPE[i], 'tot_pe_'    , False, c, [710, 1590])
            do_plot (i, real_hists_totPE[i], fake_hists_totPE[i], 'tot_pe_logy', False, c, [710, 1590])
            gr_mean_real.SetPoint(index, float(i.split('_')[3])/1000, real_hists_totPE[i].GetMean())     
            gr_mean_fake.SetPoint(index, float(i.split('_')[3])/1000, fake_hists_totPE[i].GetMean())     
            index += 1
        do_plot_2gr_v1(gr_mean_real, gr_mean_fake,'','NNValid_meanNpeVsR_', [0, 18], [1010,1500], ['z (m)', 'total PE'], 1 , plot_path, False, ['G4', 'NN'], ['G4', 'NN'])
    
    
    gr_x = []
    gr_y = []
    gr_z = []
    for i in r_theta_list:
        if i not in real_hists: continue
        gr_x.append(float(i.split('_')[0])) # r       0-20
        gr_y.append(float(i.split('_')[1])) # theta , 0-180
        c = chi2_ndf(real_hists[i], fake_hists[i], True)
        gr_z.append(c)
    if len(gr_x) == 0 : sys.exit()
    gr_2D        = rt.TGraph2D(len(gr_x), np.array(gr_x), np.array(gr_y), np.array(gr_z))
    gr_r_chi     = rt.TGraph(len(gr_z), np.array(gr_x), np.array(gr_z))
    gr_theta_chi = rt.TGraph(len(gr_z), np.array(gr_y), np.array(gr_z))
    if For_NPE:
        do_plot_gr(gr_2D, '' ,'n_pe_fit')
        do_plot_gr_1D(gr_r_chi    , '' ,'n_pe_fit_r'    ,[0,20 ],[0,10], ['r',"#chi^{2}/NDF"     ],1,plot_path, False)
        do_plot_gr_1D(gr_theta_chi, '' ,'n_pe_fit_theta',[0,180],[0,10], ['#theta',"#chi^{2}/NDF"],1,plot_path, False)
    if For_Time:
        do_plot_gr(gr_2D, '' ,'hit_time_fit')
        do_plot_gr_1D(gr_r_chi    , '' ,'hit_time_fit_r'    ,[0,20 ],[0,10], ['r',"#chi^{2}/NDF"     ], 1,plot_path, False)
        do_plot_gr_1D(gr_theta_chi, '' ,'hit_time_fit_theta',[0,180],[0,10], ['#theta',"#chi^{2}/NDF"], 1,plot_path, False)
    
    print('total chi2/NDF:',sum(gr_z))

print('done!')

