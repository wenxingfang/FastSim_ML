import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import argparse
rt.gROOT.SetBatch(rt.kTRUE)

def do_plot_h2(h1, h2, out_name, title, plot_label):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.12)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)

    x_min=h1.GetXaxis().GetXmin()
    x_max=h1.GetXaxis().GetXmax()
    y_min=0
    y_max=h2.GetBinContent(h2.GetMaximumBin())*1.5 
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
    h2 .SetMarkerStyle(24)
    h1.Draw("same:pe")
    h2.Draw("same:pe")
    dummy.Draw("AXISSAME")
    x_l = 0.6
    x_dl = 0.25
    y_h = 0.85
    y_dh = 0.15
    legend = rt.TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    legend.AddEntry(h1 ,'real','lep')
    legend.AddEntry(h2 ,"new",'lep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()

    label = rt.TLatex(0.46 , 0.9, plot_label)
    label.SetTextAlign(32)
    label.SetTextSize(0.035)
    label.SetNDC(rt.kTRUE)
    #label.Draw() 
    canvas.SaveAs("%s.png"%(out_name))
    del canvas
    gc.collect()

if __name__ == '__main__':

    h_s  = rt.TH1F('h_real' , '', 200, -10, 10)
    h_s1 = rt.TH1F('h_new'  , '', 200, -10, 10)
    s_mu    = 1
    s_sigma = 2
    b_mu    = 0.5
    b_sigma = 1
    for i in range(0, 100000):
        s = np.random.normal(s_mu, s_sigma, 1)
        b0 = np.random.normal(b_mu, b_sigma, 1)
        b1 = np.random.normal(b_mu, b_sigma, 1)
        s1 = s + b0 - b1
        h_s .Fill(s ) 
        h_s1.Fill(s1) 
    do_plot_h2(h1=h_s, h2=h_s1, out_name='test', title={'X':'','Y':'Event'}, plot_label='')
    print('done')
