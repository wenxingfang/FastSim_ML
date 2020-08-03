import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import json
from array import array
rt.gROOT.SetBatch(rt.kTRUE)
#rt.TGaxis.SetMaxDigits(3);

def do_plot_h_list(hlist, out_name, title, plot_label):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.12)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)

    x_min=0
    x_max=110
    y_min=0.9
    y_max=1.05 
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
    for h in hlist:
        h.SetLineWidth(2)
        h.SetLineColor(rt.kBlue)
        h.SetMarkerColor(rt.kBlue)
        h.SetMarkerStyle(20)
        h.Draw("same:pe")
    dummy.Draw("AXISSAME")
    x_l = 0.5
    x_dl = 0.25
    y_h = 0.85
    y_dh = 0.15
    if 'gr_Ebin_E_mean' in out_name:
        x_l = 0.2
        y_h = 0.85
    legend = rt.TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    legend.AddEntry(hlist[0] ,'Matrix ECAL','lep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    #legend.Draw()

    label = rt.TLatex(0.26 , 0.9, plot_label)
    label.SetTextAlign(32)
    label.SetTextSize(0.035)
    label.SetNDC(rt.kTRUE)
    label.Draw() 
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()



def do_plot_g2(h1, h2, out_name, title, plot_label):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.12)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)

#    h1.Scale(1/h1.GetSumOfWeights())
#    h2.Scale(1/h2.GetSumOfWeights())
    x_min=h1.GetXaxis().GetXmin()
    x_max=h1.GetXaxis().GetXmax()
    y_min=0
    y_max=1.05 
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
    #legend.AddEntry(h1 ,'Full Sim','lep')
    legend.AddEntry(h1 ,'K4 Calo only','lep')
    legend.AddEntry(h2 ,"Calo only",'lep')
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


def do_plot_h3(h1, h2, h3, out_name, title, plot_label):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.12)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)

#    h1.Scale(1/h1.GetSumOfWeights())
#    h2.Scale(1/h2.GetSumOfWeights())
    x_min=h1.GetXaxis().GetXmin()
    x_max=h1.GetXaxis().GetXmax()
    y_min=0
    y_max=1.05 
    if 'eff_E' in out_name:
        #y_min=0.9
        y_min=0.84
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
    h3 .SetLineWidth(2)
    h1 .SetLineColor(rt.kRed)
    h2 .SetLineColor(rt.kBlue)
    h3 .SetLineColor(rt.kBlack)
    h1 .SetMarkerColor(rt.kRed)
    h2 .SetMarkerColor(rt.kBlue)
    h3 .SetMarkerColor(rt.kBlack)
    h1 .SetMarkerStyle(20)
    h2 .SetMarkerStyle(21)
    h3 .SetMarkerStyle(24)
    h1.Draw("same:pe")
    h2.Draw("same:pe")
    h3.Draw("same:pe")
    dummy.Draw("AXISSAME")
    #x_l = 0.6
    x_l = 0.2
    x_dl = 0.25
    y_h = 0.85
    y_dh = 0.15
    if 'gr_Ebin_E_mean' in out_name:
        x_l = 0.2
        y_h = 0.85
    legend = rt.TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    legend.AddEntry(h1 ,'Full Sim','lep')
    legend.AddEntry(h2 ,"Calo only",'lep')
    legend.AddEntry(h3 ,"Full Sim no #gamma conversion",'lep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    #legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()

    label = rt.TLatex(0.5 , 0.9, plot_label)
    label.SetTextAlign(32)
    label.SetTextSize(0.035)
    label.SetNDC(rt.kTRUE)
    label.Draw() 
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()


def do_plot(h1, out_name, title, plot_label):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.12)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)

#    h1.Scale(1/h1.GetSumOfWeights())
#    h2.Scale(1/h2.GetSumOfWeights())
    x_min=h1.GetXaxis().GetXmin()
    x_max=h1.GetXaxis().GetXmax()
    y_min=0
    y_max=1.05 
    if 'eff_E' in out_name:
        y_min=0.9
    elif 'eff_phi' in out_name:
        y_min=0.9
    elif 'eff_theta' in out_name:
        y_min=0.9
    elif 'E_mean' in out_name:
        y_min=-0.15
        y_max=  0.
    elif 'E_std' in out_name:
        y_min= 0.
        y_max= 0.02
    elif 'Phi_mean' in out_name:
        y_min= -0.005
        y_max=  0.005
    elif 'Phi_std' in out_name:
        y_min= 0.
        y_max= 0.06
    elif 'Theta_mean' in out_name:
        y_min= -0.01
        y_max=  0.01
    elif 'Theta_std' in out_name:
        y_min= 0.
        y_max= 0.06
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
    h1 .SetLineColor(rt.kBlue)
    h1 .SetMarkerColor(rt.kBlue)
    h1 .SetMarkerStyle(20)
    h1.Draw("same:pe")
    dummy.Draw("AXISSAME")
    x_l = 0.5
    x_dl = 0.25
    y_h = 0.85
    y_dh = 0.15
    if 'gr_Ebin_E_mean' in out_name:
        x_l = 0.2
        y_h = 0.85
    legend = rt.TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    #legend.AddEntry(h1 ,'Full Sim','lep')
    legend.AddEntry(h1 ,'Matrix Ecal','ep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    #legend.Draw()

    label = rt.TLatex(0.26 , 0.9, plot_label)
    label.SetTextAlign(32)
    label.SetTextSize(0.035)
    label.SetNDC(rt.kTRUE)
    label.Draw() 
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()



if __name__ == '__main__':

    ############## for drawing reco efficiency ####################################
    plot_path = './plots_matrix/'
    f_in = rt.TFile('MatrixGamma_out.root','read')
    para_dict = {}
    eff_list = []
    for ih in range(0,f_in.GetListOfKeys().GetSize()):
        hname=f_in.GetListOfKeys()[ih].GetName()
        if 'Pan_reco_eff_E_' not in hname:continue
        hist = f_in.Get(hname)
        eff_list.append(hist)
    str_x = 'E_{mc}^{#gamma} (GeV)'
    str_label = ''
    do_plot_h_list(hlist=eff_list, out_name="reco_eff", title={'X':str_x, 'Y':'Efficiency'}, plot_label='')
    ############## for drawing res ####################################
    para_dict = {}
    for ih in range(0,f_in.GetListOfKeys().GetSize()):
        hname=f_in.GetListOfKeys()[ih].GetName()
        if 'gr_Ebin_' not in hname: continue
        hist = f_in.Get(hname)
        str_x = ''
        str_y = ''
        str_label = ''
        if 'E_mean' in hname:
            str_y = 'mean(#DeltaE/E)'
        elif 'E_std' in hname:
            str_y = '#sigma(#DeltaE/E)'
        if 'Theta_mean' in hname:
            str_y = 'mean(#Delta#theta)'
        elif 'Theta_std' in hname:
            str_y = '#sigma(#Delta#theta)'
        elif 'Phi_mean' in hname:
            str_y = 'mean(#Delta#phi)'
        elif 'Phi_std' in hname:
            str_y = '#sigma(#Delta#phi)'
        if 'Ebin' in hname:
            str_x = 'E_{pfo}^{#gamma} (GeV)'
        elif 'Phibin' in hname:
            str_x = '#phi_{pfo}^{#gamma} (degree)'
        elif 'Thetabin' in hname:
            str_x = '#theta_{pfo}^{#gamma} (degree)'
        do_plot(h1=hist, out_name=hname, title={'X':str_x, 'Y':str_y}, plot_label=str_label)
    f_in.Close()
    print('done')
