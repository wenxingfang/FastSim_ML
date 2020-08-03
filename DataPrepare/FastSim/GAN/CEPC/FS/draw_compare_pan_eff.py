import ROOT as rt
import numpy as np
import h5py 
import sys 
import os 
import gc
import math
import json
from array import array
rt.gROOT.SetBatch(rt.kTRUE)
#rt.TGaxis.SetMaxDigits(3);

def addOne(gr):
    gr_new = rt.TGraphErrors()
    Npoint = 0
    for i in range(gr.GetN()):
        x_point = gr.GetX()[i]
        low  = gr.GetEXlow ()[i]
        high = gr.GetEXhigh()[i] 
        mean = gr.GetY()[i] + 1
        err  = gr.GetEYhigh()[i]
        gr_new.SetPoint  (Npoint, x_point, mean)
        gr_new.SetPointError(Npoint, low , err)
        Npoint += 1
    return gr_new

def do_plot_v2(g1, g2, Xrange,  title, out_name):

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

    xaxis =g1.GetXaxis()
    x_min=Xrange[0]
    x_max=Xrange[1]
    y_min=-0.04
    y_max= 0.05
    if "logx" in out_name:
        pad1.SetLogx()
        pad2.SetLogx()
    if "logy" in out_name:
        pad1.SetLogy()
        y_min = 1e-1
    dummy = rt.TH2D("dummy","",1,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy.GetYaxis().SetTitle(title['Y'])
    dummy.GetXaxis().SetTitle(title['X'])
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.)
    #dummy.GetYaxis().SetLabelSize()
    dummy.GetXaxis().SetLabelSize(0.)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42  )
    dummy.GetYaxis().SetNdivisions(305)
    dummy.Draw('AXIS')
    g1.SetLineWidth(2)
    g1.SetLineColor(rt.kRed)
    g1.SetMarkerColor(rt.kRed)
    g1.SetMarkerStyle(20)
    g1.Draw("pe")
    g2.SetLineWidth(2)
    g2.SetLineColor(rt.kBlue)
    g2.SetMarkerColor(rt.kBlue)
    g2.SetMarkerStyle(24)
    g2.Draw("pe")

    legend = rt.TLegend(0.6,0.7,0.85,0.85)
    legend.AddEntry(g1,'G4','lep')
    legend.AddEntry(g2,'FS','lep')
    legend.SetBorderSize(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.SetFillStyle(0)
    legend.Draw()

    pad2.cd()
    fontScale = pad1.GetHNDC()/pad2.GetHNDC()
    ratio_y_min = -0.05
    ratio_y_max = 0.05
    dummy_ratio = rt.TH2D("dummy_ratio","",1,x_min,x_max,1,ratio_y_min,ratio_y_max)
    dummy_ratio.SetStats(rt.kFALSE)
    dummy_ratio.GetYaxis().SetTitle('G4 - FS')
    dummy_ratio.GetYaxis().CenterTitle()
    dummy_ratio.GetXaxis().SetTitle(title['X'])
    dummy_ratio.GetXaxis().SetTitleSize(0.05*fontScale)
    dummy_ratio.GetXaxis().SetLabelSize(0.05*fontScale)
    dummy_ratio.GetYaxis().SetNdivisions(305)
    dummy_ratio.GetYaxis().SetTitleSize(0.04*fontScale)
    dummy_ratio.GetYaxis().SetLabelSize(0.04*fontScale)
    dummy_ratio.GetYaxis().SetTitleOffset(0.7)
    dummy_ratio.Draw("AXIS")
    #h_ratio=h1.Clone('ratio_%s'%h1.GetName())
    #h_ratio.Divide(h2)

    gr_diff = rt.TGraphErrors()
    Npoint = 0
    for i in range(g1.GetN()):
        x_point = g1.GetX()[i]
        x_point0 = g2.GetX()[i]
        low  = g1.GetEXlow ()[i]
        high = g1.GetEXhigh()[i] 
        if x_point != x_point0 :
            print('Error in making ratio graph')
            os.exit()
        g4_mean = g1.GetY()[i]
        g4_err  = g1.GetEYhigh()[i]
        fs_mean = g2.GetY()[i] 
        fs_err  = g2.GetEYhigh()[i]
        print('N=',Npoint,',mean=',x_point,',diff=',g4_mean-fs_mean)
        gr_diff.SetPoint  (Npoint, x_point         , g4_mean-fs_mean)
        ratio_err = math.sqrt(fs_err*fs_err+g4_err*g4_err)
        gr_diff.SetPointError(Npoint, low , ratio_err)
        Npoint += 1
    gr_diff.SetLineWidth(2)
    gr_diff.SetLineColor(rt.kBlack)
    gr_diff.SetMarkerColor(rt.kBlack)
    gr_diff.SetMarkerStyle(20)
    gr_diff.Draw("pe")
    fout = rt.TFile('tunning.root','update')
    fout.cd()
    gr_diff.Write(out_name)
    fout.Close()
    canvas.SaveAs("%s/%s.png"%(plot_path, out_name))
    del canvas
    gc.collect()

def do_plot_v3(g1, g2, Xrange,  title, out_name):

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

    xaxis =g1.GetXaxis()
    x_min=Xrange[0]
    x_max=Xrange[1]
    y_min=-0.04+1
    y_max= 0.05+1
    if "logx" in out_name:
        pad1.SetLogx()
        pad2.SetLogx()
    if "logy" in out_name:
        pad1.SetLogy()
        y_min = 1e-1
    dummy = rt.TH2D("dummy","",1,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy.GetYaxis().SetTitle(title['Y'])
    dummy.GetXaxis().SetTitle(title['X'])
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.)
    #dummy.GetYaxis().SetLabelSize()
    dummy.GetXaxis().SetLabelSize(0.)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42  )
    dummy.GetYaxis().SetNdivisions(305)
    dummy.Draw('AXIS')

    g11 = addOne(g1)
    g22 = addOne(g2)


    g11.SetLineWidth(2)
    g11.SetLineColor(rt.kRed)
    g11.SetMarkerColor(rt.kRed)
    g11.SetMarkerStyle(20)
    g11.Draw("pe")
    g22.SetLineWidth(2)
    g22.SetLineColor(rt.kBlue)
    g22.SetMarkerColor(rt.kBlue)
    g22.SetMarkerStyle(24)
    g22.Draw("pe")

    legend = rt.TLegend(0.6,0.7,0.85,0.85)
    legend.AddEntry(g11,'G4','lep')
    legend.AddEntry(g22,'FS','lep')
    legend.SetBorderSize(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.SetFillStyle(0)
    legend.Draw()

    pad2.cd()
    fontScale = pad1.GetHNDC()/pad2.GetHNDC()
    ratio_y_min = -0.05+1
    ratio_y_max = 0.05+1
    dummy_ratio = rt.TH2D("dummy_ratio","",1,x_min,x_max,1,ratio_y_min,ratio_y_max)
    dummy_ratio.SetStats(rt.kFALSE)
    dummy_ratio.GetYaxis().SetTitle('G4 / FS')
    dummy_ratio.GetYaxis().CenterTitle()
    dummy_ratio.GetXaxis().SetTitle(title['X'])
    dummy_ratio.GetXaxis().SetTitleSize(0.05*fontScale)
    dummy_ratio.GetXaxis().SetLabelSize(0.05*fontScale)
    dummy_ratio.GetYaxis().SetNdivisions(305)
    dummy_ratio.GetYaxis().SetTitleSize(0.04*fontScale)
    dummy_ratio.GetYaxis().SetLabelSize(0.04*fontScale)
    dummy_ratio.GetYaxis().SetTitleOffset(0.7)
    dummy_ratio.Draw("AXIS")
    #h_ratio=h1.Clone('ratio_%s'%h1.GetName())
    #h_ratio.Divide(h2)

    gr_diff = rt.TGraphErrors()
    Npoint = 0
    for i in range(g11.GetN()):
        x_point  = g11.GetX()[i]
        x_point0 = g22.GetX()[i]
        low      = g11.GetEX()[i]
        high     = g11.GetEX()[i] 
        if x_point != x_point0 :
            print('Error in making ratio graph')
            os.exit()
        g4_mean = g11.GetY()[i]
        g4_err  = g11.GetEY()[i]
        fs_mean = g22.GetY()[i]
        fs_err  = g22.GetEY()[i]
        print('N=',Npoint,',mean=',x_point,',ratio=',g4_mean/fs_mean)
        gr_diff.SetPoint  (Npoint, x_point         , g4_mean/fs_mean)
        ratio_err = math.sqrt(g4_mean*g4_mean*fs_err*fs_err+fs_mean*fs_mean*g4_err*g4_err)/(fs_mean*fs_mean)
        gr_diff.SetPointError(Npoint, low , ratio_err)
        Npoint += 1
    gr_diff.SetLineWidth(2)
    gr_diff.SetLineColor(rt.kBlack)
    gr_diff.SetMarkerColor(rt.kBlack)
    gr_diff.SetMarkerStyle(20)
    gr_diff.Draw("pe")
    fout = rt.TFile('tunning.root','update')
    fout.cd()
    gr_diff.Write(str("%s_%s"%('ratio',out_name)))
    fout.Close()
    canvas.SaveAs("%s/%s.png"%(plot_path, out_name))
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
        #y_min=0.9
        y_min=0.96
    elif 'eff_theta' in out_name:
        y_min=0.9
    elif 'E_mean' in out_name:
        y_min=-0.04
        y_max= 0.04
    elif 'E_std' in out_name:
        y_min= 0.01
        y_max= 0.1
    elif 'Phi_mean' in out_name:
        y_min= 0.0
        y_max= 0.04
    elif 'Phi_std' in out_name:
        y_min= 0.01
        y_max= 0.04
    elif 'Theta_mean' in out_name:
        y_min= -0.015
        y_max= 0.02
    elif 'Theta_std' in out_name:
        y_min= 0.006
        y_max= 0.032
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

    label = rt.TLatex(0.46 , 0.9, plot_label)
    label.SetTextAlign(32)
    label.SetTextSize(0.035)
    label.SetNDC(rt.kTRUE)
    #label.Draw() 
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


if __name__ == '__main__':
    isBarrel = True
    ############## for drawing reco efficiency ####################################
    plot_path = './pandora_eff/'
    f_in_1 = rt.TFile('G4_reco_eff.root','read')
    f_in_2 = rt.TFile('FS_reco_eff.root','read')
    if isBarrel == False:
        f_in_1 = rt.TFile('G4_Endcap_reco_eff.root','read')
        f_in_2 = rt.TFile('FS_Endcap_reco_eff.root','read')
    para_dict = {}
    for ih in range(0,f_in_1.GetListOfKeys().GetSize()):
        hname=f_in_1.GetListOfKeys()[ih].GetName()
        hist_1 = f_in_1.Get(hname)
        hist_2 = f_in_2.Get(hname)
        str_x = ''
        str_label = ''
        if 'eff_E' in hname:
            str_x = 'E_{mc}^{#gamma} (GeV)'
            str_label = '0<#theta_{mc}<170, E_{mc}>0.1 GeV'
        elif 'eff_phi' in hname:
            str_x = '#phi_{mc}^{#gamma} (degree)'
            str_label = '0<#theta_{mc}<170, E_{mc}>0.1 GeV'
        elif 'eff_theta' in hname:
            str_x = '#theta_{mc}^{#gamma} (degree)'
            str_label = '0<#theta_{mc}<170, E_{mc}>0.1 GeV'
        do_plot_g2(h1=hist_1, h2=hist_2, out_name=hname, title={'X':str_x, 'Y':'Efficiency'}, plot_label=str_label)
        #do_plot_h3(h1=hist_1, h2=hist_2, h3=hist_3, out_name=hname, title={'X':str_x, 'Y':'Efficiency'}, plot_label=str_label)
    f_in_1.Close()
    f_in_2.Close()
    ############## for drawing res ####################################
    if True:
        f_in_1 = rt.TFile('G4_res.root','read')
        f_in_2 = rt.TFile('FS_res.root','read')
        if isBarrel == False:
            f_in_1 = rt.TFile('G4_Endcap_res.root','read')
            f_in_2 = rt.TFile('FS_Endcap_res.root','read')
        para_dict = {}
        for ih in range(0,f_in_1.GetListOfKeys().GetSize()):
            hname=f_in_1.GetListOfKeys()[ih].GetName()
            hist_1 = f_in_1.Get(hname)
            hist_2 = f_in_2.Get(hname)
            str_x = ''
            str_y = ''
            str_y1 = ''
            str_label = ''
            Xrange=[-90, 90]
            if 'E_mean' in hname:
                str_y = 'mean(#DeltaE/E)'
                str_y1 = 'mean(E_{rec}/E_{mc})'
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
                Xrange=[0, 100]
            elif 'Phibin' in hname:
                str_x = '#phi_{pfo}^{#gamma} (degree)'
                Xrange=[-100, 100]
            elif 'Thetabin' in hname:
                str_x = '#theta_{pfo}^{#gamma} (degree)'
                Xrange=[0, 180]
            do_plot_g2(h1=hist_1, h2=hist_2, out_name=hname, title={'X':str_x, 'Y':str_y}, plot_label=str_label)
            if ('E_mean' in hname and 'Ebin' in hname) or ('E_mean' in hname and 'Phibin' in hname) or ('E_mean' in hname and 'Thetabin' in hname):
                #do_plot_v2(g1=hist_1, g2=hist_2, Xrange=Xrange,title={'X':str_x, 'Y':str_y}, out_name='%s_v1'%hname)
                do_plot_v3(g1=hist_1, g2=hist_2, Xrange=Xrange,title={'X':str_x, 'Y':str_y1}, out_name='%s'%hname)
    f_in_1.Close()
    f_in_2.Close()
    print('done')
