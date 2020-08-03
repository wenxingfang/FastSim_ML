import gc
import sys
import h5py
import math
import numpy as np
import ROOT as rt
rt.gROOT.SetBatch(rt.kTRUE)
rt.TH1.AddDirectory(rt.kFALSE)

def add_info(s_content):
    lowX=0.15
    lowY=0.7
    info  = rt.TPaveText(lowX, lowY+0.06, lowX+0.30, lowY+0.16, "NDC")
    info.SetBorderSize(   0 )
    info.SetFillStyle(    0 )
    info.SetTextAlign(   12 )
    info.SetTextColor(    1 )
    info.SetTextSize(0.025)
    info.SetTextFont (   42 )
    info.AddText("%s"%(str(s_content)))
    return info

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


#def do_plot_2gr_v1(gr1, gr2,  title, out_name, x_bins, y_bins, xy_title, M_size, output_path, fit_line):
#    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
#    canvas.SetGridy()
#    canvas.SetTopMargin(0.13)
#    canvas.SetBottomMargin(0.1)
#    canvas.SetLeftMargin(0.13)
#    canvas.SetRightMargin(0.15)
#    canvas.cd()
#    size = 0.312
#    pad1 = rt.TPad('pad1', '', 0.0, size, 1.0, 1.0, 0)
#    pad2 = rt.TPad('pad2', '', 0.0, 0.0, 1.0, size, 0)
#    pad2.Draw() 
#    pad1.Draw()
#    pad1.SetTickx(1)
#    pad1.SetTicky(1)
#    pad2.SetTickx(1)
#    pad2.SetTicky(1)
#    pad2.SetGridy()
#    pad1.SetBottomMargin(0.01)
#    pad1.SetRightMargin(0.05)
#    pad1.SetLeftMargin(0.13)
#    pad2.SetTopMargin(0.01)
#    pad2.SetBottomMargin(0.5)
#    pad2.SetRightMargin(0.05)
#    pad2.SetLeftMargin(0.13)
#    pad1.cd()
#
#    x_min=x_bins[0]
#    x_max=x_bins[1]
#    y_min=y_bins[0]
#    y_max=y_bins[1]
#    dummy = rt.TH2D("dummy","",1,x_min,x_max,1,y_min,y_max)
#    dummy.SetStats(rt.kFALSE)
#    dummy_X_title=xy_title[0]
#    dummy_Y_title=xy_title[1]
#    dummy.SetTitle(title)
#    dummy.GetYaxis().SetTitle(dummy_Y_title)
#    #dummy.GetXaxis().SetTitle(dummy_X_title)
#    dummy.GetXaxis().SetTitle('')
#    dummy.GetYaxis().SetTitleSize(0.04)
#    dummy.GetXaxis().SetTitleSize(0.04)
#    dummy.GetYaxis().SetLabelSize(0.04)
#    dummy.GetXaxis().SetLabelSize(0.0 )
#    dummy.GetYaxis().SetTitleOffset(1.4)
#    dummy.GetXaxis().SetTitleOffset(1.1)
#    dummy.GetXaxis().SetTitleFont(42)
#    dummy.GetXaxis().SetLabelFont(42)
#    dummy.GetYaxis().SetTitleFont(42)
#    dummy.GetYaxis().SetLabelFont(42)
#    dummy.Draw()
#    gr1.SetMarkerStyle(20)
#    gr1.SetMarkerSize(M_size)
#    gr2.SetMarkerStyle(24)
#    gr2.SetMarkerSize(M_size)
#    gr1.SetMarkerColor(1)
#    gr2.SetMarkerColor(2)
#    gr1.Draw("same:p")
#    gr2.Draw("same:p")
#    dummy.Draw("AXISSAME")
#    if fit_line:
#        f_x_min = 0
#        f_x_max = 5
#        f1 = rt.TF1("f1","[0] + [1]*x", f_x_min, f_x_max)
#        f1.SetParameters(0, 0.1) # you MUST set non-zero initial values for parameters
#        f1.SetParameters(1, 1.1) # you MUST set non-zero initial values for parameters
#        gr1.Fit("f1", "WR") #"R" = fit between "xmin" and "xmax" of the "f1"
#    legend = rt.TLegend(0.2,0.7,0.45,0.85)
#    legend.AddEntry(gr1,'full sim'    ,'p')
#    legend.AddEntry(gr2,"possion exp.",'p')
#    legend.SetBorderSize(0)
#    legend.SetTextSize(0.05)
#    legend.SetTextFont(42)
#    legend.Draw()
#
#    pad2.cd()
#    fontScale = pad1.GetHNDC()/pad2.GetHNDC()
#    ratio_y_min = 0.9
#    ratio_y_max = 1.09
#    dummy_ratio = rt.TH2D("dummy_ratio","",1,x_min,x_max,1,ratio_y_min,ratio_y_max)
#    dummy_ratio.SetStats(rt.kFALSE)
#    dummy_ratio.GetYaxis().SetTitle('full / possion')
#    dummy_ratio.GetXaxis().SetTitle(dummy_X_title)
#    dummy_ratio.GetXaxis().SetTitleSize(0.05*fontScale)
#    dummy_ratio.GetXaxis().SetLabelSize(0.05*fontScale)
#    dummy_ratio.GetYaxis().SetNdivisions(305)
#    dummy_ratio.GetYaxis().SetTitleSize(0.037*fontScale)
#    dummy_ratio.GetYaxis().SetLabelSize(0.05 * fontScale * 0.96)
#    dummy_ratio.GetYaxis().SetTitleOffset(0.7)
#    dummy_ratio.Draw()
#    g_ratio=rt.TGraph()
#    if gr1.GetN()==gr2.GetN():
#        g_ratio = get_graph_ratio(gr1, gr2)
#    g_ratio.Draw("same:p")
#    canvas.SaveAs("%s/%s.png"%(output_path, out_name))
#    del canvas
#    gc.collect()

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


def do_plot_v1(tag, h1,h2,out_name, xbin, out_path,leg_name):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    if False:
        h_real.Scale(1/h_real.GetSumOfWeights())
        h_fake.Scale(1/h_fake.GetSumOfWeights())
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
        dummy_Y_title = "Probability"
        #dummy_Y_title = "Events"
        dummy_X_title = "Hit Time (ns)"
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    dummy.GetXaxis().SetTitle(dummy_X_title)
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.03)
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

    legend = rt.TLegend(0.6,0.75,0.85,0.85)
    legend.AddEntry(h1,leg_name[0],'lep')
    legend.AddEntry(h2,leg_name[1],'lep')
    legend.SetBorderSize(0)
    legend.SetTextSize(0.035)
    legend.SetTextFont(42)
    legend.Draw()
    r_info = add_info('r=%.3f m'%(float(tag)/1000))
    r_info.Draw()
    canvas.SaveAs("%s/%s_%s.png"%(out_path, out_name, tag))
    del canvas
    gc.collect()

def do_plot(hist, title,out_name):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    x_min=0
    x_max=17.7
    y_min=0
    y_max=180
    z_min=0
    z_max=10
    dummy = rt.TH3D("dummy","",1,x_min,x_max,1,y_min,y_max, 1,z_min, z_max)
    #dummy = rt.TH2D("dummy","",hist.GetNbinsX(),x_min,x_max,hist.GetNbinsY(),y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_X_title="r"
    dummy_Y_title="#theta"
    dummy_Z_title="mean NPE"
    #dummy.SetTitle('E(npe)')
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
    #hist.Draw("same:colz")
    hist.Draw("same:LEGOZ")
    dummy.Draw("AXISSAME")
    canvas.SaveAs("%s.png"%(out_name))
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
    dummy_Z_title="mean NPE"
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
    canvas.SaveAs("%s.png"%(out_name))
    del canvas
    gc.collect()


def make_graph_r_theta_meanNPE(files, out_name):
    Dict = {}
    for ifile in files:
        hd = 0
        try:
            hd  = h5py.File(ifile, 'r')
        except:
            hd = 0
            print('bad file:',ifile)
        else:
            pass
        if hd==0:continue
        npe = hd['nPEByPMT'][:]
        for i in range(npe.shape[0]):
            r     = npe[i,0]
            #if r < 17.700:
            #    print('r=',r,',file=',ifile)
            #    sys.exit()
            theta = npe[i,1]
            if r not in Dict:
                Dict[r] = {}
            if theta not in Dict[r]:
                Dict[r][theta] = [np.sum(npe[i,2:npe.shape[1]]), npe.shape[1]-2 ]
            else :
                Dict[r][theta] = [np.sum(npe[i,2:npe.shape[1]]) + Dict[r][theta][0], npe.shape[1]-2 + Dict[r][theta][1] ]
    
        hd.close()
    f = open(out_name,'w')
    for ir in Dict:
        for itheta in Dict[ir]:
            f.write('%f %f %f\n'%(ir, itheta, float(Dict[ir][itheta][0])/Dict[ir][itheta][1]))
    f.close()

def check_r_totalNPE(files, bins):
    Dict = {}
    Dict1 = {}
    for ifile in files:
        str1 = ifile.split('/')[-1]
        str_r = str1.split('_')[3]
        if str_r in Dict: continue 
        if float(str_r)%1000 !=0: continue #only check int meter part
        hd  = h5py.File(ifile, 'r')
        npe = hd['nPEByPMT'][:]
        npe0 = npe[:,2:npe.shape[1]]
        Dict[str_r]  = [0, 0, rt.TH1F('full_%s'%str_r,'',bins[0],bins[1],bins[2])]
        Dict1[str_r] = [0, 0, rt.TH1F('fast_%s'%str_r,'',bins[0],bins[1],bins[2])]
        total_pe = np.sum(npe0, axis=0)
        Dict[str_r][0] = np.mean(total_pe)
        Dict[str_r][1] = np.std (total_pe)
        for i in range(npe0.shape[1]):
            tmp_tot_pe = np.sum(npe0[:,i])
            Dict[str_r][2].Fill(tmp_tot_pe)
        fast = np.full((npe.shape[0], npe.shape[1]), 0, dtype=np.float32)
        for i in range(npe.shape[0]):
            mean = np.mean(npe[i,2:npe.shape[1]])
            s = np.random.poisson(mean, npe.shape[1]-2)
            fast[i,0:2] = npe[i,0:2]
            fast[i,2:fast.shape[1]] = s
        fast0 = fast[:,2:fast.shape[1]]
        total_pe = np.sum(fast0, axis=0)
        Dict1[str_r][0] = np.mean(total_pe)
        Dict1[str_r][1] = np.std (total_pe)
        for i in range(fast0.shape[1]):
            tmp_tot_pe = np.sum(fast0[:,i])
            Dict1[str_r][2].Fill(tmp_tot_pe)
        hd.close()
    return (Dict, Dict1)

def check_r_totalNPE_v2(files, in_f_name):
    Dict = {}
    Dict1 = {}
    
    f_in = rt.TFile(in_f_name,'read')
    f_in.cd()
    hist2d = f_in.Get('r_theta_meanNpe')
    f_in.Close()
    print('hist2d=',hist2d.GetName())
    x_axis = hist2d.GetXaxis()
    y_axis = hist2d.GetYaxis()
    for ifile in files:
        str1 = ifile.split('/')[-1]
        str_r = str1.split('_')[3]
        if str_r in Dict: continue 
        if float(str_r)%1000 !=0: continue #only check int meter part
        hd  = h5py.File(ifile, 'r')
        npe = hd['nPEByPMT'][:]
        npe0 = npe[:,2:npe.shape[1]]
        Dict[str_r]  = [0, 0, rt.TH1F('full_%s'%str_r,'',200,0,2000)]
        Dict1[str_r] = [0, 0, rt.TH1F('fast_%s'%str_r,'',200,0,2000)]
        #Dict[str_r]  = [0, 0, rt.TH1F('full_%s'%str_r,'',300,3000,6000)]
        #Dict1[str_r] = [0, 0, rt.TH1F('fast_%s'%str_r,'',300,3000,6000)]
        total_pe = np.sum(npe0, axis=0)
        Dict[str_r][0] = np.mean(total_pe)
        Dict[str_r][1] = np.std (total_pe)
        for i in range(npe0.shape[1]):
            tmp_tot_pe = np.sum(npe0[:,i])
            Dict[str_r][2].Fill(tmp_tot_pe)
        fast = np.full((npe.shape[0], npe.shape[1]), 0, dtype=np.float32)
        for i in range(npe.shape[0]):
            #mean = np.mean(npe[i,2:npe.shape[1]])
            mean = hist2d.GetBinContent(x_axis.FindBin(npe[i,0]),y_axis.FindBin(npe[i,1]))
            s = np.random.poisson(mean, npe.shape[1]-2)
            fast[i,0:2] = npe[i,0:2]
            fast[i,2:fast.shape[1]] = s
        fast0 = fast[:,2:fast.shape[1]]
        total_pe = np.sum(fast0, axis=0)
        Dict1[str_r][0] = np.mean(total_pe)
        Dict1[str_r][1] = np.std (total_pe)
        for i in range(fast0.shape[1]):
            tmp_tot_pe = np.sum(fast0[:,i])
            Dict1[str_r][2].Fill(tmp_tot_pe)
        hd.close()
    return (Dict, Dict1)



def check_r_totalNPE_v1(r_files): ## not finish yet
    Dict = {}
    Dict1 = {}
    for r in r_files:
        files = r_files
        npe = 0
        for ifile in files:
            hd  = h5py.File(ifile, 'r')
            tmp_npe = hd['nPEByPMT'][:]
            str_r = r
            hd  = h5py.File(ifile, 'r')
            npe = hd['nPEByPMT'][:]
            npe0 = npe[:,2:npe.shape[1]]
            Dict[str_r]  = [0, 0, rt.TH1F('full_%s'%str_r,'',200,0,2000)]
            Dict1[str_r] = [0, 0, rt.TH1F('fast_%s'%str_r,'',200,0,2000)]
            total_pe = np.sum(npe0, axis=0)
            Dict[str_r][0] = np.mean(total_pe)
            Dict[str_r][1] = np.std (total_pe)
            for i in range(npe0.shape[1]):
                tmp_tot_pe = np.sum(npe0[:,i])
                Dict[str_r][2].Fill(tmp_tot_pe)
            fast = np.full((npe.shape[0], npe.shape[1]), 0, dtype=np.float32)
            for i in range(npe.shape[0]):
                mean = np.mean(npe[i,2:npe.shape[1]])
                s = np.random.poisson(mean, npe.shape[1]-2)
                fast[i,0:2] = npe[i,0:2]
                fast[i,2:fast.shape[1]] = s
            fast0 = fast[:,2:fast.shape[1]]
            total_pe = np.sum(fast0, axis=0)
            Dict1[str_r][0] = np.mean(total_pe)
            Dict1[str_r][1] = np.std (total_pe)
            for i in range(fast0.shape[1]):
                tmp_tot_pe = np.sum(fast0[:,i])
                Dict1[str_r][2].Fill(tmp_tot_pe)
            hd.close()
        return (Dict, Dict1)


def check_graph_Interpolate(datafile, points):
    g = rt.TGraph2D(datafile)
    for p in points:
        print(p,',z=', g.Interpolate(p[0],p[1]))
    return g


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


in_files = []

#h5_txt = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/make_TH2_map/npe_h5data.txt'
#h5_txt = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/make_TH2_map/npe_h5data_noDE.txt'
h5_txt = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/make_TH2_map/phtotn_z_axis.txt'
#h5_txt = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/make_TH2_map/phtotn_z_axis_noDE.txt'

with open(h5_txt,'r') as f:
    for line in f:
        if "#" in line:continue
        line = line.replace('\n','')
        in_files.append(line)

######### make dat file ###########
if False:
    make_graph_r_theta_meanNPE(in_files, '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/make_TH2_map/r_theta_noDE.dat')

###### check total npe for photon samples ################
if True:
#    in_files = []
#    in_files.append('/scratchfs/juno/wxfang/r_grid_1MeV_h5_data/user_9495_10129_15000.000000_-7961.454427_-10207.160430_7578.200273_skimed_batch0_N1000.h5')

    #(full_sim, fast_sim ) = check_r_totalNPE(in_files, [200,0,2000])
    #(full_sim, fast_sim ) = check_r_totalNPE(in_files, [300,3000,6000])
    (full_sim, fast_sim ) = check_r_totalNPE_v2(in_files, '../r_theta_TH2.root')
    #(full_sim, fast_sim ) = check_r_totalNPE_v2(in_files, 'r_theta_TH2_noDE.root')
    
    out_path = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/make_TH2_map/full_vs_fast_possion_plots/'
    gr_full_mean = rt.TGraph()
    gr_fast_mean = rt.TGraph()
    gr_full_std = rt.TGraph()
    gr_fast_std = rt.TGraph()
    index = 0
    xbin = [1010,1500]
    #xbin = [3000,6000]
    for ir in full_sim:
        if ir not in fast_sim: continue
        gr_full_mean.SetPoint(index , float(ir)/1000, full_sim[ir][0])
        gr_fast_mean.SetPoint(index , float(ir)/1000, fast_sim[ir][0])
        gr_full_std .SetPoint(index , float(ir)/1000, full_sim[ir][1])
        gr_fast_std .SetPoint(index , float(ir)/1000, fast_sim[ir][1])
        index += 1
        #print('full_sim mean=',full_sim[ir][0])
        #print('fast_sim mean=',fast_sim[ir][0])
        #print('full_sim[ir][2],',type(full_sim[ir][2]))
        do_plot_v1(str(ir),full_sim[ir][2], fast_sim[ir][2], 'tot_pe', xbin, out_path,['full sim', 'possion exp'])
    #do_plot_2gr_v1(gr_full_mean,gr_fast_mean,'','meanNpe_Vs_R', [0,18], xbin, ['z (m)', '#mu (total PE)'], 1 , out_path, False)
    do_plot_2gr_v1(gr_full_mean,gr_fast_mean,'','meanNpe_Vs_R', [0,18], [1010,1500] , ['z (m)', 'total PE'], 1 , out_path, False, ['G4', 'Pois.'], ['G4', 'Pois.'])
    #do_plot_2gr_v1(gr_full_mean,gr_fast_mean,'','meanNpe_Vs_R', [0,18], [3000,6000], ['z (m)', '#mu (total PE)'], 1 , out_path, False)
    #do_plot_2gr_v1(gr_full_std ,gr_fast_std ,'','stdNpe_Vs_R' , [0,18], [0.1  ,50] , ['z (m)', '#sigma (total PE)'], 1 , out_path, False)
    do_plot_2gr_v1(gr_full_std ,gr_fast_std ,'','stdNpe_Vs_R', [0, 18], [0.1 , 50] , ['z (m)', '#sigma (total PE)'], 1 , out_path, False, ['G4', 'Pois.'], ['G4', 'Pois.'])

if False:
    out_path = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/make_TH2_map/r_theta_vs_NPE_plots/'
    dict_r_gr,dict_theta_gr = read_graph_r_theta_vs_meanNPE(in_files)
    #for r in dict_r_gr: 
    #    #do_plot_gr_1D(dict_r_gr[r], '','r=%s_thetaVsNpe'%str(r), [0,180], [0,0.5], ['#theta', 'mean NPE'], 0.5 ,out_path, False)
    #    do_plot_gr_1D(dict_r_gr[r], '','r=%s_thetaVsNpe'%str(r), [45,55], [0,0.5], ['#theta', 'mean NPE'], 0.5 , out_path, False)
    for theta in dict_theta_gr: 
        if float(theta)%1 !=0 : continue
        do_plot_gr_1D(dict_theta_gr[theta], '','theta=%s_rVsNpe'%str(theta), [0,20], [0,5], ['r', 'mean NPE'], 0.5 , out_path, False)


######### save  TGraph2D #####################    
if False:
    a_test   = [[15.510000, 28.700001],[15.510000, 30.900000], [15.510000, 29.900000], [1.1,90.1], [1.2,90.1], [1.15, 90.1] ]
    gr = check_graph_Interpolate('r_theta_expNpe_final.dat', a_test)
    print('x min=',gr.GetXaxis().GetXmin(), ',x max=',gr.GetXaxis().GetXmax(), ',y min=',gr.GetYaxis().GetXmin(), ',y max=',gr.GetYaxis().GetXmax() )
    f_out = rt.TFile('r_theta_graph.root','recreate')
    f_out.cd()
    gr.Write('r_theta_meanNpe')
    f_out.Close()

    f_out = rt.TFile('r_theta_TH2.root','recreate')
    f_out.cd()
    h2 = gr.GetHistogram()
    print('x min=',h2.GetXaxis().GetXmin(), ',x max=',h2.GetXaxis().GetXmax(), ',y min=',h2.GetYaxis().GetXmin(), ',y max=',h2.GetYaxis().GetXmax() )
    for i in a_test:
        nbin = h2.FindFixBin(i[0],i[1])
        print(i,',=',h2.GetBinContent(nbin))
    h2.Write('r_theta_meanNpe')
    f_out.Close()

######### save final TH2D #####################    
if False:
    #data_name = 'r_theta_expNpe_final_add.dat'
    data_name = 'r_theta_noDE.dat'
    a_test   = [[15.510000, 28.700001 ],[15.500000, 0.200000], [15.510000, 29.900000], [1.1,90.1], [1.2,90.1], [1.15, 90.1] ]
    f_in = open(data_name,'r')
    r_list     = []
    theta_list = []
    for line in f_in:
        r     = float(line.split(' ')[0])
        theta = float(line.split(' ')[1])
        r     = round(r,3)
        theta = round(theta,1)
        r_list    .append(r)
        theta_list.append(theta)
    f_in.close()
    r_list     = list(set(r_list))
    theta_list = list(set(theta_list))
    r_list     . sort()
    theta_list . sort()
    ################# refine the bining ########### 
    r_list_new = []
    for i in range(len(r_list)):
        if i == 0:
            r_list_new.append(r_list[i]/2)
        elif i == (len(r_list)-1):
            r_list_new.append((r_list[i-1]+r_list[i])/2)
            r_list_new.append(r_list[i]*(1.01))
        else: r_list_new.append((r_list[i-1]+r_list[i])/2)

    theta_list_new = []
    for i in range(len(theta_list)):
        if i == 0:
            theta_list_new.append(theta_list[i]/2)
        elif i == (len(theta_list)-1):
            theta_list_new.append((theta_list[i-1]+theta_list[i])/2)
            theta_list_new.append(theta_list[i]*(1.01))
        else: theta_list_new.append((theta_list[i-1]+theta_list[i])/2)
    #####################################################
    #h2 = rt.TH2D('r_theta_meanNpe','',len(r_list)-1,np.array(r_list), len(theta_list)-1,np.array(theta_list))
    h2 = rt.TH2D('r_theta_meanNpe','',len(r_list_new)-1,np.array(r_list_new), len(theta_list_new)-1,np.array(theta_list_new))
    f_in = open(data_name,'r')
    for line in f_in:
        r = float(line.split(' ')[0])
        theta = float(line.split(' ')[1])
        r     = round(r,3)
        theta = round(theta,1)
        nbin = h2.FindFixBin(r,theta)
        mean = float(line.split(' ')[2])
        #print('nbin=',nbin,',mean=',mean)
        h2.SetBinContent(nbin,mean)
    ######### if some bin is empty, fill it with neighbour value#####################
    n_empty_0 = 0
    n_empty = 0
    for ix in range(1, h2.GetNbinsX()+1):
        for iy in range(1, h2.GetNbinsY()+1):
            if h2.GetBinContent(ix, iy) != 0: continue
            n_empty_0 += 1 
            for z in range(1,40):
                if   (ix+z) <= h2.GetNbinsX()  and (h2.GetBinContent(ix+z, iy) != 0) :  
                    h2.SetBinContent(ix,iy, h2.GetBinContent(ix+z, iy))
                    break
                elif (ix-z)>=1                 and (h2.GetBinContent(ix-z, iy) != 0) :  
                    h2.SetBinContent(ix,iy, h2.GetBinContent(ix-z, iy))
                    break
                elif (iy+z)  <= h2.GetNbinsY() and (h2.GetBinContent(ix, iy+z) != 0) :  
                    h2.SetBinContent(ix,iy, h2.GetBinContent(ix, iy+z))
                    break
                elif (iy-z)>=1                 and (h2.GetBinContent(ix, iy-z) != 0) :  
                    h2.SetBinContent(ix,iy, h2.GetBinContent(ix, iy-z))
                    break
            
            if h2.GetBinContent(ix, iy) == 0: n_empty += 1
    print('empty bin num, orig=',n_empty_0,',final=',n_empty)
    f_in.close()
    f_out = rt.TFile('r_theta_TH2_noDE.root','recreate')
    f_out.cd()
    h2.Write('r_theta_meanNpe')
    f_out.Close()
    print('x min=',h2.GetXaxis().GetXmin(), ',x max=',h2.GetXaxis().GetXmax(), ',y min=',h2.GetYaxis().GetXmin(), ',y max=',h2.GetYaxis().GetXmax() )
    for i in a_test:
        nbin = h2.FindFixBin(i[0],i[1])
        print(i,',=',h2.GetBinContent(nbin))
    do_plot(h2, '','r_theta_hist')
    


if False:
    f_in = rt.TFile('r_theta_graph.root','read')
    gr = f_in.Get('r_theta_meanNpe')
    for p in [[15.510000, 28.700001],[15.510000, 30.900000], [15.510000, 29.900000], [1.1,90.], [1.2,90.], [1.3, 90.] ]:
        print(p,',z=', gr.Interpolate(p[0],p[1]))


if False:
    do_plot_gr(gr, '', 'gr_final')
print('done!')

if False:
    a = rt.TVector3()
    a.SetXYZ(1,1,1)
    b = rt.TVector3()
    b.SetXYZ(0,0,-1)
    print(a.Angle(b))
    print(math.cos(a.Angle(b)))
    print( (a.X()*b.X()+ a.Y()*b.Y() + a.Z()*b.Z())/(math.sqrt(3)*1))
