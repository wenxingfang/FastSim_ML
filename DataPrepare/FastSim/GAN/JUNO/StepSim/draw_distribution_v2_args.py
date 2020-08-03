import argparse
import ast
import h5py
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

def get_parser():
    parser = argparse.ArgumentParser(
        description='Run CalGAN training. '
        'Sensible defaults come from [arXiv/1511.06434]',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--real_input', action='store', type=str, default='',
                        help='real input.')
    parser.add_argument('--fake_input', action='store', type=str, default='',
                        help='fake input.')
    parser.add_argument('--for_npe', action='store', type=ast.literal_eval, default=False,
                        help='for_npe.')
    parser.add_argument('--for_time', action='store', type=ast.literal_eval, default=False,
                        help='for_time.')
    parser.add_argument('--for_pmt_theta', action='store', type=ast.literal_eval, default=False,
                        help='for_pmt_theta.')
    parser.add_argument('--for_r_theta_vs_npeMean', action='store', type=ast.literal_eval, default=False,
                        help='for_r_theta_vs_npeMean.')
    parser.add_argument('--GlobalTag', action='store', type=str, default='',
                        help='GlobalTag')
    return parser

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
    lowX=0.13
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


def do_plot(tag, h_real,h_fake,out_name, do_fit, c):
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
    x_min=h_real.GetBinLowEdge(1)
    x_max=h_real.GetBinLowEdge(nbin)+h_real.GetBinWidth(nbin)
    if "hit_time" in out_name:
        x_min=0
        x_max=1000
    elif "n_pe" in out_name:
        x_min=0
        x_max=10
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
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    dummy.GetXaxis().SetTitle(dummy_X_title)
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    if "tot_pe" in out_name:
        dummy.GetXaxis().SetLabelSize(0.03)
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
        f1 = rt.TF1("f1","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma((x)+1.)", x_min, x_max)
        f1.SetParameters(1, 100.0) # you MUST set non-zero initial values for parameters
        h_real.Fit("f1", "R") #"R" = fit between "xmin" and "xmax" of the "f1"

    legend = rt.TLegend(0.65,0.7,0.85,0.85)
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
    canvas.SaveAs("%s/%s_%s_%s.png"%(plot_path, out_name, tag, GlobalTag))
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

def do_plot_gr_1D(gr, title,out_name, x_bins, y_bins, xy_title, M_size):
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
    canvas.SaveAs("%s/%s_%s.png"%(plot_path, out_name,GlobalTag))
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

def read_hist(file_name, cat, bins, tag, doRound, scale):
    dict_hist = {}
    for ifile in file_name:
        hd = h5py.File(ifile, 'r')
        for i in hd.keys():
            if cat not in i: continue
            h = hd[i][:]/scale
            r_theta = i.split(cat)[-1]
            if r_theta not in dict_hist:
                dict_hist[r_theta] = rt.TH1F('%s_%s'%(str(tag),str(i)),'',bins[0],bins[1],bins[2])
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
            #print('j=',j,',npe=',np.sum(npe[:,j]))
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


#def read_graph_r_vs_meanNPE(file_name):
#    dict_theta_graph = {}
#    for ifile in file_name:
#        hd  = h5py.File(ifile, 'r')
#        npe = hd['nPEByPMT'][:]
#        for i in range(npe.shape[0]):
#            r     = npe[i,0]
#            theta = npe[i,1]
#            if str(theta) not in dict_theta_graph:
#                dict_theta_graph[str(theta)] = rt.TGraph()
#            potins = dict_theta_graph[str(theta)].GetN()
#            dict_theta_graph[str(theta)].Set(potins,r,np.mean(npe[i,2:npe.shape[1]]))
#        hd.close()
#    return hist

if __name__ == '__main__':

    parser = get_parser()
    parse_args = parser.parse_args()
    real_input = parse_args.real_input
    fake_input = parse_args.fake_input
    For_NPE    = parse_args.for_npe
    For_Time   = parse_args.for_time
    For_PMT_Theta   = parse_args.for_pmt_theta
    For_r_theta_vs_npeMean   = parse_args.for_r_theta_vs_npeMean
    GlobalTag    = parse_args.GlobalTag

    plot_path = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/r_theta_comp_plots/'
    
    
    real_file = []
    real_file.append(real_input)
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1001_10000.000000_batch0_N1000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1002_10000.000000_batch0_N1000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1003_10000.000000_batch0_N1000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1004_10000.000000_batch0_N1000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1005_10000.000000_batch0_N1000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1006_10000.000000_batch0_N1000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1007_10000.000000_batch0_N1000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1008_10000.000000_batch0_N1000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1009_10000.000000_batch0_N1000.h5')
    
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_78_761.054753_256.432825_-131.772148_-704.331345_batch0_N5000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_109_1972.354650_-170.609789_-1324.798385_1451.201022_batch0_N5000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_108_4143.010671_-27.302429_-27.039244_4142.832470_batch0_N5000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_114_5720.626101_2069.080878_-5293.362866_651.749089_batch0_N5000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_371_6006.914466_-332.456396_47.417349_5997.519966_batch0_N5000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_100_7265.516575_-673.529875_-1760.499121_-7016.746501_batch0_N5000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_104_10193.326270_3103.778049_-253.003481_9706.000799_batch0_N5000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_1_11042.389770_4397.912803_-8872.284301_-4886.236393_batch0_N5000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_102_14525.704016_10407.992577_-9981.839210_-1741.451597_batch0_N5000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_111_15703.688716_-2435.069757_445.717084_-15507.340548_batch0_N5000.h5')
    
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_99_13632.817202_11541.104208_-7142.638124_1280.366804_batch0_N5000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_98_16139.643014_1990.807772_15005.702117_5599.434348_batch0_N5000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_96_3891.286260_-2077.839854_3065.224272_1195.445719_batch0_N5000.h5')
    #real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user_95_10721.900721_9665.514458_4630.962318_301.949217_batch0_N5000.h5')
    
    if For_PMT_Theta:
        #hist_PMT_theta = read_hist_PMT_theta(real_file, [1800,0,180], 'real')
        hist_PMT_theta = read_hist_PMT_theta(real_file, [180,0,180], 'real')
        do_plot_1('real', hist_PMT_theta, 'pmt_theta')
        
    #real_hists = read_hist(real_file, 'HitTimeByPMT_', [310, -10, 3000], 'real', False, 1)
    #real_hists = read_hist(real_file, 'nPEByPMT_', [10, -5, 5], 'real', True, 1)
    #real_hists = read_hist_v1(real_file, [10, -5, 5], 'real', False)
    if For_r_theta_vs_npeMean:
        dict_r_gr,dict_theta_gr = read_graph_r_theta_vs_meanNPE(real_file)
        for r in dict_r_gr: 
            #do_plot_gr_1D(dict_r_gr[r], '','r=%s_thetaVsNpe'%str(r), [0,180], [0,0.5], ['#theta', 'mean NPE'], 0.5 )
            do_plot_gr_1D(dict_r_gr[r], '','r=%s_thetaVsNpe'%str(r), [45,55], [0,0.5], ['#theta', 'mean NPE'], 0.5 )
        
    
    fake_file = []
    if For_NPE:
        fake_file.append(fake_input)
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_part_b5000//user_78_761.054753_256.432825_-131.772148_-704.331345_batch0_N5000_pred.h5')
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_part_b5000//user_109_1972.354650_-170.609789_-1324.798385_1451.201022_batch0_N5000_pred.h5')
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_part_b5000//user_108_4143.010671_-27.302429_-27.039244_4142.832470_batch0_N5000_pred.h5')
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_part_b5000//user_114_5720.626101_2069.080878_-5293.362866_651.749089_batch0_N5000_pred.h5')
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_part_b5000//user_371_6006.914466_-332.456396_47.417349_5997.519966_batch0_N5000_pred.h5')
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_part_b5000//user_100_7265.516575_-673.529875_-1760.499121_-7016.746501_batch0_N5000_pred.h5')
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_part_b5000//user_104_10193.326270_3103.778049_-253.003481_9706.000799_batch0_N5000_pred.h5')
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_part_b5000//user_1_11042.389770_4397.912803_-8872.284301_-4886.236393_batch0_N5000_pred.h5')
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_part_b5000//user_102_14525.704016_10407.992577_-9981.839210_-1741.451597_batch0_N5000_pred.h5')
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_part_b5000//user_111_15703.688716_-2435.069757_445.717084_-15507.340548_batch0_N5000_pred.h5')
    
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_part_b5000//user_99_13632.817202_11541.104208_-7142.638124_1280.366804_batch0_N5000_pred.h5')
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_part_b5000//user_98_16139.643014_1990.807772_15005.702117_5599.434348_batch0_N5000_pred.h5')
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_part_b5000//user_96_3891.286260_-2077.839854_3065.224272_1195.445719_batch0_N5000_pred.h5')
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_part_b5000//user_95_10721.900721_9665.514458_4630.962318_301.949217_batch0_N5000_pred.h5')
    
    
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_full_possion//user_78_761.054753_256.432825_-131.772148_-704.331345_batch0_N5000_pred.h5')
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_full_possion//user_109_1972.354650_-170.609789_-1324.798385_1451.201022_batch0_N5000_pred.h5')
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_full_possion//user_108_4143.010671_-27.302429_-27.039244_4142.832470_batch0_N5000_pred.h5')
        #fake_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/predFiles_full_possion//user_371_6006.914466_-332.456396_47.417349_5997.519966_batch0_N5000_pred.h5')
    
        #fake_hists = read_hist_v1(fake_file, [15, -5, 10], 'fake', True)
        #real_hists = read_hist_v1(real_file, [15, -5, 10], 'real', False)
        fake_hists = {}
        real_hists = {}
        fake_hists_totPE = read_hist_totPE(fake_file, [80, 1000, 1800], 'fake','_pred.h5')
        real_hists_totPE = read_hist_totPE(real_file, [80, 1000, 1800], 'real','.h5')
    
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
        fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1210_part_elu.h5']
        fake_hists = read_hist(fake_file, 'predHitTimeByPMT_', [310, -10, 3000], 'fake', False, 1)
        #fake_hists = read_hist(fake_file, 'prednPEByPMT_', [10, -5, 5], 'fake', True, 1)
        real_hists = read_hist(real_file, 'HitTimeByPMT_', [310, -10, 3000], 'real', False, 1)
    
    if For_NPE or For_Time:
    
        count = 0
        r_theta_list = list(fake_hists.keys())
        r_theta_list = shuffle(r_theta_list, random_state=1)
        for i in r_theta_list:
            if i not in real_hists: continue
            print('r_theta=',i)
            if float(i.split('_')[0])>17.7: continue ### out of CD
            c = chi2_ndf(real_hists[i], fake_hists[i], True)
            if For_NPE:
                do_plot (i, real_hists[i], fake_hists[i], 'n_pe_', False, c)
                do_plot (i, real_hists[i], fake_hists[i], 'n_pe_logy', False, c)
            if For_Time:
                do_plot (i, real_hists[i], fake_hists[i], 'hit_time_'    , False, c)
                do_plot (i, real_hists[i], fake_hists[i], 'hit_time_logy', False, c)
            count = count + 1
            if count > 100: break
        
        if For_NPE:
            #print('fake_hists_totPE=',fake_hists_totPE) 
            #print('real_hists_totPE=',real_hists_totPE) 
            gr_meam_real = rt.TGraph2D()
            gr_mean_fake = rt.TGraph2D()
            index = 0
            for i in fake_hists_totPE:
                if i not in real_hists_totPE: continue
                c = chi2_ndf(real_hists_totPE[i], fake_hists_totPE[i], True)
                do_plot (i, real_hists_totPE[i], fake_hists_totPE[i], 'tot_pe_'    , False, c)
                do_plot (i, real_hists_totPE[i], fake_hists_totPE[i], 'tot_pe_logy', False, c)
                gr_mean_real.SetPoint(index, float(i.split('_')[2])/1000, real_hists_totPE[i].GetMean())     
                gr_mean_fake.SetPoint(index, float(i.split('_')[2])/1000, fake_hists_totPE[i].GetMean())     
                index += 1
            #do_plot_2gr_v1(gr_meam_real, gr_mean_fake,'','NNValid_meanNpeVsR_', [0, 18], [1010,1500], ['z (m)', 'total PE'], 1 , plot_path, False, ['G4', 'NN'], ['G4', 'NN'])
        
        
        
        gr_x = []
        gr_y = []
        gr_z = []
        for i in r_theta_list:
            if i not in real_hists: continue
            gr_x.append(float(i.split('_')[0])) # r       0-20
            gr_y.append(float(i.split('_')[1])) # theta , 0-180
            c = chi2_ndf(real_hists[i], fake_hists[i], True)
            gr_z.append(c)
        gr_2D        = rt.TGraph2D(len(gr_x), np.array(gr_x), np.array(gr_y), np.array(gr_z))
        gr_r_chi     = rt.TGraph(len(gr_z), np.array(gr_x), np.array(gr_z))
        gr_theta_chi = rt.TGraph(len(gr_z), np.array(gr_y), np.array(gr_z))
        if For_NPE:
            do_plot_gr(gr_2D, '' ,'n_pe_fit')
            do_plot_gr_1D(gr_r_chi    , '' ,'n_pe_fit_r'    ,[0,20 ],[0,10], ['r',"#chi^{2}/NDF"     ],1)
            do_plot_gr_1D(gr_theta_chi, '' ,'n_pe_fit_theta',[0,180],[0,10], ['#theta',"#chi^{2}/NDF"],1)
        if For_Time:
            do_plot_gr(gr_2D, '' ,'hit_time_fit')
            do_plot_gr_1D(gr_r_chi    , '' ,'hit_time_fit_r'    ,[0,20 ],[0,10], ['r',"#chi^{2}/NDF"     ], 1)
            do_plot_gr_1D(gr_theta_chi, '' ,'hit_time_fit_theta',[0,180],[0,10], ['#theta',"#chi^{2}/NDF"], 1)
        
        print('total chi2/NDF:',sum(gr_z))
    
    print('done!')

