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

def get_npe_info_v1(tag, input_files, binning_t, binning_n):
    dicts = {}
    for ifile in input_files: 
        dicts[input_files[ifile]] = [0, 0, 0]
        dicts[input_files[ifile]][1] = rt.TH1D('time_%s_%s'%(str(tag),str(input_files[ifile])),'',binning_t[0],binning_t[1],binning_t[2])
        dicts[input_files[ifile]][2] = rt.TH1D('npe_%s_%s'%(str(tag),str(input_files[ifile])),'' ,binning_n[0],binning_n[1],binning_n[2])
        chain =rt.TChain('Event/Sim/SimEvent')
        #chain =rt.TChain('Event/Sim/SimEvent/m_cd_hits')
        chain.Add(ifile)
        tree = chain
        #m_cd_hits       = getattr(tree, "m_cd_hits")
        #print('m_cd_hits=',m_cd_hits)
        totalEntries=tree.GetEntries()
        print(ifile,',totalEntries=',totalEntries)
        #sys.exit()
        for ie in range(totalEntries):
            tree.GetEntry(ie)
            pmtId       = getattr(tree, "pmtid")
            npe         = getattr(tree, "npe")
            hittime     = getattr(tree, "hittime")
            nphoton = 0
            tmp_dict = {}
            for ip in range(hittime):
                if pmtID[ip] > 17612 : continue
                nphoton +=npe[ip] ## only large pmt
                #if pmtID[ip] not in tmp_dict: tmp_dict[pmtID[ip]] = npe[ip]
                #else: tmp_dict[pmtID[ip]] += npe[ip]
                dicts[input_files[ifile]][2].Fill(hittime[ip])
                
                #if pmtID[ip] <= 8803 : nphoton +=1 ## only large pmt at positive Z
                #if pmtID[ip] <= 17612 and pmtID[ip] > 8803 : nphoton +=1 ## only large pmt at negative Z
            dicts[input_files[ifile]][0] = dicts[input_files[ifile]][0] + float(nphoton)/totalEntries
            dicts[input_files[ifile]][1].Fill(nphoton)
            #for i in range(17612):
            #    if i in tmp_dict: dicts[input_files[ifile]][2].Fill(tmp_dict[i])
            #    else            : dicts[input_files[ifile]][2].Fill(0)
    return dicts


if True:
    #full_file_name ='/junofs/users/wxfang/JUNO/read_muon/hist_full_muon_1000_LPMT.root' 
    #full_file_name ='/junofs/users/wxfang/JUNO/read_muon/hist_full_muon_1000_Qdep.root' 
    full_file_name ='/junofs/users/wxfang/JUNO/read_muon/hist_full_muon_noOP_Qdep.root' 
    full_file = rt.TFile(full_file_name,'read')
    #fast0_file_name = '/junofs/users/wxfang/JUNO/read_muon/hist_fast_pois_muon_1000_LPMT.root'
    #fast0_file_name = '/junofs/users/wxfang/JUNO/read_muon/hist_fast_pois_muon_1000_Qdep.root'
    fast0_file_name = '/junofs/users/wxfang/JUNO/read_muon/hist_fast_pois_muon_noOP_Qdep.root'
    fast0_file = rt.TFile(fast0_file_name,'read')
    full_npe     = full_file.Get('tot_pe')
    #full_hittime = full_file.Get('hittime')
    fast0_npe     = fast0_file.Get('tot_pe')
    #fast0_hittime = fast0_file.Get('hittime')
    ie = '250'
    out_path = './plots_muon/' 
    do_plot_v2(str(ie), str('E=%s GeV'%ie),full_npe    , fast0_npe    , 'tot_Qdep'  , [5e3,3e4], out_path,['full sim', 'fast sim'], False, 0.04, ['full', 'fast'])
    #do_plot_v2(str(ie), str('E=%s GeV'%ie),full_npe    , fast0_npe    , 'tot_Qdep'  , [5e3,3e4], out_path,['full sim', 'fast sim'], True, 0.04, ['full', 'fast'])
    #do_plot_v2(str(ie), str('E=%s GeV'%ie),full_hittime, fast0_hittime, 'hit_time', [0,500], out_path,['full sim', 'fast sim'], True, 0.04, ['full', 'fast'])
    #do_plot_v2(str(ie), str('E=%s GeV'%ie),full_npe    , fast0_npe    , 'tot_pe'  , [8e6,3e7], out_path,['full sim', 'fast sim'], True, 0.04, ['full', 'fast'])
    #print('mean full=',full_npe.GetMean(),',fast0=',fast0_npe.GetMean(),',ratio=',float(fast0_npe.GetMean()/full_npe.GetMean()))
    

if False:
    full_inputs = {}
    full_inputs['/cefs/higgs/wxfang/sim_ouput/official_muon_sample/detsim_1_215000_19000.000000_139.488841_34.551014_18999.456547.root'] = '250' 
    reals = get_npe_info_v1(tag='full', input_files=full_inputs, binning_t=[500,0,500], binning_n=[1000,0,100000])
    fast_inputs = {}
    fast_inputs['/cefs/higgs/wxfang/sim_ouput/official_muon_sample/detsim_1_215000_19000.000000_139.488841_34.551014_18999.456547.root'] = '250' 
    reals = get_npe_info_v1(tag='fast', input_files=fast_inputs, binning_t=[500,0,500], binning_n=[1000,0,100000])
    ie = '250'
    out_path = './plots_muon/' 
    do_plot_v2(str(ie), str('E=%s GeV'%ie),full_inputs[ie][1], fast_inputs[ie][1], 'hit_time', [0,500], out_path,['full sim', 'fast sim'], True, 0.04, ['full', 'fast'])
    do_plot_v2(str(ie), str('E=%s GeV'%ie),full_inputs[ie][2], fast_inputs[ie][2], 'npe'     , [0,100000], out_path,['full sim', 'fast sim'], True, 0.04, ['full', 'fast'])



