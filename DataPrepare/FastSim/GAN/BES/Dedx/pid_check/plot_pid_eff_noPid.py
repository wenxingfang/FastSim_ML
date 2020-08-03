#!/usr/bin/env python
"""
Plot pid efficiency
"""

__author__ = "Maoqiang JING <jingmq@ihep.ac.cn>"
__copyright__ = "Copyright (c) Maoqiang JING"
__created__ = "[2020-03-27 Fri 10:53]"

import numpy as np
import sys, os
import logging
import ROOT
import math
import gc
from ROOT import TChain,TH1F,TH2F,TCanvas,gStyle, TFile, TGraphAsymmErrors, TLegend
from params import pid_eff, data_path
from array import array
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s- %(message)s')
gStyle.SetOptTitle(0)
gStyle.SetOptTitle(0)
gStyle.SetCanvasDefH(900)
gStyle.SetCanvasDefW(450)
gStyle.SetNdivisions(505,"xy")
gStyle.SetStatFontSize(0.05)

def usage():
    sys.stdout.write('''
NAME
    plot_pid_eff.py

SYNOPSIS
    ./plot_pid_eff.py [var] [particle] [ptrk_min] [ptrk_max]

AUTHOR
    Maoqiang JING <jingmq@ihep.ac.cn>

DATE
    March 2020
\n''')

def set_canvas_style(mbc):
    mbc.SetFillColor(0)
    mbc.SetLeftMargin(0.15)
    mbc.SetRightMargin(0.15)
    mbc.SetTopMargin(0.1)
    mbc.SetBottomMargin(0.15)

def make_weight_map(f1, f2, ch, tag):
    theta_min = -1
    theta_max =  1
    theta_bin =  20
    p_min =  0
    p_max =  2.5
    p_bin =  25
    f1_h2 = TH2F('%s_data'  %tag, '', p_bin, p_min, p_max, theta_bin, theta_min, theta_max)
    f2_h2 = TH2F('%s_mc'    %tag, '', p_bin, p_min, p_max, theta_bin, theta_min, theta_max)
    w_map = TH2F('%s_weight'%tag, '', p_bin, p_min, p_max, theta_bin, theta_min, theta_max)
    chain1 = TChain('n103')
    chain1.Add(f1)
    for i in range(chain1.GetEntries()):
        chain1.GetEntry(i)
        ptrk     = getattr(chain1, 'ptrk')
        charge   = getattr(chain1, 'charge')
        costheta = getattr(chain1, 'costheta')
        prob     = getattr(chain1, 'prob')
        if charge != ch:continue
        #if len(prob) != 5:continue ##for proton data, some time it has zero prob
        f1_h2.Fill(ptrk, costheta)
    chain2 = TChain('n103')
    chain2.Add(f2)
    for i in range(chain2.GetEntries()):
        chain2.GetEntry(i)
        ptrk     = getattr(chain2, 'ptrk')
        charge   = getattr(chain2, 'charge')
        costheta = getattr(chain2, 'costheta')
        prob     = getattr(chain2, 'prob')
        if charge != ch:continue
        i#if len(prob) != 5:continue
        f2_h2.Fill(ptrk, costheta)
    w_map.Divide(f1_h2, f2_h2)
    mbc =TCanvas("H2_%s"%tag,"",1500,500)
    set_canvas_style(mbc)
    mbc.Divide(3, 1)
    mbc.cd(1)
    f1_h2.Draw("COLZ")
    mbc.cd(2)
    f2_h2.Draw("COLZ")
    mbc.cd(3)
    w_map.Draw("COLZ")
    str_c={1:'+',-1:'-'}
    print('ch=',ch)
    print('str=',str_c[ch])
    mbc.SaveAs('%s/data_mc_%s_%s%s.png'%(plot_path, tag, particle, str(str_c[ch])) )
    return w_map

def process(particle, fname, cg, tag, wmap):

    xaxis = wmap.GetXaxis()
    yaxis = wmap.GetYaxis()
    p_N_bin   = xaxis.GetNbins()
    p_bin_max = xaxis.GetXmax() 
    p_bin_min = xaxis.GetXmin() 
    costheta_N_bin   = yaxis.GetNbins()
    costheta_bin_max = yaxis.GetXmax() 
    costheta_bin_min = yaxis.GetXmin() 

    h_p_all         = TH1F('%s_h_p_all'        %tag, '', p_N_bin, p_bin_min, p_bin_max)
    h_p_pass        = TH1F('%s_h_p_pass'       %tag, '', p_N_bin, p_bin_min, p_bin_max)
    h_costheta_all  = TH1F('%s_h_costheta_all' %tag, '', costheta_N_bin, costheta_bin_min, costheta_bin_max)
    h_costheta_pass = TH1F('%s_h_costheta_pass'%tag, '', costheta_N_bin, costheta_bin_min, costheta_bin_max)
    h_theta_all     = TH1F('%s_h_theta_all' %tag, '', 18, 0, 180)
    h_theta_pass    = TH1F('%s_h_theta_pass'%tag, '', 18, 0, 180)
    h_p_all.Sumw2()
    h_p_pass.Sumw2()
    h_costheta_all.Sumw2()
    h_costheta_pass.Sumw2()
    h_theta_all.Sumw2()
    h_theta_pass.Sumw2()
    ###################################3
    h_ndedx       = TH1F('%s_ndedx'      %tag , '', 100, 0, 100)
    h_dedx        = TH1F('%s_dedx'       %tag , '', 1000, 0, 3000)
    h_dedx_p      = TH2F('%s_dedx_p'     %tag , '', 1000, 0, 3000, 250, 0, 2.5)
    h_dedx_theta  = TH2F('%s_dedx_theta' %tag , '', 1000, 0, 3000, 180, 0, 180)
    p_bin     = np.arange(0,3,0.5)
    theta_bin = np.arange(0,180,10)
    h_p_list = []
    h_theta_list = []
    for i in range(len(p_bin)-1):
        h_p_list.append(TH1F('%s__p_%.3f_%.3f__dedx'%(tag, p_bin[i], p_bin[i+1])                    , '', 300, 0, 3000))
    for i in range(len(theta_bin)-1):
        h_theta_list.append(TH1F('%s__theta_%.3f_%.3f__dedx'%(tag, theta_bin[i], theta_bin[i+1])    , '', 300, 0, 3000))
    ###################################3


    chain = TChain('n103')
    chain.Add(fname)
    for i in range(chain.GetEntries()):
        chain.GetEntry(i)
        ptrk     = getattr(chain, 'ptrk')
        charge   = getattr(chain, 'charge')
        costheta = getattr(chain, 'costheta')
        prob     = getattr(chain, 'prob')
        dEdx_meas= getattr(chain, 'dEdx_meas')
        ndedxhit = getattr(chain, 'ndedxhit')
        dEdx_hit = getattr(chain, 'dEdx_hit')
        tmp_theta = math.acos(costheta)*180/math.pi
        if charge != cg:continue
        #if len(prob) != 5:continue
        we = 1
        if tag != 'data': 
            binx = xaxis.FindBin(ptrk)
            biny = yaxis.FindBin(costheta)
            we= wmap.GetBinContent(binx, biny)
        h_p_all       .Fill(ptrk    ,we)
        h_costheta_all.Fill(costheta,we)
        h_theta_all.Fill(math.acos(costheta)*180/math.pi,we)
        '''
        pass_cut = False
        if particle == 'e':
            pass_cut = (prob[0]>prob[1]) and (prob[0]>prob[2]) and (prob[0]>prob[3]) and (prob[0]>prob[4])
        elif particle == 'mu':
            pass_cut = (prob[1]>prob[0]) and (prob[1]>prob[2]) and (prob[1]>prob[3]) and (prob[1]>prob[4])
        elif particle == 'pi':
            pass_cut = (prob[2]>prob[3]) and (prob[2]>prob[4])
        elif particle == 'K':
            pass_cut = (prob[3]>prob[2]) and (prob[3]>prob[4])
        elif particle == 'p':
            pass_cut = (prob[4]>prob[2]) and (prob[4]>prob[3])
        else:
            print('wrong patricle name:',particle)
            os.exit()
        if pass_cut:
            h_p_pass       .Fill(ptrk    ,we)
            h_costheta_pass.Fill(costheta,we)
            h_theta_pass.Fill(math.acos(costheta)*180/math.pi,we)
        '''
        ########################################3
        h_ndedx.Fill(ndedxhit, we)
        for n in range(len(dEdx_hit)):
            h_dedx      .Fill(dEdx_hit[n], we)
            h_dedx_p    .Fill(dEdx_hit[n], ptrk, we)
            h_dedx_theta.Fill(dEdx_hit[n], math.acos(costheta)*180/math.pi, we)
            for ip in range(len(p_bin)-1):
                if ptrk > p_bin[ip] and ptrk < p_bin[ip+1]:
                    h_p_list[ip].Fill(dEdx_hit[n], we)
                    break
            '''
            for it in range(len(theta_bin)-1):
                if tmp_theta > theta_bin[it] and tmp_theta < theta_bin[it+1]:
                    h_theta_list[it].Fill(dEdx_hit[n], we)
                    break
            '''
        ########################################3

                    
    p_eff = TGraphAsymmErrors()
    p_eff.Divide(h_p_pass, h_p_all,"cl=0.683 b(1,1) mode")
    costheta_eff = TGraphAsymmErrors()
    costheta_eff.Divide(h_costheta_pass, h_costheta_all,"cl=0.683 b(1,1) mode")
    theta_eff = TGraphAsymmErrors()
    theta_eff.Divide(h_theta_pass, h_theta_all,"cl=0.683 b(1,1) mode")
    #return (h_p_pass, h_p_all, p_eff, h_costheta_pass, h_costheta_all, costheta_eff, h_theta_pass, h_theta_all, theta_eff) 
    return (h_p_pass, h_p_all, p_eff, h_costheta_pass, h_costheta_all, costheta_eff, h_theta_pass, h_theta_all, theta_eff, [h_dedx, h_dedx_p, h_dedx_theta], h_p_list, h_theta_list, h_ndedx) 
        


    
def do_plot_h3(Type, h1, h2, h3, out_name, title, plot_label):
    canvas=TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.12)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)
    if Type == 2:
        h1.GetYaxis().SetTitle(title['Y'])
        h1.GetXaxis().SetTitle(title['X'])
        h1.Draw("COLZ")
        canvas.SaveAs("%s/data_%s.png"%(plot_path,out_name))
        h2.GetYaxis().SetTitle(title['Y'])
        h2.GetXaxis().SetTitle(title['X'])
        h2.Draw("COLZ")
        canvas.SaveAs("%s/mc_off_%s.png"%(plot_path,out_name))
        h3.GetYaxis().SetTitle(title['Y'])
        h3.GetXaxis().SetTitle(title['X'])
        h3.Draw("COLZ")
        canvas.SaveAs("%s/mc_NN_%s.png"%(plot_path,out_name))
        return 0 
#    h1.Scale(1/h1.GetSumOfWeights())
#    h2.Scale(1/h2.GetSumOfWeights())
    x_min=h1.GetXaxis().GetXmin()
    x_max=h1.GetXaxis().GetXmax()
    y_min=0
    y_max=1.5
    if Type == 1:
        y_max=1.5*h1.GetBinContent(h1.GetMaximumBin())
    if 'eff_p' in out_name:
        #y_min=0.9
        y_min=0.84
    elif 'eff_phi' in out_name:
        y_min=0.9
    elif 'eff_costheta' in out_name:
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
    dummy = ROOT.TH2D("dummy","",1,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(ROOT.kFALSE)
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
    h1 .SetLineWidth(2)
    h2 .SetLineWidth(2)
    h3 .SetLineWidth(2)
    h3 .SetLineColor(ROOT.kRed)
    h2 .SetLineColor(ROOT.kBlue)
    h1 .SetLineColor(ROOT.kBlack)
    h3 .SetMarkerColor(ROOT.kRed)
    h2 .SetMarkerColor(ROOT.kBlue)
    h1 .SetMarkerColor(ROOT.kBlack)
    h1 .SetMarkerStyle(20)
    h2 .SetMarkerStyle(21)
    h3 .SetMarkerStyle(24)
    if Type == 1:
        dummy.Draw()
        h1.Draw("same:pe")
        h2.Draw("same:pe")
        h3.Draw("same:pe")
        dummy.Draw("AXISSAME")
    else:
        dummy.Draw("AXIS")
        h1.Draw("pe")
        h2.Draw("pe")
        h3.Draw("pe")
    #x_l = 0.6
    x_l = 0.2
    x_dl = 0.25
    y_h = 0.85
    y_dh = 0.15
    if 'h_costheta_all' in out_name or 'h_costheta_pass' in out_name:
        x_l = 0.55
        y_h = 0.85
    elif 'h_theta_all' in out_name or 'h_theta_pass' in out_name:
        x_l = 0.55
        y_h = 0.85
    legend = TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    legend.AddEntry(h1 ,'data','lep')
    legend.AddEntry(h2 ,"mc official",'lep')
    legend.AddEntry(h3 ,"mc NN",'lep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()

    label = ROOT.TLatex(0.5 , 0.9, plot_label)
    label.SetTextAlign(32)
    label.SetTextSize(0.035)
    label.SetNDC(ROOT.kTRUE)
    label.Draw() 
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

def ana(particle, cg):
    f_data, f_off, f_NN = data_path(particle)
    off_map = make_weight_map(f_data, f_off, cg, 'off')
    NN_map  = make_weight_map(f_data, f_NN , cg, 'NN' )
    h_p_pass_data, h_p_all_data, p_eff_data, h_costheta_pass_data, h_costheta_all_data, costheta_eff_data, h_theta_pass_data, h_theta_all_data, theta_eff_data, hlist0_data, hlist1_data, hlist2_data, hndedx_data =  process(particle, f_data, cg, 'data',off_map)
    h_p_pass_off , h_p_all_off , p_eff_off , h_costheta_pass_off , h_costheta_all_off , costheta_eff_off , h_theta_pass_off , h_theta_all_off , theta_eff_off , hlist0_off , hlist1_off , hlist2_off , hndedx_off  =  process(particle, f_off , cg, 'off' ,off_map)
    h_p_pass_NN  , h_p_all_NN  , p_eff_NN  , h_costheta_pass_NN  , h_costheta_all_NN  , costheta_eff_NN  , h_theta_pass_NN  , h_theta_all_NN  , theta_eff_NN  , hlist0_NN  , hlist1_NN  , hlist2_NN  , hndedx_NN   =  process(particle, f_NN  , cg, 'NN'  ,NN_map)
    
    do_plot_h3(Type=1, h1=h_p_pass_data, h2=h_p_pass_off, h3=h_p_pass_NN, out_name="h_p_pass"                            , title={'X':'P (GeV)','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_p_all_data , h2=h_p_all_off , h3=h_p_all_NN , out_name="h_p_all"                             , title={'X':'P (GeV)','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=0, h1=p_eff_data   , h2=p_eff_off   , h3=p_eff_NN   , out_name="h_p_eff"                             , title={'X':'P (GeV)','Y':'pid eff'}, plot_label='')
    do_plot_h3(Type=1, h1=h_costheta_pass_data, h2=h_costheta_pass_off, h3=h_costheta_pass_NN, out_name="h_costheta_pass", title={'X':'cos#theta','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_costheta_all_data , h2=h_costheta_all_off , h3=h_costheta_all_NN , out_name="h_costheta_all" , title={'X':'cos#theta','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=0, h1=costheta_eff_data   , h2=costheta_eff_off   , h3=costheta_eff_NN   , out_name="h_costheta_eff" , title={'X':'cos#theta','Y':'pid eff'}, plot_label='')
    do_plot_h3(Type=1, h1=h_theta_pass_data, h2=h_theta_pass_off, h3=h_theta_pass_NN, out_name="h_theta_pass", title={'X':'#theta','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_theta_all_data , h2=h_theta_all_off , h3=h_theta_all_NN , out_name="h_theta_all" , title={'X':'#theta','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=0, h1=theta_eff_data   , h2=theta_eff_off   , h3=theta_eff_NN   , out_name="h_theta_eff" , title={'X':'#theta','Y':'pid eff'}, plot_label='')

    do_plot_h3(Type=1, h1=hndedx_data, h2=hndedx_off, h3=hndedx_NN, out_name="h_N_dedx"                  , title={'X':'N dedx','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=2, h1=hlist0_data[0], h2=hlist0_off[0], h3=hlist0_NN[0], out_name="h_hit_dedx"       , title={'X':'Dedx','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=2, h1=hlist0_data[1], h2=hlist0_off[1], h3=hlist0_NN[1], out_name="h_hit_dedx_p"     , title={'X':'Dedx','Y':'ptrk (GeV)'}, plot_label='')
    do_plot_h3(Type=2, h1=hlist0_data[2], h2=hlist0_off[2], h3=hlist0_NN[2], out_name="h_hit_dedx_theta" , title={'X':'Dedx','Y':'#theta trk (degree)'}, plot_label='')
    for i in range(len(hlist1_data)):
        h_p_data = hlist1_data[i]
        h_p_off  = hlist1_off [i]
        h_p_NN   = hlist1_NN  [i]
        tmp_str = h_p_data.GetName().split('__')[1] 
        do_plot_h3(Type=1, h1=h_p_data, h2=h_p_off, h3=h_p_NN, out_name="h_hit_dedx_%s"%tmp_str , title={'X':'Dedx','Y':'Entries'}, plot_label='')
    for i in range(len(hlist2_data)):
        h_p_data = hlist2_data[i]
        h_p_off  = hlist2_off [i]
        h_p_NN   = hlist2_NN  [i]
        tmp_str = h_p_data.GetName().split('__')[1] 
        do_plot_h3(Type=1, h1=h_p_data, h2=h_p_off, h3=h_p_NN, out_name="h_hit_dedx_%s"%tmp_str , title={'X':'Dedx','Y':'Entries'}, plot_label='')

def plot(var, particle, ptrk_min, ptrk_max):
    try:
        chain = TChain('n103')
        chain.Add(data_path(particle))
    except:
        logging.error(data_path(particle) + ' is invalid!')
        sys.exit()

    if var == 'costheta' or var == 'phi':
        cut_ptrkp = '('+ptrk_min+'<ptrk<'+ptrk_max+')&&'
        cut_ptrkm = '('+ptrk_min+'<ptrk<'+ptrk_max+')&&'
    elif var == 'ptrk':
        cut_ptrkp = ''
        cut_ptrkm = ''
    mbc = TCanvas('mbc', 'mbc')
    set_canvas_style(mbc)
    mbc.Divide(2, 3)
    xmin, xmax, xbins = pid_eff(var)
    h_plus_nume = TH1F('h_plus_nume', ' ', xbins, xmin, xmax)
    h_plus_deno = TH1F('h_plus_deno', ' ', xbins, xmin, xmax)
    h_minus_nume = TH1F('h_minus_nume', ' ', xbins, xmin, xmax)
    h_minus_deno = TH1F('h_minus_deno', ' ', xbins, xmin, xmax)
    if particle == 'e':
        cut_prob = '(prob[0]>prob[1]&&prob[0]>prob[2]&&prob[0]>prob[3]&&prob[0]>prob[4])'
    if particle == 'mu':
        cut_prob = '(prob[1]>prob[0]&&prob[1]>prob[2]&&prob[1]>prob[3]&&prob[1]>prob[4])'
    if particle == 'pi':
        cut_prob = '(prob[2]>prob[3]&&prob[2]>prob[4])'
    if particle == 'K':
        cut_prob = '(prob[3]>prob[2]&&prob[3]>prob[4])'
    if particle == 'p':
        cut_prob = '(prob[4]>prob[2]&&prob[4]>prob[3])'
    mbc.cd(1)
    chain.Draw(var + '>>h_plus_nume', cut_ptrkp + '(charge==1)&&' + cut_prob)
    mbc.cd(2)
    chain.Draw(var + '>>h_plus_deno', cut_ptrkp + '(charge==1)')
    mbc.cd(3)
    chain.Draw(var + '>>h_minus_nume', cut_ptrkm + '(charge==-1)&&' + cut_prob)
    mbc.cd(4)
    chain.Draw(var + '>>h_minus_deno', cut_ptrkm + '(charge==-1)')

    mbc.cd(5)
    h_plus_nume.Sumw2()
    h_plus_deno.Sumw2()
    h_plus_pid_eff = TH1F('h_plus_eff', ' ', xbins, xmin, xmax)
    h_plus_pid_eff.Divide(h_plus_nume, h_plus_deno)
    if particle == 'e':
        frame1 = TH2F('frame1', 'Eff of e^{+} for data', xbins, xmin, xmax, 22, 0, 1.1)
    if particle == 'mu':
        frame1 = TH2F('frame1', 'Eff of #mu^{+} for data', xbins, xmin, xmax, 22, 0, 1.1)
    if particle == 'pi':
        frame1 = TH2F('frame1', 'Eff of #pi^{+} for data', xbins, xmin, xmax, 22, 0, 1.1)
    elif particle == 'K':
        frame1 = TH2F('frame1', 'Eff of K^{+} for data', xbins, xmin, xmax, 22, 0, 1.1)
    elif particle == 'p':
        frame1 = TH2F('frame1', 'Eff of p for data', xbins, xmin, xmax, 22, 0, 1.1)
    frame1.Draw()
    h_plus_pid_eff.SetLineColor(2)
    h_plus_pid_eff.Draw('same')

    mbc.cd(6)
    h_minus_nume.Sumw2()
    h_minus_deno.Sumw2()
    h_minus_pid_eff = TH1F('h_minus_eff', ' ', xbins, xmin, xmax)
    h_minus_pid_eff.Divide(h_minus_nume, h_minus_deno)
    if particle == 'e':
        frame2 = TH2F('frame2', 'Eff of e^{-} for data', xbins, xmin, xmax, 22, 0, 1.1)
    if particle == 'mu':
        frame2 = TH2F('frame2', 'Eff of #mu^{-} for data', xbins, xmin, xmax, 22, 0, 1.1)
    if particle == 'pi':
        frame2 = TH2F('frame2', 'Eff of #pi^{-} for data', xbins, xmin, xmax, 22, 0, 1.1)
    elif particle == 'K':
        frame2 = TH2F('frame2', 'Eff of K^{-} for data', xbins, xmin, xmax, 22, 0, 1.1)
    elif particle == 'p':
        frame2 = TH2F('frame2', 'Eff of #bar{p} for data', xbins, xmin, xmax, 22, 0, 1.1)
    frame2.Draw()
    h_minus_pid_eff.SetLineColor(2)
    h_minus_pid_eff.Draw('same')

    if not os.path.exists('./files/'):
        os.makedirs('./files/')
    out_root = TFile('./files/dedx_pideff_' + var + '_' + particle + '_data.root', 'RECREATE')
    h_plus_pid_eff.Write()
    h_minus_pid_eff.Write()
    out_root.Write()

    if not os.path.exists('./figs/'):
        os.makedirs('./figs/')
    mbc.SaveAs('./figs/pid_eff_' + var + '_' + particle + '.pdf')

'''
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args)<4:
        usage()
        sys.exit()
    var = args[0]
    particle = args[1]
    ptrk_min = args[2]
    ptrk_max = args[3]

    plot(var, particle, ptrk_min, ptrk_max)
'''

if __name__ == '__main__':

    args = sys.argv[1:]
    plot_path = './pid_plots/'
    particle = args[0]
    cg       = args[1]
    particle = particle.strip(' ')
    cg = int(cg)
    print('particle=',particle,',charge=',cg)
    ana(particle, cg)
    print('done')
