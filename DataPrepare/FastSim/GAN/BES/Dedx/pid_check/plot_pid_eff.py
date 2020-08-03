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
gStyle.SetPaintTextFormat("4.2f")

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

def make_weight_map(f1, f2, ch, tag, mom_min, mom_max, costheta_min, costheta_max):
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
        dEdx_hit = getattr(chain1, 'dEdx_hit')
        if charge != ch:continue
        if ptrk < mom_min or ptrk > mom_max :continue
        if costheta < costheta_min or costheta > costheta_max :continue
        if len(prob) != 5:continue ##for proton data, some time it has zero prob
        ######################## for proton background remove #
        '''
        mean_dedx = 0
        for idedx in range(len(dEdx_hit)):
            mean_dedx += dEdx_hit[idedx]
        mean_dedx = mean_dedx/len(dEdx_hit) if len(dEdx_hit) != 0 else 0
        if ptrk < 0.6 and mean_dedx < 600: continue
        '''
        #######################################################
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
        if ptrk < mom_min or ptrk > mom_max :continue
        if costheta < costheta_min or costheta > costheta_max :continue
        if len(prob) != 5:continue
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

def process(particle, fname, cg, tag, wmap, mom_min, mom_max, costheta_min, costheta_max):

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
    h_ndedx       = TH1F('%s_ndedx'      %tag , '', 60, 0, 60)
    h_usedhit     = TH1F('%s_usedhit'    %tag , '', 60, 0, 60)
    h_dedx        = TH1F('%s_dedx'       %tag , '', 300 , 0, 3000)
    h_dEdx_meas   = TH1F('%s_dEdx_meas'  %tag , '', 300 , 0, 900 )
    h_dedx_p      = TH2F('%s_dedx_p'     %tag , '', 1000, 0, 3000, 250, 0, 2.5)
    h_dedx_theta  = TH2F('%s_dedx_theta' %tag , '', 1000, 0, 3000, 180, 0, 180)
    p_bin     = np.arange(0,3,0.5)
    theta_bin = np.arange(0,180,10)
    h_p_list = []
    h_dEdx_meas_p_list = []
    h_theta_list = []
    for i in range(len(p_bin)-1):
        h_p_list          .append(TH1F('%s__p_%.3f_%.3f__dedx'%(tag, p_bin[i], p_bin[i+1])                    , '', 300, 0, 3000))
        h_dEdx_meas_p_list.append(TH1F('%s__dEdx_meas_p_%.3f_%.3f__dedx'%(tag, p_bin[i], p_bin[i+1])          , '', 300, 0, 900))
    for i in range(len(theta_bin)-1):
        h_theta_list.append(TH1F('%s__theta_%.3f_%.3f__dedx'%(tag, theta_bin[i], theta_bin[i+1])    , '', 300, 0, 3000))
    ###################################3
    h_prob_e       = TH1F('%s_prob_e'      %tag , '', 50, 0, 1)
    h_prob_mu      = TH1F('%s_prob_mu'     %tag , '', 50, 0, 1)
    h_prob_p       = TH1F('%s_prob_p'      %tag , '', 50, 0, 1)
    h_prob_k       = TH1F('%s_prob_k'      %tag , '', 50, 0, 1)
    h_prob_pi      = TH1F('%s_prob_pi'     %tag , '', 50, 0, 1)
    h_chi_e        = TH1F('%s_chi_e'       %tag , '', 50, -5, 20)
    h_chi_mu       = TH1F('%s_chi_mu'      %tag , '', 50, -5, 20)
    h_chi_p        = TH1F('%s_chi_p'       %tag , '', 50, -5, 20)
    h_chi_k        = TH1F('%s_chi_k'       %tag , '', 50, -5, 20)
    h_chi_pi       = TH1F('%s_chi_pi'      %tag , '', 50, -5, 20)
    ###################################3
    h_hit_dedx = {}
    theta_min = -1
    theta_max =  1.1
    theta_step =  0.1
    p_min =  0.3
    p_max =  2.1
    p_step =  0.2
    p_list = np.arange(p_min, p_max, p_step)
    theta_list = np.arange(theta_min, theta_max, theta_step)
    for i in range(p_list.shape[0]-1):
        for j in range(theta_list.shape[0]-1):
            str_key = "%s_%s__%s_%s"%( str(p_list[i]), str(p_list[i+1]), str(theta_list[j]), str(theta_list[j+1]) )
            h_hit_dedx[str_key] = TH1F('%s_%s_hit_dedx'%(tag, str_key) , '', 200, 0, 2000)
    h2_dedx_mean = TH2F('%s_dedx_mean'%tag, '', len(theta_list)-1, theta_list, len(p_list)-1, p_list)
    h2_dedx_std  = TH2F('%s_dedx_std' %tag, '', len(theta_list)-1, theta_list, len(p_list)-1, p_list)
    h2_dedx_mean_std = [h2_dedx_mean, h2_dedx_std]
    ###################################3


    chain = TChain('n103')
    chain.Add(fname)
    for i in range(chain.GetEntries()):
        chain.GetEntry(i)
        ptrk     = getattr(chain, 'ptrk')
        charge   = getattr(chain, 'charge')
        costheta = getattr(chain, 'costheta')
        prob     = getattr(chain, 'prob')
        chi      = getattr(chain, 'chi' )
        dEdx_meas= getattr(chain, 'dEdx_meas')
        usedhit  = getattr(chain, 'usedhit')
        ndedxhit = getattr(chain, 'ndedxhit')
        dEdx_hit = getattr(chain, 'dEdx_hit')
        tmp_theta = math.acos(costheta)*180/math.pi
        if charge != cg:continue
        if len(prob) != 5:continue
        if ptrk < mom_min or ptrk > mom_max :continue
        if costheta < costheta_min or costheta > costheta_max :continue
        '''
        if tag == 'data': 
            ######################## for proton background remove #
            mean_dedx = 0
            for idedx in range(len(dEdx_hit)):
                mean_dedx += dEdx_hit[idedx]
            mean_dedx = mean_dedx/len(dEdx_hit) if len(dEdx_hit) != 0 else 0
            if ptrk < 0.6 and mean_dedx < 600: continue
            #######################################################
        '''
        we = 1
        if tag != 'data': 
            binx = xaxis.FindBin(ptrk)
            biny = yaxis.FindBin(costheta)
            we= wmap.GetBinContent(binx, biny)
        h_p_all       .Fill(ptrk    ,we)
        h_costheta_all.Fill(costheta,we)
        h_theta_all.Fill(math.acos(costheta)*180/math.pi,we)
        '''
        #if (prob[0]>prob[1]) and (prob[0]>prob[2]) and (prob[0]>prob[3]): 
        #if (prob[0]>prob[1]) and (prob[0]>prob[4]) and (prob[0]>prob[3]): 
        #if (prob[0]>prob[2]) and (prob[0]>prob[4]) and (prob[0]>prob[3]): 
        if (prob[0]>prob[1]) and (prob[0]>prob[2]) and (prob[0]>prob[4]): 
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
        ########################################3
        h_ndedx.Fill(ndedxhit, we)
        h_usedhit.Fill(usedhit, we)
        h_dEdx_meas.Fill(dEdx_meas, we)
        for n in range(len(dEdx_hit)):
            h_dedx      .Fill(dEdx_hit[n], we)
            h_dedx_p    .Fill(dEdx_hit[n], ptrk, we)
            h_dedx_theta.Fill(dEdx_hit[n], math.acos(costheta)*180/math.pi, we)

        for ip in range(len(p_bin)-1):
            if ptrk > p_bin[ip] and ptrk < p_bin[ip+1]:
                for n in range(len(dEdx_hit)):
                    h_p_list[ip].Fill(dEdx_hit[n], we)
                break
            '''
            for it in range(len(theta_bin)-1):
                if tmp_theta > theta_bin[it] and tmp_theta < theta_bin[it+1]:
                    h_theta_list[it].Fill(dEdx_hit[n], we)
                    break
            '''
        '''
        for ip in range(len(p_bin)-1):
            if ptrk > p_bin[ip] and ptrk < p_bin[ip+1]:
                h_dEdx_meas_p_list[ip].Fill(dEdx_meas, we)
                break
        '''
        for ip in h_hit_dedx:
            tmp_0 = ip.split('__')[0]
            tmp_1 = ip.split('__')[1]
            p_min     = float( tmp_0.split('_')[0] )
            p_max     = float( tmp_0.split('_')[1] )
            theta_min = float( tmp_1.split('_')[0] )
            theta_max = float( tmp_1.split('_')[1] )
            if p_min < ptrk and ptrk < p_max and theta_min < costheta and costheta < theta_max:
                for n in range(len(dEdx_hit)):
                    h_hit_dedx[ip].Fill(dEdx_hit[n], we)
                break
        ########################################3
        h_prob_e .Fill(prob[0], we)
        h_prob_mu.Fill(prob[1], we)
        h_prob_pi.Fill(prob[2], we)
        h_prob_k .Fill(prob[3], we)
        h_prob_p .Fill(prob[4], we)
        h_chi_e  .Fill(chi [0], we)
        h_chi_mu .Fill(chi [1], we)
        h_chi_pi .Fill(chi [2], we)
        h_chi_k  .Fill(chi [3], we)
        h_chi_p  .Fill(chi [4], we)
    h_prob_list = [h_prob_e, h_prob_mu, h_prob_pi, h_prob_k, h_prob_p] 
    h_chi_list  = [h_chi_e , h_chi_mu , h_chi_pi , h_chi_k , h_chi_p ] 
                    
    p_eff = TGraphAsymmErrors()
    p_eff.Divide(h_p_pass, h_p_all,"cl=0.683 b(1,1) mode")
    costheta_eff = TGraphAsymmErrors()
    costheta_eff.Divide(h_costheta_pass, h_costheta_all,"cl=0.683 b(1,1) mode")
    theta_eff = TGraphAsymmErrors()
    theta_eff.Divide(h_theta_pass, h_theta_all,"cl=0.683 b(1,1) mode")
    #return (h_p_pass, h_p_all, p_eff, h_costheta_pass, h_costheta_all, costheta_eff, h_theta_pass, h_theta_all, theta_eff) 
    return (h_p_pass, h_p_all, p_eff, h_costheta_pass, h_costheta_all, costheta_eff, h_theta_pass, h_theta_all, theta_eff, [h_dedx, h_dedx_p, h_dedx_theta], h_p_list, h_theta_list, h_ndedx, h_usedhit, h_dEdx_meas, h_prob_list, h_chi_list, h_dEdx_meas_p_list, h_hit_dedx, h2_dedx_mean_std)
        


    
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
        h1.SetStats(ROOT.kFALSE)
        h1.Draw("COLZ")
        canvas.SaveAs("%s/data_%s.png"%(plot_path,out_name))
        h2.GetYaxis().SetTitle(title['Y'])
        h2.GetXaxis().SetTitle(title['X'])
        h2.SetStats(ROOT.kFALSE)
        h2.Draw("COLZ")
        canvas.SaveAs("%s/mc_off_%s.png"%(plot_path,out_name))
        h3.GetYaxis().SetTitle(title['Y'])
        h3.GetXaxis().SetTitle(title['X'])
        h3.SetStats(ROOT.kFALSE)
        h3.Draw("COLZ")
        canvas.SaveAs("%s/mc_NN_%s.png"%(plot_path,out_name))
        return 0 
    elif Type == 22:
        canvas=TCanvas("%s"%(out_name),"",1600,800)
        canvas.cd()
        canvas.SetTopMargin(0.13)
        canvas.SetBottomMargin(0.12)
        canvas.SetLeftMargin(0.15)
        canvas.SetRightMargin(0.15)
        h1.GetYaxis().SetTitle(title['Y'])
        h1.GetXaxis().SetTitle(title['X'])
        h1.SetStats(ROOT.kFALSE)
        h1.Draw("COLZ TEXTE")
        canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
        return 0 
    elif Type == 33:
        canvas=TCanvas("%s"%(out_name),"",1600,800)
        canvas.cd()
        canvas.SetTopMargin(0.13)
        canvas.SetBottomMargin(0.12)
        canvas.SetLeftMargin(0.15)
        canvas.SetRightMargin(0.15)
        h1.GetYaxis().SetTitle(title['Y'])
        h1.GetXaxis().SetTitle(title['X'])
        h1.SetStats(ROOT.kFALSE)
        h1.Draw("COLZ")
        canvas.SaveAs("%s/data_%s.png"%(plot_path,out_name))
        h2.GetYaxis().SetTitle(title['Y'])
        h2.GetXaxis().SetTitle(title['X'])
        h2.SetStats(ROOT.kFALSE)
        h2.Draw("COLZ")
        canvas.SaveAs("%s/mc_off_%s.png"%(plot_path,out_name))
        h3.GetYaxis().SetTitle(title['Y'])
        h3.GetXaxis().SetTitle(title['X'])
        h3.SetStats(ROOT.kFALSE)
        h3.Draw("COLZ")
        canvas.SaveAs("%s/mc_NN_%s.png"%(plot_path,out_name))
        return 0 
#    h1.Scale(1/h1.GetSumOfWeights())
#    h2.Scale(1/h2.GetSumOfWeights())
    x_min=h1.GetXaxis().GetXmin()
    x_max=h1.GetXaxis().GetXmax()
    y_min=0
    y_max=1.5
    if Type == 1 or  Type == 3:
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
    #h1 .SetMarkerStyle(20)
    #h2 .SetMarkerStyle(21)
    h1 .SetMarkerStyle(24)
    h2 .SetMarkerStyle(24)
    h3 .SetMarkerStyle(24)
    hit_dedx_mean_std = []
    if Type == 1:
        dummy.Draw()
        h1.Draw("same:pe")
        h2.Draw("same:pe")
        h3.Draw("same:pe")
        dummy.Draw("AXISSAME")
    elif Type == 3:
        dummy.Draw()
        h1.Draw("same:pe")
        h2.Draw("same:pe")
        h3.Draw("same:pe")
        dummy.Draw("AXISSAME")
        color_list = [2,3,4]
        hist_list = [h1, h2, h3]
        for hist in hist_list:
            if hist.GetSumOfWeights() < 100:
                hit_dedx_mean_std.append(0)
                hit_dedx_mean_std.append(0)
                hit_dedx_mean_std.append(0)
                hit_dedx_mean_std.append(0)
                continue
             
            mean = hist.GetMean()
            rms  = hist.GetRMS()
            lower  = mean - 2*rms
            higher = mean + 2*rms
            f1 = ROOT.TF1("f1", "gaus", lower, higher)
            f1.SetParameters(600.0, 200.0)
            result = hist.Fit('f1','RLS0')
            status = result.Status()
            if status == 0:
                print(result)
                par0   = result.Parameter(0)
                err0   = result.ParError(0)
                par1   = result.Parameter(1)
                err1   = result.ParError(1)
                par2   = result.Parameter(2)
                err2   = result.ParError (2)
                ######### do it again ###### 
                mean = par1
                rms  = par2
                lower  = mean - 1.5*rms
                higher = mean + 1*rms
                f2 = ROOT.TF1("f2", "gaus", lower, higher)
                f2.SetParameters(par1, par2)
                #f1.SetLineColor(color_list[hist_list.index(hist)])
                f2.SetLineColor(hist.GetLineColor())
                result = hist.Fit('f2','RLS')
                print('fit 1 for %s with entry %f mean=%f, rms=%f, lower=%f, higher=%f'%(hist.GetName(), hist.GetSumOfWeights(), hist.GetMean(), hist.GetRMS(), lower, higher))
                status = result.Status()
                if status == 0:
                    par0   = result.Parameter(0)
                    err0   = result.ParError(0)
                    par1   = result.Parameter(1)
                    err1   = result.ParError(1)
                    par2   = result.Parameter(2)
                    err2   = result.ParError (2)
                    hit_dedx_mean_std.append(par1)
                    hit_dedx_mean_std.append(err1)
                    hit_dedx_mean_std.append(par2)
                    hit_dedx_mean_std.append(err2)
                    #print('%s:mean=%f,mean err=%f,sigma=%f, sigma err=%f, status=%d'%(hist.GetName(), par1, err1, par2, err2, status))
                else:
                    print('failed fit 1 for %s with entry %f, lower=%f, higher=%f'%(hist.GetName(), hist.GetSumOfWeights(), lower, higher))
            else:
                print('failed fit 0 for %s with entry %f'%(hist.GetName(), hist.GetSumOfWeights()))
 
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
    elif '_chi_' in out_name:
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
    label1 = None
    label2 = None
    label3 = None
    if Type == 3:
        label1 = ROOT.TLatex(0.8 , 0.94, "data:mean=%.1f#pm%.1f,#sigma=%.1f#pm%.1f"%(hit_dedx_mean_std[0], hit_dedx_mean_std[1], hit_dedx_mean_std[2], hit_dedx_mean_std[3]))
        label1.SetTextAlign(32)
        label1.SetTextSize(0.035)
        label1.SetNDC(ROOT.kTRUE)
        label1.Draw() 
        label2 = ROOT.TLatex(0.8 , 0.91, "mc official:mean=%.1f#pm%.1f,#sigma=%.1f#pm%.1f"%(hit_dedx_mean_std[4], hit_dedx_mean_std[5], hit_dedx_mean_std[6], hit_dedx_mean_std[7]))
        label2.SetTextAlign(32)
        label2.SetTextSize(0.035)
        label2.SetNDC(ROOT.kTRUE)
        label2.Draw() 
        label3 = ROOT.TLatex(0.8 , 0.88, "mc NN:mean=%.1f#pm%.1f,#sigma=%.1f#pm%.1f"%(hit_dedx_mean_std[8], hit_dedx_mean_std[9], hit_dedx_mean_std[10], hit_dedx_mean_std[11]))
        label3.SetTextAlign(32)
        label3.SetTextSize(0.035)
        label3.SetNDC(ROOT.kTRUE)
        label3.Draw() 
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()
    return hit_dedx_mean_std

def ana(particle, cg, mom_min, mom_max, costheta_min, costheta_max):
    out_tag = "_%s_%s_%s_%s_%s_%s"%(str(particle), str(cg), str(mom_min), str(mom_max), str(costheta_min), str(costheta_max))
    f_data, f_off, f_NN = data_path(particle)
    off_map = make_weight_map(f_data, f_off, cg, 'off', mom_min, mom_max, costheta_min, costheta_max)
    NN_map  = make_weight_map(f_data, f_NN , cg, 'NN' , mom_min, mom_max, costheta_min, costheta_max)
    h_p_pass_data, h_p_all_data, p_eff_data, h_costheta_pass_data, h_costheta_all_data, costheta_eff_data, h_theta_pass_data, h_theta_all_data, theta_eff_data, hlist0_data, hlist1_data, hlist2_data, hndedx_data,h_usedhit_data, h_dEdx_meas_data, h_prob_list_data, h_chi_list_data, h_dEdx_meas_list_data, h2_hit_dedx_dict_data, h2_dedx_mean_std_data =  process(particle, f_data, cg, 'data',off_map, mom_min, mom_max, costheta_min, costheta_max)
    h_p_pass_off , h_p_all_off , p_eff_off , h_costheta_pass_off , h_costheta_all_off , costheta_eff_off , h_theta_pass_off , h_theta_all_off , theta_eff_off , hlist0_off , hlist1_off , hlist2_off , hndedx_off ,h_usedhit_off , h_dEdx_meas_off , h_prob_list_off, h_chi_list_off  , h_dEdx_meas_list_off , h2_hit_dedx_dict_off , h2_dedx_mean_std_off  =  process(particle, f_off , cg, 'off' ,off_map, mom_min, mom_max, costheta_min, costheta_max)
    h_p_pass_NN  , h_p_all_NN  , p_eff_NN  , h_costheta_pass_NN  , h_costheta_all_NN  , costheta_eff_NN  , h_theta_pass_NN  , h_theta_all_NN  , theta_eff_NN  , hlist0_NN  , hlist1_NN  , hlist2_NN  , hndedx_NN  ,h_usedhit_NN  , h_dEdx_meas_NN  , h_prob_list_NN, h_chi_list_NN    , h_dEdx_meas_list_NN  , h2_hit_dedx_dict_NN  , h2_dedx_mean_std_NN   =  process(particle, f_NN  , cg, 'NN'  ,NN_map, mom_min, mom_max, costheta_min, costheta_max)
    
    do_plot_h3(Type=1, h1=h_p_pass_data, h2=h_p_pass_off, h3=h_p_pass_NN                     , out_name="h_p_pass_%s"       %out_tag, title={'X':'P (GeV)','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_p_all_data , h2=h_p_all_off , h3=h_p_all_NN                      , out_name="h_p_all_%s"        %out_tag, title={'X':'P (GeV)','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=0, h1=p_eff_data   , h2=p_eff_off   , h3=p_eff_NN                        , out_name="h_p_eff_%s"        %out_tag, title={'X':'P (GeV)','Y':'pid eff'}, plot_label='')
    do_plot_h3(Type=1, h1=h_costheta_pass_data, h2=h_costheta_pass_off, h3=h_costheta_pass_NN, out_name="h_costheta_pass_%s"%out_tag, title={'X':'cos#theta','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_costheta_all_data , h2=h_costheta_all_off , h3=h_costheta_all_NN , out_name="h_costheta_all_%s" %out_tag, title={'X':'cos#theta','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=0, h1=costheta_eff_data   , h2=costheta_eff_off   , h3=costheta_eff_NN   , out_name="h_costheta_eff_%s" %out_tag, title={'X':'cos#theta','Y':'pid eff'}, plot_label='')
    do_plot_h3(Type=1, h1=h_theta_pass_data, h2=h_theta_pass_off, h3=h_theta_pass_NN         , out_name="h_theta_pass_%s"   %out_tag, title={'X':'#theta','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_theta_all_data , h2=h_theta_all_off , h3=h_theta_all_NN          , out_name="h_theta_all_%s"    %out_tag, title={'X':'#theta','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=0, h1=theta_eff_data   , h2=theta_eff_off   , h3=theta_eff_NN            , out_name="h_theta_eff_%s"    %out_tag, title={'X':'#theta','Y':'pid eff'}, plot_label='')

    do_plot_h3(Type=1, h1=hndedx_data   , h2=hndedx_off   , h3=hndedx_NN                     , out_name="h_N_dedx_%s"        %out_tag, title={'X':'N dedx','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_usedhit_data, h2=h_usedhit_off, h3=h_usedhit_NN                  , out_name="h_N_usedhit_%s"     %out_tag, title={'X':'N usedhit','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_dEdx_meas_data, h2=h_dEdx_meas_off, h3=h_dEdx_meas_NN            , out_name="h_dEdx_meas_%s"     %out_tag, title={'X':'dEdx_meas','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=hlist0_data[0], h2=hlist0_off[0], h3=hlist0_NN[0]                  , out_name="h_hit_dedx_%s"      %out_tag, title={'X':'Dedx','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=2, h1=hlist0_data[1], h2=hlist0_off[1], h3=hlist0_NN[1]                  , out_name="h_hit_dedx_p_%s"    %out_tag, title={'X':'Dedx','Y':'ptrk (GeV)'}, plot_label='')
    do_plot_h3(Type=2, h1=hlist0_data[2], h2=hlist0_off[2], h3=hlist0_NN[2]                  , out_name="h_hit_dedx_theta_%s"%out_tag, title={'X':'Dedx','Y':'#theta trk (degree)'}, plot_label='')
    for i in range(len(hlist1_data)):
        h_p_data = hlist1_data[i]
        h_p_off  = hlist1_off [i]
        h_p_NN   = hlist1_NN  [i]
        tmp_str = h_p_data.GetName().split('__')[1] 
        do_plot_h3(Type=1, h1=h_p_data, h2=h_p_off, h3=h_p_NN, out_name="h_hit_dedx_%s_%s"%(tmp_str,out_tag) , title={'X':'Dedx','Y':'Entries'}, plot_label='')
    for i in range(len(hlist2_data)):
        h_p_data = hlist2_data[i]
        h_p_off  = hlist2_off [i]
        h_p_NN   = hlist2_NN  [i]
        tmp_str = h_p_data.GetName().split('__')[1] 
        do_plot_h3(Type=1, h1=h_p_data, h2=h_p_off, h3=h_p_NN, out_name="h_hit_dedx_%s_%s"%(tmp_str,out_tag) , title={'X':'Dedx','Y':'Entries'}, plot_label='')
    for i in range(len(h_dEdx_meas_list_data)):
        h_p_data = h_dEdx_meas_list_data[i]
        h_p_off  = h_dEdx_meas_list_off [i]
        h_p_NN   = h_dEdx_meas_list_NN  [i]
        tmp_str = h_p_data.GetName().split('__')[1] 
        do_plot_h3(Type=1, h1=h_p_data, h2=h_p_off, h3=h_p_NN, out_name="h_%s_%s"%(tmp_str,out_tag) , title={'X':'Dedx','Y':'Entries'}, plot_label='')


    do_plot_h3(Type=1, h1=h_prob_list_data[0], h2=h_prob_list_off[0], h3=h_prob_list_NN[0] , out_name="h_prob_e_%s"  %out_tag, title={'X':'prob e','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_prob_list_data[1], h2=h_prob_list_off[1], h3=h_prob_list_NN[1] , out_name="h_prob_mu_%s" %out_tag, title={'X':'prob #mu','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_prob_list_data[2], h2=h_prob_list_off[2], h3=h_prob_list_NN[2] , out_name="h_prob_pi_%s" %out_tag, title={'X':'prob #pi','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_prob_list_data[3], h2=h_prob_list_off[3], h3=h_prob_list_NN[3] , out_name="h_prob_k_%s"  %out_tag, title={'X':'prob k','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_prob_list_data[4], h2=h_prob_list_off[4], h3=h_prob_list_NN[4] , out_name="h_prob_p_%s"  %out_tag, title={'X':'prob p','Y':'Entries'}, plot_label='')

    do_plot_h3(Type=1, h1=h_chi_list_data[0], h2=h_chi_list_off[0], h3=h_chi_list_NN[0] , out_name="h_chi_e_%s"  %out_tag, title={'X':'#chi e','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_chi_list_data[1], h2=h_chi_list_off[1], h3=h_chi_list_NN[1] , out_name="h_chi_mu_%s" %out_tag, title={'X':'#chi #mu','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_chi_list_data[2], h2=h_chi_list_off[2], h3=h_chi_list_NN[2] , out_name="h_chi_pi_%s" %out_tag, title={'X':'#chi #pi','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_chi_list_data[3], h2=h_chi_list_off[3], h3=h_chi_list_NN[3] , out_name="h_chi_k_%s"  %out_tag, title={'X':'#chi k','Y':'Entries'}, plot_label='')
    do_plot_h3(Type=1, h1=h_chi_list_data[4], h2=h_chi_list_off[4], h3=h_chi_list_NN[4] , out_name="h_chi_p_%s"  %out_tag, title={'X':'#chi p','Y':'Entries'}, plot_label='')

    costheta_axis = h2_dedx_mean_std_data[0].GetXaxis()
    mom_axis      = h2_dedx_mean_std_data[0].GetYaxis()
    h2_dedx_mean_rel1 = h2_dedx_mean_std_data[0].Clone('h2_dedx_mean_NN_data') 
    h2_dedx_std_rel1  = h2_dedx_mean_std_data[1].Clone('h2_dedx_std_NN_data') 
    h2_dedx_mean_rel2 = h2_dedx_mean_std_data[0].Clone('h2_dedx_mean_off_data') 
    h2_dedx_std_rel2  = h2_dedx_mean_std_data[1].Clone('h2_dedx_std_off_data') 
    h2_dedx_mean_rel1.Scale(0)
    h2_dedx_std_rel1 .Scale(0)
    h2_dedx_mean_rel2.Scale(0)
    h2_dedx_std_rel2 .Scale(0)
    for ih in h2_hit_dedx_dict_data:
       para_list = do_plot_h3(Type=3, h1=h2_hit_dedx_dict_data[ih], h2=h2_hit_dedx_dict_off[ih], h3=h2_hit_dedx_dict_NN[ih], out_name="h_%s_hit_dedx"%ih, title={'X':'hit Dedx','Y':'Entries'}, plot_label='')
       tmp_p_min = float( (ih.split('__')[0]).split('_')[0] )
       tmp_p_max = float( (ih.split('__')[0]).split('_')[1] )
       tmp_costheta_min = float( (ih.split('__')[1]).split('_')[0] )
       tmp_costheta_max = float( (ih.split('__')[1]).split('_')[1] )
       tmp_binx = costheta_axis.FindBin(0.5*tmp_costheta_min+0.5*tmp_costheta_max)
       tmp_biny = mom_axis.FindBin(0.5*tmp_p_min+0.5*tmp_p_max)

       h2_dedx_mean_std_data[0].SetBinContent(tmp_binx, tmp_biny, para_list[0])
       h2_dedx_mean_std_data[0].SetBinError  (tmp_binx, tmp_biny, para_list[1])
       h2_dedx_mean_std_data[1].SetBinContent(tmp_binx, tmp_biny, para_list[2])
       h2_dedx_mean_std_data[1].SetBinError  (tmp_binx, tmp_biny, para_list[3])
       h2_dedx_mean_std_off [0].SetBinContent(tmp_binx, tmp_biny, para_list[4])
       h2_dedx_mean_std_off [0].SetBinError  (tmp_binx, tmp_biny, para_list[5])
       h2_dedx_mean_std_off [1].SetBinContent(tmp_binx, tmp_biny, para_list[6])
       h2_dedx_mean_std_off [1].SetBinError  (tmp_binx, tmp_biny, para_list[7])
       h2_dedx_mean_std_NN  [0].SetBinContent(tmp_binx, tmp_biny, para_list[8])
       h2_dedx_mean_std_NN  [0].SetBinError  (tmp_binx, tmp_biny, para_list[9])
       h2_dedx_mean_std_NN  [1].SetBinContent(tmp_binx, tmp_biny, para_list[10])
       h2_dedx_mean_std_NN  [1].SetBinError  (tmp_binx, tmp_biny, para_list[11])
      
       mean_rel1 = abs(para_list[8] - para_list[0]) / para_list[0] if para_list[0] != 0 else 0
       mean_rel1_err = math.sqrt(para_list[0]*para_list[0]*para_list[9]*para_list[9] + para_list[8]*para_list[8]*para_list[1]*para_list[1])/(para_list[0]*para_list[0]) if para_list[0] != 0 else 0
       std_rel1 = abs(para_list[10] - para_list[2]) / para_list[2] if para_list[2] != 0 else 0
       std_rel1_err = math.sqrt(para_list[2]*para_list[2]*para_list[11]*para_list[11] + para_list[10]*para_list[10]*para_list[3]*para_list[3])/(para_list[2]*para_list[2]) if para_list[2] != 0 else 0
       mean_rel2 = abs(para_list[4] - para_list[0]) / para_list[0] if para_list[0] != 0 else 0
       mean_rel2_err = math.sqrt(para_list[0]*para_list[0]*para_list[5]*para_list[5] + para_list[4]*para_list[4]*para_list[1]*para_list[1])/(para_list[0]*para_list[0]) if para_list[0] != 0 else 0
       std_rel2 = abs(para_list[6] - para_list[2]) / para_list[2] if para_list[2] != 0 else 0
       std_rel2_err = math.sqrt(para_list[2]*para_list[2]*para_list[7]*para_list[7] + para_list[6]*para_list[6]*para_list[3]*para_list[3])/(para_list[2]*para_list[2]) if para_list[2] != 0 else 0
       
       h2_dedx_mean_rel1.SetBinContent(tmp_binx, tmp_biny, mean_rel1)
       h2_dedx_mean_rel1.SetBinError  (tmp_binx, tmp_biny, mean_rel1_err)
       h2_dedx_std_rel1 .SetBinContent(tmp_binx, tmp_biny, std_rel1)
       h2_dedx_std_rel1 .SetBinError  (tmp_binx, tmp_biny, std_rel1_err)
       h2_dedx_mean_rel2.SetBinContent(tmp_binx, tmp_biny, mean_rel2)
       h2_dedx_mean_rel2.SetBinError  (tmp_binx, tmp_biny, mean_rel2_err)
       h2_dedx_std_rel2 .SetBinContent(tmp_binx, tmp_biny, std_rel2)
       h2_dedx_std_rel2 .SetBinError  (tmp_binx, tmp_biny, std_rel2_err)

    do_plot_h3(Type=33 , h1=h2_dedx_mean_std_data[0], h2=h2_dedx_mean_std_off[0], h3=h2_dedx_mean_std_NN[0] , out_name="h_hit_dedx_mean", title={'X':'cos#theta','Y':'ptrk (GeV)'}, plot_label='')
    do_plot_h3(Type=33 , h1=h2_dedx_mean_std_data[1], h2=h2_dedx_mean_std_off[1], h3=h2_dedx_mean_std_NN[1] , out_name="h_hit_dedx_std" , title={'X':'cos#theta','Y':'ptrk (GeV)'}, plot_label='')
    do_plot_h3(Type=22, h1=h2_dedx_mean_rel1       , h2=h2_dedx_mean_rel1      , h3=h2_dedx_mean_rel1      , out_name="h_hit_dedx_mean_rel_NN_data", title={'X':'cos#theta','Y':'ptrk (GeV)'}, plot_label='')
    do_plot_h3(Type=22, h1=h2_dedx_std_rel1        , h2=h2_dedx_std_rel1       , h3=h2_dedx_std_rel1       , out_name="h_hit_dedx_std_rel_NN_data" , title={'X':'cos#theta','Y':'ptrk (GeV)'}, plot_label='')
    do_plot_h3(Type=22, h1=h2_dedx_mean_rel2       , h2=h2_dedx_mean_rel2      , h3=h2_dedx_mean_rel2      , out_name="h_hit_dedx_mean_rel_off_data", title={'X':'cos#theta','Y':'ptrk (GeV)'}, plot_label='')
    do_plot_h3(Type=22, h1=h2_dedx_std_rel2        , h2=h2_dedx_std_rel2       , h3=h2_dedx_std_rel2       , out_name="h_hit_dedx_std_rel_off_data" , title={'X':'cos#theta','Y':'ptrk (GeV)'}, plot_label='')

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

def readTxt(path, out_name, title):
    #h_dedx     = TH1F('h_dedx'    , ' ', 200, -10, 10)
    #h_dedx_off = TH1F('h_dedx_off', ' ', 200, -10, 10)
    h_dedx     = TH1F('h_dedx'    , ' ', 1000, -500, 3500)
    h_dedx_off = TH1F('h_dedx_off', ' ', 1000, -500, 3500)
    h_dedx_col = TH1F('h_dedx_col', ' ', 1000, -500, 3500)
    h_dedx_ins = TH1F('h_dedx_ins', ' ', 1000, -500, 3500)
    #h_ins_pdg  = TH1F('h_ins_pdg', ' ', 500, 0, 500)
    h_ins_pdg  = TH1F('h_ins_pdg', ' ', 20, 0, 20)
    #h_dedx     = TH1F('h_dedx'    , ' ', 1000, 0, 3000)
    #h_dedx_off = TH1F('h_dedx_off', ' ', 1000, 0, 3000)
    files = os.listdir(path)
    for i in files:
        if 'nohup.out' != i : continue
        #if 'nohup.out' not in i or 'swp'in i: continue
        #if 'nohup_NN.out' not in i : continue
        #if '0.sh.out.' not in i: continue
        #if '.out.' not in i: continue
        print('%s/%s'%(path,i))
        f_in = open('%s/%s'%(path,i),'r')
        lines = f_in.readlines()
        f_in.close()
        for line in lines:
            if "costheta" in line and 'ndedxhit' not in line: 
                str0 = line.split(',')
                mom      = float(str0[0].replace('mom=',''))
                costheta = float(str0[1].replace('costheta=',''))
                dedx     = float(str0[2].replace('pred=',''))
                #dedx     = float(str0[2].replace('edep=',''))
                dedx_off = float(str0[3].replace('off=',''))
                pid      = float(str0[4].replace('pdg=',''))
                #if pid != 211: continue
                #if mom <= 0: continue
                #if mom > 0.5: continue
                #if dedx > 0: continue
                h_dedx    .Fill(dedx)
                h_dedx_off.Fill(dedx_off)
            elif "energy deposit" in line: 
                if 'MeV' in line:
                    str0 = line.split('MeV')[0]
                    str1 = str0.split('energy deposit:')[-1]
                    h_dedx_col .Fill(float(str1))
                elif 'GeV' in line:
                    str0 = line.split('GeV')[0]
                    str1 = str0.split('energy deposit:')[-1]
                    h_dedx_col .Fill(float(str1)*1000)
                else:
                    print(line)
            elif "insert:" in line : 
                str0 = line.split(',')
                str1 = str0[0].replace('insert:edep=','')
                str2 = str0[1].replace('pdg=','')
                h_dedx_ins .Fill(float(str1))
                h_ins_pdg  .Fill(float(str2))
                  
    canvas=TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.12)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)
    x_min=h_dedx.GetXaxis().GetXmin()
    x_max=h_dedx.GetXaxis().GetXmax()
    y_min=0
    y_max=1.5*h_dedx.GetBinContent(h_dedx.GetMaximumBin())
    y_max=12000
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
    h_dedx.SetLineWidth(2)
    h_dedx.SetLineColor(ROOT.kBlack)
    h_dedx.SetMarkerColor(ROOT.kBlack)
    h_dedx.SetMarkerStyle(20)
    h_dedx_off.SetLineWidth(2)
    h_dedx_off.SetLineColor  (ROOT.kRed)
    h_dedx_off.SetMarkerColor(ROOT.kRed)
    h_dedx_off.SetMarkerStyle(21)
    h_dedx_col.SetLineWidth(2)
    h_dedx_col.SetLineColor  (ROOT.kBlue)
    h_dedx_col.SetMarkerColor(ROOT.kBlue)
    h_dedx_col.SetMarkerStyle(22)
    h_dedx_ins.SetLineWidth(2)
    h_dedx_ins.SetLineColor  (ROOT.kGreen)
    h_dedx_ins.SetMarkerColor(ROOT.kGreen)
    h_dedx_ins.SetMarkerStyle(23)
    dummy.Draw()
    h_dedx.Draw("same:pe")
    h_dedx_off.Draw("same:pe")
    h_dedx_col.Draw("same:pe")
    h_dedx_ins.Draw("same:pe")
    dummy.Draw("AXISSAME")
    x_l = 0.2
    x_dl = 0.25
    y_h = 0.85
    y_dh = 0.15
    legend = TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    legend.AddEntry(h_dedx     ,'NN'      ,'lep')
    legend.AddEntry(h_dedx_off ,"official",'lep')
    legend.AddEntry(h_dedx_col ,"NN col",'lep')
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    h_ins_pdg.Draw()
    canvas.SaveAs("%s/%s_pdg.png"%(plot_path,out_name))
    del canvas
    gc.collect()

def simpleAna(f1, f2, out_name, title):
    h_dedx     = TH1F('h_dedx'    , ' ', 1000, -500, 3500)
    h_dedx_off = TH1F('h_dedx_off', ' ', 1000, -500, 3500)
    #h_dedx     = TH1F('h_dedx'    , ' ', 1000, 0, 3000)
    #h_dedx_off = TH1F('h_dedx_off', ' ', 1000, 0, 3000)
    chain1 = TChain('n103')
    chain1.Add(f1)
    for i in range(chain1.GetEntries()):
        chain1.GetEntry(i)
        ptrk     = getattr(chain1, 'ptrk')
        charge   = getattr(chain1, 'charge')
        costheta = getattr(chain1, 'costheta')
        prob     = getattr(chain1, 'prob')
        dEdx_meas= getattr(chain1, 'dEdx_meas')
        ndedxhit = getattr(chain1, 'ndedxhit')
        dEdx_hit = getattr(chain1, 'dEdx_hit')
        tmp_theta = math.acos(costheta)*180/math.pi
        if ptrk > 0.5:continue
        #if charge != cg:continue
        #if len(prob) != 5:continue
        for j in range(len(dEdx_hit)):
            h_dedx.Fill(dEdx_hit[j])
    chain2 = TChain('n103')
    chain2.Add(f2)
    for i in range(chain2.GetEntries()):
        chain2.GetEntry(i)
        ptrk     = getattr(chain2, 'ptrk')
        charge   = getattr(chain2, 'charge')
        costheta = getattr(chain2, 'costheta')
        prob     = getattr(chain2, 'prob')
        dEdx_meas= getattr(chain2, 'dEdx_meas')
        ndedxhit = getattr(chain2, 'ndedxhit')
        dEdx_hit = getattr(chain2, 'dEdx_hit')
        tmp_theta = math.acos(costheta)*180/math.pi
        if ptrk > 0.5:continue
        #if charge != cg:continue
        #if len(prob) != 5:continue
        for j in range(len(dEdx_hit)):
            h_dedx_off.Fill(dEdx_hit[j])
    canvas=TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.12)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)
    x_min=h_dedx.GetXaxis().GetXmin()
    x_max=h_dedx.GetXaxis().GetXmax()
    y_min=0
    y_max=1.5*h_dedx.GetBinContent(h_dedx.GetMaximumBin())
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
    h_dedx.SetLineWidth(2)
    h_dedx.SetLineColor(ROOT.kBlack)
    h_dedx.SetMarkerColor(ROOT.kBlack)
    h_dedx.SetMarkerStyle(20)
    h_dedx_off.SetLineWidth(2)
    h_dedx_off.SetLineColor  (ROOT.kRed)
    h_dedx_off.SetMarkerColor(ROOT.kRed)
    h_dedx_off.SetMarkerStyle(21)
    dummy.Draw()
    h_dedx.Draw("same:pe")
    h_dedx_off.Draw("same:pe")
    dummy.Draw("AXISSAME")
    x_l = 0.2
    x_dl = 0.25
    y_h = 0.85
    y_dh = 0.15
    legend = TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    legend.AddEntry(h_dedx     ,'NN'      ,'lep')
    legend.AddEntry(h_dedx_off ,"official",'lep')
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

    
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
    mom_min  = float(args[2])
    mom_max  = float(args[3])
    costheta_min  = float(args[4])
    costheta_max  = float(args[5])
    particle = particle.strip(' ')
    cg = int(cg)
    print('particle=',particle,',charge=',cg,',mom_min=',mom_min,',mom_max=',mom_max,',costheta_min=',costheta_min,',costheta_max=',costheta_max)
    #readTxt(path='/junofs/users/wxfang/FastSim/bes3/workarea/TestRelease/TestRelease-00-00-86/DedxTest/sim/job_NN/k+/', out_name='sim_k+', title={'X':'dedx', 'Y':'Entries'})
    #simpleAna(f1='/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output_NN/Single_kp_NN_0.root', f2='/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output_off/Single_k+_off_0.root', out_name='rec_k+', title={'X':'dedx', 'Y':'Entries'})
    ana(particle, cg, mom_min, mom_max, costheta_min, costheta_max)
    print('done')
