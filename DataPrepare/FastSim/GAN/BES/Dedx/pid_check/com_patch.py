#!/usr/bin/env python
"""
Compare different patches of samples
"""

__author__ = "Maoqiang JING <jingmq@ihep.ac.cn>"
__copyright__ = "Copyright (c) Maoqiang JING"
__created__ = "[2020-04-06 Mon 16:29]"

from array import array
import sys, os
import logging
from math import *
from ROOT import *
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s- %(message)s')
gStyle.SetOptStat(000000000)
gStyle.SetNdivisions(505,"xy")
gStyle.SetCanvasDefH(500)
gStyle.SetCanvasDefW(1000)

def usage():
    sys.stdout.write('''
NAME
    com_patch.py

SYNOPSIS
    ./com_patch.py [var] [particle]

AUTHOR
    Maoqiang JING <jingmq@ihep.ac.cn>

DATE
    April 2020
\n''')

def set_hist_style(h, i, xtitle, ytitle):
    h.SetLineColor(i + 1)
    h.SetMarkerColor(i + 1)
    h.SetMarkerStyle(i + 20)
    h.SetMarkerSize(1.5)
    h.GetXaxis().SetTitle(xtitle)
    h.GetXaxis().CenterTitle()
    h.GetYaxis().SetTitle(ytitle)
    h.GetYaxis().CenterTitle()
    h.GetYaxis().SetRangeUser(0, 1)

def set_canvas_style(mbc):
    mbc.SetFillColor(0)
    mbc.SetLeftMargin(0.15)
    mbc.SetRightMargin(0.15)
    mbc.SetTopMargin(0.1)
    mbc.SetBottomMargin(0.15)

def compare(path_in, var, particle):
    mbc = TCanvas('mbc', 'mbc')
    set_canvas_style(mbc)
    mbc.Divide(2, 1)
    
    eps_out = './figs/dedx_pideff_' + var + '_' + particle + '_compare.eps'
    print '--> Begin to process file: ' + eps_out
    f_in = TFile(path_in[0])
    f_in_test = TFile(path_in[1])
    h_in_plus = f_in.Get('h_plus_eff')
    h_in_test_plus = f_in_test.Get('h_plus_eff')
    h_in_minus = f_in.Get('h_minus_eff')
    h_in_test_minus = f_in_test.Get('h_minus_eff')
    if var == 'costheta':
        xtitle = 'cos#theta'
    if var == 'ptrk':
        xtitle = 'P'
    if var == 'phi':
        xtitle = '#phi'
    if particle == 'pi':
        par_name = '#pi'
    if particle == 'K':
        par_name = 'K'
    if particle == 'p':
        par_name = 'p'
    ytitle_plus = par_name + '^{+} PID Efficiency'
    ytitle_minus = par_name + '^{-} PID Efficiency'
    set_hist_style(h_in_plus, 0, xtitle, ytitle_plus)
    set_hist_style(h_in_test_plus, 1, xtitle, ytitle_plus)
    set_hist_style(h_in_minus, 0, xtitle, ytitle_minus)
    set_hist_style(h_in_test_minus, 1, xtitle, ytitle_minus)

    mbc.cd(1)
    h_in_plus.Draw('EP')
    h_in_test_plus.Draw('EPSAME')
    mbc.cd(2)
    h_in_minus.Draw('ep')
    h_in_test_minus.Draw('epsame')

    if not os.path.exists('./figs/'):
        os.makedirs('./figs/')
    mbc.SaveAs(eps_out)
    raw_input('Enter anything to end...')

    print '--> End of processing file: ' + eps_out

def main():
    args = sys.argv[1:]
    if len(args)<2:
        return usage()
    var = args[0]
    particle = args[1]

    path_in = []
    path_in.append('/besfs/groups/cal/dedx/jingmq/bes/pid_eff_check/python/files/dedx_pideff_'+var+'_'+particle+'_705_data.root')
    path_in.append('/besfs/groups/cal/dedx/jingmq/bes/pid_eff_check/python/files/dedx_pideff_costheta_K_705_data.root')
    compare(path_in, var, particle)

if __name__ == '__main__':
    main()
