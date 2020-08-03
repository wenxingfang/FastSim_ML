import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import argparse
rt.gROOT.SetBatch(rt.kTRUE)

#######################################
# use digi step data and use B field  ##
# use cell ID for ECAL                ##
# add HCAL
#######################################

def get_parser():
    parser = argparse.ArgumentParser(
        description='root to hdf5',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', action='store', type=str,
                        help='input root file')
    parser.add_argument('--output', action='store', type=str,
                        help='output root file')
    parser.add_argument('--tag', action='store', type=str,
                        help='tag name for plots')
    parser.add_argument('--str_particle', action='store', type=str,
                        help='e^{-}')


    return parser




def plot_gr(gr,out_name,title, Type):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    #gr.GetXaxis().SetTitle("#phi(AU, 0 #rightarrow 2#pi)")
    #gr.SetTitle(title)
    if Type == 0: #graph
        gr.Draw("ap")
    elif Type ==1 : #graph
        gr.SetLineWidth(2)
        gr.SetMarkerStyle(8)
        if "HoE" in out_name:
            gr.GetXaxis().SetTitle("H/E")
            gr.GetYaxis().SetTitle("Events")
        gr.Draw("hist")
        gr.Draw("same:pe")
    elif Type ==2:
        gr.Draw("COLZ")
    else:
        print('wrong type')
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

def plot_hist(hist,out_name,title):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    canvas.SetGridy()
    canvas.SetGridx()
    #h_corr.Draw("COLZ")
    #h_corr.LabelsDeflate("X")
    #h_corr.LabelsDeflate("Y")
    #h_corr.LabelsOption("v")
    hist.SetStats(rt.kFALSE)
    #hist.GetXaxis().SetTitle("#Delta Z (mm)")
    if 'x_z' in out_name:
        #hist.GetYaxis().SetTitle("X (mm)")
        hist.GetYaxis().SetTitle("cell X")
        hist.GetXaxis().SetTitle("cell Z")
    elif 'y_z' in out_name:
        #hist.GetYaxis().SetTitle("#Delta Y (mm)")
        hist.GetYaxis().SetTitle("cell Y")
        hist.GetXaxis().SetTitle("cell Z")
    elif 'z_r' in out_name:
        hist.GetYaxis().SetTitle("bin R")
        hist.GetXaxis().SetTitle("bin Z")
    elif 'z_phi' in out_name:
        hist.GetYaxis().SetTitle("bin #phi")
        hist.GetXaxis().SetTitle("bin Z")
    hist.SetTitle(title)
    #hist.SetTitleSize(0.1)
    #hist.Draw("COLZ TEXT")
    hist.Draw("COLZ")
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

if __name__ == '__main__':

    parser = get_parser()
    parse_args = parser.parse_args()
    
    gr_xy =  rt.TGraph()
    gr_xy.SetMarkerColor(rt.kBlack)
    gr_xy.SetMarkerStyle(8)
    h_xy     = rt.TH2F('h_xy'   , '',120 ,2000, 3200, 400, -2000, 2000)
    h_xy1    = rt.TH2F('h_xy1'  , '',640 ,0, 3200, 640, -3200, 3200)
    h_HoE    = rt.TH1F('h_HoE'  , '',100, 0, 10)
    h_TotE   = rt.TH1F('h_TotE' , '',20 , 0, 20)
    
    print ('Read root file')
    plot_path='/junofs/users/wxfang/FastSim/GAN/CEPC/GAN/raw_Hit_plots'
    filePath = parse_args.input
    treeName='evt'
    chain =rt.TChain(treeName)
    chain.Add(filePath)
    tree = chain
    totalEntries=tree.GetEntries()
    print (totalEntries)

    for entryNum in range(0, tree.GetEntries()):
        tree.GetEntry(entryNum)
        tmp_Hit_x   = getattr(tree, "m_HcalHit_x")
        tmp_Hit_y   = getattr(tree, "m_HcalHit_y")
        tmp_Hit_z   = getattr(tree, "m_HcalHit_z")
        tmp_Ecal_digi_Hit   = getattr(tree, "m_Hits")
        tmp_Hcal_digi_Hit   = getattr(tree, "m_HcalHits")
        E_Hcal = 0.0
        E_Ecal = 0.0
        for i in range(len(tmp_Hit_x)):
                gr_xy.SetPoint(i, tmp_Hit_x[i], tmp_Hit_y[i])
                h_xy .Fill(tmp_Hit_x[i], tmp_Hit_y[i])
                h_xy1.Fill(tmp_Hit_x[i], tmp_Hit_y[i])
        for i in range(len(tmp_Hcal_digi_Hit)):
            for j in range(len(tmp_Hcal_digi_Hit[i])):
                E_Hcal = E_Hcal + tmp_Hcal_digi_Hit[i][j]
        for i in range(len(tmp_Ecal_digi_Hit)):
            for j in range(len(tmp_Ecal_digi_Hit[i])):
                E_Ecal = E_Ecal + tmp_Ecal_digi_Hit[i][j]
        h_HoE.Fill(E_Hcal/E_Ecal)
        h_TotE.Fill(E_Hcal+E_Ecal)
    plot_gr(gr_xy, 'gr_Hcal_xy_plane','', 0)
    plot_gr(h_xy , 'h_Hcal_xy_plane','' , 2)
    plot_gr(h_xy1, 'h_Hcal_xy1_plane','' ,2)
    plot_gr(h_HoE, 'h_HoE','' , 1)
    plot_gr(h_TotE, 'h_TotE','' , 1)
    print ('Done')
