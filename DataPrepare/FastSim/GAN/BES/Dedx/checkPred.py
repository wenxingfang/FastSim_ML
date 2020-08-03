import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import argparse
rt.gROOT.SetBatch(rt.kTRUE)
from sklearn.utils import shuffle
#######################################
# use digi step data and use B field  ##
# use cell ID for ECAL                ##
# add HCAL
# add HoE cut
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
    y_max=1.5*h1.GetBinContent(h1.GetMaximumBin()) 
    if "logx" in out_name:
        canvas.SetLogx()
    if "logy" in out_name:
        canvas.SetLogy()
        y_min = 1e-1
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
    x_l = 0.7
    x_dl = 0.2
    y_h = 0.85
    y_dh = 0.15
    legend = rt.TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    legend.AddEntry(h1 ,'G4','lep')
    legend.AddEntry(h2 ,"NN",'lep')
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
    label.Draw() 
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()


def plot_gr(gr,out_name,title):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    if 'logy' in out_name:
        canvas.SetLogy()
    #gr.GetXaxis().SetTitle("#phi(AU, 0 #rightarrow 2#pi)")
    #gr.GetYaxis().SetTitle("Z(AU) (-19.5 #rightarrow 19.5 m)")
    #gr.SetTitle(title)
    #gr.Draw("pcol")
    gr.Draw("hist")
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

    test_percent = 0.2
    for_barrel = False
    for_endcap = True

    plot_path = './plots_pred/'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred_em_v1_0520.h5' 
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_endcap_2.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_endcap_2_D_1e-4.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_156_180.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_ep_barrel.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_ep_endcap.h5'

    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_barrel_k500_D1e-4.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_ep_barrel_k500_D1e-4.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_145_156.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_156_180.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_156_180_mse.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_156_180_mse_k400.h5'
    file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_156_180_mse_k600.h5'

    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_barrel.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_endcap_2.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_ep_barrel.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_ep_endcap.h5'

    hf = h5py.File(file_in, 'r')
    df = hf['Pred'][:]
    df[:,0] = df[:,0]*2
    df[:,1] = np.arccos(df[:,1])*180/math.pi
    if for_barrel:
        df = df [ np.logical_and(df[:,1] < 145, df[:,1] > 35), : ]
    if for_endcap:
        df = df [ np.logical_or(df[:,1] > 145, df[:,1] < 35), : ]
    #df = df [ np.logical_and(df[:,1] > 140, df[:,1] < 155), : ]
    hf.close()
    print('total=',df.shape[0])
    h_real_dedx0        = rt.TH1F('H_real_dedx0'        , '', 400, -2, 2)
    h_fake_dedx0        = rt.TH1F('H_fake_dedx0'        , '', 400, -2, 2)
    h_real_dedx1        = rt.TH1F('H_real_dedx1'        , '', 900, 0, 900)
    h_fake_dedx1        = rt.TH1F('H_fake_dedx1'        , '', 900, 0, 900)
    pt_bin    = np.arange(0,3,0.5)
    theta_bin = np.arange(0,180,10)
    h_real_pt_list = []
    h_fake_pt_list = []
    h_real_theta_list = []
    h_fake_theta_list = []
    for i in range(len(pt_bin)-1):
        h_real_pt_list.append(rt.TH1F('H__pt_%.3f_%.3f__real_dedx1'%(pt_bin[i], pt_bin[i+1])        , '', 900, 0, 900))
        h_fake_pt_list.append(rt.TH1F('H__pt_%.3f_%.3f__fake_dedx1'%(pt_bin[i], pt_bin[i+1])        , '', 900, 0, 900))
    for i in range(len(theta_bin)-1):
        h_real_theta_list.append(rt.TH1F('H__theta_%.3f_%.3f__real_dedx1'%(theta_bin[i], theta_bin[i+1])        , '', 900, 0, 900))
        h_fake_theta_list.append(rt.TH1F('H__theta_%.3f_%.3f__fake_dedx1'%(theta_bin[i], theta_bin[i+1])        , '', 900, 0, 900))


    for i in range(df.shape[0]):
        h_real_dedx0.Fill(df[i,2])
        h_fake_dedx0.Fill(df[i,3])
        h_real_dedx1.Fill(df[i,2]*3*32 + 546)
        h_fake_dedx1.Fill(df[i,3]*3*32 + 546)
    
    for i in range(len(pt_bin)-1):
        new_df = df [ np.logical_and(df[:,0] > pt_bin[i] , df[:,0] < pt_bin[i+1]) , : ]
        print('new_df shape=', new_df.shape)
        for j in range(new_df.shape[0]):
            h_real_pt_list[i].Fill(new_df[j,2]*3*32 + 546)
            h_fake_pt_list[i].Fill(new_df[j,3]*3*32 + 546)

    for i in range(len(theta_bin)-1):
        new_df = df [ np.logical_and(df[:,1] > theta_bin[i] , df[:,1] < theta_bin[i+1]) , : ]
        print('new_df shape=', new_df.shape)
        for j in range(new_df.shape[0]):
            h_real_theta_list[i].Fill(new_df[j,2]*3*32 + 546)
            h_fake_theta_list[i].Fill(new_df[j,3]*3*32 + 546)

    do_plot_g2(h1=h_real_dedx0, h2=h_fake_dedx0, out_name="dedx0"     , title={"X":'dedx0',"Y":'Entries'}, plot_label="")
    do_plot_g2(h1=h_real_dedx1, h2=h_fake_dedx1, out_name="dedx1"     , title={"X":'dedx',"Y":'Entries'} , plot_label="")
    do_plot_g2(h1=h_real_dedx1, h2=h_fake_dedx1, out_name="dedx1_logy", title={"X":'dedx',"Y":'Entries'} , plot_label="")
    for i in range(len(h_fake_pt_list)):
        str1 = h_real_pt_list[i].GetName().split('__')[1]
        do_plot_g2(h1=h_real_pt_list[i], h2=h_fake_pt_list[i], out_name=(h_real_pt_list[i].GetName()).replace('real',''), title={"X":'dedx',"Y":'Entries'}     , plot_label="%s"%(str1))
        do_plot_g2(h1=h_real_pt_list[i], h2=h_fake_pt_list[i], out_name=(h_real_pt_list[i].GetName()).replace('real','logy'), title={"X":'dedx',"Y":'Entries'} , plot_label="%s"%(str1))
    for i in range(len(h_fake_theta_list)):
        str1 = h_real_theta_list[i].GetName().split('__')[1]
        str1 = str1.replace('theta','#theta')
        do_plot_g2(h1=h_real_theta_list[i], h2=h_fake_theta_list[i], out_name=(h_real_theta_list[i].GetName()).replace('real',''), title={"X":'dedx',"Y":'Entries'}     , plot_label="%s"%(str1))
        do_plot_g2(h1=h_real_theta_list[i], h2=h_fake_theta_list[i], out_name=(h_real_theta_list[i].GetName()).replace('real','logy'), title={"X":'dedx',"Y":'Entries'} , plot_label="%s"%(str1))
    print ('Done')
