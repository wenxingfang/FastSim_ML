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
    hist.GetXaxis().SetTitle(title['X'])
    hist.GetYaxis().SetTitle(title['Y'])
    #hist.Draw("COLZ TEXT")
    hist.Draw("COLZ")
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()


def Ana(df, tag, option): 
    df[:,0] = df[:,0]*2
    df[:,1] = np.arccos(df[:,1])*180/math.pi
    if for_barrel:
        df = df [ np.logical_and(df[:,1] < 145, df[:,1] > 35), : ]
    if for_endcap:
        df = df [ np.logical_or(df[:,1] > 145, df[:,1] < 35), : ]
    #df = df [ np.logical_and(df[:,1] > 140, df[:,1] < 155), : ]
    #print('total=',df.shape[0])
    h_real_dedx0        = rt.TH1F('%s_H_real_dedx0'      %tag  , '', 600, -3, 3)
    h_fake_dedx0        = rt.TH1F('%s_H_fake_dedx0'      %tag  , '', 600, -3, 3)
    h_real_dedx1        = rt.TH1F('%s_H_real_dedx1'      %tag  , '', 1000, 0, 2000)
    h_fake_dedx1        = rt.TH1F('%s_H_fake_dedx1'      %tag  , '', 1000, 0, 2000)
    h_real_dedx_pt      = rt.TH2F('%s_H_real_dedx_pt'    %tag  , '', 1000, 0, 2000, 250, 0, 2.5)
    h_real_dedx_theta   = rt.TH2F('%s_H_real_dedx_theta' %tag  , '', 1000, 0, 2000, 180, 0, 180)
    h_fake_dedx_pt      = rt.TH2F('%s_H_fake_dedx_pt'    %tag  , '', 1000, 0, 2000, 250, 0, 2.5)
    h_fake_dedx_theta   = rt.TH2F('%s_H_fake_dedx_theta' %tag  , '', 1000, 0, 2000, 180, 0, 180)

    for i in range(df.shape[0]):
        h_real_dedx0.Fill(df[i,2])
        h_fake_dedx0.Fill(df[i,3])
        h_real_dedx1.Fill(df[i,2]*const_scale + const_shift)
        h_fake_dedx1.Fill(df[i,3]*const_scale + const_shift)
    #do_plot_g2(h1=h_real_dedx0, h2=h_fake_dedx0, out_name="%s_dedx0"            %tag,title={"X":'dedx0',"Y":'Entries'}, plot_label="")
    do_plot_g2(h1=h_real_dedx1, h2=h_fake_dedx1, out_name="%s_dedx1"            %tag,title={"X":'dedx',"Y":'Entries'} , plot_label="")
    do_plot_g2(h1=h_real_dedx1, h2=h_fake_dedx1, out_name="%s_dedx1_logy"       %tag,title={"X":'dedx',"Y":'Entries'} , plot_label="")
    if option == 1: return
    for i in range(df.shape[0]):
        h_real_dedx_pt.Fill(df[i,2]*const_scale + const_shift, df[i,0])
        h_fake_dedx_pt.Fill(df[i,3]*const_scale + const_shift, df[i,0])
        h_real_dedx_theta.Fill(df[i,2]*const_scale + const_shift, df[i,1])
        h_fake_dedx_theta.Fill(df[i,3]*const_scale + const_shift, df[i,1])
    plot_hist(hist=h_real_dedx_pt               ,out_name="%s_h_real_dedx_pt"   %tag,title={'X':'dedx','Y':'Pt (GeV)'})
    plot_hist(hist=h_fake_dedx_pt               ,out_name="%s_h_fake_dedx_pt"   %tag,title={'X':'dedx','Y':'Pt (GeV)'})
    plot_hist(hist=h_real_dedx_theta            ,out_name="%s_h_real_dedx_theta"%tag,title={'X':'dedx','Y':'#theta (degree)'})
    plot_hist(hist=h_fake_dedx_theta            ,out_name="%s_h_fake_dedx_theta"%tag,title={'X':'dedx','Y':'#theta (degree)'})
    if option == 2: return
    
    pt_bin    = np.arange(0,3,0.5)
    theta_bin = np.arange(0,180,10)
    h_real_pt_list = []
    h_fake_pt_list = []
    h_real_theta_list = []
    h_fake_theta_list = []
    for i in range(len(pt_bin)-1):
        h_real_pt_list.append(rt.TH1F('%s_H__pt_%.3f_%.3f__real_dedx1'%(tag, pt_bin[i], pt_bin[i+1])                    , '', 200, 0, 2000))
        h_fake_pt_list.append(rt.TH1F('%s_H__pt_%.3f_%.3f__fake_dedx1'%(tag, pt_bin[i], pt_bin[i+1])                    , '', 200, 0, 2000))
    for i in range(len(theta_bin)-1):
        h_real_theta_list.append(rt.TH1F('%s_H__theta_%.3f_%.3f__real_dedx1'%(tag, theta_bin[i], theta_bin[i+1])        , '', 200, 0, 2000))
        h_fake_theta_list.append(rt.TH1F('%s_H__theta_%.3f_%.3f__fake_dedx1'%(tag, theta_bin[i], theta_bin[i+1])        , '', 200, 0, 2000))

    for i in range(len(pt_bin)-1):
        new_df = df [ np.logical_and(df[:,0] > pt_bin[i] , df[:,0] < pt_bin[i+1]) , : ]
        print('new_df shape=', new_df.shape)
        for j in range(new_df.shape[0]):
            h_real_pt_list[i].Fill(new_df[j,2]*const_scale + const_shift)
            h_fake_pt_list[i].Fill(new_df[j,3]*const_scale + const_shift)

    for i in range(len(theta_bin)-1):
        new_df = df [ np.logical_and(df[:,1] > theta_bin[i] , df[:,1] < theta_bin[i+1]) , : ]
        print('new_df shape=', new_df.shape)
        for j in range(new_df.shape[0]):
            h_real_theta_list[i].Fill(new_df[j,2]*const_scale + const_shift)
            h_fake_theta_list[i].Fill(new_df[j,3]*const_scale + const_shift)

    for i in range(len(h_fake_pt_list)):
        str1 = h_real_pt_list[i].GetName().split('__')[1]
        do_plot_g2(h1=h_real_pt_list[i], h2=h_fake_pt_list[i], out_name=(h_real_pt_list[i].GetName()).replace('real',''), title={"X":'dedx',"Y":'Entries'}     , plot_label="%s"%(str1))
        do_plot_g2(h1=h_real_pt_list[i], h2=h_fake_pt_list[i], out_name=(h_real_pt_list[i].GetName()).replace('real','logy'), title={"X":'dedx',"Y":'Entries'} , plot_label="%s"%(str1))
    for i in range(len(h_fake_theta_list)):
        str1 = h_real_theta_list[i].GetName().split('__')[1]
        str1 = str1.replace('theta','#theta')
        do_plot_g2(h1=h_real_theta_list[i], h2=h_fake_theta_list[i], out_name=(h_real_theta_list[i].GetName()).replace('real',''), title={"X":'dedx',"Y":'Entries'}     , plot_label="%s"%(str1))
        do_plot_g2(h1=h_real_theta_list[i], h2=h_fake_theta_list[i], out_name=(h_real_theta_list[i].GetName()).replace('real','logy'), title={"X":'dedx',"Y":'Entries'} , plot_label="%s"%(str1))

if __name__ == '__main__':

    test_percent = 0.2
    for_barrel = False
    for_endcap = False

    plot_path = './plots_pred_gan/'
    const_scale = 228#3*32 
    const_shift = 638#546
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_gan.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_gan_0705.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_gan_gsize20.h5'
    file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_gan_gsize20v1.h5'
    df_dict = {}
    hf = h5py.File(file_in, 'r')
    for key in hf.keys():
        print('key=',key)
        df_dict[key] = hf[key][:]
    hf.close()
    for i in df_dict:
        epoch = float(i.split('epoch')[-1])
        if epoch < 40: continue
        Ana(df_dict[i], i, 1) 
        
    print ('Done')
