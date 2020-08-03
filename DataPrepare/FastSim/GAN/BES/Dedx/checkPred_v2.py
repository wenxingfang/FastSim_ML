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
## using theta not costheta
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
    legend.AddEntry(h1 ,'Data','lep')
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

if __name__ == '__main__':

    for_barrel = False
    for_endcap = False

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
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_156_180_mse_k600.h5'

    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_barrel.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_endcap_2.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_ep_barrel.h5'
    #file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_ep_endcap.h5'
    file_in = []
    '''
    file_in.append('/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/muon/Pred/Pred_mu+.h5')
    const_scale = 600 
    const_shift = 550
    const_mom_scale = 2
    '''
    '''
    file_in.append('/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/pion/Pred/Pred_pi+.h5')
    const_scale = 600 
    const_shift = 550
    const_mom_scale = 1
    '''
    '''
    file_in.append('/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/kaon/Pred/Pred_k+.h5')
    const_scale = 900 
    const_shift = 750
    const_mom_scale = 1
    '''
    '''
    #file_in.append('/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/proton/Pred/Pred_p+.h5')
    #file_in.append('/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/proton/Pred/Pred_p+_IndexEpoch.h5')
    file_in.append('/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/proton/Pred/Pred_p+_sh.h5')
    const_scale = 1000 
    const_shift = 1000
    const_mom_scale = 1
    '''
    file_in.append('/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_twoSPv1.h5')
    #file_in.append('/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_v1.h5')
    #file_in.append('/hpcfs/juno/junogpu/fangwx/FastSim/BES/Dedx/electron/Pred/Pred_em_200K.h5')
    const_scale = 228#3*32 
    const_shift = 638#546
    const_mom_scale = 2
    df = None
    First = True
    for i in  file_in:
        hf = h5py.File(i, 'r')
        if First : 
            df = hf['Pred'][:]
            First = False
        else: df = np.concatenate((df, hf['Pred'][:]), axis=0)
        hf.close()
      
    df[:,0] = df[:,0]*const_mom_scale
    df[:,1] = df[:,1]*180
    if for_barrel:
        df = df [ np.logical_and(df[:,1] < 145, df[:,1] > 35), : ]
    if for_endcap:
        df = df [ np.logical_or(df[:,1] > 145, df[:,1] < 35), : ]
    #df = df [ np.logical_and(df[:,1] > 140, df[:,1] < 155), : ]
    print('total=',df.shape[0])
    h_real_dedx0        = rt.TH1F('H_real_dedx0'        , '', 600, -3, 3)
    h_fake_dedx0        = rt.TH1F('H_fake_dedx0'        , '', 600, -3, 3)
    h_real_dedx1        = rt.TH1F('H_real_dedx1'        , '', 1000, 0, 3000)
    h_fake_dedx1        = rt.TH1F('H_fake_dedx1'        , '', 1000, 0, 3000)
    h_real_dedx_p      = rt.TH2F('H_real_dedx_p'      , '', 1000, 0, 3000, 250, 0, 2.5)
    h_real_dedx_theta   = rt.TH2F('H_real_dedx_theta'   , '', 1000, 0, 3000, 180, 0, 180)
    h_fake_dedx_p      = rt.TH2F('H_fake_dedx_p'      , '', 1000, 0, 3000, 250, 0, 2.5)
    h_fake_dedx_theta   = rt.TH2F('H_fake_dedx_theta'   , '', 1000, 0, 3000, 180, 0, 180)
    p_bin    = np.arange(0,3,0.5)
    theta_bin = np.arange(0,180,10)
    h_real_p_list = []
    h_fake_p_list = []
    h_real_theta_list = []
    h_fake_theta_list = []
    for i in range(len(p_bin)-1):
        h_real_p_list.append(rt.TH1F('H__p_%.3f_%.3f__real_dedx1'%(p_bin[i], p_bin[i+1])                    , '', 300, 0, 3000))
        h_fake_p_list.append(rt.TH1F('H__p_%.3f_%.3f__fake_dedx1'%(p_bin[i], p_bin[i+1])                    , '', 300, 0, 3000))
    for i in range(len(theta_bin)-1):
        h_real_theta_list.append(rt.TH1F('H__theta_%.3f_%.3f__real_dedx1'%(theta_bin[i], theta_bin[i+1])        , '', 300, 0, 3000))
        h_fake_theta_list.append(rt.TH1F('H__theta_%.3f_%.3f__fake_dedx1'%(theta_bin[i], theta_bin[i+1])        , '', 300, 0, 3000))


    for i in range(df.shape[0]):
        h_real_dedx0.Fill(df[i,2])
        h_fake_dedx0.Fill(df[i,3])
        h_real_dedx1.Fill(df[i,2]*const_scale + const_shift)
        h_fake_dedx1.Fill(df[i,3]*const_scale + const_shift)
    
    for i in range(len(p_bin)-1):
        new_df = df [ np.logical_and(df[:,0] > p_bin[i] , df[:,0] < p_bin[i+1]) , : ]
        print('new_df shape=', new_df.shape)
        for j in range(new_df.shape[0]):
            h_real_p_list[i].Fill(new_df[j,2]*const_scale + const_shift)
            h_fake_p_list[i].Fill(new_df[j,3]*const_scale + const_shift)

    for i in range(len(theta_bin)-1):
        new_df = df [ np.logical_and(df[:,1] > theta_bin[i] , df[:,1] < theta_bin[i+1]) , : ]
        print('new_df shape=', new_df.shape)
        for j in range(new_df.shape[0]):
            h_real_theta_list[i].Fill(new_df[j,2]*const_scale + const_shift)
            h_fake_theta_list[i].Fill(new_df[j,3]*const_scale + const_shift)

    for i in range(df.shape[0]):
        h_real_dedx_p.Fill(df[i,2]*const_scale + const_shift, df[i,0])
        h_fake_dedx_p.Fill(df[i,3]*const_scale + const_shift, df[i,0])
        h_real_dedx_theta.Fill(df[i,2]*const_scale + const_shift, df[i,1])
        h_fake_dedx_theta.Fill(df[i,3]*const_scale + const_shift, df[i,1])

    plot_hist(hist=h_real_dedx_p,out_name="h_real_dedx_p",title={'X':'dedx','Y':'P (GeV)'})
    plot_hist(hist=h_fake_dedx_p,out_name="h_fake_dedx_p",title={'X':'dedx','Y':'P (GeV)'})
    plot_hist(hist=h_real_dedx_theta,out_name="h_real_dedx_theta",title={'X':'dedx','Y':'#theta (degree)'})
    plot_hist(hist=h_fake_dedx_theta,out_name="h_fake_dedx_theta",title={'X':'dedx','Y':'#theta (degree)'})
    do_plot_g2(h1=h_real_dedx0, h2=h_fake_dedx0, out_name="dedx0"     , title={"X":'dedx0',"Y":'Entries'}, plot_label="")
    do_plot_g2(h1=h_real_dedx1, h2=h_fake_dedx1, out_name="dedx1"     , title={"X":'dedx',"Y":'Entries'} , plot_label="")
    do_plot_g2(h1=h_real_dedx1, h2=h_fake_dedx1, out_name="dedx1_logy", title={"X":'dedx',"Y":'Entries'} , plot_label="")
    for i in range(len(h_fake_p_list)):
        str1 = h_real_p_list[i].GetName().split('__')[1]
        do_plot_g2(h1=h_real_p_list[i], h2=h_fake_p_list[i], out_name=(h_real_p_list[i].GetName()).replace('real',''), title={"X":'dedx',"Y":'Entries'}     , plot_label="%s"%(str1))
        do_plot_g2(h1=h_real_p_list[i], h2=h_fake_p_list[i], out_name=(h_real_p_list[i].GetName()).replace('real','logy'), title={"X":'dedx',"Y":'Entries'} , plot_label="%s"%(str1))
    for i in range(len(h_fake_theta_list)):
        str1 = h_real_theta_list[i].GetName().split('__')[1]
        str1 = str1.replace('theta','#theta')
        do_plot_g2(h1=h_real_theta_list[i], h2=h_fake_theta_list[i], out_name=(h_real_theta_list[i].GetName()).replace('real',''), title={"X":'dedx',"Y":'Entries'}     , plot_label="%s"%(str1))
        do_plot_g2(h1=h_real_theta_list[i], h2=h_fake_theta_list[i], out_name=(h_real_theta_list[i].GetName()).replace('real','logy'), title={"X":'dedx',"Y":'Entries'} , plot_label="%s"%(str1))
    print ('Done')
