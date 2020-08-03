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

    test_percent = 0.5
    for_em = True
    for_ep = False

    plot_path = './plots/'
    f_in = rt.TFile("/besfs/groups/cal/dedx/zhuk/calib/663/26577-27090/Simulation/hadron_track/electron/electron.root","READ")
    tree = f_in.Get('n103')
    print('entries=',tree.GetEntries())
    h_pt        = rt.TH1F('H_pt'        , '', 220, 0, 2.2)
    h_pt0       = rt.TH1F('H_pt0'       , '', 220, 0, 2.2)
    h_charge    = rt.TH1F('H_charge'    , '', 20 , -2, 2)
    h_costheta  = rt.TH1F('H_costheta'  , '', 20 , -2, 2)
    h_theta     = rt.TH1F('H_theta'     , '', 200 , 0, 200)
    h_dEdx_meas = rt.TH1F('H_dEdx_meas' , '', 901,-1, 900)
    h_dedx_theta= rt.TH2F('H_dedx_theta', '', 900, 0, 900, 200 , 0, 200)
    h_dedx_pt   = rt.TH2F('H_dedx_pt'   , '', 900, 0, 900, 220 , 0, 2.2)
    maxEvent = tree.GetEntries()
    Data = np.full((maxEvent, 3), 0 ,dtype=np.float32)#init 
    for i in range(maxEvent):
        tree.GetEntry(i)
        ptrk   = getattr(tree, 'ptrk')
        charge = getattr(tree, 'charge')
        costheta = getattr(tree, 'costheta')
        dEdx_meas = getattr(tree, 'dEdx_meas')
        if for_em and charge != -1: continue
        if for_ep and charge != 1 : continue
        Data[i,0] = ptrk/2.0 
        #Data[i,1] = charge 
        Data[i,1] = costheta 
        Data[i,2] = (dEdx_meas - 546)/(3*32)
        h_pt       .Fill(ptrk)
        h_pt0      .Fill(math.sqrt(ptrk))
        h_charge   .Fill(charge)
        h_costheta .Fill(costheta)
        tmp_theta = math.acos(costheta)*180/math.pi
        h_theta    .Fill(tmp_theta)
        h_dEdx_meas.Fill(dEdx_meas)
        h_dedx_theta.Fill(dEdx_meas, tmp_theta)
        h_dedx_pt   .Fill(dEdx_meas, ptrk)
    if True:
        dele_list = []
        for i in range(Data.shape[0]):
            if Data[i,0]==0:
                dele_list.append(i) ## remove the empty event 
        Data = np.delete(Data, dele_list, axis = 0)
    print('final size=', Data.shape[0])        
    plot_gr(h_pt       , "h_pt_track" ,"")
    plot_gr(h_pt0      , "h_pt_track0","")
    plot_gr(h_charge   , "h_charge"   ,"")
    plot_gr(h_costheta , "h_costheta" ,"")
    plot_gr(h_theta    , "h_theta"    ,"")
    plot_gr(h_dEdx_meas, "h_dEdx_meas","")
    plot_gr(h_dEdx_meas, "h_dEdx_meas_logy","")
    plot_hist(h_dedx_theta, "h_dedx_theta","")
    plot_hist(h_dedx_pt   , "h_dedx_pt"   ,"")
    Data = shuffle(Data)
    all_evt = Data.shape[0]
    training_data = Data[0:int((1-test_percent)*all_evt)      ,:] 
    test_data     = Data[int((1-test_percent)*all_evt):all_evt,:] 
    theta_range0 = 35
    theta_range1 = 145
    training_data_barrel = training_data [ np.logical_and( np.arccos(training_data[:,1])*180/math.pi < 145, np.arccos(training_data[:,1])*180/math.pi > 35), : ]
    test_data_barrel     = test_data     [ np.logical_and( np.arccos(test_data    [:,1])*180/math.pi < 145, np.arccos(test_data    [:,1])*180/math.pi > 35), : ]
    training_data_endcap = training_data [ np.logical_or ( np.arccos(training_data[:,1])*180/math.pi > 145, np.arccos(training_data[:,1])*180/math.pi < 35), : ]
    test_data_endcap     = test_data     [ np.logical_or ( np.arccos(test_data    [:,1])*180/math.pi > 145, np.arccos(test_data    [:,1])*180/math.pi < 35), : ]
    hf = h5py.File('electron_train_barrel.h5', 'w')
    hf.create_dataset('dataset', data=training_data_barrel)
    print ('training_data_barrel shape=',training_data_barrel.shape)
    hf.close()
    hf = h5py.File('electron_train_endcap.h5', 'w')
    hf.create_dataset('dataset', data=training_data_endcap)
    print ('training_data_endcap shape=',training_data_endcap.shape)
    hf.close()
    hf = h5py.File('electron_test_barrel.h5', 'w')
    hf.create_dataset('dataset' , data=test_data_barrel)
    print ('test_data_barrel shape=',test_data_barrel.shape)
    hf.close()
    hf = h5py.File('electron_test_endcap.h5', 'w')
    hf.create_dataset('dataset' , data=test_data_endcap)
    print ('test_data_endcap shape=',test_data_endcap.shape)
    hf.close()
    print ('Done')
