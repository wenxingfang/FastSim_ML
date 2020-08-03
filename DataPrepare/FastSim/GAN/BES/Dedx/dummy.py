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




def plot_gr(event,gr,out_name,title):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    #gr.GetXaxis().SetTitle("#phi(AU, 0 #rightarrow 2#pi)")
    #gr.GetYaxis().SetTitle("Z(AU) (-19.5 #rightarrow 19.5 m)")
    #gr.SetTitle(title)
    gr.Draw("pcol")
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

    maxEvent = 1000000
    
    Data = np.full((maxEvent, 3), 0 ,dtype=np.float32)#init 
    #shape = np.random.uniform(2,3 ,maxEvent)
    #scale = np.random.uniform(1,2 ,maxEvent)
    shape = np.repeat(3, maxEvent)
    scale = np.repeat(2, maxEvent)
    s     = np.random.gamma(shape, scale)
    Data[:,0]=shape
    Data[:,1]=scale
    Data[:,2]=s
    ''' 
    h_Hit_B_x_z = rt.TH2F('Hit_B_x_z' , '', 31, 0, 31 , 29, 0, 29)
    plot_hist(h_Hit_B_x_z ,'%s_Hit_barrel_x_z_plane'%(parse_args.tag)      , '%s (2-20 GeV)'%(str_e))
    if True:
        dele_list = []
        for i in range(MC_info.shape[0]):
            if MC_info[i][0]==0:
                dele_list.append(i) ## remove the empty event 
        MC_info         = np.delete(MC_info        , dele_list, axis = 0)
        Barrel_Hit      = np.delete(Barrel_Hit     , dele_list, axis = 0)
        Barrel_Hit_HCAL = np.delete(Barrel_Hit_HCAL, dele_list, axis = 0)
    print('final size=', MC_info.shape[0])        
    ''' 
    hf = h5py.File('DummyGammaSingle.h5', 'w')
    hf.create_dataset('Gamma'     , data=Data)
    hf.close()
    print ('Done')
