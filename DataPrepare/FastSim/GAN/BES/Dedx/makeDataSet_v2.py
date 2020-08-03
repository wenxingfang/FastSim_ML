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
## add theta range select 
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
    parser.add_argument('--particle', action='store', type=str,
                        help='e-, e+, ...')
    parser.add_argument('--region', nargs='+', help='<Required> Set flag', required=False)
    parser.add_argument('--percentage', action='store', type=float, default=0.5, help='test dataset percentage')
    parser.add_argument('--thetaMin', action='store', type=float, default=0, help='min theta ')
    parser.add_argument('--thetaMax', action='store', type=float, default=180, help='max theta ')


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
    gr.GetXaxis().SetTitle(title['X'])
    gr.GetYaxis().SetTitle(title['Y'])
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
    hist.GetXaxis().SetTitle(title['X'])
    hist.GetYaxis().SetTitle(title['Y'])
    #hist.SetTitleSize(0.1)
    #hist.Draw("COLZ TEXT")
    hist.Draw("COLZ")
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

if __name__ == '__main__':
    
    g_parser = get_parser()
    parser = g_parser.parse_args()
    test_percent = parser.percentage
    thetaMin = parser.thetaMin
    thetaMax = parser.thetaMax
    for_em = True if parser.particle == 'e-' else False
    for_ep = True if parser.particle == 'e+' else False
    regions = parser.region
    theta_range0 = 35
    theta_range1 = 145
    plot_path = './plots/'
    f_in = rt.TFile(parser.input,"READ")
    tree = f_in.Get('n103')
    print('for_em=',for_em, ',for_ep=',for_ep, ',test_percent=',test_percent, ',regions=',regions, ',input=',parser.input)
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
        #tmp_region = "-1"
        #if math.acos(costheta)*180/math.pi > theta_range0 and math.acos(costheta)*180/math.pi < theta_range1 : tmp_region = '1' 
        #elif math.acos(costheta)*180/math.pi < theta_range0: tmp_region = '0' 
        #elif math.acos(costheta)*180/math.pi > theta_range0: tmp_region = '2'
        #if tmp_region not in regions: continue
        if  math.acos(costheta)*180/math.pi < thetaMin or math.acos(costheta)*180/math.pi > thetaMax: continue
        Data[i,0] = ptrk/2.0
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
    plot_gr(h_pt       , "h_pt_track_%s"      %parser.tag ,{'X':'Pt (GeV)','Y':'Entries'})
    plot_gr(h_pt0      , "h_pt_track0_%s"     %parser.tag ,{'X':'Pt (GeV)','Y':'Entries'})
    plot_gr(h_charge   , "h_charge_%s"        %parser.tag ,{'X':'charge','Y':'Entries'})
    plot_gr(h_costheta , "h_costheta_%s"      %parser.tag ,{'X':'cos#theta','Y':'Entries'})
    plot_gr(h_theta    , "h_theta_%s"         %parser.tag ,{'X':'#theta (degree)','Y':'Entries'})
    plot_gr(h_dEdx_meas, "h_dEdx_meas_%s"     %parser.tag ,{'X':'dEdx_meas','Y':'Entries'})
    plot_gr(h_dEdx_meas, "h_dEdx_meas_logy_%s"%parser.tag ,{'X':'dEdx_meas','Y':'Entries'})
    plot_hist(h_dedx_theta, "h_dedx_theta_%s" %parser.tag ,{'X':'dEdx_meas','Y':'#theta (degree)'})
    plot_hist(h_dedx_pt   , "h_dedx_pt_%s"    %parser.tag ,{'X':'dEdx_meas','Y':'Pt (GeV)'})
    Data = shuffle(Data)
    all_evt = Data.shape[0]
    training_data = Data[0:int((1-test_percent)*all_evt)      ,:] 
    test_data     = Data[int((1-test_percent)*all_evt):all_evt,:] 
    hf = h5py.File('%s'%(parser.output).replace('.h5','_train.h5'), 'w')
    hf.create_dataset('dataset', data=training_data)
    print ('%s shape='%hf.filename, hf['dataset'].shape)
    hf.close()
    hf = h5py.File('%s'%(parser.output).replace('.h5','_test.h5'), 'w')
    hf.create_dataset('dataset' , data=test_data)
    print ('%s shape='%hf.filename, hf['dataset'].shape)
    hf.close()
    print ('Done')
