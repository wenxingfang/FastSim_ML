import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import argparse
rt.gROOT.SetBatch(rt.kTRUE)
from sklearn.utils import shuffle
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
#######################################
## try to use cluster to remove background in proton and kaon
## using hit dedx 
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
    parser.add_argument('--max_event', action='store', type=int, default=-1, help='max event ')
    parser.add_argument('--p_scale', action='store', type=float, default=1, help='mom scale ')
    parser.add_argument('--dedx_scale', action='store', type=float, default=1, help='dedx scale ')
    parser.add_argument('--dedx_shift', action='store', type=float, default=0, help='dedx shift ')


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
    p_scale = parser.p_scale
    dedx_scale = parser.dedx_scale
    dedx_shift = parser.dedx_shift
    thetaMax = parser.thetaMax
    max_event = parser.max_event
    for_neg = True if '-' in parser.particle else False
    for_pos = True if '+' in parser.particle else False
    regions = parser.region
    #theta_range0 = 35
    #theta_range1 = 145
    plot_path = './plots/'
    f_in = rt.TFile(parser.input,"READ")
    tree = f_in.Get('n103')
    print('for_neg=',for_neg, ',for_pos=',for_pos, ',test_percent=',test_percent, ',regions=',regions, ',input=',parser.input)
    print('entries=',tree.GetEntries())
    h_p         = rt.TH1F('H_p '        , '', 220, 0, 2.2)
    h_charge    = rt.TH1F('H_charge'    , '', 20 , -2, 2)
    h_costheta  = rt.TH1F('H_costheta'  , '', 20 , -2, 2)
    h_theta     = rt.TH1F('H_theta'     , '', 200 , 0, 200)
    h_dEdx_hit  = rt.TH1F('H_dEdx_hit'  , '', 1000, 0, 3000)
    h_dedx_theta= rt.TH2F('H_dedx_theta', '', 1000, 0, 3000, 180, 0, 180)
    h_dedx_p    = rt.TH2F('H_dedx_p'    , '', 1000, 0, 3000, 250, 0, 2.5)
    h_dedx_mean_p = rt.TH2F('H_dedx_mean_p' , '', 1000, 0, 3000, 250, 0, 2.5)
    h_dedx_std_p  = rt.TH2F('H_dedx_std_p'  , '', 100 , 0, 1000, 250, 0, 2.5)
    entries = tree.GetEntries()
    maxEvent = entries*40 if max_event == -1 else max_event
    Data = np.full((maxEvent, 3), 0 ,dtype=np.float32)#init 
    Data_Cluster = np.full((entries, 2), 0 ,dtype=np.float32)#init 
    Nd = 0
    for i in range(entries):
        tree.GetEntry(i)
        ptrk   = getattr(tree, 'ptrk')
        charge = getattr(tree, 'charge')
        costheta = getattr(tree, 'costheta')
        dEdx_meas = getattr(tree, 'dEdx_meas')
        ndedxhit = getattr(tree, 'ndedxhit')
        dEdx_hit = getattr(tree, 'dEdx_hit')
        prob     = getattr(tree, 'prob')
    #    if len(prob) != 5:continue ##pid failed
        if for_neg and charge != -1: continue
        if for_pos and charge != 1 : continue
        if  math.acos(costheta)*180/math.pi < thetaMin or math.acos(costheta)*180/math.pi > thetaMax: continue
        total_dedx = 0
        if(len(dEdx_hit)==0): continue
        for n in range(len(dEdx_hit)):
            tmp_dedx = dEdx_hit[n]
            if tmp_dedx <= 0:continue
            total_dedx += tmp_dedx
            Data[Nd,0] = ptrk/p_scale
            Data[Nd,1] = costheta
            #Data[Nd,2] = (tmp_dedx - 638)/(228)
            Data[Nd,2] = (tmp_dedx - dedx_shift)/(dedx_scale)
            h_p        .Fill(ptrk)
            h_charge   .Fill(charge)
            h_costheta .Fill(costheta)
            tmp_theta = math.acos(costheta)*180/math.pi
            h_theta    .Fill(tmp_theta)
            h_dEdx_hit  .Fill(tmp_dedx)
            h_dedx_theta.Fill(tmp_dedx, tmp_theta)
            h_dedx_p    .Fill(tmp_dedx, ptrk)
            Nd += 1
            if Nd >= Data.shape[0]: break
        dedx_mean = total_dedx/len(dEdx_hit)
        if dedx_mean < 0 : 
            print('dedx_mean=',dedx_mean,',len(dEdx_hit)=',len(dEdx_hit))
            continue
        h_dedx_mean_p.Fill(dedx_mean, ptrk)
        #Data_Cluster[i,0] = ptrk  
        #Data_Cluster[i,1] = dedx_mean  
        Data_Cluster[i,0] = ptrk/2  
        Data_Cluster[i,1] = (dedx_mean-dedx_shift)/dedx_scale  
        sum_w2 = 0
        for n in range(len(dEdx_hit)):
            tmp_dedx = dEdx_hit[n]
            sum_w2 += (tmp_dedx-dedx_mean)*(tmp_dedx-dedx_mean)
        dedx_std = math.sqrt(sum_w2/len(dEdx_hit))
        h_dedx_std_p.Fill(dedx_std, ptrk)
        if Nd >= Data.shape[0]: break
    if True:
        dele_list = []
        for i in range(Data.shape[0]):
            if Data[i,0]==0:
                dele_list.append(i) ## remove the empty event 
        Data = np.delete(Data, dele_list, axis = 0)

        dele_list = []
        for i in range(Data_Cluster.shape[0]):
            if Data_Cluster[i,0]==0:
                dele_list.append(i) ## remove the empty event 
        Data_Cluster = np.delete(Data_Cluster, dele_list, axis = 0)
    print('final Data size=', Data.shape[0])        
    print('final Data_Cluster size=', Data_Cluster.shape[0])        
    ############
    #aa = np.random.rand(100,2)
    #y_pred = DBSCAN().fit_predict(aa)
    #y_pred = DBSCAN(eps = 0.04, min_samples = 30).fit_predict(Data_Cluster)
    #eps = 0.04
    #min_samples = 27
    eps = 0.03
    min_samples = 15
    y_pred = DBSCAN(eps = eps, min_samples = min_samples).fit_predict(Data_Cluster)
    print('y_pred shape=', y_pred.shape,',min=',np.min(y_pred),',max=',np.max(y_pred))        
    plt.scatter(Data_Cluster[:, 1], Data_Cluster[:, 0], s=0.1 ,c=y_pred+2)
    plt.xlim(-2, 10)
    plt.ylim(0, 1)
    plt.savefig("y_pred_eps%s_mins%s.png"%(str(eps),str(min_samples)))

    grs = {}
    #print(gr_pred)
    #print('y_pred shape=', y_pred.shape,',min=',np.min(y_pred),',max=',np.max(y_pred))        
    for i in range(np.min(y_pred), np.max(y_pred)+1):
        grs[i]=rt.TGraph()
    for i in range(y_pred.shape[0]):
        grs[y_pred[i]].SetPoint(grs[y_pred[i]].GetN(),Data_Cluster[i,1], Data_Cluster[i,0]) 
    clus_dict ={}
    for i in grs:
        clus_dict[i]=grs[i].GetN()
    print( sorted(clus_dict.items(), key=lambda item: item[1]))
    ############


    print('final size=', Data.shape[0])        
    plot_gr(h_p        , "h_p_track_%s"      %parser.tag ,{'X':'P (GeV)','Y':'Entries'})
    plot_gr(h_charge   , "h_charge_%s"        %parser.tag ,{'X':'charge','Y':'Entries'})
    plot_gr(h_costheta , "h_costheta_%s"      %parser.tag ,{'X':'cos#theta','Y':'Entries'})
    plot_gr(h_theta    , "h_theta_%s"         %parser.tag ,{'X':'#theta (degree)','Y':'Entries'})
    plot_gr(h_dEdx_hit, "h_dEdx_hit_%s"     %parser.tag ,{'X':'dEdx_hit','Y':'Entries'})
    plot_gr(h_dEdx_hit, "h_dEdx_hit_logy_%s"%parser.tag ,{'X':'dEdx_hit','Y':'Entries'})
    plot_hist(h_dedx_theta, "h_dedx_theta_%s" %parser.tag ,{'X':'dEdx_hit','Y':'#theta (degree)'})
    plot_hist(h_dedx_p    , "h_dedx_p_%s"    %parser.tag ,{'X':'dEdx_hit','Y':'P (GeV)'})
    plot_hist(h_dedx_mean_p, "h_dedx_mean_p_%s"    %parser.tag ,{'X':'dEdx_mean','Y':'P (GeV)'})
    plot_hist(h_dedx_std_p , "h_dedx_std_p_%s"    %parser.tag ,{'X':'dEdx_std','Y':'P (GeV)'})
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
