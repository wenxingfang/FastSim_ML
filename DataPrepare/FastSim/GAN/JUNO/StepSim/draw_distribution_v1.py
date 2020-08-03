import h5py
import gc
import random
import ROOT as rt
rt.gROOT.SetBatch(rt.kTRUE)
from scipy.stats import anderson, ks_2samp
from sklearn.utils import shuffle
############ #################
## pdf(time or npe | r theta)#
##############################
def add_info(s_content):
    lowX=0.15
    lowY=0.7
    info  = rt.TPaveText(lowX, lowY+0.06, lowX+0.30, lowY+0.16, "NDC")
    info.SetBorderSize(   0 )
    info.SetFillStyle(    0 )
    info.SetTextAlign(   12 )
    info.SetTextColor(    1 )
    info.SetTextSize(0.025)
    info.SetTextFont (   42 )
    info.AddText("%s"%(str(s_content)))
    return info


def do_plot(tag, h_real,h_fake,out_name, do_fit):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    if True:
        h_real.Scale(1/h_real.GetSumOfWeights())
        h_fake.Scale(1/h_fake.GetSumOfWeights())
    nbin =h_real.GetNbinsX()
    x_min=h_real.GetBinLowEdge(1)
    x_max=h_real.GetBinLowEdge(nbin)+h_real.GetBinWidth(nbin)
    if "hit_time" in out_name:
        x_min=0
        x_max=1000
    elif "n_pe" in out_name:
        x_min=0
        x_max=5
    y_min=0
    y_max=1.5*h_real.GetBinContent(h_real.GetMaximumBin())
    if "logx" in out_name:
        canvas.SetLogx()
    if "logy" in out_name:
        canvas.SetLogy()
        y_min = 1e-3
    dummy = rt.TH2D("dummy","",nbin,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_Y_title=""
    dummy_X_title=""
    if "n_pe" in out_name:
        #dummy_Y_title = "Events"
        dummy_Y_title = "Probability"
        dummy_X_title = "N PE"
    elif "hit_time" in out_name:
        dummy_Y_title = "Probability"
        #dummy_Y_title = "Events"
        dummy_X_title = "Hit Time (ns)"
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    dummy.GetXaxis().SetTitle(dummy_X_title)
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw()
    h_real.SetLineWidth(2)
    h_fake.SetLineWidth(2)
    h_real.SetLineColor(rt.kRed)
    h_fake.SetLineColor(rt.kBlue)
    h_real.SetMarkerColor(rt.kRed)
    h_fake.SetMarkerColor(rt.kBlue)
    h_real.SetMarkerStyle(20)
    h_fake.SetMarkerStyle(21)
    h_real.Draw("same:pe")
    h_fake.Draw("same:histe")
    dummy.Draw("AXISSAME")
    if do_fit:
        f1 = rt.TF1("f1","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma((x)+1.)", x_min, x_max)
        f1.SetParameters(1, 100.0) # you MUST set non-zero initial values for parameters
        h_real.Fit("f1", "R") #"R" = fit between "xmin" and "xmax" of the "f1"

    legend = rt.TLegend(0.6,0.7,0.85,0.85)
    legend.AddEntry(h_real,'G4','lep')
    legend.AddEntry(h_fake,"NN",'lep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    #legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()
    label_theta = add_info(str('(r=%s,theta=%s)'%(tag.split('_')[0], tag.split('_')[-1])))
    label_theta.Draw()
    canvas.SaveAs("%s/%s_%s.png"%(plot_path, out_name, tag))
    del canvas
    gc.collect()



def make_hist_0(tag, hd):
    dict_hist = {}
    tag = tag*180 # back to real theta
    for i in range(tag.shape[0]):
        #dict_hist[tag[i]] = rt.TH1F('v0_%s'%str(tag[i]),'',5,0,5)
        dict_hist[tag[i]] = rt.TH1F('v0_%s'%str(tag[i]),'',10,-5,5)
        for j in range(hd.shape[0]):
            #dict_hist[tag[i]].Fill(hd[j,i])
            dict_hist[tag[i]].Fill(round(hd[j,i]))
    return dict_hist

def make_hist_1(tag, hd):
    dict_hist = {}
    for i in range(tag.shape[1]):
        if tag[0,i] in dict_hist: continue
        #dict_hist[tag[0,i]] = rt.TH1F('v1_%s'%str(tag[0,i]),'',5,0,5)
        dict_hist[tag[0,i]] = rt.TH1F('v1_%s'%str(tag[0,i]),'',10,-5,5)
    for i in range(tag.shape[1]):
        for j in range(hd.shape[0]):
            dict_hist[tag[0,i]].Fill(hd[j,i])
    return dict_hist


def make_hist_time_0(tag, hd):
    dict_hist = {}
    tag = tag*180 # back to real theta
    for i in range(tag.shape[0]):
        dict_hist[tag[i]] = rt.TH1F('time_v0_%s'%str(tag[i]),'',51,-10,500)
        for j in range(hd.shape[0]):
            dict_hist[tag[i]].Fill(hd[j,i])
            #dict_hist[tag[i]].Fill(round(hd[j,i]))
    return dict_hist


def make_hist_time_1(tag, hd):
    dict_hist = {}
    for i in range(tag.shape[1]):
        if tag[0,i] in dict_hist: continue
        dict_hist[tag[0,i]] = rt.TH1F('time_v1_%s'%str(tag[0,i]),'',51,-10,500)
    for i in range(tag.shape[1]):
        for j in range(hd.shape[0]):
            if hd[j,i] <= 0: continue # remove 0 
            dict_hist[tag[0,i]].Fill(hd[j,i])
    return dict_hist

def read_hist(file_name, cat, bins, tag, doRound, scale):
    dict_hist = {}
    for ifile in file_name:
        hd = h5py.File(ifile, 'r')
        for i in hd.keys():
            if cat not in i: continue
            h = hd[i][:]/scale
            r_theta = i.split(cat)[-1]
            if r_theta not in dict_hist:
                dict_hist[r_theta] = rt.TH1F('%s_%s'%(str(tag),str(i)),'',bins[0],bins[1],bins[2])
            for j in range(h.shape[0]):
                dict_hist[r_theta].Fill(h[j] if doRound == False else round(h[j]))
        hd.close()
    return dict_hist


def read_hist_v1(file_name, bins, tag, doRound):
    dict_hist = {}
    for ifile in file_name:
        hd  = h5py.File(ifile, 'r')
        npe = hd['nPEByPMT'][:]
        for i in range(npe.shape[0]):
            r_theta = str('%s_%s'%(npe[i,0], npe[i,1]))
            if r_theta not in dict_hist:
                dict_hist[r_theta] = rt.TH1F('%s_%s'%(str(tag),str(r_theta)),'',bins[0],bins[1],bins[2])
            for j in range(2, len(npe[i,:])):
                dict_hist[r_theta].Fill(npe[i,j] if doRound == False else round(npe[i,j]))
        hd.close()
    return dict_hist

plot_path = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/r_theta_comp_plots/'

real_file = []
#real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user-detsim-103_batch0_N1000.h5')
#real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user-detsim-105_batch0_N1000.h5')
#real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user-detsim-107_batch0_N1000.h5')
#real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user-detsim-108_batch0_N1000.h5')
#real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user-detsim-10_batch0_N1000.h5')
#real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user-detsim-111_batch0_N1000.h5')
#real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user-detsim-113_batch0_N1000.h5')
#real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user-detsim-116_batch0_N1000.h5')
#real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_r_theta/user-detsim-119_batch0_N1000.h5')

real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1001_10000.000000_batch0_N1000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1002_10000.000000_batch0_N1000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1003_10000.000000_batch0_N1000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1004_10000.000000_batch0_N1000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1005_10000.000000_batch0_N1000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1006_10000.000000_batch0_N1000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1007_10000.000000_batch0_N1000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1008_10000.000000_batch0_N1000.h5')
real_file.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_fixR_theta/user-detsim-1009_10000.000000_batch0_N1000.h5')



real_hists = read_hist(real_file, 'HitTimeByPMT_', [310, -10, 3000], 'real', False, 1)
#real_hists = read_hist(real_file, 'nPEByPMT_', [10, -5, 5], 'real', True, 1)
#real_hists = read_hist_v1(real_file, [10, -5, 5], 'real', False)

#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1202.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1204mini.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1204mini_v1.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1204mini_v2.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1204mini_v3.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1205.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_npe_1205.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_npe_1205mae.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1206ks.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_npe_1206ks.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_npe_1206ksNoise.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_npe_1207fix.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1207fix.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_npe_1207v1.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1208.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1208_ep40.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1208_ep100.h5']
#fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1208_ne13.h5']
fake_file = ['/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/produced_r_theta_time_1208_ne13ep100.h5']
fake_hists = read_hist(fake_file, 'predHitTimeByPMT_', [310, -10, 3000], 'fake', False, 1)
#fake_hists = read_hist(fake_file, 'prednPEByPMT_', [10, -5, 5], 'fake', True, 1)

count = 0
r_theta_list = list(fake_hists.keys())
#random.shuffle(r_theta_list)
r_theta_list = shuffle(r_theta_list, random_state=1)
for i in r_theta_list:
    if i not in real_hists: continue
    print('r_theta=',i)
    if float(i.split('_')[0])>17.7: continue ### out of CD
    #do_plot (i, real_hists[i], fake_hists[i], 'n_pe_', False)
    #do_plot (i, real_hists[i], fake_hists[i], 'n_pe_logy', False)
    do_plot (i, real_hists[i], fake_hists[i], 'hit_time_', False)
    do_plot (i, real_hists[i], fake_hists[i], 'hit_time_logy', False)
    count = count + 1
    if count > 100: break
print('done!')

