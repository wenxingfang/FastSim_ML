import h5py
import gc
import ROOT as rt
rt.gROOT.SetBatch(rt.kTRUE)
from scipy.stats import anderson, ks_2samp

def add_info(s_name, s_content):
    lowX=0.15
    lowY=0.7
    info  = rt.TPaveText(lowX, lowY+0.06, lowX+0.30, lowY+0.16, "NDC")
    info.SetBorderSize(   0 )
    info.SetFillStyle(    0 )
    info.SetTextAlign(   12 )
    info.SetTextColor(    1 )
    info.SetTextSize(0.025)
    info.SetTextFont (   42 )
    info.AddText("%s%s"%(str(s_name), str(s_content)))
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
        x_min=50
        x_max=400
    elif "n_pe" in out_name:
        x_min=0
        x_max=5
    y_min=0
    y_max=1.5*h_real.GetBinContent(h_real.GetMaximumBin())
    if "logx" in out_name:
        canvas.SetLogx()
    if "logy" in out_name:
        canvas.SetLogy()
        y_min = 1e-1
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
        dummy_X_title = "First Hit Time (ns)"
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
    label_theta = add_info('theta=', tag)
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

plot_path = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/comp_plots/'

real_file = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/data/Laser_theta/laser_batch9_N5000.h5' 
#fake_file = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/Learn_pdf.h5'
#fake_file = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/Learn_time_pdf_1202.h5'
#fake_file = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/Learn_time_pdf_1202_sh.h5'
#fake_file = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/Learn_time_pdf_1203.h5'
fake_file = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/Learn_time_pdf_1204.h5'

fake_hd = h5py.File(fake_file,'r')
#fake_hist = make_hist_0(fake_hd['theta_set'][:], fake_hd['pred_n_pe'][:])
fake_hist = make_hist_time_0(fake_hd['theta_set'][:], fake_hd['pred_hit_time'][:])
#print('fake_hist=',fake_hist)
real_hd = h5py.File(real_file,'r')
#real_hist = make_hist_1(real_hd['infoPMT'][:], real_hd['nPEByPMT'][:])
real_hist = make_hist_time_1(real_hd['infoPMT'][:], real_hd['firstHitTimeByPMT'][:])

#print('real_hist=',real_hist)
#real_hist = make_hist_2(real_hd['infoPMT'][:], real_hd['firstHitTimeByPMT'][:])
#fake_hist = real_hist
for i in real_hist:
    if i not in fake_hist: continue
    #do_plot(i, real_hist[i], fake_hist[i], 'n_pe_', False)
    do_plot(i, real_hist[i], fake_hist[i], 'hit_time_', False)
print('done!')

