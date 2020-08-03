import ROOT as rt
import numpy as np
import h5py
import gc
rt.gROOT.SetBatch(rt.kTRUE)

def get_sum_E(df, label):
    H_sum_E = rt.TH1F('H_sum_E_%s'%(label)  , '', 100, 0, 100)
    print(df.shape)
    d = df[:,:,3] # hit energy
    for i in range(d.shape[0]):
        tmp =np.sum(d[i], keepdims=False)
        H_sum_E.Fill(tmp)
    return H_sum_E

def get_hit_xyzE(df, label):
    H_hit_x = rt.TH1F('H_hit_x_%s'%(label)  , '', 40, 1800, 2200)
    H_hit_y = rt.TH1F('H_hit_y_%s'%(label)  , '', 20, -100, 100)
    H_hit_z = rt.TH1F('H_hit_z_%s'%(label)  , '', 20, -100, 100)
    H_hit_E = rt.TH1F('H_hit_E_%s'%(label)  , '', 1000, 1, 1e3)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            H_hit_x.Fill(df[i,j,0])
            H_hit_y.Fill(df[i,j,1])
            H_hit_z.Fill(df[i,j,2])
            H_hit_E.Fill(df[i,j,3]*1000)
    return (H_hit_x, H_hit_y, H_hit_z, H_hit_E)

def do_plot_v1(h_real,h_fake,out_name,title, str_particle):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)

    #h_real.Scale(1/h_real.GetSumOfWeights())
    #h_fake.Scale(1/h_fake.GetSumOfWeights())
    nbin =h_real.GetNbinsX()
    x_min=h_real.GetBinLowEdge(1)
    x_max=h_real.GetBinLowEdge(nbin)+h_real.GetBinWidth(nbin)
    y_min=0
    y_max=1.5*h_real.GetBinContent(h_real.GetMaximumBin())
    if "logx" in out_name:
        canvas.SetLogx()
    if "logy" in out_name:
        canvas.SetLogy()
        y_min = 1e-3
    if "hit_energy" in out_name:
        y_min = 1e-4
        y_max = 1
    elif "prob" in out_name:
        x_min=0.4
        x_max=0.6
    dummy = rt.TH2D("dummy","",nbin,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_Y_title=""
    '''
    if "z_showershape" in out_name:
        dummy_Y_title = "Normalized energy"
        dummy_X_title = "cell Z"
    elif "y_showershape" in out_name:
        dummy_Y_title = "Normalized energy"
        dummy_X_title = "cell Y"
    elif "dep_showershape" in out_name:
        dummy_Y_title = "Normalized energy"
        dummy_X_title = "cell X"
    elif "cell_energy" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "cell energy deposition (MeV)"
    elif "cell_sum_energy" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "#sum hit energy (GeV)"
    elif "diff_sum_energy" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "E_{#sumhit}-E_{true} (GeV)"
    elif "ratio_energy" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "E_{#sumhit}/E_{true}"
    elif "ratio_e3x3" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "E_{3x3}/E_{true}"
    elif "ratio_e5x5" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "E_{5x5}/E_{true}"
    elif "e3x3_energy" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "E_{3x3} (GeV)"
    elif "e5x5_energy" in out_name:
        dummy_Y_title = "Normalized"
        dummy_X_title = "E_{5x5} (GeV)"
    elif "prob" in out_name:
        dummy_Y_title = "Normalized Count"
        dummy_X_title = "Real/Fake"
    '''
    dummy.GetYaxis().SetTitle(title[1])
    dummy.GetXaxis().SetTitle(title[0])
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.5)
    dummy.GetXaxis().SetTitleOffset(1.0)
    #if "cell_energy" not in out_name:
    #    dummy.GetXaxis().SetMoreLogLabels()
    #    dummy.GetXaxis().SetNoExponent()  
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw()
    h_real.SetLineColor(rt.kRed)
    h_fake.SetLineColor(rt.kBlue)
    h_real.SetMarkerColor(rt.kRed)
    h_fake.SetMarkerColor(rt.kBlue)
    h_real.SetMarkerStyle(20)
    h_fake.SetMarkerStyle(21)
    h_real.Draw("same:pe")
    h_fake.Draw("same:histe")
    dummy.Draw("AXISSAME")
    legend = rt.TLegend(0.75,0.7,0.85,0.85)
    legend.AddEntry(h_real,'G4','lep')
    legend.AddEntry(h_fake,"NN",'lep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    #legend.SetFillStyle(0)
    #legend.SetTextFont(42)
    legend.Draw()
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

if __name__ == '__main__':

    plot_path = './plots'


    file_real = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/learn_pdf/gamma/gamma_test.h5'
    d_real = h5py.File(file_real, 'r')
    hit_real  = d_real['Barrel_Hit'][:]
    info_real = d_real['MC_info'   ][:]

    #file_fake = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/learn_pdf//Pred.h5'
    file_fake = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/learn_pdf//pred_0212.h5'
    d_fake = h5py.File(file_fake, 'r')
    hit_fake  = d_fake['Barrel_Hit'][:]
    info_fake = d_fake['MC_info'   ][:]

    h_real_sum_E = get_sum_E(hit_real, 'real')
    h_fake_sum_E = get_sum_E(hit_fake, 'fake')
    h_real_hit_x, h_real_hit_y, h_real_hit_z, h_real_hit_E = get_hit_xyzE(hit_real, 'real') 
    h_fake_hit_x, h_fake_hit_y, h_fake_hit_z, h_fake_hit_E = get_hit_xyzE(hit_fake, 'fake') 


    do_plot_v1(h_real_sum_E, h_fake_sum_E, "Hit_sum_E",['sum E', 'Events'], '')
    do_plot_v1(h_real_hit_x, h_fake_hit_x, "Hit_x"    ,['Hit x', 'Entries'], '')
    do_plot_v1(h_real_hit_y, h_fake_hit_y, "Hit_y"    ,['Hit y', 'Entries'], '')
    do_plot_v1(h_real_hit_z, h_fake_hit_z, "Hit_z"    ,['Hit z', 'Entries'], '')
    do_plot_v1(h_real_hit_E, h_fake_hit_E, "Hit_E_logx"    ,['Hit E', 'Entries'], '')

