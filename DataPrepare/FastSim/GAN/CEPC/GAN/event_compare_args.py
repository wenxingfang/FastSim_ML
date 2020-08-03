import ROOT as rt
import numpy as np
import h5py 
#import pandas as pd
import sys 
import gc
import argparse
rt.gROOT.SetBatch(rt.kTRUE)
#rt.gStyle.SetMaxDigitsY(4)
rt.TGaxis.SetMaxDigits(4);

def readInput(datafile):
    print('reading input')
    df1 = None
    df2 = None
    df3 = None
    First = True
    f_DataSet = open(datafile, 'r')
    for line in f_DataSet:
        idata = line.strip('\n')
        idata = idata.strip(' ')
        if "#" in idata: continue ##skip the commented one
        d = h5py.File(str(idata), 'r')
        if First:
            df1 = d['Barrel_Hit'][:]
            df2 = d['Barrel_Hit_HCAL'][:]
            df3 = d['MC_info'][:]
            First = False
        else:
            df1 = np.concatenate ((df1, d['Barrel_Hit'][:])     , axis=0)
            df2 = np.concatenate ((df2, d['Barrel_Hit_HCAL'][:]), axis=0)
            df3 = np.concatenate ((df3, d['MC_info'][:])        , axis=0)
        d.close()
    f_DataSet.close()
    print('df1=',df1.shape)
    return df1, df2, df3



class Obj:
    def __init__(self, name, fileName, is_real, evt_start, evt_end):
        self.name = name
        self.is_real = is_real
        self.fileName = fileName
        self.nB   = None
        self.nB_H = None
        self.info = None
        
        if '.txt' in self.fileName:
            self.nB, self.nB_H, self.info = readInput(self.fileName)
        else:
            hf = h5py.File(self.fileName, 'r')
            self.nB   = hf['Barrel_Hit']     [:]
            #self.nB_H = hf['Barrel_Hit_HCAL'][:]
            self.nB_H = hf['Barrel_Hit'][:]
            self.info = hf['MC_info'   ]     [:]
            hf.close()
        if evt_end != 0: 
            if (evt_start < self.nB.shape[0]) and (evt_end < self.nB.shape[0]):
                self.nB   = self.nB   [evt_start:evt_end]
                self.nB_H = self.nB_H [evt_start:evt_end]
                self.info = self.info [evt_start:evt_end]
            else:
                print('wrong !!')
        self.nEvt = self.nB.shape[0]
        self.nRow = self.nB.shape[1]
        self.nCol = self.nB.shape[2]
        self.nDep = self.nB.shape[3]
        self.nRow_H = self.nB_H.shape[1]
        self.nCol_H = self.nB_H.shape[2]
        self.nDep_H = self.nB_H.shape[3]
        
    def produce_z_sp(self):## produce showershape in z direction
        str1 = "" if self.is_real else "_gen"
        H_z_sp = rt.TH1F('H_z_sp_%s'%(str1)  , '', self.nCol, 0, self.nCol)
        for i in range(self.nEvt):
            for j in range(0, self.nCol):
                H_z_sp.Fill(j+0.01, np.sum(self.nB[i,:,j,:]))
        return H_z_sp

    def produce_y_sp(self):## produce showershape in y direction
        str1 = "" if self.is_real else "_gen"
        H_y_sp = rt.TH1F('H_y_sp_%s'%(str1)  , '', self.nRow, 0, self.nRow)
        for i in range(self.nEvt):
            for j in range(0, self.nRow):
                H_y_sp.Fill(j+0.01, np.sum(self.nB[i,j,:,:]))
        return H_y_sp
         
    def produce_dep_sp(self):## produce showershape in dep direction
        str1 = "" if self.is_real else "_gen"
        H_dep_sp = rt.TH1F('H_dep_sp_%s'%(str1)  , '', self.nDep, 0, self.nDep)
        for i in range(self.nEvt):
            for j in range(0, self.nDep):
                H_dep_sp.Fill(j+0.01, np.sum(self.nB[i,:,:,j]))
        return H_dep_sp

    def produce_H_z_sp(self):## produce showershape in z direction
        str1 = "" if self.is_real else "_gen"
        H_z_sp = rt.TH1F('H_H_z_sp_%s'%(str1)  , '', self.nRow_H, 0, self.nRow_H)
        for i in range(self.nEvt):
            for j in range(0, self.nRow_H):
                H_z_sp.Fill(j+0.01, np.sum(self.nB_H[i,j,:,:]))
        return H_z_sp

    def produce_H_phi_sp(self):## produce showershape in z direction
        str1 = "" if self.is_real else "_gen"
        H_phi_sp = rt.TH1F('H_H_phi_sp_%s'%(str1)  , '', self.nCol_H, 0, self.nCol_H)
        for i in range(self.nEvt):
            for j in range(0, self.nCol_H):
                H_phi_sp.Fill(j+0.01, np.sum(self.nB_H[i,:,j,:]))
        return H_phi_sp

    def produce_H_r_sp(self):## produce showershape in z direction
        str1 = "" if self.is_real else "_gen"
        H_r_sp = rt.TH1F('H_H_r_sp_%s'%(str1)  , '', self.nDep_H, 0, self.nDep_H)
        for i in range(self.nEvt):
            for j in range(0, self.nDep_H):
                H_r_sp.Fill(j+0.01, np.sum(self.nB_H[i,:,:,j]))
        return H_r_sp

    def produce_cell_energy(self):## 
        str1 = "" if self.is_real else "_gen"
        H_cell_E = rt.TH1F('H_cell_E_%s'%(str1)  , '', 1000, 1, 10e3)
        for i in range(self.nEvt):
            for j in range(0, self.nRow):
                for k in range(0, self.nCol):
                    for z in range(0, self.nDep):
                        H_cell_E.Fill(self.nB[i,j,k,z]*1000)# to MeV
        return H_cell_E

    def produce_cell_sum_energy(self, part):## 
        str1 = "" if self.is_real else "_gen"
        H_cell_sum_E = 0
        if part == 0:
            H_cell_sum_E = rt.TH1F('H_Ecal_cell_sum_E_%s'%(str1)  , '', 20, 0, 20)
        elif part == 1:
            H_cell_sum_E = rt.TH1F('H_Hcal_cell_sum_E_%s'%(str1)  , '', 20, 0, 20)
        elif part == 2:
            H_cell_sum_E = rt.TH1F('H_EHcal_cell_sum_E_%s'%(str1)  , '', 22, 0, 22)
        hit_sum = np.sum(self.nB, axis=(1,2,3), keepdims=False)
        for i in range(self.nEvt):
            if part == 0:
                H_cell_sum_E.Fill( np.sum(self.nB[i,:,:,:]) )#  GeV
            elif part == 1:
                H_cell_sum_E.Fill( np.sum(self.nB_H[i,:,:,:]) )#  GeV
            elif part == 2:
                H_cell_sum_E.Fill( np.sum(self.nB[i,:,:,:]) + np.sum(self.nB_H[i,:,:,:]) )#  GeV
        return H_cell_sum_E

    def produce_e3x3_energy(self):## 
        str1 = "" if self.is_real else "_gen"
        H_e3x3_E = rt.TH1F('H_e3x3_E_%s'%(str1)  , '', 100, 0, 100)
        for i in range(self.nEvt):
            result = self.nB[i,14:17,14:17,:]
            H_e3x3_E.Fill(np.sum(result))#  GeV
        return H_e3x3_E

    def produce_e5x5_energy(self):## 
        str1 = "" if self.is_real else "_gen"
        H_e5x5_E = rt.TH1F('H_e5x5_E_%s'%(str1)  , '', 100, 0, 100)
        for i in range(self.nEvt):
            result = self.nB[i,13:18,13:18,:]
            H_e5x5_E.Fill(np.sum(result))#  GeV
        return H_e5x5_E

    def produce_e3x3_ratio(self):## 
        str1 = "" if self.is_real else "_gen"
        H_e3x3_ratio = rt.TH1F('H_e3x3_ratio_%s'%(str1)  , '', 100, 0, 1)
        for i in range(self.nEvt):
            result = self.nB[i,14:17,14:17,:]
            H_e3x3_ratio.Fill(np.sum(result)/self.info[i,0])
        return H_e3x3_ratio

    def produce_e5x5_ratio(self):## 
        str1 = "" if self.is_real else "_gen"
        H_e5x5_ratio = rt.TH1F('H_e5x5_ratio_%s'%(str1)  , '', 100, 0, 1)
        for i in range(self.nEvt):
            result = self.nB[i,13:18,13:18,:]
            H_e5x5_ratio.Fill(np.sum(result)/self.info[i,0])
        return H_e5x5_ratio

    def produce_ennergy_diff(self):## 
        str1 = "" if self.is_real else "_gen"
        H_diff_sum_E = rt.TH1F('H_diff_sum_E_%s'%(str1)  , '', 100, -50, 50)
        hit_sum = np.sum(self.nB, axis=(1,2,3), keepdims=False)
        for i in range(self.nEvt):
            H_diff_sum_E.Fill(hit_sum[i]-self.info[i,0])#  GeV
        return H_diff_sum_E

    def produce_ennergy_ratio(self, part):## 
        str1 = "" if self.is_real else "_gen"
        H_ratio_sum_E = 0
        if part == 0 :
            H_ratio_sum_E = rt.TH1F('H_Ecal_ratio_sum_E_%s'%(str1)  , '', 100, 0.0, 1.0)
        elif part == 1 :
            H_ratio_sum_E = rt.TH1F('H_Hcal_ratio_sum_E_%s'%(str1)  , '', 100, 0.0, 1.0)
        elif part == 2 :
            H_ratio_sum_E = rt.TH1F('H_EHcal_ratio_sum_E_%s'%(str1)  , '', 100, 0.0, 1.0)
        for i in range(self.nEvt):
            if part == 0:
                H_ratio_sum_E.Fill( np.sum(self.nB[i,:,:,:])/self.info[i,0] )#  GeV
            elif part == 1:
                H_ratio_sum_E.Fill( np.sum(self.nB_H[i,:,:,:])/self.info[i,0] )#  GeV
            elif part == 2:
                H_ratio_sum_E.Fill( (np.sum(self.nB[i,:,:,:]) + np.sum(self.nB_H[i,:,:,:]))/self.info[i,0] )#  GeV
        return H_ratio_sum_E

    def produce_HoE(self):## 
        str1 = "" if self.is_real else "_gen"
        H_HoE = rt.TH1F('H_HoE_%s'%(str1)  , '', 100, 0.0, 100.0)
        for i in range(self.nEvt):
                H_HoE.Fill( np.sum(self.nB_H[i,:,:,:])/np.sum(self.nB[i,:,:,:]))#  GeV
        return H_HoE

    def produce_TotE(self):## 
        str1 = "" if self.is_real else "_gen"
        H_TotE = rt.TH1F('H_TotE_%s'%(str1)  , '', 20, 0.0, 20.0)
        for i in range(self.nEvt):
                H_TotE.Fill( np.sum(self.nB_H[i,:,:,:])+np.sum(self.nB[i,:,:,:]))#  GeV
        return H_TotE

    def produce_prob(self, data, label, evt_start, evt_end):## produce discriminator prob
        str1 = "" if self.is_real else "_gen"
        hf = h5py.File(data, 'r')
        da = hf[label][:]
        if evt_end !=0 :
            da = hf[label][evt_start:evt_end]
        H_prob = rt.TH1F('H_prob_%s'%(str1)  , '', 120, -0.1, 1.1)
        for i in range(da.shape[0]):
            H_prob.Fill(da[i]) 
        return H_prob
        


def mc_info(particle, theta_mom, phi_mom, energy):
    lowX=0.1
    lowY=0.8
    info  = rt.TPaveText(lowX, lowY+0.06, lowX+0.30, lowY+0.16, "NDC")
    info.SetBorderSize(   0 )
    info.SetFillStyle(    0 )
    info.SetTextAlign(   12 )
    info.SetTextColor(    1 )
    info.SetTextSize(0.04)
    info.SetTextFont (   42 )
    info.AddText("%s (#theta=%.1f, #phi=%.1f, E=%.1f GeV)"%(particle, theta_mom, phi_mom, energy))
    return info

def layer_info(layer):
    lowX=0.85
    lowY=0.8
    info  = rt.TPaveText(lowX, lowY+0.06, lowX+0.30, lowY+0.16, "NDC")
    info.SetBorderSize(   0 )
    info.SetFillStyle(    0 )
    info.SetTextAlign(   12 )
    info.SetTextColor(    1 )
    info.SetTextSize(0.04)
    info.SetTextFont (   42 )
    info.AddText("%s"%(layer))
    return info

def do_plot(event,hist,out_name,title, str_particle):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    #h_corr.Draw("COLZ")
    #h_corr.LabelsDeflate("X")
    #h_corr.LabelsDeflate("Y")
    #h_corr.LabelsOption("v")
    hist.SetStats(rt.kFALSE)
    if "Barrel_z_y" in out_name:
        hist.GetYaxis().SetTitle("cell Y")
        hist.GetXaxis().SetTitle("cell Z")
    elif "Barrel_z_dep" in out_name:
        hist.GetYaxis().SetTitle("cell X")
        hist.GetXaxis().SetTitle("cell Z")
    elif "Barrel_y_dep" in out_name:
        hist.GetYaxis().SetTitle("cell X")
        hist.GetXaxis().SetTitle("cell Y")
    hist.SetTitle(title)
    #hist.SetTitleSize(0.1)
    #hist.Draw("COLZ TEXT")
    hist.Draw("COLZ")
    if n3 is not None:
        Info = mc_info(str_particle, n3[event][0], n3[event][1], n3[event][2])
        Info.Draw()
    if 'layer' in out_name:
        str_layer = out_name.split('_')[3] 
        l_info = layer_info(str_layer)
        l_info.Draw()
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

def do_plot_v1(h_real,h_fake,out_name,tag, str_particle):
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
        y_min = 1
        y_max = 4e5
    if "cell_energy" in out_name:
        #y_min = 1e-4
        #y_max = 1
        y_min = 1e-1
        y_max = 1e7
    #elif "prob" in out_name:
    #    x_min=0.4
    #    x_max=0.6
    dummy = rt.TH2D("dummy","",nbin,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy_Y_title=""
    dummy_X_title = ""
    if "z_showershape" in out_name:
        dummy_Y_title = "Energy (GeV)"
        dummy_X_title = "cell Z"
    elif "y_showershape" in out_name:
        dummy_Y_title = "Energy (GeV)"
        dummy_X_title = "cell Y"
    elif "dep_showershape" in out_name:
        dummy_Y_title = "Energy (GeV)"
        dummy_X_title = "cell X"
    elif "r_showershape" in out_name:
        dummy_Y_title = "Energy (GeV)"
        dummy_X_title = "cell R"
    elif "phi_showershape" in out_name:
        dummy_Y_title = "Energy (GeV)"
        dummy_X_title = "cell #phi"
    elif "cell_energy" in out_name:
        dummy_Y_title = "Hits"
        dummy_X_title = "Energy deposit per Hit (MeV)"
    elif "cell_sum_energy" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "#sum hit energy (GeV)"
    elif "diff_sum_energy" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{#sumhit}-E_{true} (GeV)"
    elif "ratio_energy" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{#sumhit}/E_{true}"
    elif "ratio_e3x3" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{3x3}/E_{true}"
    elif "ratio_e5x5" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{5x5}/E_{true}"
    elif "e3x3_energy" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{3x3} (GeV)"
    elif "e5x5_energy" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "E_{5x5} (GeV)"
    elif "prob" in out_name:
        dummy_Y_title = "Events"
        dummy_X_title = "Real/Fake"
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    str_part = ''
    if 'EHcal' in out_name:
        pass
    elif 'Ecal' in out_name:
        str_part = 'Ecal '
    elif 'Hcal' in out_name:
        str_part = 'Hcal '
    dummy.GetXaxis().SetTitle(str_part + dummy_X_title)
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetMaxDigits(4)
    if "cell_energy" not in out_name:
        dummy.GetXaxis().SetMoreLogLabels()
        dummy.GetXaxis().SetNoExponent()  
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
    legend = rt.TLegend(0.6,0.7,0.85,0.85)
    legend.AddEntry(h_real,'G4','lep')
    legend.AddEntry(h_fake,"GAN",'lep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    #legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.05)
    legend.Draw()
    canvas.SaveAs("%s/%s_%s.png"%(plot_path,out_name,tag))
    del canvas
    gc.collect()


def get_parser():
    parser = argparse.ArgumentParser(
        description='Run',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--event', action='store', type=int, default=0,
                        help='Number of epochs to train for.')

    parser.add_argument('--real_file'    , action='store', type=str, default='',  help='')
    parser.add_argument('--fake_file'    , action='store', type=str, default='',  help='')
    parser.add_argument('--tag'          , action='store', type=str, default='',  help='')
    parser.add_argument('--particle'     , action='store', type=str, default='',  help='')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    parse_args = parser.parse_args()
    data_real  = parse_args.real_file
    data_fake  = parse_args.fake_file
    N_event    = parse_args.event
    tag        = parse_args.tag
    particle   = parse_args.particle


    plot_path='plot_comparision'
    
    str_particle = particle
    #str_particle = 'e^{-}'
    #real_file = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/data/e-/em_10.h5'
    #fake_file = '/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/generator/Gen_em_1104_epoch15.h5'
    
    real = Obj('real', data_real, True , 0, N_event)
    fake = Obj('real', data_fake, False, 0, N_event)

    '''
    real_HoE = real.produce_HoE()
    print('sum=',real_HoE.GetSumOfWeights())
    do_plot_v1(real_HoE, real_HoE,'HoE',tag, str_particle)
    real_TotE = real.produce_TotE()
    do_plot_v1(real_TotE, real_TotE,'TotE',tag, str_particle)
    '''

    real_h_z_ps = real.produce_z_sp()
    fake_h_z_ps = fake.produce_z_sp()
    do_plot_v1(real_h_z_ps, fake_h_z_ps,'Ecal_z_showershape',tag, str_particle)
    do_plot_v1(real_h_z_ps, fake_h_z_ps,'Ecal_z_showershape_logy',tag, str_particle)
    real_h_y_ps = real.produce_y_sp()
    fake_h_y_ps = fake.produce_y_sp()
    do_plot_v1(real_h_y_ps, fake_h_y_ps,'Ecal_y_showershape',tag, str_particle)
    do_plot_v1(real_h_y_ps, fake_h_y_ps,'Ecal_y_showershape_logy',tag, str_particle)
    real_h_dep_ps = real.produce_dep_sp()
    fake_h_dep_ps = fake.produce_dep_sp()
    do_plot_v1(real_h_dep_ps, fake_h_dep_ps,'Ecal_dep_showershape',tag, str_particle)
    do_plot_v1(real_h_dep_ps, fake_h_dep_ps,'Ecal_dep_showershape_logy',tag, str_particle)

    #real_h_cell_E = real.produce_cell_energy()
    #fake_h_cell_E = fake.produce_cell_energy()
    #do_plot_v1(real_h_cell_E, fake_h_cell_E,'cell_energy_logxlogy',tag, str_particle)
    real_h_Ecal_cell_sum_E = real.produce_cell_sum_energy(0)
    fake_h_Ecal_cell_sum_E = fake.produce_cell_sum_energy(0)
    do_plot_v1(real_h_Ecal_cell_sum_E, fake_h_Ecal_cell_sum_E,'Ecal_cell_sum_energy',tag, str_particle)

    real_h_diff_sum_E = real.produce_ennergy_diff()
    fake_h_diff_sum_E = fake.produce_ennergy_diff()
    do_plot_v1(real_h_diff_sum_E, fake_h_diff_sum_E,'diff_sum_energy',tag, str_particle)
    
    real_Ecal_ratio = real.produce_ennergy_ratio(0)
    fake_Ecal_ratio = fake.produce_ennergy_ratio(0)
    do_plot_v1(real_Ecal_ratio, fake_Ecal_ratio,'Ecal_ratio_energy',tag, str_particle)
    
    real_e3x3_ratio = real.produce_e3x3_ratio()
    fake_e3x3_ratio = fake.produce_e3x3_ratio()
    do_plot_v1(real_e3x3_ratio, fake_e3x3_ratio,'ratio_e3x3',tag, str_particle)
    real_e5x5_ratio = real.produce_e5x5_ratio()
    fake_e5x5_ratio = fake.produce_e5x5_ratio()
    do_plot_v1(real_e5x5_ratio, fake_e5x5_ratio,'ratio_e5x5',tag, str_particle)
    
    real_e3x3 = real.produce_e3x3_energy()
    fake_e3x3 = fake.produce_e3x3_energy()
    do_plot_v1(real_e3x3, fake_e3x3,'e3x3_energy',tag, str_particle)
    real_e5x5 = real.produce_e5x5_energy()
    fake_e5x5 = fake.produce_e5x5_energy()
    do_plot_v1(real_e5x5, fake_e5x5,'e5x5_energy',tag, str_particle)
    ''' 
    real_h_prob = real.produce_prob(data_fake, 'Disc_real', 0, N_event)
    fake_h_prob = fake.produce_prob(data_fake, 'Disc_fake', 0, N_event)
    do_plot_v1(real_h_prob, fake_h_prob,'prob',tag, str_particle)
    ''' 

    '''
    real_h_H_z_ps = real.produce_H_z_sp()
    fake_h_H_z_ps = fake.produce_H_z_sp()
    do_plot_v1(real_h_H_z_ps, fake_h_H_z_ps,'Hcal_z_showershape',tag, str_particle)
    real_h_H_phi_ps = real.produce_H_phi_sp()
    fake_h_H_phi_ps = fake.produce_H_phi_sp()
    do_plot_v1(real_h_H_phi_ps, fake_h_H_phi_ps,'Hcal_phi_showershape',tag, str_particle)
    real_h_H_r_ps = real.produce_H_r_sp()
    fake_h_H_r_ps = fake.produce_H_r_sp()
    do_plot_v1(real_h_H_r_ps, fake_h_H_r_ps,'Hcal_r_showershape',tag, str_particle)


    real_h_Hcal_cell_sum_E = real.produce_cell_sum_energy(1)
    fake_h_Hcal_cell_sum_E = fake.produce_cell_sum_energy(1)
    do_plot_v1(real_h_Hcal_cell_sum_E, fake_h_Hcal_cell_sum_E,'Hcal_cell_sum_energy',tag, str_particle)
    real_h_EHcal_cell_sum_E = real.produce_cell_sum_energy(2)
    fake_h_EHcal_cell_sum_E = fake.produce_cell_sum_energy(2)
    do_plot_v1(real_h_EHcal_cell_sum_E, fake_h_EHcal_cell_sum_E,'EHcal_cell_sum_energy',tag, str_particle)
    
    real_Hcal_ratio = real.produce_ennergy_ratio(1)
    fake_Hcal_ratio = fake.produce_ennergy_ratio(1)
    do_plot_v1(real_Hcal_ratio, fake_Hcal_ratio,'Hcal_ratio_energy',tag, str_particle)
    real_EHcal_ratio = real.produce_ennergy_ratio(2)
    fake_EHcal_ratio = fake.produce_ennergy_ratio(2)
    do_plot_v1(real_EHcal_ratio, fake_EHcal_ratio,'EHcal_ratio_energy',tag, str_particle)
    '''
