import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import json
from array import array
rt.gROOT.SetBatch(rt.kTRUE)





def getPhi(x, y):
    if x == 0 and y == 0: return 0
    elif x == 0 and y > 0: return 90
    elif x == 0 and y < 0: return 270
    phi = math.atan(y/x)
    phi = 180*phi/math.pi
    if x < 0 : phi = phi + 180
    elif x > 0 and y < 0 : phi = phi + 360
    return phi

def getTheta(r, z):
    if z == 0: return 90
    phi = math.atan(r/z)
    phi = 180*phi/math.pi
    if phi < 0 : phi = phi + 180
    return phi

def getID(x, y, z, lookup):
    tmp_ID = 0
    id_z = int(z/10)
    id_z = str(id_z)
    id_phi = int(getPhi(x, y))
    id_phi = str(id_phi)
    if id_z not in lookup:
        print('exception id_z=', id_z)
        return tmp_ID
    min_distance = 999
    for ID in lookup[id_z][id_phi]:
        c_x = float(lookup[id_z][id_phi][ID][0])
        c_y = float(lookup[id_z][id_phi][ID][1])
        c_z = float(lookup[id_z][id_phi][ID][2])
        distance = math.sqrt( math.pow(x-c_x,2) + math.pow(y-c_y,2) + math.pow(z-c_z,2) )
        if  distance < min_distance :
            min_distance = distance
            tmp_ID = ID
    return int(tmp_ID) 


def do_plot(hist,out_name,title):
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
    ystr = ''
    if 'GeV/c^{2}' in title['X']:
        ystr = 'GeV/c^{2}'
    elif 'GeV/c' in title['X']:
        ystr = 'GeV/c'
    elif 'GeV' in title['X']:
        ystr = 'GeV'
    hist.GetYaxis().SetTitle(str(title['Y']+' / %.1f %s'%(hist.GetBinWidth(1), ystr)))
    hist.GetXaxis().SetTitle(title['X'])
    hist.SetLineColor(rt.kBlue)
    hist.SetMarkerColor(rt.kBlue)
    hist.SetMarkerStyle(20)
    hist.Draw("histep")
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()


def do_plot_h3(h_real, h_real1, h_fake, out_name, title, str_particle):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)

#    h_real.Scale(1/h_real.GetSumOfWeights())
#    h_fake.Scale(1/h_fake.GetSumOfWeights())
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
    if "cell_energy" in out_name:
        y_min = 1e-4
        y_max = 1
    elif "prob" in out_name:
        x_min=0.4
        x_max=0.6
    dummy = rt.TH2D("dummy","",nbin,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    ystr = ''
    if 'GeV/c^{2}' in title['X']:
        ystr = 'GeV/c^{2}'
    elif 'GeV/c' in title['X']:
        ystr = 'GeV/c'
    elif 'GeV' in title['X']:
        ystr = 'GeV'
    dummy.GetYaxis().SetTitle(str(title['Y']+' / %.1f %s'%(h_real.GetBinWidth(1), ystr)))
    dummy.GetXaxis().SetTitle(title['X'])
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.5)
    dummy.GetXaxis().SetTitleOffset(1.5)
    if "cell_energy" not in out_name:
        dummy.GetXaxis().SetMoreLogLabels()
        dummy.GetXaxis().SetNoExponent()  
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw()
    h_real .SetLineColor(rt.kRed)
    h_real1.SetLineColor(rt.kGreen)
    h_fake .SetLineColor(rt.kBlue)
    h_real .SetMarkerColor(rt.kRed)
    h_real1.SetMarkerColor(rt.kGreen)
    h_fake .SetMarkerColor(rt.kBlue)
    h_real .SetMarkerStyle(20)
    h_real1.SetMarkerStyle(22)
    h_fake .SetMarkerStyle(21)
    h_real.Draw("same:pe")
    h_real.Draw("same:pe")
    h_fake.Draw("same:histe")
    dummy.Draw("AXISSAME")
    legend = rt.TLegend(0.75,0.7,0.85,0.85)
    legend.AddEntry(h_real ,'G4','lep')
    legend.AddEntry(h_real1,'G4 (w/o EB Hits)','lep')
    legend.AddEntry(h_fake ,"GAN",'lep')
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

def do_plot_h2(h_real, h_fake, out_name, title):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.12)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    if do_Normalize:
        #h_real.Scale(1/h_real.GetSumOfWeights())
        #h_fake.Scale(1/h_fake.GetSumOfWeights())
        h_fake.Scale(h_real.GetSumOfWeights()/h_fake.GetSumOfWeights())
    nbin =h_real.GetNbinsX()
    x_min=h_real.GetBinLowEdge(1)
    x_max=h_real.GetBinLowEdge(nbin)+h_real.GetBinWidth(nbin)
    y_min=0
    y_max=1.5*h_real.GetBinContent(h_real.GetMaximumBin())
    if "logx" in out_name:
        canvas.SetLogx()
    if "logy" in out_name:
        canvas.SetLogy()
        y_min = 1e-1
    if "cell_energy" in out_name:
        y_min = 1e-4
        y_max = 1
    elif "prob" in out_name:
        x_min=0.4
        x_max=0.6
    elif "e3x3" in out_name or 'e5x5' in out_name or 'shower_e' in out_name or 'cluster_e' in out_name:
        x_min=1
        x_max=2
    elif "shower_over_P" in out_name:
        x_min=0.8
        x_max=1.2
    elif "cluster2ndMoment" in out_name:
        x_min=0
        x_max=60
    elif "secondMoment" in out_name:
        x_min=0
        x_max=40
    elif "a20Moment" in out_name:
        x_min=0.8
        x_max=1
    dummy = rt.TH2D("dummy","",nbin,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    ystr = ''
    if 'GeV/c^{2}' in title['Y']:
        ystr = 'GeV/c^{2}'
    elif 'GeV/c' in title['Y']:
        ystr = 'GeV/c'
    elif 'GeV' in title['Y']:
        ystr = 'GeV'
    dummy.GetYaxis().SetTitle(str(title['Y']+' / %.1f %s'%(h_real.GetBinWidth(1), ystr)))
    dummy.GetXaxis().SetTitle(title['X'])
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    if "cell_energy" not in out_name:
        dummy.GetXaxis().SetMoreLogLabels()
        dummy.GetXaxis().SetNoExponent()  
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw()
    h_real .SetLineWidth(2)
    h_fake .SetLineWidth(2)
    h_real .SetLineColor(rt.kRed)
    h_fake .SetLineColor(rt.kBlue)
    h_real .SetMarkerColor(rt.kRed)
    h_fake .SetMarkerColor(rt.kBlue)
    h_real .SetMarkerStyle(20)
    h_fake .SetMarkerStyle(21)
    h_real.Draw("same:pe")
    h_real.Draw("same:pe")
    h_fake.Draw("same:histe")
    dummy.Draw("AXISSAME")
    legend = rt.TLegend(0.6,0.7,0.85,0.85)
    legend.AddEntry(h_real ,'G4','lep')
    legend.AddEntry(h_fake ,"GAN",'lep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    #legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

###############
class Obj:
    def __init__(self, name, fileName):
        self.name = name
        self.file_name = fileName
        str_ext = self.name
        theta_L =  0; 
        theta_H =  3.2; 
        theta_n = 32;
        phi_L = -3.2; 
        phi_H =  3.2; 
        phi_n = 64;
        self.h_ep_mdc_mom_shower      = rt.TH1F('%s_ep_mdc_mom_shower'%(str_ext),'',200,0,2)
        self.h_ep_mdc_mom             = rt.TH1F('%s_ep_mdc_mom'%(str_ext),'',200,0,2)
        self.h_ep_mdc_theta           = rt.TH1F('%s_ep_mdc_theta'%(str_ext),'',theta_n,theta_L,theta_H)
        self.h_ep_mdc_phi             = rt.TH1F('%s_ep_mdc_phi'%(str_ext),'',phi_n,phi_L,phi_H)
        self.h_em_mdc_mom_shower      = rt.TH1F('%s_em_mdc_mom_shower'%(str_ext),'',200,0,2)
        self.h_em_mdc_mom             = rt.TH1F('%s_em_mdc_mom'%(str_ext),'',200,0,2)
        self.h_em_mdc_theta           = rt.TH1F('%s_em_mdc_theta'%(str_ext),'',theta_n,theta_L,theta_H)
        self.h_em_mdc_phi             = rt.TH1F('%s_em_mdc_phi'%(str_ext),'',phi_n,phi_L,phi_H)
        self.h_ep_e3x3                = rt.TH1F('%s_ep_e3x3'%(str_ext),'',200,0,2)
        self.h_ep_e5x5                = rt.TH1F('%s_ep_e5x5'%(str_ext),'',200,0,2)
        self.h_ep_etof2x1             = rt.TH1F('%s_ep_etof2x1'%(str_ext),'',200,0,2)
        self.h_ep_etof2x3             = rt.TH1F('%s_ep_etof2x3'%(str_ext),'',200,0,2)
        self.h_ep_cluster2ndMoment    = rt.TH1F('%s_ep_cluster2ndMoment'%(str_ext),'',100,0,100)
        self.h_ep_cluster_e           = rt.TH1F('%s_ep_cluster_e'%(str_ext),'',200,0,2)
        self.h_ep_secondMoment        = rt.TH1F('%s_ep_secondMoment'%(str_ext),'',50,0,50)
        self.h_ep_latMoment           = rt.TH1F('%s_ep_latMoment'%(str_ext),'',50,0.1,0.6)
        self.h_ep_a20Moment           = rt.TH1F('%s_ep_a20Moment'%(str_ext),'',30,0.7,1)
        self.h_ep_a42Moment           = rt.TH1F('%s_ep_a42Moment'%(str_ext),'',20,0,0.2)
        self.h_ep_track_N             = rt.TH1F('%s_ep_track_N'%(str_ext), '', 100,0,10) 
        self.h_ep_shower_N            = rt.TH1F('%s_ep_shower_N'%(str_ext), '', 100,0,10) 
        self.h_ep_shower_N_v1         = rt.TH1F('%s_ep_shower_N_v1'%(str_ext), '', 100,0,10) 
        self.h_ep_cluster_N           = rt.TH1F('%s_ep_cluster_N'%(str_ext), '', 100,0,10) 
        self.h_ep_trkext_N            = rt.TH1F('%s_ep_trkext_N'%(str_ext), '', 100,0,10) 
        self.h_em_track_N             = rt.TH1F('%s_em_track_N'%(str_ext), '', 100,0,10) 
        self.h_em_shower_N            = rt.TH1F('%s_em_shower_N'%(str_ext), '', 100,0,10) 
        self.h_em_shower_N_v1         = rt.TH1F('%s_em_shower_N_v1'%(str_ext), '', 100,0,10) 
        self.h_em_cluster_N           = rt.TH1F('%s_em_cluster_N'%(str_ext), '', 100,0,10) 
        self.h_em_trkext_N            = rt.TH1F('%s_em_trkext_N'%(str_ext), '', 100,0,10) 

        self.h_em_e3x3                = rt.TH1F('%s_em_e3x3'%(str_ext),'',200,0,2)
        self.h_em_e5x5                = rt.TH1F('%s_em_e5x5'%(str_ext),'',200,0,2)
        self.h_em_etof2x1             = rt.TH1F('%s_em_etof2x1'%(str_ext),'',200,0,2)
        self.h_em_etof2x3             = rt.TH1F('%s_em_etof2x3'%(str_ext),'',200,0,2)
        self.h_em_cluster2ndMoment    = rt.TH1F('%s_em_cluster2ndMoment'%(str_ext),'',100,0,100)
        self.h_em_cluster_e           = rt.TH1F('%s_em_cluster_e'%(str_ext),'',200,0,2)
        self.h_em_secondMoment        = rt.TH1F('%s_em_secondMoment'%(str_ext),'',50,0,50)
        self.h_em_latMoment           = rt.TH1F('%s_em_latMoment'%(str_ext),'',50,0.1,0.6)
        self.h_em_a20Moment           = rt.TH1F('%s_em_a20Moment'%(str_ext),'',30,0.7,1)
        self.h_em_a42Moment           = rt.TH1F('%s_em_a42Moment'%(str_ext),'',20,0,0.2)
        self.h_ep_ext_x               = rt.TH1F('%s_ep_ext_x'%(str_ext),'',200,0,2)
        self.h_ep_ext_y               = rt.TH1F('%s_ep_ext_y'%(str_ext),'',200,0,2)
        self.h_ep_ext_z               = rt.TH1F('%s_ep_ext_z'%(str_ext),'',200,0,2)
        self.h_em_ext_x               = rt.TH1F('%s_em_ext_x'%(str_ext),'',200,0,2)
        self.h_em_ext_y               = rt.TH1F('%s_em_ext_y'%(str_ext),'',200,0,2)
        self.h_em_ext_z               = rt.TH1F('%s_em_ext_z'%(str_ext),'',200,0,2)
        self.h_ep_ext_Px              = rt.TH1F('%s_ep_ext_Px'%(str_ext),'',200,0,2)
        self.h_ep_ext_Py              = rt.TH1F('%s_ep_ext_Py'%(str_ext),'',200,0,2)
        self.h_ep_ext_Pz              = rt.TH1F('%s_ep_ext_Pz'%(str_ext),'',200,0,2)
        self.h_em_ext_Px              = rt.TH1F('%s_em_ext_Px'%(str_ext),'',200,0,2)
        self.h_em_ext_Py              = rt.TH1F('%s_em_ext_Py'%(str_ext),'',200,0,2)
        self.h_em_ext_Pz              = rt.TH1F('%s_em_ext_Pz'%(str_ext),'',200,0,2)
        self.h_ep_emc_shower_energy   = rt.TH1F('%s_ep_emc_shower_energy'%(str_ext),'',200,0,2)
        self.h_ep_emc_shower_x        = rt.TH1F('%s_ep_emc_shower_x'%(str_ext),'',20,-100,100)
        self.h_ep_emc_shower_y        = rt.TH1F('%s_ep_emc_shower_y'%(str_ext),'',20,-100,100)
        self.h_ep_emc_shower_z        = rt.TH1F('%s_ep_emc_shower_z'%(str_ext),'',30,-150,150)
        self.h_ep_emc_shower_theta    = rt.TH1F('%s_ep_emc_shower_theta'%(str_ext),'',90,0,180)
        self.h_ep_emc_shower_phi      = rt.TH1F('%s_ep_emc_shower_phi'%(str_ext),'',36,0,360)
        self.h_ep_M_dtheta            = rt.TH1F('%s_ep_M_dtheta'%(str_ext),'',200,0,2)
        self.h_ep_M_dphi              = rt.TH1F('%s_ep_M_dphi'%(str_ext),'',200,0,2)
        self.h_ep_P_dz                = rt.TH1F('%s_ep_P_dz'%(str_ext),'',200,0,2)
        self.h_ep_P_dphi              = rt.TH1F('%s_ep_P_dphi'%(str_ext),'',200,0,2)
        self.h_ep_emc_hit_energy      = rt.TH1F('%s_ep_emc_hit_energy'%(str_ext),'',100,1,2)
        self.h_em_emc_shower_energy   = rt.TH1F('%s_em_emc_shower_energy'%(str_ext),'',200,0,2)
        self.h_em_emc_shower_x        = rt.TH1F('%s_em_emc_shower_x'%(str_ext),'',20,-100,100)
        self.h_em_emc_shower_y        = rt.TH1F('%s_em_emc_shower_y'%(str_ext),'',20,-100,100)
        self.h_em_emc_shower_z        = rt.TH1F('%s_em_emc_shower_z'%(str_ext),'',30,-150,150)
        self.h_em_emc_shower_theta    = rt.TH1F('%s_em_emc_shower_theta'%(str_ext),'',90,0,180)
        self.h_em_emc_shower_phi      = rt.TH1F('%s_em_emc_shower_phi'%(str_ext),'',36,0,360)
        self.h_em_M_dtheta            = rt.TH1F('%s_em_M_dtheta'%(str_ext),'',200,0,2)
        self.h_em_M_dphi              = rt.TH1F('%s_em_M_dphi'%(str_ext),'',200,0,2)
        self.h_em_P_dz                = rt.TH1F('%s_em_P_dz'%(str_ext),'',200,0,2)
        self.h_em_P_dphi              = rt.TH1F('%s_em_P_dphi'%(str_ext),'',200,0,2)
        self.h_em_emc_hit_energy      = rt.TH1F('%s_em_emc_hit_energy'%(str_ext),'',100,1,2)
        self.h_ep_mc_mom              = rt.TH1F('%s_ep_mc_mom'%(str_ext),'',200,0,2)
        self.h_ep_mc_theta            = rt.TH1F('%s_ep_mc_theta'%(str_ext),'',theta_n,theta_L,theta_H)
        self.h_ep_mc_phi              = rt.TH1F('%s_ep_mc_phi'%(str_ext),'',phi_n,phi_L,phi_H)
        self.h_em_mc_mom              = rt.TH1F('%s_em_mc_mom'%(str_ext),'',200,0,2)
        self.h_em_mc_theta            = rt.TH1F('%s_em_mc_theta'%(str_ext),'',theta_n,theta_L,theta_H)
        self.h_em_mc_phi              = rt.TH1F('%s_em_mc_phi'%(str_ext),'',phi_n,phi_L,phi_H)
        self.h_SumHitE                = rt.TH1F('%s_SumHitE'%(str_ext),'',200,0,4)
        self.h_E_shower_total         = rt.TH1F('%s_E_shower_total'%(str_ext),'',200,2,4)

    def fill(self):
        #P4_em = rt.TLorentzVector()
        FileName = self.file_name
        treeName='Bhabha'
        chain =rt.TChain(treeName)
        chain.Add(FileName)
        tree = chain
        totalEntries=tree.GetEntries()
        print (totalEntries)
        for entryNum in range(0, tree.GetEntries()):
            tree.GetEntry(entryNum)

            m_ep_mdc_mom             = getattr(tree, 'ep_mdc_mom')
            m_ep_mdc_theta           = getattr(tree, 'ep_mdc_theta')
            m_ep_mdc_phi             = getattr(tree, 'ep_mdc_phi')
            m_em_mdc_mom             = getattr(tree, 'em_mdc_mom')
            m_em_mdc_theta           = getattr(tree, 'em_mdc_theta')
            m_em_mdc_phi             = getattr(tree, 'em_mdc_phi')
            m_ep_e3x3                = getattr(tree, 'ep_e3x3')
            m_ep_e5x5                = getattr(tree, 'ep_e5x5')
            m_ep_etof2x1             = getattr(tree, 'ep_etof2x1')
            m_ep_etof2x3             = getattr(tree, 'ep_etof2x3')
            m_ep_cluster2ndMoment    = getattr(tree, 'ep_cluster2ndMoment')
            m_ep_cluster_e           = getattr(tree, 'ep_cluster_e')
            m_ep_secondMoment        = getattr(tree, 'ep_secondMoment')
            m_ep_latMoment           = getattr(tree, 'ep_latMoment')
            m_ep_a20Moment           = getattr(tree, 'ep_a20Moment')
            m_ep_a42Moment           = getattr(tree, 'ep_a42Moment')
            m_ep_track_N             = getattr(tree, 'ep_track_N')
            m_ep_shower_N            = getattr(tree, 'ep_shower_N')
            m_ep_shower_N_v1         = getattr(tree, 'ep_shower_N_v1')
            m_ep_cluster_N           = getattr(tree, 'ep_cluster_N')
            m_ep_trkext_N            = getattr(tree, 'ep_trkext_N')
            m_em_e3x3                = getattr(tree, 'em_e3x3')
            m_em_e5x5                = getattr(tree, 'em_e5x5')
            m_em_etof2x1             = getattr(tree, 'em_etof2x1')
            m_em_etof2x3             = getattr(tree, 'em_etof2x3')
            m_em_cluster2ndMoment    = getattr(tree, 'em_cluster2ndMoment')
            m_em_cluster_e           = getattr(tree, 'em_cluster_e')
            m_em_secondMoment        = getattr(tree, 'em_secondMoment')
            m_em_latMoment           = getattr(tree, 'em_latMoment')
            m_em_a20Moment           = getattr(tree, 'em_a20Moment')
            m_em_a42Moment           = getattr(tree, 'em_a42Moment')
            m_em_track_N             = getattr(tree, 'em_track_N')
            m_em_shower_N            = getattr(tree, 'em_shower_N')
            m_em_shower_N_v1         = getattr(tree, 'em_shower_N_v1')
            m_em_cluster_N           = getattr(tree, 'em_cluster_N')
            m_em_trkext_N            = getattr(tree, 'em_trkext_N')
            m_ep_ext_x               = getattr(tree, 'ep_ext_x')
            m_ep_ext_y               = getattr(tree, 'ep_ext_y')
            m_ep_ext_z               = getattr(tree, 'ep_ext_z')
            m_em_ext_x               = getattr(tree, 'em_ext_x')
            m_em_ext_y               = getattr(tree, 'em_ext_y')
            m_em_ext_z               = getattr(tree, 'em_ext_z')
            m_ep_ext_Px              = getattr(tree, 'ep_ext_Px')
            m_ep_ext_Py              = getattr(tree, 'ep_ext_Py')
            m_ep_ext_Pz              = getattr(tree, 'ep_ext_Pz')
            m_em_ext_Px              = getattr(tree, 'em_ext_Px')
            m_em_ext_Py              = getattr(tree, 'em_ext_Py')
            m_em_ext_Pz              = getattr(tree, 'em_ext_Pz')
            m_ep_emc_shower_energy   = getattr(tree, 'ep_emc_shower_energy')
            m_ep_emc_shower_x        = getattr(tree, 'ep_emc_shower_x')
            m_ep_emc_shower_y        = getattr(tree, 'ep_emc_shower_y')
            m_ep_emc_shower_z        = getattr(tree, 'ep_emc_shower_z')
            m_ep_M_dtheta            = getattr(tree, 'ep_M_dtheta')
            m_ep_M_dphi              = getattr(tree, 'ep_M_dphi')
            m_ep_P_dz                = getattr(tree, 'ep_P_dz')
            m_ep_P_dphi              = getattr(tree, 'ep_P_dphi')
            m_ep_emc_hit_energy      = getattr(tree, 'ep_emc_hit_energy')
            m_em_emc_shower_energy   = getattr(tree, 'em_emc_shower_energy')
            m_em_emc_shower_x        = getattr(tree, 'em_emc_shower_x')
            m_em_emc_shower_y        = getattr(tree, 'em_emc_shower_y')
            m_em_emc_shower_z        = getattr(tree, 'em_emc_shower_z')
            m_em_M_dtheta            = getattr(tree, 'em_M_dtheta')
            m_em_M_dphi              = getattr(tree, 'em_M_dphi')
            m_em_P_dz                = getattr(tree, 'em_P_dz')
            m_em_P_dphi              = getattr(tree, 'em_P_dphi')
            m_em_emc_hit_energy      = getattr(tree, 'em_emc_hit_energy')
            m_ep_mc_mom              = getattr(tree, 'ep_mc_mom')
            m_ep_mc_theta            = getattr(tree, 'ep_mc_theta')
            m_ep_mc_phi              = getattr(tree, 'ep_mc_phi')
            m_em_mc_mom              = getattr(tree, 'em_mc_mom')
            m_em_mc_theta            = getattr(tree, 'em_mc_theta')
            m_em_mc_phi              = getattr(tree, 'em_mc_phi')
            m_SumHitE                = getattr(tree, 'SumHitE')
            m_E_shower_total         = getattr(tree, 'E_shower_total')
            ep_hit_e_sum = 0
            em_hit_e_sum = 0
            for i in range(len(m_ep_emc_hit_energy)):
                ep_hit_e_sum = ep_hit_e_sum + m_ep_emc_hit_energy[i]
            for i in range(len(m_em_emc_hit_energy)):
                em_hit_e_sum = em_hit_e_sum + m_em_emc_hit_energy[i]
            if for_em_High_Z and abs(m_em_ext_z)<100 : continue
            if for_em_Low_Z  and abs(m_em_ext_z)>=100: continue
            if for_ep_High_Z and abs(m_ep_ext_z)<100 : continue
            if for_ep_Low_Z  and abs(m_ep_ext_z)>=100: continue
            #if m_ep_shower_N_v1 !=2: continue

            self.h_ep_mdc_mom_shower      .Fill(m_ep_emc_shower_energy/m_ep_mdc_mom)
            self.h_ep_mdc_mom             .Fill(m_ep_mdc_mom             )
            self.h_ep_mdc_theta           .Fill(m_ep_mdc_theta           )
            self.h_ep_mdc_phi             .Fill(m_ep_mdc_phi             )
            self.h_em_mdc_mom_shower      .Fill(m_em_emc_shower_energy/m_em_mdc_mom)
            self.h_em_mdc_mom             .Fill(m_em_mdc_mom             )
            self.h_em_mdc_theta           .Fill(m_em_mdc_theta           )
            self.h_em_mdc_phi             .Fill(m_em_mdc_phi             )
            self.h_ep_e3x3                .Fill(m_ep_e3x3                )
            self.h_ep_e5x5                .Fill(m_ep_e5x5                )
            self.h_ep_etof2x1             .Fill(m_ep_etof2x1             )
            self.h_ep_etof2x3             .Fill(m_ep_etof2x3             )
            self.h_ep_cluster2ndMoment    .Fill(m_ep_cluster2ndMoment    )
            self.h_ep_cluster_e           .Fill(m_ep_cluster_e           )
            self.h_ep_secondMoment        .Fill(m_ep_secondMoment        )
            self.h_ep_latMoment           .Fill(m_ep_latMoment           )
            self.h_ep_a20Moment           .Fill(m_ep_a20Moment           )
            self.h_ep_a42Moment           .Fill(m_ep_a42Moment           )
            self.h_ep_track_N             .Fill(m_ep_track_N   )
            self.h_ep_shower_N            .Fill(m_ep_shower_N  )
            self.h_ep_shower_N_v1         .Fill(m_ep_shower_N_v1)
            self.h_ep_cluster_N           .Fill(m_ep_cluster_N )
            self.h_ep_trkext_N            .Fill(m_ep_trkext_N  )
            self.h_em_track_N             .Fill(m_em_track_N   )
            self.h_em_shower_N            .Fill(m_em_shower_N  )
            self.h_em_shower_N_v1         .Fill(m_em_shower_N_v1)
            self.h_em_cluster_N           .Fill(m_em_cluster_N )
            self.h_em_trkext_N            .Fill(m_em_trkext_N  )


            self.h_em_e3x3                .Fill(m_em_e3x3                )
            self.h_em_e5x5                .Fill(m_em_e5x5                )
            self.h_em_etof2x1             .Fill(m_em_etof2x1             )
            self.h_em_etof2x3             .Fill(m_em_etof2x3             )
            self.h_em_cluster2ndMoment    .Fill(m_em_cluster2ndMoment    )
            self.h_em_cluster_e           .Fill(m_em_cluster_e           )
            self.h_em_secondMoment        .Fill(m_em_secondMoment        )
            self.h_em_latMoment           .Fill(m_em_latMoment           )
            self.h_em_a20Moment           .Fill(m_em_a20Moment           )
            self.h_em_a42Moment           .Fill(m_em_a42Moment           )
            self.h_ep_ext_x               .Fill(m_ep_ext_x               )
            self.h_ep_ext_y               .Fill(m_ep_ext_y               )
            self.h_ep_ext_z               .Fill(m_ep_ext_z               )
            self.h_em_ext_x               .Fill(m_em_ext_x               )
            self.h_em_ext_y               .Fill(m_em_ext_y               )
            self.h_em_ext_z               .Fill(m_em_ext_z               )
            self.h_ep_ext_Px              .Fill(m_ep_ext_Px              )
            self.h_ep_ext_Py              .Fill(m_ep_ext_Py              )
            self.h_ep_ext_Pz              .Fill(m_ep_ext_Pz              )
            self.h_em_ext_Px              .Fill(m_em_ext_Px              )
            self.h_em_ext_Py              .Fill(m_em_ext_Py              )
            self.h_em_ext_Pz              .Fill(m_em_ext_Pz              )
            self.h_ep_emc_shower_energy   .Fill(m_ep_emc_shower_energy   )
            self.h_ep_emc_shower_x        .Fill(m_ep_emc_shower_x        )
            self.h_ep_emc_shower_y        .Fill(m_ep_emc_shower_y        )
            self.h_ep_emc_shower_z        .Fill(m_ep_emc_shower_z        )
            self.h_ep_emc_shower_theta    .Fill(getTheta(math.sqrt(m_ep_emc_shower_x*m_ep_emc_shower_x + m_ep_emc_shower_y*m_ep_emc_shower_y), m_ep_emc_shower_z) )
            self.h_ep_emc_shower_phi      .Fill( getPhi(m_ep_emc_shower_x, m_ep_emc_shower_y) )
            self.h_ep_M_dtheta            .Fill(m_ep_M_dtheta            )
            self.h_ep_M_dphi              .Fill(m_ep_M_dphi              )
            self.h_ep_P_dz                .Fill(m_ep_P_dz                )
            self.h_ep_P_dphi              .Fill(m_ep_P_dphi              )
            self.h_ep_emc_hit_energy      .Fill(ep_hit_e_sum             )
            self.h_em_emc_shower_energy   .Fill(m_em_emc_shower_energy   )
            self.h_em_emc_shower_x        .Fill(m_em_emc_shower_x        )
            self.h_em_emc_shower_y        .Fill(m_em_emc_shower_y        )
            self.h_em_emc_shower_z        .Fill(m_em_emc_shower_z        )
            self.h_em_emc_shower_theta    .Fill(getTheta(math.sqrt(m_em_emc_shower_x*m_em_emc_shower_x + m_em_emc_shower_y*m_em_emc_shower_y), m_em_emc_shower_z) )
            self.h_em_emc_shower_phi      .Fill( getPhi(m_em_emc_shower_x, m_em_emc_shower_y) )
            self.h_em_M_dtheta            .Fill(m_em_M_dtheta            )
            self.h_em_M_dphi              .Fill(m_em_M_dphi              )
            self.h_em_P_dz                .Fill(m_em_P_dz                )
            self.h_em_P_dphi              .Fill(m_em_P_dphi              )
            self.h_em_emc_hit_energy      .Fill(em_hit_e_sum             )
            self.h_ep_mc_mom              .Fill(m_ep_mc_mom              )
            self.h_ep_mc_theta            .Fill(m_ep_mc_theta            )
            self.h_ep_mc_phi              .Fill(m_ep_mc_phi              )
            self.h_em_mc_mom              .Fill(m_em_mc_mom              )
            self.h_em_mc_theta            .Fill(m_em_mc_theta            )
            self.h_em_mc_phi              .Fill(m_em_mc_phi              )
            self.h_SumHitE                .Fill(m_SumHitE                )
            self.h_E_shower_total         .Fill(m_E_shower_total         )

###### BEGIN #######

do_Normalize = False

for_em_Low_Z  = False
for_em_High_Z  =True
for_ep_High_Z = False
for_ep_Low_Z  = False

plot_path = './plots'

obj_orgin        = Obj('orgin','/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_baba_0to9_NshowerCheck.root')
#obj_orgin        = Obj('orgin','/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_baba_0to9.root')
#obj_orgin        = Obj('orgin','/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_220_noMix_rereco.root')
#obj_fake         = Obj('fake' ,'/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_220_noMix_tf_20191106.root')
#obj_fake         = Obj('fake' ,'/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_0to9_noMix_tf_20191106.root')
#obj_fake         = Obj('fake' ,'/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_0to9_noMix_tf_20191109.root')
#obj_fake         = Obj('fake' ,'/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_0to9_noMix_tf_20191110.root')
#obj_fake         = Obj('fake' ,'/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_0to9_noMix_tf_20191112_NshowerCheck.root')
#obj_fake         = Obj('fake' ,'/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_0to9_noMix_tf_20191112_Comp.root')
obj_fake         = Obj('fake' ,'/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_0to9_noMix_tf_20191113_epoch59.root')
#obj_fake         = Obj('fake' ,'/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_0to9_noMix_tf_20191113_epoch57.root')
obj_orgin.fill()
obj_fake.fill()


do_plot(obj_orgin.h_ep_mc_mom         ,"mc_ep_P"       ,{'X':"P_{mc}^{e^{+}} GeV/c"       ,'Y':'Events'} )
do_plot(obj_orgin.h_ep_mc_theta       ,"mc_ep_theta"   ,{'X':"#theta_{mc}^{e^{+}} GeV/c"  ,'Y':'Events'} )
do_plot(obj_orgin.h_ep_mc_phi         ,"mc_ep_phi"     ,{'X':"#phi_{mc}^{e^{+}} GeV/c"    ,'Y':'Events'} )
do_plot(obj_orgin.h_em_mc_mom         ,"mc_em_P"       ,{'X':"P_{mc}^{e^{-}} GeV/c"       ,'Y':'Events'} )
do_plot(obj_orgin.h_em_mc_theta       ,"mc_em_theta"   ,{'X':"#theta_{mc}^{e^{-}} GeV/c"  ,'Y':'Events'} )
do_plot(obj_orgin.h_em_mc_phi         ,"mc_em_phi"     ,{'X':"#phi_{mc}^{e^{-}} GeV/c"    ,'Y':'Events'} )

do_plot_h2(obj_orgin.h_ep_mdc_mom_shower ,obj_fake.h_ep_mdc_mom_shower         ,"ep_shower_over_P"       ,{'X':"E_{shower}/P_{mdc}^{e^{+}}"       ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_mdc_mom     ,obj_fake.h_ep_mdc_mom         ,"mdc_ep_P"       ,{'X':"P_{mdc}^{e^{+}} GeV/c"       ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_mdc_theta   ,obj_fake.h_ep_mdc_theta       ,"mdc_ep_theta"   ,{'X':"#theta_{mdc}^{e^{+}} GeV/c"  ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_mdc_phi     ,obj_fake.h_ep_mdc_phi         ,"mdc_ep_phi"     ,{'X':"#phi_{mdc}^{e^{+}} GeV/c"    ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_mdc_mom_shower ,obj_fake.h_em_mdc_mom_shower         ,"em_shower_over_P"       ,{'X':"E_{shower}/P_{mdc}^{e^{-}}"       ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_mdc_mom     ,obj_fake.h_em_mdc_mom         ,"mdc_em_P"       ,{'X':"P_{mdc}^{e^{-}} GeV/c"       ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_mdc_theta   ,obj_fake.h_em_mdc_theta       ,"mdc_em_theta"   ,{'X':"#theta_{mdc}^{e^{-}} GeV/c"  ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_mdc_phi     ,obj_fake.h_em_mdc_phi         ,"mdc_em_phi"     ,{'X':"#phi_{mdc}^{e^{-}} GeV/c"    ,'Y':'Events'} )


do_plot_h2(obj_orgin.h_ep_e3x3               ,obj_fake.h_ep_e3x3               , "ep_e3x3"              ,{'X':"e_{3x3}^{e^{+}} GeV"             ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_e5x5               ,obj_fake.h_ep_e5x5               , "ep_e5x5"              ,{'X':"e_{5x5}^{e^{+}} GeV"             ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_etof2x1            ,obj_fake.h_ep_etof2x1            , "ep_etof2x1"           ,{'X':"ET_{2x1}^{e^{+}} GeV"            ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_etof2x3            ,obj_fake.h_ep_etof2x3            , "ep_etof2x3"           ,{'X':"ET_{2x3}^{e^{+}} GeV"            ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_cluster2ndMoment   ,obj_fake.h_ep_cluster2ndMoment   , "ep_cluster2ndMoment"  ,{'X':"2ndMom_{cluster}^{e^{+}}"        ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_cluster_e          ,obj_fake.h_ep_cluster_e          , "ep_cluster_e"         ,{'X':"E_{cluster}^{e^{+}} GeV"         ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_emc_hit_energy     ,obj_fake.h_ep_emc_hit_energy     , "ep_hit_sum_e"         ,{'X':"#sumE_{hit}^{e^{+}} GeV"         ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_secondMoment       ,obj_fake.h_ep_secondMoment       , "ep_secondMoment"      ,{'X':"2ndMom_{shower}^{e^{+}}"         ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_latMoment          ,obj_fake.h_ep_latMoment          , "ep_latMoment"         ,{'X':"latMom_{shower}^{e^{+}} GeV/c"   ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_a20Moment          ,obj_fake.h_ep_a20Moment          , "ep_a20Moment"         ,{'X':"a20Mom_{shower}^{e^{+}} GeV/c"   ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_a42Moment          ,obj_fake.h_ep_a42Moment          , "ep_a42Moment"         ,{'X':"a42Mom_{shower}^{e^{+}} GeV/c"   ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_emc_shower_energy  ,obj_fake.h_ep_emc_shower_energy  , "ep_shower_e"          ,{'X':"E_{shower}^{e^{+}} GeV"          ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_emc_shower_x       ,obj_fake.h_ep_emc_shower_x       , "ep_shower_x"          ,{'X':"x_{shower}^{e^{+}} cm"           ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_emc_shower_y       ,obj_fake.h_ep_emc_shower_y       , "ep_shower_y"          ,{'X':"y_{shower}^{e^{+}} cm"           ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_emc_shower_z       ,obj_fake.h_ep_emc_shower_z       , "ep_shower_z"          ,{'X':"z_{shower}^{e^{+}} cm"           ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_emc_shower_theta   ,obj_fake.h_ep_emc_shower_theta   , "ep_shower_theta"      ,{'X':"#theta_{shower}^{e^{+}} degree"      ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_emc_shower_phi     ,obj_fake.h_ep_emc_shower_phi     , "ep_shower_phi"        ,{'X':"#phi_{shower}^{e^{+}} degree"        ,'Y':'Events'} )

do_plot_h2(obj_orgin.h_em_e3x3               ,obj_fake.h_em_e3x3               , "em_e3x3"              ,{'X':"e_{3x3}^{e^{-}} GeV"             ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_e5x5               ,obj_fake.h_em_e5x5               , "em_e5x5"              ,{'X':"e_{5x5}^{e^{-}} GeV"             ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_etof2x1            ,obj_fake.h_em_etof2x1            , "em_etof2x1"           ,{'X':"ET_{2x1}^{e^{-}} GeV"            ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_etof2x3            ,obj_fake.h_em_etof2x3            , "em_etof2x3"           ,{'X':"ET_{2x3}^{e^{-}} GeV"            ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_cluster2ndMoment   ,obj_fake.h_em_cluster2ndMoment   , "em_cluster2ndMoment"  ,{'X':"2ndMom_{cluster}^{e^{-}}"        ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_cluster_e          ,obj_fake.h_em_cluster_e          , "em_cluster_e"         ,{'X':"E_{cluster}^{e^{-}} GeV"         ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_emc_hit_energy     ,obj_fake.h_em_emc_hit_energy     , "em_hit_sum_e"         ,{'X':"#sumE_{hit}^{e^{-}} GeV"         ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_secondMoment       ,obj_fake.h_em_secondMoment       , "em_secondMoment"      ,{'X':"2ndMom_{shower}^{e^{-}}"         ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_latMoment          ,obj_fake.h_em_latMoment          , "em_latMoment"         ,{'X':"latMom_{shower}^{e^{-}} GeV/c"   ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_a20Moment          ,obj_fake.h_em_a20Moment          , "em_a20Moment"         ,{'X':"a20Mom_{shower}^{e^{-}} GeV/c"   ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_a42Moment          ,obj_fake.h_em_a42Moment          , "em_a42Moment"         ,{'X':"a42Mom_{shower}^{e^{-}} GeV/c"   ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_emc_shower_energy  ,obj_fake.h_em_emc_shower_energy  , "em_shower_e"          ,{'X':"E_{shower}^{e^{-}} GeV"          ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_emc_shower_x       ,obj_fake.h_em_emc_shower_x       , "em_shower_x"          ,{'X':"x_{shower}^{e^{-}} cm"           ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_emc_shower_y       ,obj_fake.h_em_emc_shower_y       , "em_shower_y"          ,{'X':"y_{shower}^{e^{-}} cm"           ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_emc_shower_z       ,obj_fake.h_em_emc_shower_z       , "em_shower_z"          ,{'X':"z_{shower}^{e^{-}} cm"           ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_emc_shower_theta   ,obj_fake.h_em_emc_shower_theta   , "em_shower_theta"      ,{'X':"#theta_{shower}^{e^{-}} degree"      ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_emc_shower_phi     ,obj_fake.h_em_emc_shower_phi     , "em_shower_phi"        ,{'X':"#phi_{shower}^{e^{-}} degree"        ,'Y':'Events'} )

do_plot_h2(obj_orgin.h_ep_track_N            ,obj_fake.h_ep_track_N            , "ep_track_N_logy"           ,{'X':"N track^{e^{+}}"            ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_shower_N           ,obj_fake.h_ep_shower_N           , "ep_shower_N_logy"          ,{'X':"N shower^{e^{+}}"           ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_ep_cluster_N          ,obj_fake.h_ep_cluster_N          , "ep_cluster_N_logy"         ,{'X':"N cluster^{e^{+}}"          ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_track_N            ,obj_fake.h_em_track_N            , "em_track_N_logy"           ,{'X':"N track^{e^{-}}"            ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_shower_N           ,obj_fake.h_em_shower_N           , "em_shower_N_logy"          ,{'X':"N shower^{e^{-}}"           ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_shower_N_v1        ,obj_fake.h_em_shower_N_v1        , "em_shower_N_v1_logy"       ,{'X':"N shower^{e^{-}}"           ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_shower_N_v1        ,obj_fake.h_em_shower_N_v1        , "em_shower_N_v1"            ,{'X':"N shower^{e^{-}}"           ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_em_cluster_N          ,obj_fake.h_em_cluster_N          , "em_cluster_N_logy"         ,{'X':"N cluster^{e^{-}}"          ,'Y':'Events'} )

do_plot_h2(obj_orgin.h_SumHitE               ,obj_fake.h_SumHitE               , "EvtSumHitE"           ,{'X':"#sumE_{evt hit} GeV"             ,'Y':'Events'} )
do_plot_h2(obj_orgin.h_E_shower_total        ,obj_fake.h_E_shower_total        , "E_shower_total"           ,{'X':"#sumE_{shower} GeV"             ,'Y':'Events'} )
print('done')



