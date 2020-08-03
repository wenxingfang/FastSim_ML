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

def do_plot_h2(h_real, h_fake, out_name, title, str_particle):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.12)
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
    legend = rt.TLegend(0.75,0.7,0.85,0.85)
    legend.AddEntry(h_real ,'G4','lep')
    legend.AddEntry(h_fake ,"GAN",'lep')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    #legend.SetFillStyle(0)
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
        self.h_em_px          = rt.TH1F('%s_h_em_px'        %(str_ext) ,'',10,0,100)
        self.h_em_py          = rt.TH1F('%s_h_em_py'        %(str_ext) ,'',10,0,100)
        self.h_em_pz          = rt.TH1F('%s_h_em_pz'        %(str_ext) ,'',10,0,100)
        self.h_em_p           = rt.TH1F('%s_h_em_p'         %(str_ext) ,'',10,0,100)
        self.h_ep_px          = rt.TH1F('%s_h_ep_px'        %(str_ext) ,'',10,0,100)
        self.h_ep_py          = rt.TH1F('%s_h_ep_py'        %(str_ext) ,'',10,0,100)
        self.h_ep_pz          = rt.TH1F('%s_h_ep_pz'        %(str_ext) ,'',10,0,100)
        self.h_ep_p           = rt.TH1F('%s_h_ep_p'         %(str_ext) ,'',10,0,100)
        self.h_m_ee           = rt.TH1F('%s_h_m_ee'         %(str_ext) ,'',60,60,120)
        self.h_m_reco         = rt.TH1F('%s_h_m_reco'       %(str_ext) ,'',50,100,150)
        self.h_rec_eLed_px    = rt.TH1F('%s_h_rec_eLed_px'  %(str_ext) ,'',10,0,100)
        self.h_rec_eLed_py    = rt.TH1F('%s_h_rec_eLed_py'  %(str_ext) ,'',10,0,100)
        self.h_rec_eLed_pz    = rt.TH1F('%s_h_rec_eLed_pz'  %(str_ext) ,'',10,0,100)
        self.h_rec_eLed_E     = rt.TH1F('%s_h_rec_eLed_E'   %(str_ext) ,'',10,0,100)
        self.h_rec_eLed_cluE  = rt.TH1F('%s_h_rec_eLed_cluE'%(str_ext) ,'',10,0,100)
        self.h_rec_eSub_px    = rt.TH1F('%s_h_rec_eSub_px'  %(str_ext) ,'',10,0,100)
        self.h_rec_eSub_py    = rt.TH1F('%s_h_rec_eSub_py'  %(str_ext) ,'',10,0,100)
        self.h_rec_eSub_pz    = rt.TH1F('%s_h_rec_eSub_pz'  %(str_ext) ,'',10,0,100)
        self.h_rec_eSub_E     = rt.TH1F('%s_h_rec_eSub_E'   %(str_ext) ,'',10,0,100)
        self.h_rec_eSub_cluE  = rt.TH1F('%s_h_rec_eSub_cluE'%(str_ext) ,'',10,0,100)
        self.h_rec_ep_px      = rt.TH1F('%s_h_rec_ep_px'    %(str_ext) ,'',10,0,100)
        self.h_rec_ep_py      = rt.TH1F('%s_h_rec_ep_py'    %(str_ext) ,'',10,0,100)
        self.h_rec_ep_pz      = rt.TH1F('%s_h_rec_ep_pz'    %(str_ext) ,'',10,0,100)
        self.h_rec_ep_E       = rt.TH1F('%s_h_rec_ep_E'     %(str_ext) ,'',10,0,100)
        self.h_rec_ep_cluE    = rt.TH1F('%s_h_rec_ep_cluE'  %(str_ext) ,'',10,0,100)
        self.h_rec_em_px      = rt.TH1F('%s_h_rec_em_px'    %(str_ext) ,'',10,0,100)
        self.h_rec_em_py      = rt.TH1F('%s_h_rec_em_py'    %(str_ext) ,'',10,0,100)
        self.h_rec_em_pz      = rt.TH1F('%s_h_rec_em_pz'    %(str_ext) ,'',10,0,100)
        self.h_rec_em_E       = rt.TH1F('%s_h_rec_em_E'     %(str_ext) ,'',10,0,100)
        self.h_rec_em_cluE    = rt.TH1F('%s_h_rec_em_cluE'  %(str_ext) ,'',10,0,100)
        self.h_rec_muLed_px   = rt.TH1F('%s_h_rec_muLed_px' %(str_ext) ,'',10,0,100)
        self.h_rec_muLed_py   = rt.TH1F('%s_h_rec_muLed_py' %(str_ext) ,'',10,0,100)
        self.h_rec_muLed_pz   = rt.TH1F('%s_h_rec_muLed_pz' %(str_ext) ,'',10,0,100)
        self.h_rec_muLed_E    = rt.TH1F('%s_h_rec_muLed_E'  %(str_ext) ,'',10,0,100)
        self.h_rec_muSub_px   = rt.TH1F('%s_h_rec_muSub_px' %(str_ext) ,'',10,0,100)
        self.h_rec_muSub_py   = rt.TH1F('%s_h_rec_muSub_py' %(str_ext) ,'',10,0,100)
        self.h_rec_muSub_pz   = rt.TH1F('%s_h_rec_muSub_pz' %(str_ext) ,'',10,0,100)
        self.h_rec_muSub_E    = rt.TH1F('%s_h_rec_muSub_E'  %(str_ext) ,'',10,0,100)
        self.h_rec_m_ee       = rt.TH1F('%s_h_rec_m_ee'     %(str_ext) ,'',60,60,120)
        self.h_rec_m_mumu     = rt.TH1F('%s_h_rec_m_mumu'   %(str_ext) ,'',50,100,150)

    def fill(self):
        P4_em = rt.TLorentzVector()
        P4_ep = rt.TLorentzVector()
        P4_rec_eLed = rt.TLorentzVector()
        P4_rec_eSub = rt.TLorentzVector()
        P4_rec_muLed = rt.TLorentzVector()
        P4_rec_muSub = rt.TLorentzVector()
        P4_rec_ep = rt.TLorentzVector()
        P4_rec_em = rt.TLorentzVector()
        FileName = self.file_name
        treeName='evt'
        chain =rt.TChain(treeName)
        chain.Add(FileName)
        tree = chain
        totalEntries=tree.GetEntries()
        print (totalEntries)
        for entryNum in range(0, tree.GetEntries()):
            tree.GetEntry(entryNum)
            tmp_mc_Px   = getattr(tree, "m_mc_Px")
            tmp_mc_Py   = getattr(tree, "m_mc_Py")
            tmp_mc_Pz   = getattr(tree, "m_mc_Pz")
            tmp_mc_M   = getattr(tree, "m_mc_M")
            tmp_mc_C    = getattr(tree, "m_mc_Charge")
            tmp_rec_eLed_px    = getattr(tree, "m_rec_eLed_px")
            tmp_rec_eLed_py    = getattr(tree, "m_rec_eLed_py")
            tmp_rec_eLed_pz    = getattr(tree, "m_rec_eLed_pz")
            tmp_rec_eLed_E     = getattr(tree, "m_rec_eLed_E")
            tmp_rec_eLed_cluE  = getattr(tree, "m_rec_eLed_cluE")
            tmp_rec_eLed_charge= getattr(tree, "m_rec_eLed_charge")
            tmp_rec_eSub_px    = getattr(tree, "m_rec_eSub_px")
            tmp_rec_eSub_py    = getattr(tree, "m_rec_eSub_py")
            tmp_rec_eSub_pz    = getattr(tree, "m_rec_eSub_pz")
            tmp_rec_eSub_E     = getattr(tree, "m_rec_eSub_E")
            tmp_rec_eSub_cluE  = getattr(tree, "m_rec_eSub_cluE")
            tmp_rec_eSub_charge= getattr(tree, "m_rec_eSub_charge")
            tmp_rec_muLed_px   = getattr(tree, "m_rec_muLed_px")
            tmp_rec_muLed_py   = getattr(tree, "m_rec_muLed_py")
            tmp_rec_muLed_pz   = getattr(tree, "m_rec_muLed_pz")
            tmp_rec_muLed_E    = getattr(tree, "m_rec_muLed_E")
            tmp_rec_muSub_px   = getattr(tree, "m_rec_muSub_px")
            tmp_rec_muSub_py   = getattr(tree, "m_rec_muSub_py")
            tmp_rec_muSub_pz   = getattr(tree, "m_rec_muSub_pz")
            tmp_rec_muSub_E    = getattr(tree, "m_rec_muSub_E")
            if len(tmp_mc_Px) == 2 and ((tmp_mc_C[0] + tmp_mc_C[1]) == 0):
                if tmp_mc_C[0] > 0:
                    ep_E = math.sqrt(tmp_mc_Px[0]*tmp_mc_Px[0] + tmp_mc_Py[0]*tmp_mc_Py[0] + tmp_mc_Pz[0]*tmp_mc_Pz[0] + tmp_mc_M[0]*tmp_mc_M[0] )
                    P4_ep.SetPxPyPzE(tmp_mc_Px[0],tmp_mc_Py[0],tmp_mc_Pz[0],ep_E);
                    em_E = math.sqrt(tmp_mc_Px[1]*tmp_mc_Px[1] + tmp_mc_Py[1]*tmp_mc_Py[1] + tmp_mc_Pz[1]*tmp_mc_Pz[1] + tmp_mc_M[1]*tmp_mc_M[1] )
                    P4_em.SetPxPyPzE(tmp_mc_Px[1],tmp_mc_Py[1],tmp_mc_Pz[1],em_E);
                else:
                    ep_E = math.sqrt(tmp_mc_Px[1]*tmp_mc_Px[1] + tmp_mc_Py[1]*tmp_mc_Py[1] + tmp_mc_Pz[1]*tmp_mc_Pz[1] + tmp_mc_M[1]*tmp_mc_M[1] )
                    P4_ep.SetPxPyPzE(tmp_mc_Px[1],tmp_mc_Py[1],tmp_mc_Pz[1],ep_E);
                    em_E = math.sqrt(tmp_mc_Px[0]*tmp_mc_Px[0] + tmp_mc_Py[0]*tmp_mc_Py[0] + tmp_mc_Pz[0]*tmp_mc_Pz[0] + tmp_mc_M[0]*tmp_mc_M[0] )
                    P4_em.SetPxPyPzE(tmp_mc_Px[0],tmp_mc_Py[0],tmp_mc_Pz[0],em_E);
                P4_ee = P4_ep + P4_em
                P4_cms = rt.TLorentzVector(0,0,0,240)   
                P4_reco = P4_cms - P4_ee
                self.h_em_px.Fill(P4_em.Px())
                self.h_em_py.Fill(P4_em.Py())
                self.h_em_pz.Fill(P4_em.Pz())
                self.h_em_p .Fill(P4_em.P())
                self.h_ep_px.Fill(P4_ep.Px())
                self.h_ep_py.Fill(P4_ep.Py())
                self.h_ep_pz.Fill(P4_ep.Pz())
                self.h_ep_p .Fill(P4_ep.P())
                self.h_m_ee .Fill(P4_ee.M())
                self.h_m_reco.Fill(P4_reco.M())
            P4_rec_eLed.SetPxPyPzE(tmp_rec_eLed_px,tmp_rec_eLed_py,tmp_rec_eLed_pz,tmp_rec_eLed_E);
            P4_rec_eSub.SetPxPyPzE(tmp_rec_eSub_px,tmp_rec_eSub_py,tmp_rec_eSub_pz,tmp_rec_eSub_E);
            P4_rec_muLed.SetPxPyPzE(tmp_rec_muLed_px,tmp_rec_muLed_py,tmp_rec_muLed_pz,tmp_rec_muLed_E);
            P4_rec_muSub.SetPxPyPzE(tmp_rec_muSub_px,tmp_rec_muSub_py,tmp_rec_muSub_pz,tmp_rec_muSub_E);

            if tmp_rec_eLed_E !=0 and tmp_rec_eSub_E !=0:
                self.h_rec_eLed_px  .Fill(P4_rec_eLed.Px()) 
                self.h_rec_eLed_py  .Fill(P4_rec_eLed.Py()) 
                self.h_rec_eLed_pz  .Fill(P4_rec_eLed.Pz()) 
                self.h_rec_eLed_E   .Fill(P4_rec_eLed.E() ) 
                self.h_rec_eLed_cluE.Fill(tmp_rec_eLed_cluE)
                self.h_rec_eSub_px  .Fill(P4_rec_eSub.Px()) 
                self.h_rec_eSub_py  .Fill(P4_rec_eSub.Py()) 
                self.h_rec_eSub_pz  .Fill(P4_rec_eSub.Pz()) 
                self.h_rec_eSub_E   .Fill(P4_rec_eSub.E() ) 
                self.h_rec_eSub_cluE.Fill(tmp_rec_eSub_cluE)
                self.h_rec_m_ee     .Fill((P4_rec_eLed+P4_rec_eSub)  .M())
            if tmp_rec_muLed_E !=0 and tmp_rec_muSub_E !=0:
                self.h_rec_muLed_px .Fill(P4_rec_muLed.Px())
                self.h_rec_muLed_py .Fill(P4_rec_muLed.Py())
                self.h_rec_muLed_pz .Fill(P4_rec_muLed.Pz())
                self.h_rec_muLed_E  .Fill(P4_rec_muLed.E() )
                self.h_rec_muSub_px .Fill(P4_rec_muSub.Px())
                self.h_rec_muSub_py .Fill(P4_rec_muSub.Py())
                self.h_rec_muSub_pz .Fill(P4_rec_muSub.Pz())
                self.h_rec_muSub_E  .Fill(P4_rec_muSub.E() )
                self.h_rec_m_mumu   .Fill((P4_rec_muLed+P4_rec_muSub).M())
            if tmp_rec_eLed_charge + tmp_rec_eSub_charge == 0:
                if tmp_rec_eLed_charge > 0:
                    P4_rec_ep.SetPxPyPzE(tmp_rec_eLed_px,tmp_rec_eLed_py,tmp_rec_eLed_pz,tmp_rec_eLed_E);
                    P4_rec_em.SetPxPyPzE(tmp_rec_eSub_px,tmp_rec_eSub_py,tmp_rec_eSub_pz,tmp_rec_eSub_E);
                else:
                    P4_rec_em.SetPxPyPzE(tmp_rec_eLed_px,tmp_rec_eLed_py,tmp_rec_eLed_pz,tmp_rec_eLed_E);
                    P4_rec_ep.SetPxPyPzE(tmp_rec_eSub_px,tmp_rec_eSub_py,tmp_rec_eSub_pz,tmp_rec_eSub_E);
                if tmp_rec_eLed_E !=0 and tmp_rec_eSub_E !=0:
                    self.h_rec_ep_px  .Fill(P4_rec_ep.Px()) 
                    self.h_rec_ep_py  .Fill(P4_rec_ep.Py()) 
                    self.h_rec_ep_pz  .Fill(P4_rec_ep.Pz()) 
                    self.h_rec_ep_E   .Fill(P4_rec_ep.E() ) 
                    self.h_rec_em_px  .Fill(P4_rec_em.Px()) 
                    self.h_rec_em_py  .Fill(P4_rec_em.Py()) 
                    self.h_rec_em_pz  .Fill(P4_rec_em.Pz()) 
                    self.h_rec_em_E   .Fill(P4_rec_em.E() ) 

                    self.h_rec_ep_cluE.Fill(tmp_rec_eLed_cluE if tmp_rec_eLed_charge > 0 else tmp_rec_eSub_cluE)
                    self.h_rec_em_cluE.Fill(tmp_rec_eSub_cluE if tmp_rec_eLed_charge > 0 else tmp_rec_eLed_cluE)


plot_path = './plots'

obj_orgin        = Obj('orgin','/junofs/users/wxfang/CEPC/CEPCOFF/ana/build/real_e1e1h_e2e2_0.root')
#obj_orgin        = Obj('orgin','/junofs/users/wxfang/CEPC/CEPCOFF/ana/reco_final_orgin.root')
#obj_orgin_dropEB = Obj('orgin_dropEB','/junofs/users/wxfang/CEPC/CEPCOFF/ana/reco_final_orgin_dropEB.root')
#obj_fake         = Obj('fake','/junofs/users/wxfang/CEPC/CEPCOFF/ana/reco_final_fakeEB.root')
#obj_fake         = Obj('fake','/junofs/users/wxfang/CEPC/CEPCOFF/ana/reco_final_fakeEB_corrFlag.root')
obj_fake         = Obj('fake','/junofs/users/wxfang/CEPC/CEPCOFF/ana/build/fake_e1e1h_e2e2_0.root')
obj_orgin.fill()
#obj_orgin_dropEB.fill()
obj_fake.fill()


do_plot(obj_orgin.h_em_p         ,"orgin_mc_em_P"       ,{'X':"P_{mc}^{e^{-}} GeV/c"       ,'Y':'Events'} )
do_plot(obj_orgin.h_em_px        ,"orgin_mc_em_Px"      ,{'X':"Px_{mc}^{e^{-}} GeV/c"      ,'Y':'Events'} )
do_plot(obj_orgin.h_em_py        ,"orgin_mc_em_Py"      ,{'X':"Py_{mc}^{e^{-}} GeV/c"      ,'Y':'Events'} )
do_plot(obj_orgin.h_em_pz        ,"orgin_mc_em_Pz"      ,{'X':"Pz_{mc}^{e^{-}} GeV/c"      ,'Y':'Events'} )
do_plot(obj_orgin.h_ep_p         ,"orgin_mc_ep_P"       ,{'X':"P_{mc}^{e^{+}} GeV/c"       ,'Y':'Events'} )
do_plot(obj_orgin.h_ep_px        ,"orgin_mc_ep_Px"      ,{'X':"Px_{mc}^{e^{+}} GeV/c"      ,'Y':'Events'} )
do_plot(obj_orgin.h_ep_py        ,"orgin_mc_ep_Py"      ,{'X':"Py_{mc}^{e^{+}} GeV/c"      ,'Y':'Events'} )
do_plot(obj_orgin.h_ep_pz        ,"orgin_mc_ep_Pz"      ,{'X':"Pz_{mc}^{e^{+}} GeV/c"      ,'Y':'Events'} )
do_plot(obj_orgin.h_m_ee         ,"orgin_mc_ee_Mass"    ,{'X':"M_{mc}^{ee} GeV/c^{2}"      ,'Y':'Events'} )
do_plot(obj_orgin.h_m_reco       ,"orgin_mc_reco_Mass"  ,{'X':"M_{mc}^{ee reco} GeV/c^{2}" ,'Y':'Events'} )


#do_plot_h2(obj_orgin.h_em_p         ,obj_fake.h_em_p         ,"mc_em_Mom"     ,{'X':"P_{mc}^{e^{-}} GeV/c"       ,'Y':'Events'}, 'e' )
#do_plot_h2(obj_orgin.h_ep_p         ,obj_fake.h_ep_p         ,"mc_ep_Mom"     ,{'X':"P_{mc}^{e^{+}} GeV/c"       ,'Y':'Events'}, 'e' )
#do_plot_h2(obj_orgin.h_m_ee         ,obj_fake.h_m_ee         ,"mc_ee_Mass"    ,{'X':"M_{mc}^{ee} GeV/c^{2}"      ,'Y':'Events'}, 'e' )
#do_plot_h2(obj_orgin.h_m_reco       ,obj_fake.h_m_reco       ,"mc_reco_Mass"  ,{'X':"M_{mc}^{ee reco} GeV/c^{2}" ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_eLed_px  ,obj_fake.h_rec_eLed_px  ,'rec_eLed_px'   ,{'X':'P_{x}^{led e} GeV/c'     ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_eLed_py  ,obj_fake.h_rec_eLed_py  ,'rec_eLed_py'   ,{'X':'P_{y}^{led e} GeV/c'     ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_eLed_pz  ,obj_fake.h_rec_eLed_pz  ,'rec_eLed_pz'   ,{'X':'P_{z}^{led e} GeV/c'     ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_eLed_E   ,obj_fake.h_rec_eLed_E   ,'rec_eLed_E'    ,{'X':'E^{led e} GeV'        ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_eLed_cluE,obj_fake.h_rec_eLed_cluE,'rec_eLed_cluE' ,{'X':'E_{cluster}^{led e} GeV','Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_eSub_px  ,obj_fake.h_rec_eSub_px  ,'rec_eSub_px'   ,{'X':'P_{x}^{sub e} GeV/c'     ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_eSub_py  ,obj_fake.h_rec_eSub_py  ,'rec_eSub_py'   ,{'X':'P_{y}^{sub e} GeV/c'     ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_eSub_pz  ,obj_fake.h_rec_eSub_pz  ,'rec_eSub_pz'   ,{'X':'P_{z}^{sub e} GeV/c'     ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_eSub_E   ,obj_fake.h_rec_eSub_E   ,'rec_eSub_E'    ,{'X':'E^{sub e} GeV'        ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_eSub_cluE,obj_fake.h_rec_eSub_cluE,'rec_eSub_cluE' ,{'X':'E_{cluster}^{sub e} GeV','Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_muLed_px ,obj_fake.h_rec_muLed_px ,'rec_muLed_px'  ,{'X':'P_{x}^{led #mu} GeV/c'   ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_muLed_py ,obj_fake.h_rec_muLed_py ,'rec_muLed_py'  ,{'X':'P_{y}^{led #mu} GeV/c'   ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_muLed_pz ,obj_fake.h_rec_muLed_pz ,'rec_muLed_pz'  ,{'X':'P_{z}^{led #mu} GeV/c'   ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_muLed_E  ,obj_fake.h_rec_muLed_E  ,'rec_muLed_E'   ,{'X':'E^{led #mu} GeV'      ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_muSub_px ,obj_fake.h_rec_muSub_px ,'rec_muSub_px'  ,{'X':'P_{x}^{sub #mu} GeV/c'   ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_muSub_py ,obj_fake.h_rec_muSub_py ,'rec_muSub_py'  ,{'X':'P_{y}^{sub #mu} GeV/c'   ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_muSub_pz ,obj_fake.h_rec_muSub_pz ,'rec_muSub_pz'  ,{'X':'P_{z}^{sub #mu} GeV/c'   ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_muSub_E  ,obj_fake.h_rec_muSub_E  ,'rec_muSub_E'   ,{'X':'E^{sub #mu} GeV'      ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_m_ee     ,obj_fake.h_rec_m_ee     ,'rec_m_ee'      ,{'X':'M^{ee} GeV/c^{2}'    ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_m_mumu   ,obj_fake.h_rec_m_mumu   ,'rec_m_mumu'    ,{'X':'M^{#mu#mu} GeV/c^{2}','Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_ep_px    ,obj_fake.h_rec_ep_px    ,'rec_ep_px'     ,{'X':'P_{x}^{e^{+}} GeV/c'     ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_ep_py    ,obj_fake.h_rec_ep_py    ,'rec_ep_py'     ,{'X':'P_{y}^{e^{+}} GeV/c'     ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_ep_pz    ,obj_fake.h_rec_ep_pz    ,'rec_ep_pz'     ,{'X':'P_{z}^{e^{+}} GeV/c'     ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_ep_E     ,obj_fake.h_rec_ep_E     ,'rec_ep_E'      ,{'X':'E^{e^{+}} GeV'        ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_ep_cluE  ,obj_fake.h_rec_ep_cluE  ,'rec_ep_cluE'   ,{'X':'E_{cluster}^{e^{+}} GeV','Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_em_px    ,obj_fake.h_rec_em_px    ,'rec_em_px'     ,{'X':'P_{x}^{e^{-}} GeV/c'     ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_em_py    ,obj_fake.h_rec_em_py    ,'rec_em_py'     ,{'X':'P_{y}^{e^{-}} GeV/c'     ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_em_pz    ,obj_fake.h_rec_em_pz    ,'rec_em_pz'     ,{'X':'P_{z}^{e^{-}} GeV/c'     ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_em_E     ,obj_fake.h_rec_em_E     ,'rec_em_E'      ,{'X':'E^{e^{-}} GeV'        ,'Y':'Events'}, 'e' )
do_plot_h2(obj_orgin.h_rec_em_cluE  ,obj_fake.h_rec_em_cluE  ,'rec_em_cluE'   ,{'X':'E_{cluster}^{e^{-}} GeV','Y':'Events'}, 'e' )

print('done')



