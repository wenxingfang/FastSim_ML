import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import json
from array import array
rt.gROOT.SetBatch(rt.kTRUE)
rt.TGaxis.SetMaxDigits(3);

def getTheta(r, z):
    if z == 0: return 90
    phi = math.atan(r/z)
    phi = 180*phi/math.pi
    if phi < 0 : phi = phi + 180
    return phi

def findLed(X, Y, Z, vPDG, pdg ):
    max_ = -1
    index = -1
    for i in range(len(X)):
        if vPDG[i] != pdg:continue
        if (X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i]) > max_ :
            max_ = (X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i])
            index = i
    return index



def findLedv1(E, Type, type_, y, z ):
    max_ = -1
    index = -1
    for i in range(len(E)):
        if Type[i] != type_:continue
        if abs(y[i]) > 600 or abs(z[i]) > 2000 :continue
        if E[i] > max_ :
            max_ = E[i]
            index = i
    return index


def findSub(X, Y, Z, led, vPDG, pdg):
    max_ = -1
    index = -1
    for i in range(len(X)):
        if i == led: continue
        if vPDG[i] != pdg:continue
        if (X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i]) > max_ :
            max_ = (X[i]*X[i] + Y[i]*Y[i] + Z[i]*Z[i])
            index = i
    return index


def findSubv1(E, led, Type, type_, y, z ):
    max_ = -1
    index = -1
    for i in range(len(E)):
        if i == led: continue
        if Type[i] != type_:continue
        if abs(y[i]) > 600 or abs(z[i]) > 2000 :continue
        if E[i] > max_ :
            max_ = E[i]
            index = i
    return index


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


def do_plot(hist,out_name,title, doFit):
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
    hist.GetXaxis().SetTitleOffset(1.2)
    hist.SetLineColor(rt.kBlue)
    hist.SetMarkerColor(rt.kBlue)
    hist.SetMarkerStyle(20)
    hist.Draw("histep")
    ############ 
    str_para = ''
    func = 0
    if doFit: 
        f1 = rt.TF1("f1", "gaus", 118, 132)
        #f1 = rt.TF1("f1", "gaus", 115, 135)
        f1.SetParameters(100,125,1)
        result = hist.Fit('f1','RLS')
        par0   = result.Parameter(0)
        err0   = result.ParError(0)
        par1   = result.Parameter(1)
        err1   = result.ParError(1)
        par2   = result.Parameter(2)
        err2   = result.ParError (2)
        status = result.Status()
        #print('%s:mean=%f,mean err=%f,sigma=%f, sigma err=%f, status=%d'%(hname, par1, err1, par2, err2, status))
        str_para = 'mean=%.3f#pm%.3f, #sigma=%.3f#pm%.3f'%(par1, err1, par2, err2)
        func = hist.GetFunction("f1")
        func.Draw('SAME')
    label_f=rt.TLatex(0.15, 0.89 , "%s"%(str(str_para)))
    label_f.SetTextSize(0.03)
    label_f.SetNDC()
    label_f.Draw()
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

def do_plot_h2(h_real, h_fake, out_name, title, plot_label):
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
    y_max=1.5*h_real.GetBinContent(h_real.GetMaximumBin()) if h_real.GetBinContent(h_real.GetMaximumBin()) > h_fake.GetBinContent(h_fake.GetMaximumBin()) else 1.5*h_fake.GetBinContent(h_fake.GetMaximumBin()) 
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
    if 'cm_x' in out_name or 'cm_y' in out_name or 'cm_z' in out_name:
        dummy.GetXaxis().SetLabelSize(0.03)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    if "cell_energy" not in out_name:
        dummy.GetXaxis().SetMoreLogLabels()
        dummy.GetXaxis().SetNoExponent()  
    dummy.GetXaxis().SetMoreLogLabels()
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

    label = rt.TLatex(0.38 , 0.82, plot_label)
    label.SetTextAlign(32)
    label.SetTextSize(0.035)
    label.SetNDC(rt.kTRUE)
    label.Draw() 
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

###############
class Obj:
    def __init__(self, name, fileName):
        self.name = name
        self.file_name = fileName
        str_ext = self.name
        self.Led_PID = 22
        self.Sub_PID = 22
        self.doAll = False
        self.h_mc_led_g_px          = rt.TH1F('%s_h_mc_led_g_px'        %(str_ext) ,'',20,-100,100)
        self.h_mc_led_g_py          = rt.TH1F('%s_h_mc_led_g_py'        %(str_ext) ,'',20,-100,100)
        self.h_mc_led_g_pz          = rt.TH1F('%s_h_mc_led_g_pz'        %(str_ext) ,'',20,-100,100)
        self.h_mc_led_g_p           = rt.TH1F('%s_h_mc_led_g_p'         %(str_ext) ,'',15,0,150)
        self.h_mc_sub_g_px          = rt.TH1F('%s_h_mc_sub_g_px'        %(str_ext) ,'',20,-100,100)
        self.h_mc_sub_g_py          = rt.TH1F('%s_h_mc_sub_g_py'        %(str_ext) ,'',20,-100,100)
        self.h_mc_sub_g_pz          = rt.TH1F('%s_h_mc_sub_g_pz'        %(str_ext) ,'',20,-100,100)
        self.h_mc_sub_g_p           = rt.TH1F('%s_h_mc_sub_g_p'         %(str_ext) ,'',15,0,150)
        self.h_mc_m_gg           = rt.TH1F('%s_h_mc_m_gg'         %(str_ext) ,'',100,100,150)
        self.h_mc_gg_reco         = rt.TH1F('%s_h_mc_gg_reco'       %(str_ext) ,'',60,60,120)
        self.h_rec_led_g_px    = rt.TH1F('%s_h_rec_led_g_px'  %(str_ext) ,'',20,-100,100)
        self.h_rec_led_g_py    = rt.TH1F('%s_h_rec_led_g_py'  %(str_ext) ,'',20,-100,100)
        self.h_rec_led_g_pz    = rt.TH1F('%s_h_rec_led_g_pz'  %(str_ext) ,'',20,-100,100)
        self.h_rec_led_g_E     = rt.TH1F('%s_h_rec_led_g_E'   %(str_ext) ,'',30,0,200)
        self.h_rec_led_g_E_res = rt.TH1F('%s_h_rec_led_g_E_res'   %(str_ext) ,'',40,-0.2,0.2)
        self.h_rec_led_g_cluE  = rt.TH1F('%s_h_rec_led_g_cluE'%(str_ext) ,'',30,0,150)
        self.h_rec_sub_g_px    = rt.TH1F('%s_h_rec_sub_g_px'  %(str_ext) ,'',20,-100,100)
        self.h_rec_sub_g_py    = rt.TH1F('%s_h_rec_sub_g_py'  %(str_ext) ,'',20,-100,100)
        self.h_rec_sub_g_pz    = rt.TH1F('%s_h_rec_sub_g_pz'  %(str_ext) ,'',20,-100,100)
        self.h_rec_sub_g_E     = rt.TH1F('%s_h_rec_sub_g_E'   %(str_ext) ,'',20,0,100)
        self.h_rec_sub_g_E_res = rt.TH1F('%s_h_rec_sub_g_E_res'   %(str_ext) ,'',40,-0.2,0.2)
        self.h_rec_sub_g_cluE  = rt.TH1F('%s_h_rec_sub_g_cluE'%(str_ext) ,'',20,0,100)
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
        self.h_rec_m_gg       = rt.TH1F('%s_h_rec_m_gg'     %(str_ext) ,'',100,100,150)
        self.h_rec_m_gg_reco  = rt.TH1F('%s_h_rec_m_gg_reco'%(str_ext) ,'',60,60,120)
        self.h_rec_m_mumu     = rt.TH1F('%s_h_rec_m_mumu'   %(str_ext) ,'',50,100,150)
        self.h_HitEn_tot      = rt.TH1F('%s_h_HitEn_tot'    %(str_ext) ,'',25,0, 250)
        self.h_En_ratio       = rt.TH1F('%s_h_En_ratio'     %(str_ext) ,'',20,0, 2)
        self.h_Hit_cm_x       = rt.TH1F('%s_h_Hit_cm_x'     %(str_ext) ,'',25,-2500,2500)
        self.h_Hit_cm_y       = rt.TH1F('%s_h_Hit_cm_y'     %(str_ext) ,'',25,-2500,2500)
        self.h_Hit_cm_z       = rt.TH1F('%s_h_Hit_cm_z'     %(str_ext) ,'',40,-4000,4000)
        self.h_rec_led_g_cm_x = rt.TH1F('%s_h_rec_led_g_cm_x'     %(str_ext) ,'',21,-2100,2100)
        self.h_rec_led_g_cm_y = rt.TH1F('%s_h_rec_led_g_cm_y'     %(str_ext) ,'',21,-2100,2100)
        self.h_rec_led_g_cm_z = rt.TH1F('%s_h_rec_led_g_cm_z'     %(str_ext) ,'',21,-2100,2100)
        self.h_rec_led_g_cm_theta = rt.TH1F('%s_h_rec_led_g_cm_theta'     %(str_ext) ,'',45,0,180)
        self.h_rec_led_g_cm_phi   = rt.TH1F('%s_h_rec_led_g_cm_phi'       %(str_ext) ,'',36,0,360)
        self.h_rec_sub_g_cm_x = rt.TH1F('%s_h_rec_sub_g_cm_x'     %(str_ext) ,'',21,-2100,2100)
        self.h_rec_sub_g_cm_y = rt.TH1F('%s_h_rec_sub_g_cm_y'     %(str_ext) ,'',21,-2100,2100)
        self.h_rec_sub_g_cm_z = rt.TH1F('%s_h_rec_sub_g_cm_z'     %(str_ext) ,'',21,-2100,2100)
        self.h_rec_sub_g_cm_theta = rt.TH1F('%s_h_rec_sub_g_cm_theta'     %(str_ext) ,'',45,0,180)
        self.h_rec_sub_g_cm_phi   = rt.TH1F('%s_h_rec_sub_g_cm_phi'       %(str_ext) ,'',36,0,360)

    def setPID(self, p1, p2):
        self.Led_PID = p1
        self.Sub_PID = p2
    def setDoAll(self, doAll):
        self.doAll = doAll
    def fill(self):
        P4_mc_led_g = rt.TLorentzVector()
        P4_mc_sub_g = rt.TLorentzVector()
        P4_mc_gg      = rt.TLorentzVector()
        P4_mc_gg_reco = rt.TLorentzVector()
        P4_rec_led_g = rt.TLorentzVector()
        P4_rec_sub_g = rt.TLorentzVector()
        P4_cms = rt.TLorentzVector(0,0,0,240)   
        FileName = self.file_name
        treeName='evt'
        chain =rt.TChain(treeName)
        chain.Add(FileName)
        tree = chain
        totalEntries=tree.GetEntries()
        print (totalEntries)
        for entryNum in range(0, tree.GetEntries()):
            tree.GetEntry(entryNum)
            tmp_PID   = getattr(tree, "m_pReco_PID")
            tmp_mass  = getattr(tree, "m_pReco_mass")
            tmp_charge= getattr(tree, "m_pReco_charge")
            tmp_E     = getattr(tree, "m_pReco_energy")
            tmp_px    = getattr(tree, "m_pReco_px")
            tmp_py    = getattr(tree, "m_pReco_py")
            tmp_pz    = getattr(tree, "m_pReco_pz")

            if self.doAll==True:
                total_px = 0
                total_py = 0
                total_pz = 0
                total_E  = 0
                for ii in range(len(tmp_PID)):
                    total_px += tmp_px[ii] 
                    total_py += tmp_py[ii] 
                    total_pz += tmp_pz[ii] 
                    total_E  += tmp_E [ii] 
                P4_rec_led_g.SetPxPyPzE(total_px,total_py,total_pz,total_E);
                self.h_rec_led_g_px  .Fill(P4_rec_led_g.Px()) 
                self.h_rec_led_g_py  .Fill(P4_rec_led_g.Py()) 
                self.h_rec_led_g_pz  .Fill(P4_rec_led_g.Pz()) 
                self.h_rec_led_g_E   .Fill(P4_rec_led_g.E() ) 
                self.h_rec_m_gg      .Fill(P4_rec_led_g.M())
                self.h_rec_m_gg_reco .Fill((P4_cms-P4_rec_led_g).M())
            else:
                hasEle = False
                for ii in range(len(tmp_PID)): 
                    if abs(tmp_PID[ii])==11: 
                        hasEle=True
                        break
                #if hasEle:continue
                led_gamma = -1 
                sub_gamma = -1 
                if self.Led_PID == self.Sub_PID:
                    led_gamma = findLed(tmp_px, tmp_py, tmp_pz, tmp_PID, self.Led_PID)
                    sub_gamma = findSub(tmp_px, tmp_py, tmp_pz, led_gamma, tmp_PID, self.Sub_PID)
                else:
                    led_gamma = findLed(tmp_px, tmp_py, tmp_pz, tmp_PID, self.Led_PID)
                    sub_gamma = findLed(tmp_px, tmp_py, tmp_pz, tmp_PID, self.Sub_PID)

                if led_gamma !=-1 and sub_gamma != -1:
                    P4_rec_led_g.SetPxPyPzE(tmp_px[led_gamma],tmp_py[led_gamma],tmp_pz[led_gamma],tmp_E[led_gamma]);
                    P4_rec_sub_g.SetPxPyPzE(tmp_px[sub_gamma],tmp_py[sub_gamma],tmp_pz[sub_gamma],tmp_E[sub_gamma]);
                    #if P4_rec_led_g.DeltaR(P4_mc_led_g)>0.3 or P4_rec_sub_g.DeltaR(P4_mc_sub_g)>0.3: continue  #matching
                    self.h_rec_led_g_px  .Fill(P4_rec_led_g.Px()) 
                    self.h_rec_led_g_py  .Fill(P4_rec_led_g.Py()) 
                    self.h_rec_led_g_pz  .Fill(P4_rec_led_g.Pz()) 
                    self.h_rec_led_g_E   .Fill(P4_rec_led_g.E() ) 
                    #self.h_rec_led_g_E_res .Fill( (P4_rec_led_g.E()-P4_mc_led_g.E())/P4_mc_led_g.E() ) 
                    #self.h_rec_led_g_cluE.Fill(tmp_rec_cluE[led_gamma])
                    self.h_rec_sub_g_px  .Fill(P4_rec_sub_g.Px()) 
                    self.h_rec_sub_g_py  .Fill(P4_rec_sub_g.Py()) 
                    self.h_rec_sub_g_pz  .Fill(P4_rec_sub_g.Pz()) 
                    self.h_rec_sub_g_E   .Fill(P4_rec_sub_g.E() ) 
                    #self.h_rec_sub_g_E_res .Fill( (P4_rec_sub_g.E()-P4_mc_sub_g.E())/P4_mc_sub_g.E() ) 
                    #self.h_rec_sub_g_cluE.Fill(tmp_rec_cluE[sub_gamma])
                    self.h_rec_m_gg      .Fill((P4_rec_led_g+P4_rec_sub_g).M())
                    self.h_rec_m_gg_reco .Fill((P4_cms-P4_rec_led_g-P4_rec_sub_g).M())

############ BEGIN ##############

plot_path = './plots'

#obj_pan        = Obj('PanPfo','/junofs/users/wxfang/MyGit/CEPCSW/Reconstruction/PFA/Pandora/GaudiPandora/Ana_nnHaa_NewDigi.root')
obj_pan        = Obj('PanPfo','/junofs/users/wxfang/MyGit/CEPCSW/Reconstruction/PFA/Pandora/GaudiPandora/Ana_nnHaa_G2CDEM0p9.root')
obj_pan.setPID(p1=22, p2=22)
str_led_pa = "#gamma"
str_sub_pa = "#gamma"
#obj_pan        = Obj('PanPfo','/junofs/users/wxfang/MyGit/CEPCSW/Reconstruction/PFA/Pandora/GaudiPandora/Ana_nnHe2e2_G2CDEM0p9.root')
#obj_pan.setPID(p1=13, p2=-13)
#str_led_pa = "#mu^{-}"
#str_sub_pa = "#mu^{+}"

#obj_pan        = Obj('PanPfo','/junofs/users/wxfang/MyGit/CEPCSW/Reconstruction/PFA/Pandora/GaudiPandora/Ana_nnHbb_G2CDEM0p9.root')
#obj_pan.setDoAll(True)
#str_led_pa = "bb"
#str_sub_pa = ""
#obj_pan        = Obj('PanPfo','/junofs/users/wxfang/MyGit/CEPCSW/Reconstruction/PFA/Pandora/GaudiPandora/Ana_nnHgg_G2CDEM0p9.root')
#obj_pan.setDoAll(True)
#str_led_pa = "gg"
#str_sub_pa = ""


obj_pan.fill()
#obj_fake         = Obj('fake' ,'/junofs/users/wxfang/CEPC/CEPCOFF/ana/fake_nnh_aa.root')
#obj_fake.fill()


do_plot(obj_pan.h_rec_led_g_E         ,"Pan_reco_led_g_E"       ,{'X':"E_{pfo}^{%s} GeV"%str_led_pa                           ,'Y':'Events'}, False )
do_plot(obj_pan.h_rec_led_g_px        ,"Pan_reco_led_g_Px"      ,{'X':"Px_{pfo}^{%s} GeV/c"%str_led_pa                        ,'Y':'Events'}, False )
do_plot(obj_pan.h_rec_led_g_py        ,"Pan_reco_led_g_Py"      ,{'X':"Py_{pfo}^{%s} GeV/c"%str_led_pa                        ,'Y':'Events'}, False )
do_plot(obj_pan.h_rec_led_g_pz        ,"Pan_reco_led_g_Pz"      ,{'X':"Pz_{pfo}^{%s} GeV/c"%str_led_pa                        ,'Y':'Events'}, False )
do_plot(obj_pan.h_rec_sub_g_E         ,"Pan_reco_sub_g_E"       ,{'X':"E_{pfo}^{%s} GeV"%str_sub_pa                           ,'Y':'Events'}, False )
do_plot(obj_pan.h_rec_sub_g_px        ,"Pan_reco_sub_g_Px"      ,{'X':"Px_{pfo}^{%s} GeV/c"%str_sub_pa                        ,'Y':'Events'}, False )
do_plot(obj_pan.h_rec_sub_g_py        ,"Pan_reco_sub_g_Py"      ,{'X':"Py_{pfo}^{%s} GeV/c"%str_sub_pa                        ,'Y':'Events'}, False )
do_plot(obj_pan.h_rec_sub_g_pz        ,"Pan_reco_sub_g_Pz"      ,{'X':"Pz_{pfo}^{%s} GeV/c"%str_sub_pa                        ,'Y':'Events'}, False )
do_plot(obj_pan.h_rec_m_gg            ,"Pan_reco_gg_Mass"       ,{'X':"M_{pfo}^{%s%s} GeV/c^{2}"%(str_led_pa,str_sub_pa)      ,'Y':'Events'}, True  )
do_plot(obj_pan.h_rec_m_gg_reco       ,"Pan_reco_gg_reco_Mass"  ,{'X':"M_{pfo}^{%s%s reco} GeV/c^{2}"%(str_led_pa,str_sub_pa) ,'Y':'Events'}, False )




print('saved led E hist with events=',obj_pan.h_rec_led_g_E.GetSumOfWeights())
print('done')
'''
do_plot_h2(obj_pan.h_rec_led_g_px  ,obj_fake.h_rec_led_g_px  ,'rec_led_g_px'   ,{'X':'P_{x} GeV/c'     ,'Y':'Events'}, 'leading #gamma' )
do_plot_h2(obj_pan.h_rec_led_g_py  ,obj_fake.h_rec_led_g_py  ,'rec_led_g_py'   ,{'X':'P_{y} GeV/c'     ,'Y':'Events'}, 'leading #gamma' )
do_plot_h2(obj_pan.h_rec_led_g_pz  ,obj_fake.h_rec_led_g_pz  ,'rec_led_g_pz'   ,{'X':'P_{z} GeV/c'     ,'Y':'Events'}, 'leading #gamma' )
do_plot_h2(obj_pan.h_rec_led_g_E   ,obj_fake.h_rec_led_g_E   ,'rec_led_g_E'    ,{'X':'E GeV'           ,'Y':'Events'}, 'leading #gamma' )
do_plot_h2(obj_pan.h_rec_led_g_E_res   ,obj_fake.h_rec_led_g_E_res   ,'rec_led_g_E_res' ,{'X':'(E^{rec}-E_{mc})/E_{mc} '           ,'Y':'Events'}, 'leading #gamma' )
do_plot_h2(obj_pan.h_rec_led_g_cluE,obj_fake.h_rec_led_g_cluE,'rec_led_g_cluE' ,{'X':'E_{cluster} GeV' ,'Y':'Events'}, 'leading #gamma' )
do_plot_h2(obj_pan.h_rec_sub_g_px  ,obj_fake.h_rec_sub_g_px  ,'rec_sub_g_px'   ,{'X':'P_{x} GeV/c'     ,'Y':'Events'}, 'sub-leading #gamma' )
do_plot_h2(obj_pan.h_rec_sub_g_py  ,obj_fake.h_rec_sub_g_py  ,'rec_sub_g_py'   ,{'X':'P_{y} GeV/c'     ,'Y':'Events'}, 'sub-leading #gamma' )
do_plot_h2(obj_pan.h_rec_sub_g_pz  ,obj_fake.h_rec_sub_g_pz  ,'rec_sub_g_pz'   ,{'X':'P_{z} GeV/c'     ,'Y':'Events'}, 'sub-leading #gamma' )
do_plot_h2(obj_pan.h_rec_sub_g_E   ,obj_fake.h_rec_sub_g_E   ,'rec_sub_g_E'    ,{'X':'E GeV'           ,'Y':'Events'}, 'sub-leading #gamma' )
do_plot_h2(obj_pan.h_rec_sub_g_E_res   ,obj_fake.h_rec_sub_g_E_res   ,'rec_sub_g_E_res'    ,{'X':'(E^{rec}-E_{mc})/E_{mc} '           ,'Y':'Events'}, 'sub-leading #gamma' )
do_plot_h2(obj_pan.h_rec_sub_g_cluE,obj_fake.h_rec_sub_g_cluE,'rec_sub_g_cluE' ,{'X':'E_{cluster} GeV' ,'Y':'Events'}, 'sub-leading #gamma' )
do_plot_h2(obj_pan.h_rec_m_gg      ,obj_fake.h_rec_m_gg      ,"rec_m_gg"       ,{'X':"M^{#gamma#gamma} GeV/c^{2}"   ,'Y':'Events'}, '' )
do_plot_h2(obj_pan.h_rec_m_gg_reco ,obj_fake.h_rec_m_gg_reco ,"rec_m_gg_reco"  ,{'X':"M_{reco}^{#gamma#gamma} GeV/c^{2}"   ,'Y':'Events'}, '' )
do_plot_h2(obj_pan.h_HitEn_tot     ,obj_fake.h_HitEn_tot     ,"HitEn_tot"      ,{'X':"#sumHit_{evt} GeV"            ,'Y':'Events'}, '' )
do_plot_h2(obj_pan.h_En_ratio      ,obj_fake.h_En_ratio      ,"En_ratio"       ,{'X':"#sumHit_{evt}/E_{#gamma#gamma}"            ,'Y':'Events'}, '' )
do_plot_h2(obj_pan.h_Hit_cm_x      ,obj_fake.h_Hit_cm_x      ,"Hit_cm_x"       ,{'X':"#sumHit_{cm x} cm"               ,'Y':'Events'}, '' )
do_plot_h2(obj_pan.h_Hit_cm_y      ,obj_fake.h_Hit_cm_y      ,"Hit_cm_y"       ,{'X':"#sumHit_{cm y} cm"               ,'Y':'Events'}, '' )
do_plot_h2(obj_pan.h_Hit_cm_z      ,obj_fake.h_Hit_cm_z      ,"Hit_cm_z"       ,{'X':"#sumHit_{cm z} cm"               ,'Y':'Events'}, '' )

do_plot_h2(obj_pan.h_rec_led_g_cm_x    ,obj_fake.h_rec_led_g_cm_x     ,"rec_led_g_cm_x"       ,{'X':"cm x_{cluster} (cm)"               ,'Y':'Events'}, 'leading #gamma' )
do_plot_h2(obj_pan.h_rec_led_g_cm_y    ,obj_fake.h_rec_led_g_cm_y     ,"rec_led_g_cm_y"       ,{'X':"cm y_{cluster} (cm)"               ,'Y':'Events'}, 'leading #gamma' )
do_plot_h2(obj_pan.h_rec_led_g_cm_z    ,obj_fake.h_rec_led_g_cm_z     ,"rec_led_g_cm_z"       ,{'X':"cm z_{cluster} (cm)"               ,'Y':'Events'}, 'leading #gamma' )
do_plot_h2(obj_pan.h_rec_led_g_cm_theta    ,obj_fake.h_rec_led_g_cm_theta     ,"rec_led_g_cm_theta"       ,{'X':"#theta_{cluster cm} (degree)"             ,'Y':'Events'}, 'leading #gamma' )
do_plot_h2(obj_pan.h_rec_led_g_cm_phi      ,obj_fake.h_rec_led_g_cm_phi       ,"rec_led_g_cm_phi"         ,{'X':"#phi_{cluster cm} (degree)"               ,'Y':'Events'}, 'leading #gamma' )
do_plot_h2(obj_pan.h_rec_sub_g_cm_x    ,obj_fake.h_rec_sub_g_cm_x     ,"rec_sub_g_cm_x"       ,{'X':"cm x_{cluster} (cm)"               ,'Y':'Events'}, 'sub-leading #gamma' )
do_plot_h2(obj_pan.h_rec_sub_g_cm_y    ,obj_fake.h_rec_sub_g_cm_y     ,"rec_sub_g_cm_y"       ,{'X':"cm y_{cluster} (cm)"               ,'Y':'Events'}, 'sub-leading #gamma' )
do_plot_h2(obj_pan.h_rec_sub_g_cm_z    ,obj_fake.h_rec_sub_g_cm_z     ,"rec_sub_g_cm_z"       ,{'X':"cm z_{cluster} (cm)"               ,'Y':'Events'}, 'sub-leading #gamma' )
do_plot_h2(obj_pan.h_rec_sub_g_cm_theta    ,obj_fake.h_rec_sub_g_cm_theta     ,"rec_sub_g_cm_theta"       ,{'X':"#theta_{cluster cm} (degree)"             ,'Y':'Events'}, 'leading #gamma' )
do_plot_h2(obj_pan.h_rec_sub_g_cm_phi      ,obj_fake.h_rec_sub_g_cm_phi       ,"rec_sub_g_cm_phi"         ,{'X':"#phi_{cluster cm} (degree)"               ,'Y':'Events'}, 'leading #gamma' )
'''
