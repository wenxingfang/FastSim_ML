import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import json
import random
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
    hist.GetXaxis().SetTitleOffset(1.2)
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

def do_plot_h2(h_real, h_fake, out_name, title, plot_label, scaleToG4):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.12)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    if scaleToG4:
#    h_real.Scale(1/h_real.GetSumOfWeights())
        h_fake.Scale(h_real.GetSumOfWeights()/h_fake.GetSumOfWeights())
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
    h_fake .SetMarkerStyle(24)
    h_real.Draw("same:pe")
    h_real.Draw("same:pe")
    h_fake.Draw("same:histe")
    dummy.Draw("AXISSAME")
    legend = rt.TLegend(0.6,0.7,0.85,0.85)
    legend.AddEntry(h_real ,'G4','lep')
    legend.AddEntry(h_fake ,"FS",'lep')
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
    def __init__(self, name, fileName, max_evt):
        self.name = name
        self.file_name = fileName
        str_ext = self.name
        self.max_evt = max_evt
        self.h_Hit_z_x  = rt.TH2F('%s_Hit_z_x' %(str_ext)   , '', 200, -100, 100 , 200, 1840, 2040)
        self.h_Hit_y_x  = rt.TH2F('%s_Hit_y_x' %(str_ext)   , '', 200, -100, 100 , 200, 1840, 2040)
        self.h_Hit_z_y  = rt.TH2F('%s_Hit_z_y' %(str_ext)   , '', 200, -100, 100 , 200, -100, 100 )
        #self.h_Hit_E    = rt.TH1F('%s_Hit_E'   %(str_ext)   , '', 20, 0, 0.5)
        #self.h_Hit_cm_x = rt.TH1F('%s_Hit_mc_x'%(str_ext)   , '', 100, 1850, 2000)
        #self.h_Hit_cm_y = rt.TH1F('%s_Hit_mc_y'%(str_ext)   , '', 65, -500, 800)
        #self.h_Hit_cm_z = rt.TH1F('%s_Hit_mc_z'%(str_ext)   , '', 50, -1000, 1000)
        self.h_Hit_res  = rt.TH1F('%s_Hit_res' %(str_ext)   , '', 60, 0.005, 0.035)
        self.h_Hit_E    = rt.TH1F('%s_Hit_E'   %(str_ext)   , '', 100, 0, 1000)
        #self.h_Hit_res  = rt.TH1F('%s_Hit_res' %(str_ext)   , '', 40, 0.0, 0.04)
        #self.h_Hit_E    = rt.TH1F('%s_Hit_E'   %(str_ext)   , '', 100, 0, 0.05)
        self.h_Hit_cm_x = rt.TH1F('%s_Hit_mc_x'%(str_ext)   , '', 100, 0, 2000)
        self.h_Hit_cm_y = rt.TH1F('%s_Hit_mc_y'%(str_ext)   , '', 200, -2000, 2000)
        self.h_Hit_cm_z = rt.TH1F('%s_Hit_mc_z'%(str_ext)   , '', 200, -2000, 2000)
        self.h_Mom      = rt.TH1F('%s_MC_mom'  %(str_ext)   , '', 110, 0, 1100)
        self.h_theta    = rt.TH1F('%s_MC_theta'%(str_ext)   , '', 180, 0, 180)
        self.h_phi      = rt.TH1F('%s_MC_phi'  %(str_ext)   , '', 362, -1, 361)
        self.h_mc_x     = rt.TH1F('%s_MC_x'    %(str_ext)   , '', 200, 1800, 2000)
        self.h_Mom_lib  = rt.TH1F('%s_mom_lib' %(str_ext)   , '', 110, 0, 1100)
        self.h_theta_lib= rt.TH1F('%s_theta_lib'%(str_ext)  , '', 180, 0, 180)
        self.h_phi_lib  = rt.TH1F('%s_phi_lib'  %(str_ext)  , '', 362, -1, 361)
        self.h_x_lib    = rt.TH1F('%s_x_lib'    %(str_ext)  , '', 200, 1800, 2000)

    def fill(self):
        FileName = self.file_name
        treeName='evt'
        chain =rt.TChain(treeName)
        chain.Add(FileName)
        tree = chain
        totalEntries=tree.GetEntries()
        max_event = self.max_evt if self.max_evt < totalEntries else totalEntries
        print ('tot evt=',totalEntries,'.max evt=',max_event)
        for entryNum in range(0, max_event):
            tree.GetEntry(int(entryNum))
            if int(entryNum)%100000 ==0:print('processed:', 100.0*int(entryNum)/totalEntries,'%%')
            tmp_mc_Px   = getattr(tree, "m_mc_Px")
            tmp_mc_Py   = getattr(tree, "m_mc_Py")
            tmp_mc_Pz   = getattr(tree, "m_mc_Pz")
            tmp_Hit_x     = getattr(tree, "m_Hit_x")
            tmp_Hit_y     = getattr(tree, "m_Hit_y")
            tmp_Hit_z     = getattr(tree, "m_Hit_z")
            tmp_Hit_E     = getattr(tree, "m_Hit_E")
            tmp_mc_Pdg    = getattr(tree, "m_mc_Pdg")
            En = 0
            cm_x = 0
            cm_y = 0
            cm_z = 0
            #print('mc_Pdg=',tmp_mc_Pdg)
            if len(tmp_mc_Px) !=1 or len(tmp_Hit_x)==0 : continue
            if tmp_mc_Pdg[0]==11 : continue
            #if tmp_mc_Pdg[0] !=22 : continue ## for photon
            #if tmp_mc_Pdg[0] != 11 : continue ## for ele
            tmp_Mom = 1000*math.sqrt(tmp_mc_Px[0]*tmp_mc_Px[0] + tmp_mc_Py[0]*tmp_mc_Py[0] + tmp_mc_Pz[0]*tmp_mc_Pz[0])
            if tmp_Mom >500 : continue
            cos_theta = tmp_mc_Pz[0]*1000/tmp_Mom
            tmp_theta = math.acos(cos_theta)*180/math.pi
            cos_phi   = tmp_mc_Px[0]/math.sqrt(tmp_mc_Px[0]*tmp_mc_Px[0] + tmp_mc_Py[0]*tmp_mc_Py[0])
            tmp_phi   = math.acos(cos_phi)*180/math.pi if tmp_mc_Py[0] > 0 else 360-math.acos(cos_phi)*180/math.pi
            tmp_phi1  = math.acos(cos_phi)*180/math.pi 
            #if tmp_phi < 22.5 or tmp_phi>67.5: continue
            #if tmp_phi1 >= 22.5: continue
            #if tmp_phi1 >= 15: continue
            #print('tmp_phi=',tmp_phi)
            #if tmp_Mom<100:continue
            #if tmp_mc_Px[0]<0:continue
            self.h_Mom.Fill(tmp_Mom)
            self.h_theta.Fill(tmp_theta)
            self.h_phi  .Fill(tmp_phi)
            #print('evt=',entryNum,',Mom=',tmp_Mom,',theta=',tmp_theta,',phi=',tmp_phi)
            for i in range(len(tmp_Hit_x)):
                '''
                self.h_Hit_z_x.Fill(tmp_Hit_z[i], tmp_Hit_x[i])
                self.h_Hit_y_x.Fill(tmp_Hit_y[i], tmp_Hit_x[i])
                self.h_Hit_z_y.Fill(tmp_Hit_z[i], tmp_Hit_y[i])
                '''
                #if(entryNum==304) :print(tmp_Hit_x[i])
                #if(tmp_Hit_x[i] <= 1870): print(tmp_Hit_x[i])
                #if tmp_Hit_x[i] >= 1870: continue
                #if tmp_Hit_x[i] < 1880 or tmp_Hit_x[i] > 1900: continue
                #if tmp_Hit_x[i] < 1940 or tmp_Hit_x[i] > 1960: continue
                #print(tmp_Hit_x[i])
                cm_x += tmp_Hit_x[i]*tmp_Hit_E[i]
                cm_y += tmp_Hit_y[i]*tmp_Hit_E[i]
                cm_z += tmp_Hit_z[i]*tmp_Hit_E[i]
                En += tmp_Hit_E[i]
                #print('En=',En)
            #print('evt=',entryNum,',En=',En*1000)
            if(En == 0) : continue
            self.h_Hit_E.Fill(En*1000)#GeV
            self.h_Hit_res.Fill(1000*En/tmp_Mom)#GeV
            self.h_Hit_cm_x.Fill(cm_x/En)
            self.h_Hit_cm_y.Fill(cm_y/En)
            self.h_Hit_cm_z.Fill(cm_z/En)
    def fill_sp(self):
        FileName = self.file_name
        treeName='evt'
        chain =rt.TChain(treeName)
        chain.Add(FileName)
        tree = chain
        totalEntries=tree.GetEntries()
        max_event = self.max_evt if self.max_evt < totalEntries else totalEntries
        print ('tot evt=',totalEntries,'.max evt=',max_event)
        for entryNum in range(0, max_event):
            tree.GetEntry(int(entryNum))
            if int(entryNum)%100000 ==0:print('processed:', 100.0*int(entryNum)/totalEntries,'%%')
            tmp_mc_x     = getattr(tree, "m_Find_point_x")
            tmp_mc_y     = getattr(tree, "m_Find_point_y")
            tmp_mc_z     = getattr(tree, "m_Find_point_z")
            tmp_mc_Px    = getattr(tree, "m_Find_mom_x")
            tmp_mc_Py    = getattr(tree, "m_Find_mom_y")
            tmp_mc_Pz    = getattr(tree, "m_Find_mom_z")
            tmp_mc_Pdg   = getattr(tree, "m_Find_pid")
            tmp_Hit_E    = getattr(tree, "m_Find_sum_hit_E")
            tmp_Hit_e    = getattr(tree, "m_Find_sum_hit_e")
            if len(tmp_mc_x) != len(tmp_Hit_E):
                print('not same len')
                continue
            for i in range(len(tmp_mc_x)):
                if tmp_mc_Pdg[i]==11 : continue
                tmp_Mom = math.sqrt(tmp_mc_Px[i]*tmp_mc_Px[i] + tmp_mc_Py[i]*tmp_mc_Py[i] + tmp_mc_Pz[i]*tmp_mc_Pz[i])
                if tmp_Mom >500 : continue
                cos_theta = tmp_mc_Pz[i]/tmp_Mom
                tmp_theta = math.acos(cos_theta)*180/math.pi
                cos_phi   = tmp_mc_Px[i]/math.sqrt(tmp_mc_Px[i]*tmp_mc_Px[i] + tmp_mc_Py[i]*tmp_mc_Py[i])
                tmp_phi   = math.acos(cos_phi)*180/math.pi if tmp_mc_Py[0] > 0 else 360-math.acos(cos_phi)*180/math.pi
                self.h_Mom.Fill(tmp_Mom)
                self.h_theta.Fill(tmp_theta)
                self.h_phi  .Fill(tmp_phi)
                self.h_Hit_E.Fill(tmp_Hit_e[i])#MeV
                self.h_Hit_res.Fill(tmp_Hit_e[i]/tmp_Mom)#GeV
    def fill_sp_lib(self):
        f_out      = open('./start_points/sp_G4.txt'     ,'w')
        f_out_lib  = open('./start_points/sp_lib.txt'    ,'w')
        f_out_lib1 = open('./start_points/sp_lib_sxy.txt','w')
        FileName = self.file_name
        treeName='evt'
        chain =rt.TChain(treeName)
        chain.Add(FileName)
        tree = chain
        totalEntries=tree.GetEntries()
        max_event = self.max_evt if self.max_evt < totalEntries else totalEntries
        print ('tot evt=',totalEntries,'.max evt=',max_event)
        for entryNum in range(0, max_event):
            tree.GetEntry(int(entryNum))
            if int(entryNum)%100000 ==0:print('processed:', 100.0*int(entryNum)/totalEntries,'%%')
            tmp_mc_x     = getattr(tree, "m_Find_point_x")
            tmp_mc_y     = getattr(tree, "m_Find_point_y")
            tmp_mc_z     = getattr(tree, "m_Find_point_z")
            tmp_mc_Px    = getattr(tree, "m_Find_mom_x")
            tmp_mc_Py    = getattr(tree, "m_Find_mom_y")
            tmp_mc_Pz    = getattr(tree, "m_Find_mom_z")
            tmp_mc_Pdg   = getattr(tree, "m_Find_pid")
            tmp_Hit_E    = getattr(tree, "m_Find_sum_hit_E")
            tmp_Hit_e    = getattr(tree, "m_Find_sum_hit_e")
            tmp_Lib_x    = getattr(tree, "m_FindLib_x")
            tmp_Lib_mom_x= getattr(tree, "m_FindLib_mom_x")
            tmp_Lib_mom_y= getattr(tree, "m_FindLib_mom_y")
            tmp_Lib_mom_z= getattr(tree, "m_FindLib_mom_z")
            if len(tmp_mc_x) != len(tmp_Hit_E):
                print('not same len')
                continue
            for i in range(len(tmp_mc_x)):
                tmp_Mom = math.sqrt(tmp_mc_Px[i]*tmp_mc_Px[i] + tmp_mc_Py[i]*tmp_mc_Py[i] + tmp_mc_Pz[i]*tmp_mc_Pz[i])
                tmp_Mom_lib = math.sqrt(tmp_Lib_mom_x[i]*tmp_Lib_mom_x[i] + tmp_Lib_mom_y[i]*tmp_Lib_mom_y[i] + tmp_Lib_mom_z[i]*tmp_Lib_mom_z[i])
                cos_theta = tmp_mc_Pz[i]/tmp_Mom
                cos_theta_lib = tmp_Lib_mom_z[i]/tmp_Mom_lib
                tmp_theta = math.acos(cos_theta)*180/math.pi
                tmp_theta_lib = math.acos(cos_theta_lib)*180/math.pi
                cos_phi   = tmp_mc_Px[i]/math.sqrt(tmp_mc_Px[i]*tmp_mc_Px[i] + tmp_mc_Py[i]*tmp_mc_Py[i])
                cos_phi_lib   = tmp_Lib_mom_x[i]/math.sqrt(tmp_Lib_mom_x[i]*tmp_Lib_mom_x[i] + tmp_Lib_mom_y[i]*tmp_Lib_mom_y[i])
                tmp_phi   = math.acos(cos_phi)*180/math.pi if tmp_mc_Py[0] > 0 else 360-math.acos(cos_phi)*180/math.pi
                tmp_phi_lib   = math.acos(cos_phi_lib)*180/math.pi if tmp_Lib_mom_y[0] > 0 else 360-math.acos(cos_phi_lib)*180/math.pi
                self.h_Mom   .Fill(tmp_Mom)
                self.h_mc_x  .Fill(tmp_mc_x[i])
                self.h_theta .Fill(tmp_theta)
                self.h_phi   .Fill(tmp_phi)
                self.h_Mom_lib   .Fill(tmp_Mom_lib)
                self.h_x_lib     .Fill(tmp_Lib_x[i])
                self.h_theta_lib .Fill(tmp_theta_lib)
                self.h_phi_lib   .Fill(tmp_phi_lib)
                if tmp_mc_x[i] > 1850 and tmp_Lib_x[i] > 1850:
                   f_out     .write("%f %f %f %f %f %f %d\n"%(tmp_mc_x[i] , tmp_mc_y[i]             , tmp_mc_z[i]               , tmp_mc_Px[i]/1000    , tmp_mc_Py[i]/1000    , tmp_mc_Pz[i]/1000    , tmp_mc_Pdg[i]))
                   #f_out     .write("%f %f %f %f %f %f %d\n"%(tmp_mc_x[i] , tmp_mc_y[i]             , random.uniform(-100,100)               , tmp_mc_Px[i]/1000    , tmp_mc_Py[i]/1000    , tmp_mc_Pz[i]/1000    , tmp_mc_Pdg[i]))## this will give higher sum hit energy
                   #f_out     .write("%f %f %f %f %f %f %d\n"%(tmp_mc_x[i] , tmp_mc_y[i]             , tmp_mc_z[i] + random.uniform(-10,10)    , tmp_mc_Px[i]/1000    , tmp_mc_Py[i]/1000    , tmp_mc_Pz[i]/1000    , tmp_mc_Pdg[i]))## this will give higher sum hit energy
                   #f_out     .write("%f %f %f %f %f %f %d\n"%(tmp_mc_x[i] , tmp_mc_y[i]             , tmp_mc_z[i] + 10.16666667    , tmp_mc_Px[i]/1000    , tmp_mc_Py[i]/1000    , tmp_mc_Pz[i]/1000    , tmp_mc_Pdg[i]))## this will give higher sum hit energy
                   f_out_lib .write("%f %f %f %f %f %f %d\n"%(tmp_Lib_x[i], random.uniform(-500,500), random.uniform(-1500,1500), tmp_Lib_mom_x[i]/1000, tmp_Lib_mom_y[i]/1000, tmp_Lib_mom_z[i]/1000, tmp_mc_Pdg[i]))## this will give higher sum hit energy 
                   #f_out_lib .write("%f %f %f %f %f %f %d\n"%(tmp_Lib_x[i], random.uniform(-100,100), random.uniform(-100,100), tmp_Lib_mom_x[i]/1000, tmp_Lib_mom_y[i]/1000, tmp_Lib_mom_z[i]/1000, tmp_mc_Pdg[i]))## this also gives the similar energy bias
                   #f_out_lib .write("%f %f %f %f %f %f %d\n"%(tmp_Lib_x[i], tmp_mc_y[i], random.uniform(-100,100), tmp_Lib_mom_x[i]/1000, tmp_Lib_mom_y[i]/1000, tmp_Lib_mom_z[i]/1000, tmp_mc_Pdg[i]))## this also gives the similar energy bias
                   #f_out_lib .write("%f %f %f %f %f %f %d\n"%(tmp_Lib_x[i], random.uniform(-100,100), tmp_mc_z[i]               , tmp_Lib_mom_x[i]/1000, tmp_Lib_mom_y[i]/1000, tmp_Lib_mom_z[i]/1000, tmp_mc_Pdg[i]))## this gives good agreement
                   f_out_lib1.write("%f %f %f %f %f %f %d\n"%(tmp_Lib_x[i], tmp_mc_y[i]             , tmp_mc_z[i]               , tmp_Lib_mom_x[i]/1000, tmp_Lib_mom_y[i]/1000, tmp_Lib_mom_z[i]/1000, tmp_mc_Pdg[i]))
        f_out.close()
        f_out_lib.close()
        f_out_lib1.close()
############ BEGIN ##############

plot_path = './compare_sim_plots'

'''
obj_G4        = Obj('G4','/junofs/users/wxfang/CEPC/CEPCOFF/doSim/useB/FS_apply/test/sim_G4_find_0731_to.root'   ,100000)
obj_G4.fill()

obj_FS        = Obj('FS','/junofs/users/wxfang/CEPC/CEPCOFF/doSim/useB/FS_apply/test/sim_FS_0731.root' ,100000)
obj_FS.fill_sp()
scale_to_G4 = False
print('G4 saved sum Hit E=',obj_G4.h_Hit_E.GetSumOfWeights(),',FS saved sum Hit E=',obj_FS.h_Hit_E.GetSumOfWeights())
print('G4 saved sum Hit E mean=%f ,rms=%f'%(obj_G4.h_Hit_E.GetMean(), obj_G4.h_Hit_E.GetRMS() ),',FS saved sum Hit E mean=%f, rms=%f'%(obj_FS.h_Hit_E.GetMean(),obj_FS.h_Hit_E.GetRMS()) )
print('G4 saved cm x='     ,obj_G4.h_Hit_cm_x.GetSumOfWeights(),',FS saved cm x=',obj_FS.h_Hit_cm_x.GetSumOfWeights())
print('G4 saved cm y='     ,obj_G4.h_Hit_cm_y.GetSumOfWeights(),',FS saved cm y=',obj_FS.h_Hit_cm_y.GetSumOfWeights())
print('G4 saved cm z='     ,obj_G4.h_Hit_cm_z.GetSumOfWeights(),',FS saved cm z=',obj_FS.h_Hit_cm_z.GetSumOfWeights())
do_plot_h2(obj_G4.h_Hit_res   ,obj_FS.h_Hit_res   ,'h_Hit_res'    ,{'X':'#sum Hit_{E}/mc_{E}'     ,'Y':'Events'}, '' , scale_to_G4)
do_plot_h2(obj_G4.h_Hit_E     ,obj_FS.h_Hit_E     ,'h_Hit_E'      ,{'X':'#sum Hit_{E} GeV'    ,'Y':'Events'}, '' , scale_to_G4)
do_plot_h2(obj_G4.h_Hit_cm_x  ,obj_FS.h_Hit_cm_x  ,'h_Hit_cm_x'   ,{'X':'Hit_{x}^{cm} mm'     ,'Y':'Events'}, '' , scale_to_G4)
do_plot_h2(obj_G4.h_Hit_cm_y  ,obj_FS.h_Hit_cm_y  ,'h_Hit_cm_y'   ,{'X':'Hit_{y}^{cm} mm'     ,'Y':'Events'}, '' , scale_to_G4)
do_plot_h2(obj_G4.h_Hit_cm_z  ,obj_FS.h_Hit_cm_z  ,'h_Hit_cm_z'   ,{'X':'Hit_{z}^{cm} mm'     ,'Y':'Events'}, '' , scale_to_G4)
do_plot_h2(obj_G4.h_Mom       ,obj_FS.h_Mom       ,'h_Mom'        ,{'X':'P_{mc} GeV/c'        ,'Y':'Events'}, '' , scale_to_G4)
do_plot_h2(obj_G4.h_theta     ,obj_FS.h_theta     ,'h_theta'      ,{'X':'#theta_{mc} (degree)','Y':'Events'}, '' , scale_to_G4)
do_plot_h2(obj_G4.h_phi       ,obj_FS.h_phi       ,'h_phi'        ,{'X':'#phi_{mc} (degree)','Y':'Events'}  , '' , scale_to_G4)
'''
#obj_FS1        = Obj('FS1','/junofs/users/wxfang/CEPC/CEPCOFF/doSim/useB/FS_apply/test/sim_FS_0731v1.root' ,100000)
obj_FS1        = Obj('FS1','/junofs/users/wxfang/CEPC/CEPCOFF/doSim/useB/FS_apply/test/sim_FS_0803_20degree.root' ,100000)
obj_FS1.fill_sp_lib()
do_plot_h2(obj_FS1.h_Mom       ,obj_FS1.h_Mom_lib        ,'h_Mom_sp_lib'        ,{'X':'P_{mc} GeV/c'       ,'Y':'Events'}, '' , False)
do_plot_h2(obj_FS1.h_theta     ,obj_FS1.h_theta_lib      ,'h_theta_sp_lib'      ,{'X':'#theta_{mc}'        ,'Y':'Events'}, '' , False)
do_plot_h2(obj_FS1.h_phi       ,obj_FS1.h_phi_lib        ,'h_phi_sp_lib'        ,{'X':'#phi_{mc}'          ,'Y':'Events'}, '' , False)
do_plot_h2(obj_FS1.h_mc_x      ,obj_FS1.h_x_lib          ,'h_phi_x_lib'         ,{'X':'#x_{mc}'            ,'Y':'Events'}, '' , False)
print('SP saved mean=%f ,rms=%f'%(obj_FS1.h_Mom.GetMean(), obj_FS1.h_Mom.GetRMS() ),',Lib saved mean=%f, rms=%f'%(obj_FS1.h_Mom_lib.GetMean(), obj_FS1.h_Mom_lib.GetRMS()) )
print('SP saved mean=%f ,rms=%f'%(obj_FS1.h_mc_x.GetMean(), obj_FS1.h_mc_x.GetRMS() ),',Lib saved mean=%f, rms=%f'%(obj_FS1.h_x_lib.GetMean(), obj_FS1.h_x_lib.GetRMS()) )


print('done')
