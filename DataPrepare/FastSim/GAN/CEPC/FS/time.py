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
        self.h_Hit_E    = rt.TH1F('%s_Hit_E'   %(str_ext)   , '', 20, 0, 0.5)
        self.h_Hit_cm_x = rt.TH1F('%s_Hit_mc_x'%(str_ext)   , '', 100, 1850, 2000)
        self.h_Hit_cm_y = rt.TH1F('%s_Hit_mc_y'%(str_ext)   , '', 65, -500, 800)
        self.h_Hit_cm_z = rt.TH1F('%s_Hit_mc_z'%(str_ext)   , '', 50, -1000, 1000)
        self.h_Mom      = rt.TH1F('%s_MC_mom'  %(str_ext)   , '', 30, 0, 30)
        self.h_theta    = rt.TH1F('%s_MC_theta'%(str_ext)   , '', 180, 0, 180)
        self.h_phi      = rt.TH1F('%s_MC_phi'  %(str_ext)   , '', 360, 0, 360)

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
            tree.GetEntry(entryNum)
            if entryNum%100000 ==0:print('processed:', 100.0*entryNum/totalEntries,'%%')
            tmp_mc_Px   = getattr(tree, "m_mc_Px")
            tmp_mc_Py   = getattr(tree, "m_mc_Py")
            tmp_mc_Pz   = getattr(tree, "m_mc_Pz")
            tmp_Hit_x     = getattr(tree, "m_Hit_x")
            tmp_Hit_y     = getattr(tree, "m_Hit_y")
            tmp_Hit_z     = getattr(tree, "m_Hit_z")
            tmp_Hit_E     = getattr(tree, "m_Hit_E")
            #tmp_HcalHit   = getattr(tree, "m_HcalHits")
            En = 0
            cm_x = 0
            cm_y = 0
            cm_z = 0
            #print('len=',len(tmp_mc_Px),',len2=',len(tmp_Hit_x))
            if len(tmp_mc_Px) !=1 or len(tmp_Hit_x)==0 : continue
            #if len(tmp_mc_Px) !=1 : continue
            tmp_Mom = 1000*math.sqrt(tmp_mc_Px[0]*tmp_mc_Px[0] + tmp_mc_Py[0]*tmp_mc_Py[0] + tmp_mc_Pz[0]*tmp_mc_Pz[0])
            cos_theta = tmp_mc_Pz[0]*1000/tmp_Mom
            tmp_theta = math.acos(cos_theta)*180/math.pi
            cos_phi   = tmp_mc_Px[0]/math.sqrt(tmp_mc_Px[0]*tmp_mc_Px[0] + tmp_mc_Py[0]*tmp_mc_Py[0])
            tmp_phi   = math.acos(cos_phi)*180/math.pi if tmp_mc_Py[0] > 0 else 360-math.acos(cos_phi)*180/math.pi
            #if tmp_Mom<100:continue
            #if tmp_mc_Px[0]<0:continue
            self.h_Mom.Fill(tmp_Mom/1000)
            self.h_theta.Fill(tmp_theta)
            self.h_phi  .Fill(tmp_phi)
            for i in range(len(tmp_Hit_x)):
                self.h_Hit_z_x.Fill(tmp_Hit_z[i], tmp_Hit_x[i])
                self.h_Hit_y_x.Fill(tmp_Hit_y[i], tmp_Hit_x[i])
                self.h_Hit_z_y.Fill(tmp_Hit_z[i], tmp_Hit_y[i])
                cm_x += tmp_Hit_x[i]*tmp_Hit_E[i]
                cm_y += tmp_Hit_y[i]*tmp_Hit_E[i]
                cm_z += tmp_Hit_z[i]*tmp_Hit_E[i]
                En += tmp_Hit_E[i]
            self.h_Hit_E.Fill(En)#GeV
            self.h_Hit_cm_x.Fill(cm_x/En)
            self.h_Hit_cm_y.Fill(cm_y/En)
            self.h_Hit_cm_z.Fill(cm_z/En)

def readTime(File):
    phi_dict = {}
    phi_dict_final = {}
    E_dict = {}
    E_dict_final = {}
    for phi in range(-90,90,10):
        phi_dict["%s_%s"%(phi,phi+10)] = []
        phi_dict_final["%s_%s"%(phi,phi+10)] = []
    for E in range(0,100,10):
        E_dict["%s_%s"%(E,E+10)] = []
        E_dict_final["%s_%s"%(E,E+10)] = []

    E_list =[]
    phi_list =[]
    t_list =[]
    f = open(File,'r')
    lines = f.readlines()
    for line in lines:
        if 'Particle:' in line: 
            energy = float(line.split('energy:')[-1]) 
            direction = line.split('(')[-1]
            direction = direction.split(')')[0]
            px = float( direction.split(',')[0])
            py = float( direction.split(',')[1])
            pz = float( direction.split(',')[2])
            phi = math.asin(py/math.sqrt(px*px+py*py))*180/math.pi
            E_list.append(energy/1000) #to GeV
            phi_list.append(phi)
        elif 'SpeNt time' in line: 
            time =float(line.split(':')[-1])
            t_list.append(time)
    f.close()
    assert (len(t_list) == len(E_list))
    for i in range(len(t_list)):
        for sphi in phi_dict:
            phi_low  = float(sphi.split('_')[0])
            phi_high = float(sphi.split('_')[1])
            if phi_list[i] > phi_low and phi_list[i] < phi_high:
                phi_dict[sphi].append(t_list[i]) 
                break
        for sE in E_dict:
            E_low  = float(sE.split('_')[0])
            E_high = float(sE.split('_')[1])
            if E_list[i] > E_low and E_list[i] < E_high:
                E_dict[sE].append(t_list[i]) 
                break
    for i in phi_dict_final:
        mean = -1
        std = 0
        if len(phi_dict[i]) != 0:
            mean = sum(phi_dict[i])/len(phi_dict[i])
            sum_2 = 0
            for j in phi_dict[i]:
                sum_2 += (j-mean)*(j-mean)
            std = math.sqrt(sum_2)/len(phi_dict[i])
        phi_dict_final[i].append(mean)
        phi_dict_final[i].append(std )
    for i in E_dict_final:
        mean = -1
        std = 0
        if len(E_dict[i]) >= 2:
            mean = sum(E_dict[i])/len(E_dict[i])
            sum_2 = 0
            for j in E_dict[i]:
                sum_2 += (j-mean)*(j-mean)
            std = math.sqrt(sum_2)/(len(E_dict[i])-1)
        E_dict_final[i].append(mean)
        E_dict_final[i].append(std )
    #print('phi_dict_final=',phi_dict_final)
    #print('E_dict_final=',E_dict_final)
    return phi_dict_final, E_dict_final

def get_gr(G4, FS):
    gr_G4    = rt.TGraphErrors()
    gr_FS    = rt.TGraphErrors()
    gr_ratio = rt.TGraphErrors()
    Npoint = 0
    for s in G4:
        low  = float(s.split('_')[0])
        high = float(s.split('_')[1])
        mean = (low + high)/2
        g4_time_mean = G4[s][0]
        g4_time_err  = G4[s][1]
        fs_time_mean = FS[s][0]
        fs_time_err  = FS[s][1]
        if g4_time_mean == -1 or fs_time_mean == -1: continue
        gr_G4.SetPoint     (Npoint, mean         , g4_time_mean)
        gr_G4.SetPointError(Npoint, high-mean    , g4_time_err)
        gr_FS.SetPoint     (Npoint, mean         , fs_time_mean)
        gr_FS.SetPointError(Npoint, high-mean    , fs_time_err)
        gr_ratio.SetPoint  (Npoint, mean         , fs_time_mean/g4_time_mean)
        ratio_err = math.sqrt(g4_time_mean*g4_time_mean*fs_time_err*fs_time_err+fs_time_mean*fs_time_mean*g4_time_err*g4_time_err)/(g4_time_mean*g4_time_mean)
        gr_ratio.SetPointError(Npoint, high-mean , ratio_err)
        Npoint += 1
    return gr_G4,gr_FS,gr_ratio

def do_plot_v2(g1, g2, g3, Xrange,  title, out_name):

    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.SetGridy()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    canvas.cd()
    size = 0.312
    pad1 = rt.TPad('pad1', '', 0.0, size, 1.0, 1.0, 0)
    pad2 = rt.TPad('pad2', '', 0.0, 0.0, 1.0, size, 0)
    pad2.Draw() 
    pad1.Draw()
    pad1.SetTickx(1)
    pad1.SetTicky(1)
    pad2.SetTickx(1)
    pad2.SetTicky(1)
    pad2.SetGridy()
    pad1.SetBottomMargin(0.0)
    pad1.SetRightMargin(0.05)
    pad1.SetLeftMargin(0.13)
    pad2.SetTopMargin(0.02)
    pad2.SetBottomMargin(0.5)
    pad2.SetRightMargin(0.05)
    pad2.SetLeftMargin(0.13)
    pad1.cd()

    xaxis =g1.GetXaxis()
    x_min=Xrange[0]
    x_max=Xrange[1]
    y_min=0
    y_max=10
    if "logx" in out_name:
        pad1.SetLogx()
        pad2.SetLogx()
    if "logy" in out_name:
        pad1.SetLogy()
        y_min = 1e-1
    dummy = rt.TH2D("dummy","",1,x_min,x_max,1,y_min,y_max)
    dummy.SetStats(rt.kFALSE)
    dummy.GetYaxis().SetTitle(title['Y'])
    dummy.GetXaxis().SetTitle(title['X'])
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.)
    #dummy.GetYaxis().SetLabelSize()
    dummy.GetXaxis().SetLabelSize(0.)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42  )
    dummy.GetYaxis().SetNdivisions(305)
    dummy.Draw('AXIS')
    g1.SetLineWidth(2)
    g1.SetLineColor(rt.kRed)
    g1.SetMarkerColor(rt.kRed)
    g1.SetMarkerStyle(20)
    g1.Draw("pe")
    g2.SetLineWidth(2)
    g2.SetLineColor(rt.kBlue)
    g2.SetMarkerColor(rt.kBlue)
    g2.SetMarkerStyle(24)
    g2.Draw("pe")

    legend = rt.TLegend(0.6,0.7,0.85,0.85)
    legend.AddEntry(g1,'G4','lep')
    legend.AddEntry(g2,'FS','lep')
    legend.SetBorderSize(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.SetFillStyle(0)
    legend.Draw()

    pad2.cd()
    fontScale = pad1.GetHNDC()/pad2.GetHNDC()
    ratio_y_min = 0.
    ratio_y_max = 2.
    dummy_ratio = rt.TH2D("dummy_ratio","",1,x_min,x_max,1,ratio_y_min,ratio_y_max)
    dummy_ratio.SetStats(rt.kFALSE)
    dummy_ratio.GetYaxis().SetTitle('FS / G4')
    dummy_ratio.GetYaxis().CenterTitle()
    dummy_ratio.GetXaxis().SetTitle(title['X'])
    dummy_ratio.GetXaxis().SetTitleSize(0.05*fontScale)
    dummy_ratio.GetXaxis().SetLabelSize(0.05*fontScale)
    dummy_ratio.GetYaxis().SetNdivisions(305)
    dummy_ratio.GetYaxis().SetTitleSize(0.04*fontScale)
    dummy_ratio.GetYaxis().SetLabelSize(0.04*fontScale)
    dummy_ratio.GetYaxis().SetTitleOffset(0.7)
    dummy_ratio.Draw("AXIS")
    #h_ratio=h1.Clone('ratio_%s'%h1.GetName())
    #h_ratio.Divide(h2)
    g3.SetLineWidth(2)
    g3.SetLineColor(rt.kBlack)
    g3.SetMarkerColor(rt.kBlack)
    g3.SetMarkerStyle(20)
    g3.Draw("pe")
    canvas.SaveAs("%s/%s.png"%(plot_path, out_name))
    del canvas
    gc.collect()
############ BEGIN ##############

plot_path = './time_plots'

G4_phi_time, G4_E_time   = readTime('/junofs/users/wxfang/CEPC/CEPCOFF/doSim/useB/FS_apply/jobs/G4_nohup.out')
FS_phi_time, FS_E_time   = readTime('/junofs/users/wxfang/CEPC/CEPCOFF/doSim/useB/FS_apply/jobs/FSnohup.out')
gr_G4_phi, gr_FS_phi, gr_ratio_phi = get_gr(G4_phi_time, FS_phi_time)
gr_G4_E  , gr_FS_E  , gr_ratio_E   = get_gr(G4_E_time  , FS_E_time  )
do_plot_v2(g1=gr_G4_phi, g2=gr_FS_phi, g3=gr_ratio_phi, Xrange=[-90, 90],title={'X':'#phi (degree)', 'Y':'Time (s)'}, out_name='phi')
do_plot_v2(g1=gr_G4_E  , g2=gr_FS_E  , g3=gr_ratio_E  , Xrange=[0, 100 ],title={'X':'E (GeV)'      , 'Y':'Time (s)'}, out_name='E'  )

print('done')
