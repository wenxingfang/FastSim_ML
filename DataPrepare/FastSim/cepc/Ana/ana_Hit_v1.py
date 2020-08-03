import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import json
from array import array
rt.gROOT.SetBatch(rt.kTRUE)


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
    dummy.GetXaxis().SetLabelSize(0.025)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    #if "cell_energy" not in out_name:
    #    dummy.GetXaxis().SetMoreLogLabels()
    #    dummy.GetXaxis().SetNoExponent()  
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



class Obj:
    def __init__(self, name, fileName):
        print('construct')
        self.name = name
        self.file_name = fileName
        str_ext = self.name
        self.h_Hit_x          = rt.TH1F('%s_h_Hit_x'        %(str_ext) ,'',500,-2500,2500)
        self.h_Hit_y          = rt.TH1F('%s_h_Hit_y'        %(str_ext) ,'',500,-2500,2500)
        self.h_Hit_z          = rt.TH1F('%s_h_Hit_z'        %(str_ext) ,'',500,-2500,2500)
        self.h_Hit_e          = rt.TH1F('%s_h_Hit_e'        %(str_ext) ,'',100,0,10)
        self.h_Hit_id         = rt.TH1F('%s_h_Hit_id'       %(str_ext) ,'',10000,0,100000000)
        self.h_ID_S           = rt.TH1F('%s_h_ID_S'         %(str_ext) ,'',100,0,10)
        self.h_ID_M           = rt.TH1F('%s_h_ID_M'         %(str_ext) ,'',100,0,10)
        self.h_ID_I           = rt.TH1F('%s_h_ID_I'         %(str_ext) ,'',400,0,200)
        self.h_ID_J           = rt.TH1F('%s_h_ID_J'         %(str_ext) ,'',200,0,100)
        self.h_ID_K           = rt.TH1F('%s_h_ID_K'         %(str_ext) ,'',60,0,30)
        self.h_ID_S_we        = rt.TH1F('%s_h_ID_S_we'      %(str_ext) ,'',100,0,10)
        self.h_ID_M_we        = rt.TH1F('%s_h_ID_M_we'      %(str_ext) ,'',100,0,10)
        self.h_ID_I_we        = rt.TH1F('%s_h_ID_I_we'      %(str_ext) ,'',400,0,200)
        self.h_ID_J_we        = rt.TH1F('%s_h_ID_J_we'      %(str_ext) ,'',200,0,100)
        self.h_ID_K_we        = rt.TH1F('%s_h_ID_K_we'      %(str_ext) ,'',60,0,30)
        
        self.m_hit_x = 'm_Hit_x'  
        self.m_hit_y = 'm_Hit_y'  
        self.m_hit_z = 'm_Hit_z'  
        self.m_hit_e = 'm_Hit_E'  
        self.m_hit_id= 'm_Hit_ID0'
        self.m_ID_S= 'm_ID_S'
        self.m_ID_M= 'm_ID_M'
        self.m_ID_I= 'm_ID_I'
        self.m_ID_J= 'm_ID_J'
        self.m_ID_K= 'm_ID_K'

    def fill(self, event_min, event_max):
        self.h_Hit_x .Scale(0.0) 
        self.h_Hit_y .Scale(0.0) 
        self.h_Hit_z .Scale(0.0) 
        self.h_Hit_e .Scale(0.0) 
        self.h_Hit_id.Scale(0.0) 
        self.h_ID_S.Scale(0.0) 
        self.h_ID_M.Scale(0.0) 
        self.h_ID_I.Scale(0.0) 
        self.h_ID_J.Scale(0.0) 
        self.h_ID_K.Scale(0.0) 
        self.h_ID_S_we.Scale(0.0) 
        self.h_ID_M_we.Scale(0.0) 
        self.h_ID_I_we.Scale(0.0) 
        self.h_ID_J_we.Scale(0.0) 
        self.h_ID_K_we.Scale(0.0) 

        FileName = self.file_name
        treeName='evt'
        chain =rt.TChain(treeName)
        chain.Add(FileName)
        tree = chain
        totalEntries=tree.GetEntries()
        print (totalEntries)
        for entryNum in range(0, tree.GetEntries()):
            #if entryNum > event_max or entryNum < event_min: continue
            if entryNum != event_min: continue
            tree.GetEntry(entryNum)

            #tmp_hit_n   = getattr(tree, "n_hit")
            tmp_hit_x   = getattr(tree, self.m_hit_x )
            tmp_hit_y   = getattr(tree, self.m_hit_y )
            tmp_hit_z   = getattr(tree, self.m_hit_z )
            tmp_hit_e   = getattr(tree, self.m_hit_e )
            tmp_hit_id  = getattr(tree, self.m_hit_id)
            tmp_ID_S    = getattr(tree, self.m_ID_S)
            tmp_ID_M    = getattr(tree, self.m_ID_M)
            tmp_ID_I    = getattr(tree, self.m_ID_I)
            tmp_ID_J    = getattr(tree, self.m_ID_J)
            tmp_ID_K    = getattr(tree, self.m_ID_K)
            for i in range(len(tmp_hit_x)):
                self.h_Hit_x .Fill(tmp_hit_x [i])
                self.h_Hit_y .Fill(tmp_hit_y [i])
                self.h_Hit_z .Fill(tmp_hit_z [i])
                self.h_Hit_e .Fill(tmp_hit_e [i])
                self.h_Hit_id.Fill(tmp_hit_id[i])
                self.h_ID_S.Fill(tmp_ID_S[i])
                self.h_ID_M.Fill(tmp_ID_M[i])
                self.h_ID_I.Fill(tmp_ID_I[i])
                self.h_ID_J.Fill(tmp_ID_J[i])
                self.h_ID_K.Fill(tmp_ID_K[i])
                self.h_ID_S_we.Fill(tmp_ID_S[i], tmp_hit_e [i])
                self.h_ID_M_we.Fill(tmp_ID_M[i], tmp_hit_e [i])
                self.h_ID_I_we.Fill(tmp_ID_I[i], tmp_hit_e [i])
                self.h_ID_J_we.Fill(tmp_ID_J[i], tmp_hit_e [i])
                self.h_ID_K_we.Fill(tmp_ID_K[i], tmp_hit_e [i])
        print ('self.h_ID_I=',self.h_ID_I.GetSumOfWeights())
        print ('self.h_ID_J=',self.h_ID_J.GetSumOfWeights())

plot_path = './Hit_plot'

real = Obj('real', '/junofs/users/wxfang/CEPC/CEPCOFF/ana/real_nnh_aa_0.root')
#fake = Obj('fake', '/junofs/users/wxfang/CEPC/CEPCOFF/ana/fake_nnh_aa_0.root')
#fake = Obj('fake', '/junofs/users/wxfang/CEPC/CEPCOFF/ana/fake_nnh_aa_0_cut.root')
fake = Obj('fake', '/junofs/users/wxfang/CEPC/CEPCOFF/ana/fake_nnh_aa_0_cut1.root')

for i in range(20):
    real.fill(i,i+1)
    fake.fill(i,i+1)
    do_plot_h2(real.h_Hit_x, fake.h_Hit_x, 'evt_%d_hit_x'%i,    {'X':'hit_x' ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_Hit_y, fake.h_Hit_y, 'evt_%d_hit_y'%i,    {'X':'hit_y' ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_Hit_z, fake.h_Hit_z, 'evt_%d_hit_z'%i,    {'X':'hit_z' ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_Hit_e, fake.h_Hit_e, 'evt_%d_hit_e'%i,    {'X':'hit_e' ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_Hit_id, fake.h_Hit_id, 'evt_%d_hit_id'%i, {'X':'hit_id','Y':'Events'}, 'e')
    do_plot_h2(real.h_ID_S, fake.h_ID_S, 'evt_%d_ID_S'%i      , {'X':'ID_S'  ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_ID_M, fake.h_ID_M, 'evt_%d_ID_M'%i      , {'X':'ID_M'  ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_ID_I, fake.h_ID_I, 'evt_%d_ID_I'%i      , {'X':'ID_I'  ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_ID_J, fake.h_ID_J, 'evt_%d_ID_J'%i      , {'X':'ID_J'  ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_ID_K, fake.h_ID_K, 'evt_%d_ID_K'%i      , {'X':'ID_K'  ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_ID_S_we, fake.h_ID_S_we, 'evt_%d_ID_S_we'%i      , {'X':'ID_S E'  ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_ID_M_we, fake.h_ID_M_we, 'evt_%d_ID_M_we'%i      , {'X':'ID_M E'  ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_ID_I_we, fake.h_ID_I_we, 'evt_%d_ID_I_we'%i      , {'X':'ID_I E'  ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_ID_J_we, fake.h_ID_J_we, 'evt_%d_ID_J_we'%i      , {'X':'ID_J E'  ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_ID_K_we, fake.h_ID_K_we, 'evt_%d_ID_K_we'%i      , {'X':'ID_K E'  ,'Y':'Events'}, 'e')

print('done')
