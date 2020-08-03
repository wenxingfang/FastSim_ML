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
        
        self.m_hit_x = 'm_Hit_x'  if self.name == 'real' else "hit_x"
        self.m_hit_y = 'm_Hit_y'  if self.name == 'real' else "hit_y"
        self.m_hit_z = 'm_Hit_z'  if self.name == 'real' else "hit_z"
        self.m_hit_e = 'm_Hit_E'  if self.name == 'real' else "hit_e"
        self.m_hit_id= 'm_Hit_ID0'if self.name == 'real' else "hit_id"

    def fill(self, event_min, event_max):
        self.h_Hit_x .Scale(0.0) 
        self.h_Hit_y .Scale(0.0) 
        self.h_Hit_z .Scale(0.0) 
        self.h_Hit_e .Scale(0.0) 
        self.h_Hit_id.Scale(0.0) 

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
            for i in range(len(tmp_hit_x)):
                self.h_Hit_x .Fill(tmp_hit_x [i])
                self.h_Hit_y .Fill(tmp_hit_y [i])
                self.h_Hit_z .Fill(tmp_hit_z [i])
                self.h_Hit_e .Fill(tmp_hit_e [i])
                self.h_Hit_id.Fill(tmp_hit_id[i])

plot_path = './Hit_plot'

real = Obj('real', '/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/data/final_mc_info_cellID.root')
#fake = Obj('fake', '/junofs/users/wxfang/FastSim/cepc/ForReco/Hit_info_for_reco.root')
fake = Obj('fake', '/junofs/users/wxfang/FastSim/cepc/ForReco/Hit_info_for_reco_new.root')

for i in range(20):
    real.fill(i,i+1)
    fake.fill(i,i+1)
    do_plot_h2(real.h_Hit_x, fake.h_Hit_x, 'evt_%d_hit_x'%i,    {'X':'hit_x' ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_Hit_y, fake.h_Hit_y, 'evt_%d_hit_y'%i,    {'X':'hit_y' ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_Hit_z, fake.h_Hit_z, 'evt_%d_hit_z'%i,    {'X':'hit_z' ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_Hit_e, fake.h_Hit_e, 'evt_%d_hit_e'%i,    {'X':'hit_e' ,'Y':'Events'}, 'e')
    do_plot_h2(real.h_Hit_id, fake.h_Hit_id, 'evt_%d_hit_id'%i, {'X':'hit_id','Y':'Events'}, 'e')

print('done')
