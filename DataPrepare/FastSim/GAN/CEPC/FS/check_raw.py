import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import argparse
rt.gROOT.SetBatch(rt.kTRUE)

#######################################
# use sim step Hit  ##
#######################################

def get_parser():
    parser = argparse.ArgumentParser(
        description='root to hdf5',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', action='store', type=str,
                        help='input root file')
    parser.add_argument('--output', action='store', type=str,
                        help='output root file')
    parser.add_argument('--tag', action='store', type=str,
                        help='tag name for plots')
    parser.add_argument('--str_particle', action='store', type=str,
                        help='e^{-}')


    return parser


def plot_gr(gr,out_name,title, Type):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    gr.GetXaxis().SetTitle(title[0])
    gr.GetYaxis().SetTitle(title[1])
    if Type == 0: #graph
        gr.Draw("ap")
    elif Type ==1 : #h1
        gr.SetLineWidth(2)
        gr.SetMarkerStyle(8)
        gr.Draw("hist")
        gr.Draw("same:pe")
    elif Type ==2:
        gr.Draw("COLZ")
    else:
        print('wrong type')
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

def plot_gr1(gr,out_name,title, Type):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    gr.GetXaxis().SetTitle(title[0])
    gr.GetYaxis().SetTitle(title[1])
    if Type == 0: #graph
        gr.Draw("ap")
    elif Type ==1 : #h1
        gr.SetLineWidth(2)
        gr.SetMarkerStyle(8)
        gr.Draw("hist")
        gr.Draw("same:pe")
    elif Type ==2:
        gr.Draw("COLZ")
    else:
        print('wrong type')
    el = rt.TEllipse(0.,0.,1800.,1800.)
    el.SetLineColor(rt.kRed)
    el.SetLineWidth(2)
    el.SetFillStyle(0)
    el.Draw()
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()


def plot_hist(hist,out_name,title):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    canvas.SetGridy()
    canvas.SetGridx()
    #h_corr.Draw("COLZ")
    #h_corr.LabelsDeflate("X")
    #h_corr.LabelsDeflate("Y")
    #h_corr.LabelsOption("v")
    hist.SetStats(rt.kFALSE)
    #hist.GetXaxis().SetTitle("#Delta Z (mm)")
    if 'x_z' in out_name:
        #hist.GetYaxis().SetTitle("X (mm)")
        hist.GetYaxis().SetTitle("cell X")
        hist.GetXaxis().SetTitle("cell Z")
    elif 'y_z' in out_name:
        #hist.GetYaxis().SetTitle("#Delta Y (mm)")
        hist.GetYaxis().SetTitle("cell Y")
        hist.GetXaxis().SetTitle("cell Z")
    elif 'z_r' in out_name:
        hist.GetYaxis().SetTitle("bin R")
        hist.GetXaxis().SetTitle("bin Z")
    elif 'z_phi' in out_name:
        hist.GetYaxis().SetTitle("bin #phi")
        hist.GetXaxis().SetTitle("bin Z")
    hist.SetTitle(title)
    #hist.SetTitleSize(0.1)
    #hist.Draw("COLZ TEXT")
    hist.Draw("COLZ")
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

if __name__ == '__main__':

    parser = get_parser()
    parse_args = parser.parse_args()
    
    str_e = parse_args.str_particle
    
    #For_ep = False # now just use em
    #str_e = 'e^{-}'
    #if For_ep:
    #    str_e = 'e^{+}'
    print ('Start..')
    cell_x = 10.0
    cell_y = 10.0
    Depth = [1850, 1857, 1860, 1868, 1871, 1878, 1881, 1889, 1892, 1899, 1902, 1910, 1913, 1920, 1923, 1931, 1934, 1941, 1944, 1952, 1957, 1967, 1972, 1981, 1986, 1996, 2001, 2011, 2016, 2018]
    
    print ('Read root file')
    plot_path='/junofs/users/wxfang/FastSim/GAN/CEPC/FS/raw_plots'
    #filePath = '/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/build/Simu_40_100_em.root' ## will not use sim one, it has too many sim hits
    filePath = parse_args.input
    treeName='evt'
    chain =rt.TChain(treeName)
    chain.Add(filePath)
    tree = chain
    totalEntries=tree.GetEntries()
    print (totalEntries)
    nBin_dZ = 31 
    nBin_dI = 31 
    nBin_L  = 29 
    ## for HCAL ###
    nBin_r = 55
    nBin_z = 40
    nBin_phi = 40

    dz=150
    x_min= 1840
    x_max= 2020
    y_min= -150
    y_max= 150
    #h_Hit_z_x = rt.TH2F('Hit_z_x' , '', 200, -100, 100 , 200, 1840, 2040)
    #h_Hit_y_x = rt.TH2F('Hit_y_x' , '', 200, -100, 100 , 200, 1840, 2040)
    #h_Hit_z_y = rt.TH2F('Hit_z_y' , '', 200, -100, 100 , 200, -100, 100 )
    h_Hit_z_x = rt.TH2F('Hit_z_x' , '', 300,  2400, 2700 , 4400, -2200, 2200)
    h_Hit_z_y = rt.TH2F('Hit_z_y' , '', 300,  2400, 2700 , 4400, -2200, 2200)
    h_Hit_y_x = rt.TH2F('Hit_y_x' , '', 4400, -2200, 2200 , 4400, -2200, 2200)
    h_Hit_E    = rt.TH1F('Hit_E'   , '', 100, 0, 1)
    h_Hit_cm_x = rt.TH1F('Hit_mc_x'   , '', 200, 1840, 2040)
    h_Hit_cm_y = rt.TH1F('Hit_mc_y'   , '', 200, -100, 100)
    h_Hit_cm_z = rt.TH1F('Hit_mc_z'   , '', 200, -100, 100)
    h_Mom      = rt.TH1F('MC_mom'     , '', 500, 0, 500)
    h_phi      = rt.TH1F('MC_phi'     , '', 360, -180, 180)
    h_theta    = rt.TH1F('MC_theta'   , '', 180, 0, 180)
    index=0
    total_evt = tree.GetEntries()
    for entryNum in range(0, total_evt):
    #for entryNum in range(4,5):
        tree.GetEntry(entryNum)
        if entryNum%100000 ==0:print('processed:', 100.0*entryNum/total_evt,'%%')
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
        if len(tmp_mc_Px)==1:
            h_Mom.Fill(1000*math.sqrt(tmp_mc_Px[0]*tmp_mc_Px[0] + tmp_mc_Py[0]*tmp_mc_Py[0] + tmp_mc_Pz[0]*tmp_mc_Pz[0]))
            pt = math.sqrt(tmp_mc_Px[0]*tmp_mc_Px[0] + tmp_mc_Py[0]*tmp_mc_Py[0] )
            mom = math.sqrt(tmp_mc_Px[0]*tmp_mc_Px[0] + tmp_mc_Py[0]*tmp_mc_Py[0] + tmp_mc_Pz[0]*tmp_mc_Pz[0] )
            phi = math.acos(tmp_mc_Px[0]/pt)*180/math.pi if tmp_mc_Py[0] > 0 else -math.acos(tmp_mc_Px[0]/pt)*180/math.pi 
            theta = math.acos(tmp_mc_Pz[0]/mom)*180/math.pi 
            h_phi  .Fill(phi)
            h_theta.Fill(theta)
        for i in range(len(tmp_Hit_x)):
            h_Hit_z_x.Fill(tmp_Hit_z[i], tmp_Hit_x[i])
            h_Hit_y_x.Fill(tmp_Hit_y[i], tmp_Hit_x[i])
            h_Hit_z_y.Fill(tmp_Hit_z[i], tmp_Hit_y[i])
            #cm_x += tmp_Hit_x[i]*tmp_Hit_E[i]
            #cm_y += tmp_Hit_y[i]*tmp_Hit_E[i]
            #cm_z += tmp_Hit_z[i]*tmp_Hit_E[i]
            En += tmp_Hit_E[i]
        if En == 0: continue
        h_Hit_E.Fill(En)
        #h_Hit_cm_x.Fill(cm_x/En)
        #h_Hit_cm_y.Fill(cm_y/En)
        #h_Hit_cm_z.Fill(cm_z/En)
        #for i in range(len(tmp_Hit_x)):
        #    print('x=',tmp_Hit_x[i],',y=',tmp_Hit_y[i],',z=',tmp_Hit_z[i],',E=',tmp_Hit_E[i])
            

    plot_gr1(h_Hit_y_x ,'h_Hit_y_x_v1' ,['hit y','hit x'],2)
    plot_gr(h_Hit_z_x ,'h_Hit_z_x' ,['hit z','hit x'],2)
    plot_gr(h_Hit_y_x ,'h_Hit_y_x' ,['hit y','hit x'],2)
    plot_gr(h_Hit_z_y ,'h_Hit_z_y' ,['hit z','hit y'],2)
    plot_gr(h_Hit_E   ,'h_Hit_E'   ,['tot E','Event'],1)
    plot_gr(h_Hit_cm_x,'h_Hit_cm_x',['cm x' ,'Event'],1)
    plot_gr(h_Hit_cm_y,'h_Hit_cm_y',['cm y' ,'Event'],1)
    plot_gr(h_Hit_cm_z,'h_Hit_cm_z',['cm z' ,'Event'],1)
    plot_gr(h_Mom,'h_MC_Mom',['Mom (MeV)' ,'Event'],1)
    plot_gr(h_phi,'h_MC_phi',['#phi (degree)' ,'Event'],1)
    plot_gr(h_theta,'h_MC_theta',['#theta (degree)' ,'Event'],1)
    print('done')
