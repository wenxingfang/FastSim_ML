import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import random
import argparse
rt.gROOT.SetBatch(rt.kTRUE)

#######################################
# save start point  ##
#######################################

def get_parser():
    parser = argparse.ArgumentParser(
        description='root to hdf5',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', action='store', type=str,
                        help='input root file')
    parser.add_argument('--output', action='store', type=str, default='',
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
    

    do_draw_plot = False
    do_save_sp   = True
    n_sp = 0
    precentage = 1.0
    mom_low  = 100
    mom_high = 1000
    y_abs  = 500
    z_abs  = 2000
    theta_low = 50
    theta_high = 130
    phi_abs = 25
    str_e = parse_args.str_particle
    x_list = list(range(1870,2010))
    dict_f = {}
    if parse_args.output != '': 
        for x in x_list:
            f_out = open(parse_args.output+'_x%d.txt'%x,'w') 
            dict_f[x] = f_out
    
    print ('Read root file')
    plot_path='/junofs/users/wxfang/FastSim/GAN/CEPC/FS/plot_check_startpoint/'
    filePath = parse_args.input
    treeName='evt'
    chain =rt.TChain(treeName)
    chain.Add(filePath)
    tree = chain
    total_evt=tree.GetEntries()
    print (total_evt)
    h_point_x  = rt.TH1F('point_x'     , '', 160, 1850, 2010)
    h_point_y  = rt.TH1F('point_y'     , '', 800, -800, 800)
    h_point_z  = rt.TH1F('point_z'     , '', 240, -2400, 2400)
    h_Mom      = rt.TH1F('mom'         , '', 110, 0, 1100)
    h_theta    = rt.TH1F('mom_theta'   , '', 180, 0, 180)
    h_phi      = rt.TH1F('mom_phi'     , '', 180, -90, 90)
    total_evt = tree.GetEntries()
    for entryNum in range(0, total_evt):
        tree.GetEntry(entryNum)
        if entryNum%10000 ==0:print('processed:', 100.0*entryNum/total_evt,'%%')
        m_point_x   = getattr(tree, "m_point_x")
        m_point_y   = getattr(tree, "m_point_y")
        m_point_z   = getattr(tree, "m_point_z")
        m_mom_x     = getattr(tree, "m_mom_x")
        m_mom_y     = getattr(tree, "m_mom_y")
        m_mom_z     = getattr(tree, "m_mom_z")
        for i in range(len(m_mom_x)):
            tmp_Mom = math.sqrt( m_mom_x[i]*m_mom_x[i] + m_mom_y[i]*m_mom_y[i] + m_mom_z[i]*m_mom_z[i] )
            tmp_Pt  = math.sqrt( m_mom_x[i]*m_mom_x[i] + m_mom_y[i]*m_mom_y[i] )
            tmp_theta = math.acos(m_mom_z[i]/tmp_Mom)*180/math.pi
            tmp_phi = math.acos(m_mom_x[i]/tmp_Pt)*180/math.pi if m_mom_y[i] > 0 else -math.acos(m_mom_x[i]/tmp_Pt)*180/math.pi
            if do_draw_plot:
                h_point_x.Fill( m_point_x[i] )
                h_point_y.Fill( m_point_y[i] )
                h_point_z.Fill( m_point_z[i] )
                h_Mom  .Fill( tmp_Mom )
                h_theta.Fill( tmp_theta)
                h_phi  .Fill( tmp_phi )
            if mom_low < tmp_Mom and mom_high > tmp_Mom and theta_low < tmp_theta and theta_high > tmp_theta and abs(tmp_phi) < phi_abs and abs(m_point_y[i]) < y_abs and abs(m_point_z[i]) < z_abs :
                if do_save_sp and (int(m_point_x[i]) in dict_f) and (precentage > random.uniform(0, 1)):
                    dict_f[int(m_point_x[i])].write('%f %f %f %f %f %f\n'%(m_point_x[i], m_point_y[i], m_point_z[i], m_mom_x[i]/1000, m_mom_y[i]/1000, m_mom_z[i]/1000))
                    n_sp +=1
    for x in dict_f:
        dict_f[x].close()
    print('total n_sp=',n_sp)
    plot_gr(h_point_x,'h_point_x_%s'%parse_args.tag,['X (mm)' ,'Event'],1)
    plot_gr(h_point_y,'h_point_y_%s'%parse_args.tag,['Y (mm)' ,'Event'],1)
    plot_gr(h_point_z,'h_point_z_%s'%parse_args.tag,['Z (mm)' ,'Event'],1)
    plot_gr(h_Mom,'h_Mom_%s'%parse_args.tag,['Mom (MeV)' ,'Event'],1)
    plot_gr(h_theta,'h_theta_%s'%parse_args.tag,['#theta (degree)' ,'Event'],1)
    plot_gr(h_phi  ,'h_phi_%s'%parse_args.tag,['#phi (degree)' ,'Event'],1)
    print('done')
