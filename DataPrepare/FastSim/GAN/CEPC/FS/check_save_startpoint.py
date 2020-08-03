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
    parser.add_argument('--output', action='store', type=str, default='test.txt',
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
    
    ForNotFind = False
    ForFind    = False
    ForLibSP   = True

    forBarrel = True
    forEndcap = False

    do_draw_plot = False
    do_save_sp   = True
    n_sp = 0
    #precentage = 0.2
    precentage = 1.1
    mom_low  = 100
    mom_high = 1000
    ### for Ecal barrel ##
    y_abs  = 500   
    z_abs  = 2000
    theta_low = 50
    theta_high = 90#130
    phi_abs = 25
    x_list = list(range(1870,2010))
    #################### 
    ### for Ecal barrel ##
    endcap_theta_low = 20
    endcap_theta_high = 40# most sp is in 20-40 for endcap
    #################### 
    f_out = open(parse_args.output,'a') 
    
    print ('Read root file')
    plot_path='/junofs/users/wxfang/FastSim/GAN/CEPC/FS/plot_check_startpoint/'
    filePath = parse_args.input
    treeName='evt'
    chain =rt.TChain(treeName)
    chain.Add(filePath)
    tree = chain
    total_evt=tree.GetEntries()
    print (total_evt)

    po_x_bin = 160
    po_x_min = 1850
    po_x_max = 2010
    po_y_bin = 800
    po_y_min = -800
    po_y_max = 800
    po_z_bin = 240
    po_z_min = -2400
    po_z_max = 2400
    if forEndcap:
        po_x_bin = 2000
        po_x_min = -2000
        po_x_max = 2000
        po_y_bin = 2000
        po_y_min = -2000
        po_y_max = 2000
        po_z_bin = 250
        po_z_min = 2400
        po_z_max = 2650

    h_em_point_x  = rt.TH1F('em_point_x'     , '', po_x_bin, po_x_min, po_x_max)
    h_em_point_y  = rt.TH1F('em_point_y'     , '', po_y_bin, po_y_min, po_y_max)
    h_em_point_z  = rt.TH1F('em_point_z'     , '', po_z_bin, po_z_min, po_z_max)
    h_em_Mom      = rt.TH1F('em_mom'         , '', 110, 0, 1100)
    h_em_theta    = rt.TH1F('em_mom_theta'   , '', 180, 0, 180)
    h_em_phi      = rt.TH1F('em_mom_phi'     , '', 180, -180, 180)
    h_ep_point_x  = rt.TH1F('ep_point_x'     , '', po_x_bin, po_x_min, po_x_max)
    h_ep_point_y  = rt.TH1F('ep_point_y'     , '', po_y_bin, po_y_min, po_y_max)
    h_ep_point_z  = rt.TH1F('ep_point_z'     , '', po_z_bin, po_z_min, po_z_max)
    h_ep_Mom      = rt.TH1F('ep_mom'         , '', 110, 0, 1100)
    h_ep_theta    = rt.TH1F('ep_mom_theta'   , '', 180, 0, 180)
    h_ep_phi      = rt.TH1F('ep_mom_phi'     , '', 180, -180, 180)
    total_evt = tree.GetEntries()
    #total_evt = 100#int(total_evt/100.0)
    for entryNum in range(0, total_evt):
        tree.GetEntry(entryNum)
        if entryNum%10000 ==0:print('processed:', 100.0*entryNum/total_evt,'%%')
        m_point_x   = getattr(tree, "m_point_x")
        m_point_y   = getattr(tree, "m_point_y")
        m_point_z   = getattr(tree, "m_point_z")
        m_mom_x     = getattr(tree, "m_mom_x")
        m_mom_y     = getattr(tree, "m_mom_y")
        m_mom_z     = getattr(tree, "m_mom_z")
        m_pid       = getattr(tree, "m_pid")
        if ForNotFind:
            m_point_x   = getattr(tree, "m_noFind_point_x")
            m_point_y   = getattr(tree, "m_noFind_point_y")
            m_point_z   = getattr(tree, "m_noFind_point_z")
            m_mom_x     = getattr(tree, "m_noFind_mom_x")
            m_mom_y     = getattr(tree, "m_noFind_mom_y")
            m_mom_z     = getattr(tree, "m_noFind_mom_z")
            m_pid       = getattr(tree, "m_noFind_pid")
        elif ForFind:
            m_point_x   = getattr(tree, "m_Find_point_x")
            m_point_y   = getattr(tree, "m_Find_point_y")
            m_point_z   = getattr(tree, "m_Find_point_z")
            m_mom_x     = getattr(tree, "m_Find_mom_x")
            m_mom_y     = getattr(tree, "m_Find_mom_y")
            m_mom_z     = getattr(tree, "m_Find_mom_z")
            m_pid       = getattr(tree, "m_Find_pid")
        for i in range(len(m_mom_x)):
            ori_point_x = m_point_x[i]
            ori_point_y = m_point_y[i]
            ori_mom_x = m_mom_x[i]
            ori_mom_y = m_mom_y[i]
            tmp_Mom = math.sqrt( m_mom_x[i]*m_mom_x[i] + m_mom_y[i]*m_mom_y[i] + m_mom_z[i]*m_mom_z[i] )
            tmp_Pt  = math.sqrt( m_mom_x[i]*m_mom_x[i] + m_mom_y[i]*m_mom_y[i] )
            tmp_theta = math.acos(m_mom_z[i]/tmp_Mom)*180/math.pi
            tmp_phi = math.acos(m_mom_x[i]/tmp_Pt)*180/math.pi if m_mom_y[i] > 0 else -math.acos(m_mom_x[i]/tmp_Pt)*180/math.pi
            
            ''' 
            tmp_point_r   = math.sqrt( m_point_x[i]*m_point_x[i] + m_point_y[i]*m_point_y[i] )
            tmp_point_phi = math.acos(m_point_x[i]/tmp_point_r)*180/math.pi if m_point_y[i] > 0 else -math.acos(m_point_x[i]/tmp_point_r)*180/math.pi
            if(only_stave_1 and (22.5 < abs(tmp_point_phi))) : continue
            if(only_stave_2 and (22.5 > tmp_point_phi or 67.5 < tmp_point_phi)) : continue
            if(22.5 < tmp_point_phi and 67.5 < tmp_point_phi):
                tmp_point_phi = tmp_point_phi - 45
                tmp_phi = tmp_phi - 45
                m_point_x[i] = tmp_point_r*math.cos(tmp_point_phi*math.pi/180)
                m_point_y[i] = tmp_point_r*math.sin(tmp_point_phi*math.pi/180)
                m_mom_x[i]   = tmp_Pt     *math.cos(tmp_phi  *math.pi/180) 
                m_mom_y[i]   = tmp_Pt     *math.sin(tmp_phi  *math.pi/180) 
            ''' 
            if do_draw_plot:
                if m_pid[i] == 11 :
                    h_em_point_x.Fill( m_point_x[i] )
                    h_em_point_y.Fill( m_point_y[i] )
                    h_em_point_z.Fill( m_point_z[i] )
                    h_em_Mom  .Fill( tmp_Mom )
                    h_em_theta.Fill( tmp_theta)
                    h_em_phi  .Fill( tmp_phi )
                elif m_pid[i] == -11 :
                    h_ep_point_x.Fill( m_point_x[i] )
                    h_ep_point_y.Fill( m_point_y[i] )
                    h_ep_point_z.Fill( m_point_z[i] )
                    h_ep_Mom  .Fill( tmp_Mom )
                    h_ep_theta.Fill( tmp_theta)
                    h_ep_phi  .Fill( tmp_phi )
            if do_save_sp:
                save_it = False
                #if do_save_sp and (precentage > random.uniform(0, 1)):
                if forBarrel and mom_low < tmp_Mom and mom_high > tmp_Mom and theta_low < tmp_theta and theta_high > tmp_theta and abs(tmp_phi) < phi_abs and abs(m_point_y[i]) < y_abs and abs(m_point_z[i]) < z_abs :
                    save_it = True
                if forEndcap and mom_low < tmp_Mom and mom_high > tmp_Mom and endcap_theta_low < tmp_theta and endcap_theta_high > tmp_theta :
                    save_it = True
                if ForFind:
                    save_it = True
                if save_it and (precentage > random.uniform(0, 1)):
                    f_out.write('%f %f %f %f %f %f %d\n'%(ori_point_x, ori_point_y, m_point_z[i], ori_mom_x/1000, ori_mom_y/1000, m_mom_z[i]/1000, m_pid[i]))
                    n_sp +=1
    f_out.close()
    print('total n_sp=',n_sp)
    if do_draw_plot:
        plot_gr(h_em_point_x,'h_em_point_x_%s'%parse_args.tag,['X (mm)' ,'Event'],1)
        plot_gr(h_em_point_y,'h_em_point_y_%s'%parse_args.tag,['Y (mm)' ,'Event'],1)
        plot_gr(h_em_point_z,'h_em_point_z_%s'%parse_args.tag,['Z (mm)' ,'Event'],1)
        plot_gr(h_em_Mom    ,'h_em_Mom_%s'%parse_args.tag,['Mom (MeV)' ,'Event'],1)
        plot_gr(h_em_theta  ,'h_em_theta_%s'%parse_args.tag,['#theta (degree)' ,'Event'],1)
        plot_gr(h_em_phi    ,'h_em_phi_%s'%parse_args.tag,['#phi (degree)' ,'Event'],1)
        plot_gr(h_ep_point_x,'h_ep_point_x_%s'%parse_args.tag,['X (mm)' ,'Event'],1)
        plot_gr(h_ep_point_y,'h_ep_point_y_%s'%parse_args.tag,['Y (mm)' ,'Event'],1)
        plot_gr(h_ep_point_z,'h_ep_point_z_%s'%parse_args.tag,['Z (mm)' ,'Event'],1)
        plot_gr(h_ep_Mom    ,'h_ep_Mom_%s'%parse_args.tag,['Mom (MeV)' ,'Event'],1)
        plot_gr(h_ep_theta  ,'h_ep_theta_%s'%parse_args.tag,['#theta (degree)' ,'Event'],1)
        plot_gr(h_ep_phi    ,'h_ep_phi_%s'%parse_args.tag,['#phi (degree)' ,'Event'],1)
    print('done')
