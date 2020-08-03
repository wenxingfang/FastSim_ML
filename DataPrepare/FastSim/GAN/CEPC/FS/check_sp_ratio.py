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
    
    str_e = parse_args.str_particle
    f_out = None
    if parse_args.output != '': f_out = open(parse_args.output,'w') 
    #For_ep = False # now just use em
    #str_e = 'e^{-}'
    #if For_ep:
    #    str_e = 'e^{+}'
    print ('Start..')
    cell_x = 10.0
    cell_y = 10.0
    Depth = [1850, 1857, 1860, 1868, 1871, 1878, 1881, 1889, 1892, 1899, 1902, 1910, 1913, 1920, 1923, 1931, 1934, 1941, 1944, 1952, 1957, 1967, 1972, 1981, 1986, 1996, 2001, 2011, 2016, 2018]
    
    print ('Read root file')
    plot_path='/junofs/users/wxfang/FastSim/GAN/CEPC/FS/plot_check_sp_ratio/'
    filePath = parse_args.input
    treeName='evt'
    chain =rt.TChain(treeName)
    chain.Add(filePath)
    tree = chain
    total_evt=tree.GetEntries()
    print (total_evt)
    h_return  = rt.TH1F('h_return'     , '', 160, -1, 15)
    h_pass_0  = rt.TH1F('pass_0'     , '', 100, 0, 2000)
    h_pass_1  = rt.TH1F('pass_1'     , '', 100, 0, 200)
    h_pass_2  = rt.TH1F('pass_2'     , '', 100, 0, 200)
    h_ratio_1  = rt.TH1F('ratio_1'     , '',200, 0, 2)
    h_ratio_2  = rt.TH1F('ratio_2'     , '',200, 0, 2)
    h_s_x_0      = rt.TH1F('s_x_0'     , '',150, 1850, 2000)
    h_s_y_0      = rt.TH1F('s_y_0'     , '',1600, -800, 800)
    h_s_z_0      = rt.TH1F('s_z_0'     , '',3000, -1500, 1500)
    h_s_theta_0  = rt.TH1F('s_theta_0' , '',180, 0, 180)
    h_s_phi_0    = rt.TH1F('s_phi_0'   , '',80, -40, 40)
    h_s_mom_0    = rt.TH1F('s_mom_0'   , '',110, 0, 1100)
    h_s_x_1      = rt.TH1F('s_x_1'     , '',150, 1850, 2000)
    h_s_y_1      = rt.TH1F('s_y_1'     , '',1600, -800, 800)
    h_s_z_1      = rt.TH1F('s_z_1'     , '',3000, -1500, 1500)
    h_s_theta_1  = rt.TH1F('s_theta_1' , '',180, 0, 180)
    h_s_phi_1    = rt.TH1F('s_phi_1'   , '',80, -40, 40)
    h_s_mom_1    = rt.TH1F('s_mom_1'   , '',110, 0, 1100)
    h_s_x_2      = rt.TH1F('s_x_2'     , '',150, 1850, 2000)
    h_s_theta_2  = rt.TH1F('s_theta_2' , '',180, 0, 180)
    h_s_phi_2    = rt.TH1F('s_phi_2'   , '',80, -40, 40)
    h_s_mom_2    = rt.TH1F('s_mom_2'   , '',110, 0, 1100)
    total_evt = tree.GetEntries()
    tree.SetBranchStatus("m_point_*",0); # branches
    tree.SetBranchStatus("m_mom_*",0); # branches
    tree.SetBranchStatus("m_pid",0); # branches
    tree.SetBranchStatus("m_s_*",0); # branches
    #for entryNum in range(0, total_evt):
    for entryNum in range(0, 100):
        tree.GetEntry(entryNum)
        print(entryNum)
        if entryNum%100000 ==0:print('processed:', 100.0*entryNum/total_evt,'%%')
        '''
        m_s_x  = getattr(tree, "m_s_x")
        m_s_y  = getattr(tree, "m_s_y")
        m_s_z  = getattr(tree, "m_s_z")
        m_s_px = getattr(tree, "m_s_px")
        m_s_py = getattr(tree, "m_s_py")
        m_s_pz = getattr(tree, "m_s_pz")
        m_s_x  = getattr(tree, "m_point_x")
        m_s_y  = getattr(tree, "m_point_y")
        m_s_z  = getattr(tree, "m_point_z")
        m_s_px = getattr(tree, "m_mom_x")
        m_s_py = getattr(tree, "m_mom_y")
        m_s_pz = getattr(tree, "m_mom_z")
        '''
        
        m_pass_0   = getattr(tree, "m_pass0")
        m_pass_1   = getattr(tree, "m_pass1")
        m_pass_2   = getattr(tree, "m_pass2")
        m_return   = getattr(tree, "m_return")
        #if(len(m_return) == 1):
        #    if m_return[0] == 12 and f_out != None: f_out.write('%d\n'%entryNum)     
        m_fail = 0
        m_pass = 0
        print('len(m_return)=',len(m_return))
        for i in range(len(m_return)):
            #h_return .Fill(m_return[i])
            if m_return[i] == 4 or m_return[i] == 5: m_fail +=1
            elif m_return[i]>5: m_pass +=1
        '''
        x_sum = 0
        y_sum = 0
        for i in range(len(m_return)):
            x_sum += m_s_x[i]
            y_sum += m_s_y[i]
        x_sum = x_sum/len(m_return)
        y_sum = y_sum/len(m_return)
        phi = math.acos(x_sum/math.sqrt(x_sum*x_sum+y_sum*y_sum))*180/math.pi
        if phi > 22.5 : continue
        '''
        '''
        for i in range(len(m_s_x)):
            #if m_return[i] < 9 : continue
            mom = math.sqrt(m_s_px[i]*m_s_px[i] + m_s_py[i]*m_s_py[i] + m_s_pz[i]*m_s_pz[i])
            phi = math.acos(m_s_px[i]/math.sqrt(m_s_px[i]*m_s_px[i]+m_s_py[i]*m_s_py[i]))*180/math.pi
            if m_s_py[i] < 0 : phi = -phi
            theta = math.acos(m_s_pz[i]/mom)*180/math.pi
        '''
        '''
            if m_return[i] == 9 :
                h_s_x_0    .Fill(m_s_x[i])
            #    h_s_y_0    .Fill(m_s_y[i])
            #    h_s_z_0    .Fill(m_s_z[i])
                h_s_mom_0  .Fill(mom)
                h_s_phi_0  .Fill(phi)
                h_s_theta_0.Fill(theta)
            elif m_return[i] > 9 :
                h_s_x_1    .Fill(m_s_x[i])
            #    h_s_y_1    .Fill(m_s_y[i])
            #    h_s_z_1    .Fill(m_s_z[i])
                h_s_mom_1  .Fill(mom)
                h_s_phi_1  .Fill(phi)
                h_s_theta_1.Fill(theta)
        '''
        '''
            #print(mom)
            h_s_x_2    .Fill(m_s_x[i])
            h_s_mom_2  .Fill(mom)
            h_s_phi_2  .Fill(phi)
            h_s_theta_2.Fill(theta)
        '''
        #h_pass_0 .Fill(m_pass_0)
        #h_pass_1 .Fill(m_pass_1)
        #h_pass_2 .Fill(m_pass_2)
        h_ratio_1.Fill(float(m_pass)/(m_pass+m_fail) if (m_pass+m_fail) != 0 else 1.5)
        #h_ratio_1.Fill(float(m_pass_1)/m_pass_0 if m_pass_0 != 0 else 0)
        #h_ratio_2.Fill(float(m_pass_2)/m_pass_1 if m_pass_1 != 0 else 0)
    if f_out != None: f_out.close()
    #plot_gr(h_return,'h_return_%s'%parse_args.tag,['return' ,'Event'],1)
    #plot_gr(h_pass_0,'h_pass_0_%s'%parse_args.tag,['pass_0' ,'Event'],1)
    #plot_gr(h_pass_1,'h_pass_1_%s'%parse_args.tag,['pass_1' ,'Event'],1)
    #plot_gr(h_pass_2,'h_pass_2_%s'%parse_args.tag,['pass_2' ,'Event'],1)
    plot_gr(h_ratio_1,'h_ratio_1_%s'%parse_args.tag,['ratio_1' ,'Event'],1)
    #plot_gr(h_ratio_2,'h_ratio_2_%s'%parse_args.tag,['ratio_2' ,'Event'],1)
    #plot_gr(h_s_x_0,'h_s_x_0_%s'%parse_args.tag,['x_0' ,'Event'],1)
    #plot_gr(h_s_y_0,'h_s_y_0_%s'%parse_args.tag,['y_0' ,'Event'],1)
    #plot_gr(h_s_z_0,'h_s_z_0_%s'%parse_args.tag,['z_0' ,'Event'],1)
    #plot_gr(h_s_theta_0,'h_s_theta_0_%s'%parse_args.tag,['theta_0' ,'Event'],1)
    #plot_gr(h_s_phi_0,'h_s_phi_0_%s'%parse_args.tag,['phi_0' ,'Event'],1)
    #plot_gr(h_s_mom_0,'h_s_mom_0_%s'%parse_args.tag,['mom_0' ,'Event'],1)
    #plot_gr(h_s_x_1,'h_s_x_1_%s'%parse_args.tag,['x_1' ,'Event'],1)
    #plot_gr(h_s_y_1,'h_s_y_1_%s'%parse_args.tag,['y_1' ,'Event'],1)
    #plot_gr(h_s_z_1,'h_s_z_1_%s'%parse_args.tag,['z_1' ,'Event'],1)
    #plot_gr(h_s_theta_1,'h_s_theta_1_%s'%parse_args.tag,['theta_1' ,'Event'],1)
    #plot_gr(h_s_phi_1,'h_s_phi_1_%s'%parse_args.tag,['phi_1' ,'Event'],1)
    #plot_gr(h_s_mom_1,'h_s_mom_1_%s'%parse_args.tag,['mom_1' ,'Event'],1)
    #plot_gr(h_s_x_2,'h_s_x_2_%s'%parse_args.tag,['x_2' ,'Event'],1)
    #plot_gr(h_s_theta_2,'h_s_theta_2_%s'%parse_args.tag,['theta_2' ,'Event'],1)
    #plot_gr(h_s_phi_2,'h_s_phi_2_%s'%parse_args.tag,['phi_2' ,'Event'],1)
    #plot_gr(h_s_mom_2,'h_s_mom_2_%s'%parse_args.tag,['mom_2' ,'Event'],1)
    print('done')
