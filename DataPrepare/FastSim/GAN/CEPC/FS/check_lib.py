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
    
    #str_e = parse_args.str_particle
    
    #For_ep = False # now just use em
    #str_e = 'e^{-}'
    #if For_ep:
    #    str_e = 'e^{+}'
    print ('Start..')
    cell_x = 10.0
    cell_y = 10.0
    Depth = [1850, 1857, 1860, 1868, 1871, 1878, 1881, 1889, 1892, 1899, 1902, 1910, 1913, 1920, 1923, 1931, 1934, 1941, 1944, 1952, 1957, 1967, 1972, 1981, 1986, 1996, 2001, 2011, 2016, 2018]
    
    print ('Read root file')
    plot_path='/junofs/users/wxfang/FastSim/GAN/CEPC/FS/lib_plots'
    filePath = parse_args.input
    F_in = rt.TFile(filePath)

    h_E_nFS      = F_in.Get('E_nFS'     )	
    h_E_tag0     = F_in.Get('E_tag0'    )	
    h_E_tag1     = F_in.Get('E_tag1'    )	
    h_E_tag2     = F_in.Get('E_tag2'    )	
    h_x_nFS      = F_in.Get('x_nFS'     )	
    h_x_tag0     = F_in.Get('x_tag0'    )	
    h_x_tag1     = F_in.Get('x_tag1'    )	
    h_x_tag2     = F_in.Get('x_tag2'    )	
    h_theta_nFS  = F_in.Get('theta_nFS' )	
    h_theta_tag0 = F_in.Get('theta_tag0')	
    h_theta_tag1 = F_in.Get('theta_tag1')	
    h_theta_tag2 = F_in.Get('theta_tag2')	

    h2_x_SumHitE = F_in.Get('x_SumHitE')	
    #h2_x_SumHitE = F_in.Get('x_SumHitE_ori')	

    plot_gr(h_E_nFS ,'h_E_nFS' ,['Mom (MeV)' ,'N FS']  ,1)
    plot_gr(h_E_tag0,'h_E_tag0',['Mom (MeV)' ,'N tag0'],1)
    plot_gr(h_E_tag1,'h_E_tag1',['Mom (MeV)' ,'N tag1'],1)
    plot_gr(h_E_tag2,'h_E_tag2',['Mom (MeV)' ,'N tag2'],1)
    plot_gr(h_x_nFS ,'h_x_nFS' ,['x (mm)' ,'N FS']  ,1)
    plot_gr(h_x_tag0,'h_x_tag0',['x (mm)' ,'N tag0'],1)
    plot_gr(h_x_tag1,'h_x_tag1',['x (mm)' ,'N tag1'],1)
    plot_gr(h_x_tag2,'h_x_tag2',['x (mm)' ,'N tag2'],1)
    plot_gr(h_theta_nFS ,'h_theta_nFS' ,['#theta (degree)' ,'N FS']  ,1)
    plot_gr(h_theta_tag0,'h_theta_tag0',['#theta (degree)' ,'N tag0'],1)
    plot_gr(h_theta_tag1,'h_theta_tag1',['#theta (degree)' ,'N tag1'],1)
    plot_gr(h_theta_tag2,'h_theta_tag2',['#theta (degree)' ,'N tag2'],1)
    print('done')
    for x in [1900, 2000]:
        yaxis = h2_x_SumHitE.GetYaxis()
        Nbins = h2_x_SumHitE.GetNbinsY()
        #hist = rt.TH1F('sumE_x%d'%x,'',Nbins, yaxis.GetXmin(), yaxis.GetXmax())
        hist = rt.TH1F('sumE_x%d'%x,'',100, 0, 100)
        binx = h2_x_SumHitE.GetXaxis().FindBin(x)
        #for y in range(Nbins):
        for y in range(100):
            hist.SetBinContent(y+1, h2_x_SumHitE.GetBinContent(binx,y+1))
        plot_gr(hist ,'h_x%d_SumHiE'%x ,['#sum Hit E (GeV)' ,'Events']  ,1)



