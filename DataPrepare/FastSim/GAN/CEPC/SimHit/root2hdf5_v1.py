import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import argparse
rt.gROOT.SetBatch(rt.kTRUE)

#######################################
# add random noise ##
# ##
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




def plot_gr(event,gr,out_name,title):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    #gr.GetXaxis().SetTitle("#phi(AU, 0 #rightarrow 2#pi)")
    #gr.GetYaxis().SetTitle("Z(AU) (-19.5 #rightarrow 19.5 m)")
    #gr.SetTitle(title)
    gr.Draw("pcol")
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
    hist.GetXaxis().SetTitle("cell Z")
    if 'x_z' in out_name:
        #hist.GetYaxis().SetTitle("X (mm)")
        hist.GetYaxis().SetTitle("cell X")
    elif 'y_z' in out_name:
        #hist.GetYaxis().SetTitle("#Delta Y (mm)")
        hist.GetYaxis().SetTitle("cell Y")
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
    
    print ('Start..')
    #Depth = [1850, 1857, 1860, 1868, 1871, 1878, 1881, 1889, 1892, 1899, 1902, 1910, 1913, 1920, 1923, 1931, 1934, 1941, 1944, 1952, 1957, 1967, 1972, 1981, 1986, 1996, 2001, 2011, 2016, 2018]
    
    print ('Read root file')
    #plot_path='/junofs/users/wxfang/FastSim/cepc/raw_plots'
    filePath = parse_args.input
    outFileName= parse_args.output
    treeName='evt'
    chain =rt.TChain(treeName)
    chain.Add(filePath)
    tree = chain
    totalEntries=tree.GetEntries()
    print (totalEntries)
    maxEvent = totalEntries
    #maxEvent = 100
    MaxHits= 500
    Barrel_Hit = np.full((maxEvent, MaxHits, 4 ), 0 ,dtype=np.float32)#init 
    MC_info    = np.full((maxEvent, 6          ), 0 ,dtype=np.float32)#init 
    noise      = np.random.normal(0, 1, (maxEvent, 4))
    #x_min= 1840
    #x_max= 2020
    #y_min= -150
    #y_max= 150
    #h_Hit_B_x_z = rt.TH2F('Hit_B_x_z' , '', 2*dz, -1*dz, dz ,x_max-x_min, x_min, x_max)
    #h_Hit_B_y_z = rt.TH2F('Hit_B_y_z' , '', 2*dz, -1*dz, dz ,y_max-y_min, y_min, y_max)
    #h_Hit_B_x_z = rt.TH2F('Hit_B_x_z' , '', 31, 0, 31 , 29, 0, 29)
    #h_Hit_B_y_z = rt.TH2F('Hit_B_y_z' , '', 31, 0, 31 , 31, 0, 31)
    for entryNum in range(0, tree.GetEntries()):
        tree.GetEntry(entryNum)
        tmp_mc_Px   = getattr(tree, "m_mc_Px")
        tmp_mc_Py   = getattr(tree, "m_mc_Py")
        tmp_mc_Pz   = getattr(tree, "m_mc_Pz")
        tmp_HitFirst_x = getattr(tree, "m_mc_pHitx")
        tmp_HitFirst_y = getattr(tree, "m_mc_pHity")
        tmp_HitFirst_z = getattr(tree, "m_mc_pHitz")
        tmp_HitFirst_vtheta = getattr(tree, "m_mc_pHit_theta")
        tmp_HitFirst_vphi   = getattr(tree, "m_mc_pHit_phi"  )
        tmp_Hit_x   = getattr(tree, "m_Hit_x")
        tmp_Hit_y   = getattr(tree, "m_Hit_y")
        tmp_Hit_z   = getattr(tree, "m_Hit_z")
        tmp_Hit_E   = getattr(tree, "m_Hit_E")
        if (len(tmp_mc_Px)) !=1:continue
        MC_info[entryNum][0] = math.sqrt(tmp_mc_Px[0]*tmp_mc_Px[0] + tmp_mc_Py[0]*tmp_mc_Py[0] + tmp_mc_Pz[0]*tmp_mc_Pz[0])
        MC_info[entryNum][1] = tmp_HitFirst_x [0]
        MC_info[entryNum][2] = tmp_HitFirst_y [0]
        MC_info[entryNum][3] = tmp_HitFirst_z [0]
        MC_info[entryNum][4] = tmp_HitFirst_vtheta [0]
        MC_info[entryNum][5] = tmp_HitFirst_vphi   [0]



        tmp_Hit_r = {}
        for i in range(len(tmp_Hit_x)):
            tmp_Hit_r[i] = math.sqrt(tmp_Hit_x[i]*tmp_Hit_x[i] + tmp_Hit_y[i]*tmp_Hit_y[i] + tmp_Hit_z[i]*tmp_Hit_z[i])
        s_Hit_r=sorted(tmp_Hit_r.items(), key=lambda x: x[1], reverse=False) #increase
        save_index = 0
        for i_r in s_Hit_r:
            hit_index = i_r[0]
            if save_index >= Barrel_Hit.shape[1]:break
            Barrel_Hit[entryNum, save_index, 0] = tmp_Hit_x[hit_index]  
            Barrel_Hit[entryNum, save_index, 1] = tmp_Hit_y[hit_index]  
            Barrel_Hit[entryNum, save_index, 2] = tmp_Hit_z[hit_index]  
            Barrel_Hit[entryNum, save_index, 3] = tmp_Hit_E[hit_index]  
            save_index += 1
    if True:
        dele_list = []
        for i in range(MC_info.shape[0]):
            if MC_info[i][0]==0:
                dele_list.append(i) ## remove the empty event 
        MC_info    = np.delete(MC_info   , dele_list, axis = 0)
        Barrel_Hit = np.delete(Barrel_Hit, dele_list, axis = 0)
        noise      = np.delete(noise     , dele_list, axis = 0)
    print('final size=', MC_info.shape[0])        
    
    hf = h5py.File(outFileName, 'w')
    hf.create_dataset('Barrel_Hit', data=Barrel_Hit)
    hf.create_dataset('MC_info'   , data=MC_info)
    hf.create_dataset('noise'     , data=noise)
    hf.close()
    print ('Done')
