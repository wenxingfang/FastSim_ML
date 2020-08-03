import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
import argparse
rt.gROOT.SetBatch(rt.kTRUE)

#######################################
# use digi step data and use B field  ##
# use cell ID for ECAL                ##
# add HCAL
# add HoE cut
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
    plot_path='/junofs/users/wxfang/FastSim/GAN/CEPC/GAN/raw_plots'
    #filePath = '/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/build/Simu_40_100_em.root' ## will not use sim one, it has too many sim hits
    filePath = parse_args.input
    outFileName= parse_args.output
    treeName='evt'
    chain =rt.TChain(treeName)
    chain.Add(filePath)
    tree = chain
    totalEntries=tree.GetEntries()
    print (totalEntries)
    maxEvent = 2*totalEntries
    #maxEvent = 100
    nBin_dZ = 31 
    nBin_dI = 31 
    nBin_L  = 29 
    ## for HCAL ###
    nBin_r = 55
    nBin_z = 40
    nBin_phi = 40

    Barrel_Hit = np.full((maxEvent, nBin_dZ , nBin_dI, nBin_L ), 0 ,dtype=np.float32)#init 
    Barrel_Hit_HCAL = np.full((maxEvent, nBin_z , nBin_phi, nBin_r ), 0 ,dtype=np.float32)#init 
    MC_info    = np.full((maxEvent, 7 ), 0 ,dtype=np.float32)#init 
    dz=150
    x_min= 1840
    x_max= 2020
    y_min= -150
    y_max= 150
    #h_Hit_B_x_z = rt.TH2F('Hit_B_x_z' , '', 2*dz, -1*dz, dz ,x_max-x_min, x_min, x_max)
    #h_Hit_B_y_z = rt.TH2F('Hit_B_y_z' , '', 2*dz, -1*dz, dz ,y_max-y_min, y_min, y_max)
    h_Hit_B_x_z = rt.TH2F('Hit_B_x_z' , '', 31, 0, 31 , 29, 0, 29)
    h_Hit_B_y_z = rt.TH2F('Hit_B_y_z' , '', 31, 0, 31 , 31, 0, 31)
    h_Hit_HB_z_r   = rt.TH2F('Hit_HB_z_r'   , '', nBin_z, 0, nBin_z , nBin_r  , 0, nBin_r)
    h_Hit_HB_z_phi = rt.TH2F('Hit_HB_z_phi' , '', nBin_z, 0, nBin_z , nBin_phi, 0, nBin_phi)
    index=0
    for entryNum in range(0, tree.GetEntries()):
        tree.GetEntry(entryNum)
        tmp_mc_Px   = getattr(tree, "m_mc_Px")
        tmp_mc_Py   = getattr(tree, "m_mc_Py")
        tmp_mc_Pz   = getattr(tree, "m_mc_Pz")
        tmp_HitFirst_x = getattr(tree, "m_mc_pHitx")
        tmp_HitFirst_y = getattr(tree, "m_mc_pHity")
        tmp_HitFirst_z = getattr(tree, "m_mc_pHitz")
        tmp_HitFirst_dz = getattr(tree, "m_mc_pHit_dz")
        tmp_HitFirst_dy = getattr(tree, "m_mc_pHit_dy")
        tmp_HitFirst_vtheta = getattr(tree, "m_mc_pHit_theta")
        tmp_HitFirst_vphi   = getattr(tree, "m_mc_pHit_phi"  )
        #print('tmp_HitFirst_vphi=', type(tmp_HitFirst_vphi))
        tmp_Hit   = getattr(tree, "m_Hits")
        tmp_HcalHit   = getattr(tree, "m_HcalHits")
        for i in range(len(tmp_mc_Px)):
            if index >= maxEvent:
                print('out of max, exit now')
                sys.exit()
            if (len(tmp_Hit[i]) != 27869): continue 
            if (len(tmp_HcalHit[i]) != int(nBin_z*nBin_phi*nBin_r)): continue 
            MC_info[index][0] = math.sqrt(tmp_mc_Px[i]*tmp_mc_Px[i] + tmp_mc_Py[i]*tmp_mc_Py[i] + tmp_mc_Pz[i]*tmp_mc_Pz[i])
            MC_info[index][1] = tmp_HitFirst_vtheta [i]
            MC_info[index][2] = tmp_HitFirst_vphi [i]
            MC_info[index][3] = tmp_HitFirst_dz [i]
            MC_info[index][4] = tmp_HitFirst_dy [i]
            MC_info[index][5] = tmp_HitFirst_z  [i]
            MC_info[index][6] = tmp_HitFirst_y  [i]
            tot_Ecal = 0
            tot_Hcal = 0
            for j in range(len(tmp_Hit[i])):
                n_Layer = int(j/(31.0*31.0))
                n_Row   = (j - 31*31*n_Layer)%31
                n_Col   = int((j - 31*31*n_Layer)/31.0)
                Barrel_Hit[index, n_Row, n_Col, n_Layer] = tmp_Hit[i][j]  
                h_Hit_B_x_z.Fill(n_Col+0.01, n_Layer+0.01   , tmp_Hit[i][j])
                h_Hit_B_y_z.Fill(n_Col+0.01, 31-(n_Row+0.01), tmp_Hit[i][j])
                tot_Ecal = tot_Ecal + tmp_Hit[i][j]
                #print('i=',i,'j=',j,'e=',tmp_Hit[i][j])
            for j in range(len(tmp_HcalHit[i])):
                n_r = int(j/(nBin_z*nBin_phi*1.0))
                n_z = int( (j - nBin_z*nBin_phi*n_r)/(nBin_z*1.0) )
                n_phi = (j - nBin_z*nBin_phi*n_r)%nBin_z
                Barrel_Hit_HCAL[index, n_z, n_phi, n_r] = tmp_HcalHit[i][j]  
                h_Hit_HB_z_r  .Fill(n_z+0.01, n_r+0.01     , tmp_HcalHit[i][j])
                h_Hit_HB_z_phi.Fill(n_z+0.01, n_phi+0.01   , tmp_HcalHit[i][j])
                tot_Hcal = tot_Hcal + tmp_HcalHit[i][j]
            if tot_Hcal/tot_Ecal > 2: MC_info[index][0] = 0 ## for HoE cut
            index = index + 1
        '''
        for i in range(0, len(tmp_Hit_x)):
            if tmp_Hit_x[i] < Depth[0] or tmp_Hit_x[i] > Depth[-1]: continue
            #if abs(tmp_Hit_y[i]) > 600 : continue # remove the hit in other plane
            h_Hit_B_x_z.Fill(tmp_Hit_z[i]-tmp_HitFirst_z, tmp_Hit_x[i]               , tmp_Hit_E[i])
            h_Hit_B_y_z.Fill(tmp_Hit_z[i]-tmp_HitFirst_z, tmp_Hit_y[i]-tmp_HitFirst_y, tmp_Hit_E[i])
            index_dep=0
            for dp in  range(len(Depth)):
                if Depth[dp] <= tmp_Hit_x[i] and tmp_Hit_x[i] < Depth[dp+1] :
                    index_dep = dp
                    break
            if tmp_Hit_z[i] > tmp_HitFirst_z: index_col = int((tmp_Hit_z[i]-tmp_HitFirst_z)/cell_x) + int(0.5*nBin)
            else : index_col = int((tmp_Hit_z[i]-tmp_HitFirst_z)/cell_x) + int(0.5*nBin) -1
            if tmp_Hit_y[i] > tmp_HitFirst_y: index_row = int((tmp_Hit_y[i]-tmp_HitFirst_y)/cell_y) + int(0.5*nBin)
            else : index_row = int((tmp_Hit_y[i]-tmp_HitFirst_y)/cell_y) + int(0.5*nBin) -1
            if index_col >= nBin or index_col <0 or index_row >= nBin or index_row<0: continue; ##skip this hit now, maybe can merge it?
            #index_row = int(index_row)
            #index_col = int(index_col)
            Barrel_Hit[entryNum, index_row, index_col, index_dep] = Barrel_Hit[entryNum, index_row, index_col, index_dep] + tmp_Hit_E[i]  
        '''
            
    plot_hist(h_Hit_B_x_z ,'%s_Hit_barrel_x_z_plane'%(parse_args.tag)      , '%s (2-20 GeV)'%(str_e))
    plot_hist(h_Hit_B_y_z ,'%s_Hit_barrel_y_z_plane'%(parse_args.tag)      , '%s (2-20 GeV)'%(str_e))
    plot_hist(h_Hit_HB_z_r   ,'%s_Hit_Hbarrel_z_r_plane'  %(parse_args.tag), '%s (2-20 GeV)'%(str_e))
    plot_hist(h_Hit_HB_z_phi ,'%s_Hit_Hbarrel_z_phi_plane'%(parse_args.tag), '%s (2-20 GeV)'%(str_e))


    if True:
        dele_list = []
        for i in range(MC_info.shape[0]):
            if MC_info[i][0]==0:
                dele_list.append(i) ## remove the empty event 
        MC_info         = np.delete(MC_info        , dele_list, axis = 0)
        Barrel_Hit      = np.delete(Barrel_Hit     , dele_list, axis = 0)
        Barrel_Hit_HCAL = np.delete(Barrel_Hit_HCAL, dele_list, axis = 0)
    print('final size=', MC_info.shape[0])        
    
    hf = h5py.File(outFileName, 'w')
    hf.create_dataset('Barrel_Hit'     , data=Barrel_Hit)
    hf.create_dataset('Barrel_Hit_HCAL', data=Barrel_Hit_HCAL)
    hf.create_dataset('MC_info'        , data=MC_info)
    hf.close()
    print ('Done')
