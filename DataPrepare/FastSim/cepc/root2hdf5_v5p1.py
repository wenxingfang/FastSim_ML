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
    hist.GetXaxis().SetTitle("#Delta Z (mm)")
    if 'x_z' in out_name:
        hist.GetYaxis().SetTitle("X (mm)")
    elif 'y_z' in out_name:
        hist.GetYaxis().SetTitle("#Delta Y (mm)")
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
    
    
    For_ep = False # now just use em
    str_e = 'e^{-}'
    str_e1 = 'em'
    if For_ep:
        str_e = 'e^{+}'
        str_e1 = 'ep'
    print ('Start..')
    cell_x = 10.0
    cell_y = 10.0
    Depth = [1850, 1857, 1860, 1868, 1871, 1878, 1881, 1889, 1892, 1899, 1902, 1910, 1913, 1920, 1923, 1931, 1934, 1941, 1944, 1952, 1957, 1967, 1972, 1981, 1986, 1996, 2001, 2011, 2016, 2018]
    
    print ('Read root file')
    plot_path='./raw_plots'
    #filePath = '/junofs/users/wxfang/CEPC/CEPCOFF/GetSimHit/build/Simu_40_100_em.root' ## will not use sim one, it has too many sim hits
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
    nBin = 30 
    Barrel_Hit = np.full((maxEvent, nBin , nBin, len(Depth)-1 ), 0 ,dtype=np.float32)#init 
    MC_info    = np.full((maxEvent, 4 ), 0 ,dtype=np.float32)#init 
    dz=150
    x_min= 1840
    #x_max= 1930
    #x_min= 1930
    x_max= 2020
    y_min= -150
    y_max= 150
    h_Hit_B_x_z = rt.TH2F('Hit_B_x_z' , '', 2*dz, -1*dz, dz ,x_max-x_min, x_min, x_max)
    h_Hit_B_y_z = rt.TH2F('Hit_B_y_z' , '', 2*dz, -1*dz, dz ,y_max-y_min, y_min, y_max)
    n=0
    for entryNum in range(0, tree.GetEntries()):
        tree.GetEntry(entryNum)
        if entryNum>= maxEvent: break
        tmp_mc_Px   = getattr(tree, "m_mc_Px")
        tmp_mc_Py   = getattr(tree, "m_mc_Py")
        tmp_mc_Pz   = getattr(tree, "m_mc_Pz")
        tmp_mc_M    = getattr(tree, "m_mc_M")
        #tmp_mc_Pdg  = getattr(tree, "m_mc_Pdg")
        #tmp_HitCm_x = getattr(tree, "m_HitCm_x")
        #tmp_HitCm_y = getattr(tree, "m_HitCm_y")
        #tmp_HitCm_z = getattr(tree, "m_HitCm_z")
        tmp_HitFirst_x = getattr(tree, "m_HitFirst_x")
        tmp_HitFirst_y = getattr(tree, "m_HitFirst_y")
        tmp_HitFirst_z = getattr(tree, "m_HitFirst_z")
        tmp_HitFirst_x = tmp_HitFirst_x*1000 # change to mm  
        tmp_HitFirst_y = tmp_HitFirst_y*1000 # change to mm
        tmp_HitFirst_z = tmp_HitFirst_z*1000 # change to mm
        tmp_HitFirst_vtheta = getattr(tree, "m_HitFirst_vtheta")
        tmp_HitFirst_vphi   = getattr(tree, "m_HitFirst_vphi"  )
        tmp_Hit_x   = getattr(tree, "m_Hit_x")
        tmp_Hit_y   = getattr(tree, "m_Hit_y")
        tmp_Hit_z   = getattr(tree, "m_Hit_z")
        tmp_Hit_E   = getattr(tree, "m_Hit_E")
        #str_mc = 'PDG=%d, Px=%.1f, Py=%.1f, Pz=%.1f'%(tmp_mc_Pdg[0], tmp_mc_Px[0], tmp_mc_Py[0], tmp_mc_Pz[0])
        #print (str_mc)
        #MC_info[entryNum][0]=tmp_mc_Px[0] 
        #MC_info[entryNum][1]=tmp_mc_Py[0]
        #MC_info[entryNum][2]=tmp_mc_Pz[0]
        MC_info[entryNum][0] = math.sqrt(tmp_mc_Px[0]*tmp_mc_Px[0] + tmp_mc_Py[0]*tmp_mc_Py[0] + tmp_mc_Pz[0]*tmp_mc_Pz[0])
        MC_info[entryNum][1] = tmp_HitFirst_vtheta
        MC_info[entryNum][2] = tmp_HitFirst_vphi
        MC_info[entryNum][3] = tmp_HitFirst_z
        #print ('HitFirst:',tmp_HitFirst_x,';',tmp_HitFirst_y,';',tmp_HitFirst_z)
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
            
    plot_hist(h_Hit_B_x_z ,'%s_Hit_barrel_x_z_plane_%s'%(parse_args.tag, str_e1), '%s (10-100 GeV)'%(str_e))
    plot_hist(h_Hit_B_y_z ,'%s_Hit_barrel_y_z_plane_%s'%(parse_args.tag, str_e1), '%s (10-100 GeV)'%(str_e))
    
    hf = h5py.File(outFileName, 'w')
    hf.create_dataset('Barrel_Hit', data=Barrel_Hit)
    hf.create_dataset('MC_info'   , data=MC_info)
    hf.close()
    print ('Done')
    
    '''
    g2D =  rt.TGraph2D()
    g2D.SetPoint(n, tmp_Hit_x[i], tmp_Hit_y[i], tmp_Hit_z[i])
    n=n+1
    do_plot("",g2D, "total","e- 50GeV")
    '''
    
