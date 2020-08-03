import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import ast
import math
import argparse
rt.gROOT.SetBatch(rt.kTRUE)

##########################################################################
#using Mom dtheta, Mom dphi, Pos dz, Pos dphi, array with 121 cell energy#
#merge high z and low z
##########################################################################
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
    hist.GetXaxis().SetTitle("cell Z")
    if 'r_z' in out_name:
        hist.GetYaxis().SetTitle("R (cm)")
    elif 'phi_z' in out_name:
        hist.GetYaxis().SetTitle("cell #phi")
    hist.SetTitle(title)
    #hist.SetTitleSize(0.1)
    #hist.Draw("COLZ TEXT")
    hist.Draw("COLZ")
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

def get_parser():
    parser = argparse.ArgumentParser(
        description='Run',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--nb', action='store', type=int, default=0,
                        help='Number of epochs to train for.')

    parser.add_argument('--for-test'     , action='store', type=ast.literal_eval, default=False,  help='')
    parser.add_argument('--for-em'       , action='store', type=ast.literal_eval, default=False,  help='')
    parser.add_argument('--using-Z-cut'  , action='store', type=ast.literal_eval, default=True ,  help='')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    parse_args = parser.parse_args()
    for_test     = parse_args.for_test
    for_em       = parse_args.for_em
    using_Z_cut  = parse_args.using_Z_cut
    print ('Start..')
    print ('for_test=',for_test,',for_em=',for_em,',using_Z_cut=',using_Z_cut)
    str_z_cut = ''
    str_z_region = ''
    if using_Z_cut == True:
        str_z_cut = '_zcut'
    print ('Read root file')
    plot_path='./raw_plots'
    #filePath = '/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_baba_ForTrain.root'
    #filePath = '/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_baba_ForTrain_newHit.root'
    filePath = '/junofs/users/wxfang/FastSim/bes3/workarea/SingleParticle/SingleParticle-00-00-01/single_em_ForTrain.root'
    outFileName=str('mc_Hit_Barrel_ep%s%s.h5'%(str_z_cut,str_z_region))
    if for_em: outFileName=str('mc_Hit_Barrel_em%s%s.h5'%(str_z_cut,str_z_region))
    if for_test:
        #filePath = '/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_baba_ForTest.root'
        #filePath = '/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_baba_ForTest_newHit.root'
        #filePath = '/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/mc_220_noMix_rereco.root'
        filePath = '/junofs/users/wxfang/FastSim/bes3/workarea/SingleParticle/SingleParticle-00-00-01/single_em_ForTest.root'
        outFileName=str('mc_Hit_Barrel_ep_test%s%s.h5'%(str_z_cut,str_z_region))
        if for_em: outFileName=str('mc_Hit_Barrel_em_test%s%s.h5'%(str_z_cut,str_z_region))
    treeName='Bhabha'
    chain =rt.TChain(treeName)
    chain.Add(filePath)
    tree = chain
    totalEntries=tree.GetEntries()
    print (totalEntries)
    maxEvent = totalEntries
    nBin_row = 11 
    nBin_col = 11 
    Barrel_Hit = np.full((maxEvent, nBin_row , nBin_col, 1 ), 0 ,dtype=np.float32)#init 
    MC_info    = np.full((maxEvent, 6 ), 0 ,dtype=np.float32)#init 
    h_Hit_B_phi_z = rt.TH2F('Hit_B_phi_z' , '', 11, 0, 11 , 11, 0, 11)
    
    for entryNum in range(0, tree.GetEntries()):
        tree.GetEntry(entryNum)
        if entryNum>= maxEvent: break
        isBB   = getattr(tree, "isBB")
        str_e = 'ep'
        if for_em: str_e = 'em'
        str_mdc_mom        = "%s_mdc_mom"       %(str_e) 
        str_M_dtheta       = "%s_M_dtheta"      %(str_e) 
        str_M_dphi         = "%s_M_dphi"        %(str_e) 
        str_P_dz           = "%s_P_dz"          %(str_e) 
        str_P_dphi         = "%s_P_dphi"        %(str_e) 
        str_ext_Z          = "%s_ext_z"         %(str_e) 
        str_emc_hit_energy = "%s_emc_hit_energy"%(str_e)
        mdc_mom          = getattr(tree,str_mdc_mom         )
        M_dtheta         = getattr(tree,str_M_dtheta        )
        M_dphi           = getattr(tree,str_M_dphi          )
        P_dz             = getattr(tree,str_P_dz            )
        P_dphi           = getattr(tree,str_P_dphi          )
        ext_Z            = getattr(tree,str_ext_Z           )
        tmp_Hit_E        = getattr(tree,str_emc_hit_energy  )
        MC_info[entryNum][0] = mdc_mom
        MC_info[entryNum][1] = M_dtheta
        MC_info[entryNum][2] = M_dphi
        MC_info[entryNum][3] = P_dz
        MC_info[entryNum][4] = P_dphi
        #if for_em and using_Low_z:
        #    if ext_Z<=-125 : continue
        #elif for_em and using_High_z:
        #    if ext_Z>-125 : continue
        #elif for_em==False and using_Low_z:
        #    if ext_Z>=125 : continue
        #elif for_em==False and using_High_z:
        #    if ext_Z<125 : continue
        MC_info[entryNum][5] = ext_Z
        for i in range(121):
            nRow = i - 11*int(i/11.0)
            nCol =10 - int(i/11.0)
            h_Hit_B_phi_z.Fill(nCol+0.1, 10.9-nRow, tmp_Hit_E[i])
            Barrel_Hit[entryNum, nRow, nCol, 0] = tmp_Hit_E[i]
    
    if True:
        dele_list = []
        for i in range(MC_info.shape[0]):
            pass_z_cut = True
            if using_Z_cut and for_em  and MC_info[i,5]>0:
                pass_z_cut = False
            elif using_Z_cut and for_em==False  and MC_info[i,5]<0:
                pass_z_cut = False
            if np.sum(Barrel_Hit[i]) ==0 or pass_z_cut==False :
                dele_list.append(i) ## remove the event has no hits
        MC_info    = np.delete(MC_info   , dele_list, axis = 0)
        Barrel_Hit = np.delete(Barrel_Hit, dele_list, axis = 0)
    
    str_e = 'e+'        
    if for_em: str_e = 'e-'        
    str_hit = 'Hit'
    plot_hist(h_Hit_B_phi_z ,'%s_barrel_phi_z_plane_%s'%(str_hit, str_e), 'Bhabha %s'%str_e)
    
    hf = h5py.File(outFileName, 'w')
    hf.create_dataset('Barrel_Hit', data=Barrel_Hit)
    hf.create_dataset('MC_info'   , data=MC_info)
    hf.close()
    print ('Done')
    
