import ROOT as rt
import numpy as np
import h5py 
import sys 
import gc
import math
rt.gROOT.SetBatch(rt.kTRUE)

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
    hist.GetXaxis().SetTitle("#Delta Z(cm)")
    if 'r_z' in out_name:
        hist.GetYaxis().SetTitle("R (cm)")
    elif 'phi_z' in out_name:
        hist.GetYaxis().SetTitle("#Delta#phi^{#circ}")
    hist.SetTitle(title)
    #hist.SetTitleSize(0.1)
    #hist.Draw("COLZ TEXT")
    hist.Draw("COLZ")
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

print ('Start..')
cell_x = 5.0 # 5 cm in Z
cell_y = 3.0 # 3 degree in phi
Depth = [90, 100, 110, 120, 130]

print ('Read root file')
plot_path='./raw_plots'
filePath = '/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/Data_BaBa.root'
outFileName='Hit_Barrel.h5'
treeName='Bhabha'
chain =rt.TChain(treeName)
chain.Add(filePath)
tree = chain
totalEntries=tree.GetEntries()
print (totalEntries)
maxEvent = totalEntries
maxEvent = 10
nBin = 20 
Barrel_Hit = np.full((maxEvent, nBin , nBin, len(Depth)-1 ), 0 ,dtype=np.float32)#init 
MC_info    = np.full((maxEvent, 3 ), 0 ,dtype=np.float32)#init 
dz=50        #+- 50 cm
r_min= 90
r_max= 130
phi_min= -30 #+- 30 degrees
phi_max= 30
for_em = False
for_Fhit = False

for entryNum in range(0, tree.GetEntries()):
    tree.GetEntry(entryNum)
    if entryNum>= maxEvent: break
    isBB   = getattr(tree, "isBB")
    if isBB == 0 : continue
    h_Hit_B_r_z = rt.TH2F('evt%d_Hit_B_r_z'%entryNum     , '', 2*dz, -1*dz, dz ,r_max-r_min, r_min, r_max)
    h_Hit_B_phi_z = rt.TH2F('evt%d_Hit_B_phi_z'%entryNum , '', 2*dz, -1*dz, dz ,phi_max-phi_min, phi_min, phi_max)
    str_e = 'ep'
    if for_em: str_e = 'em'
    str_mdc_mom        = "%s_mdc_mom"       %(str_e) 
    str_mdc_theta      = "%s_mdc_theta"     %(str_e) 
    str_mdc_phi        = "%s_mdc_phi"       %(str_e) 
    str_ext_x          = "%s_ext_x"         %(str_e) 
    str_ext_y          = "%s_ext_y"         %(str_e) 
    str_ext_z          = "%s_ext_z"         %(str_e) 
    str_emc_hit_energy = "%s_emc_hit_energy"%(str_e)
    str_emc_hit_x      = "%s_emc_hit_x"     %(str_e) 
    str_emc_hit_y      = "%s_emc_hit_y"     %(str_e) 
    str_emc_hit_z      = "%s_emc_hit_z"     %(str_e) 
    if for_Fhit: 
        str_emc_hit_x      = "%s_emc_hit_Fx"     %(str_e) 
        str_emc_hit_y      = "%s_emc_hit_Fy"     %(str_e) 
        str_emc_hit_z      = "%s_emc_hit_Fz"     %(str_e) 
    

    mdc_mom        = getattr(tree, str_mdc_mom        )
    mdc_theta      = getattr(tree, str_mdc_theta      )
    mdc_phi        = getattr(tree, str_mdc_phi        )
    tmp_HitFirst_x = getattr(tree, str_ext_x          )
    tmp_HitFirst_y = getattr(tree, str_ext_y          )
    tmp_HitFirst_z = getattr(tree, str_ext_z          )
    tmp_Hit_x      = getattr(tree, str_emc_hit_x      )
    tmp_Hit_y      = getattr(tree, str_emc_hit_y      )
    tmp_Hit_z      = getattr(tree, str_emc_hit_z      )
    tmp_Hit_E      = getattr(tree, str_emc_hit_energy )
    MC_info[entryNum][0] = mdc_theta*180/math.pi
    MC_info[entryNum][1] = mdc_phi*180/math.pi  # phi
    MC_info[entryNum][2] = mdc_mom 
    tmp_HitFirst_phi = math.atan(tmp_HitFirst_y/tmp_HitFirst_x)*180/math.pi
    for i in range(0, len(tmp_Hit_x)):
        tmp_Hit_r = math.sqrt(tmp_Hit_x[i]*tmp_Hit_x[i] + tmp_Hit_y[i]*tmp_Hit_y[i])
        tmp_Hit_phi = math.atan(tmp_Hit_y[i]/tmp_Hit_x[i])*180/math.pi
        if tmp_Hit_r < Depth[0] or tmp_Hit_r > Depth[-1]: continue
        h_Hit_B_r_z  .Fill(tmp_Hit_z[i]-tmp_HitFirst_z, tmp_Hit_r                   , tmp_Hit_E[i])
        h_Hit_B_phi_z.Fill(tmp_Hit_z[i]-tmp_HitFirst_z, tmp_Hit_phi-tmp_HitFirst_phi, tmp_Hit_E[i])

    str_e = 'e+'        
    if for_em: str_e = 'e-'        
    str_hit = 'Hit'
    if for_Fhit:str_hit = 'FHit'
    plot_hist(h_Hit_B_r_z   ,'evt%d_%s_barrel_r_z_plane_%s'  %(entryNum, str_hit, str_e), 'Bhabha %s'%str_e)
    plot_hist(h_Hit_B_phi_z ,'evt%d_%s_barrel_phi_z_plane_%s'%(entryNum, str_hit, str_e), 'Bhabha %s'%str_e)
