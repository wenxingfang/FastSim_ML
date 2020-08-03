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
    hist.GetXaxis().SetTitle("#Delta Z (cm)")
    if 'r_z' in out_name:
        hist.GetYaxis().SetTitle("R (cm)")
    elif 'phi_z' in out_name:
        hist.GetYaxis().SetTitle("#Delta#phi (dergee)")
    hist.SetTitle(title)
    #hist.SetTitleSize(0.1)
    #hist.Draw("COLZ TEXT")
    hist.Draw("COLZ")
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

for_test = True
for_em   = False
for_Fhit = False

print ('Start..')
cell_x = 5.0 # 5 cm in Z
cell_y = 3.0 # 3 degree in phi
#Depth = [90, 100, 110, 120, 130]
Depth = [90, 130]

print ('Read root file')
plot_path='./raw_plots'
filePath = '/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/Data_BaBa_BB_100k.root'
outFileName='Hit_Barrel_ep_100k.h5'
if for_em: outFileName='Hit_Barrel_em_100k.h5'
if for_test:
    filePath = '/junofs/users/wxfang/FastSim/bes3/workarea/Luminosity/Data_BaBa_BB.root'
    outFileName='Hit_Barrel_ep_test.h5'
    if for_em: outFileName='Hit_Barrel_em_test.h5'
treeName='Bhabha'
chain =rt.TChain(treeName)
chain.Add(filePath)
tree = chain
totalEntries=tree.GetEntries()
print (totalEntries)
maxEvent = totalEntries
nBin_row = 12 
nBin_col = 16 
#nBin = 20 
Barrel_Hit = np.full((maxEvent, nBin_row , nBin_col, len(Depth)-1 ), 0 ,dtype=np.float32)#init 
Hit_Depth  = np.full((maxEvent, 1 ), 0 ,dtype=np.float32)#init 
MC_info    = np.full((maxEvent, 4 ), 0 ,dtype=np.float32)#init 
#dz=50        #+- 50 cm
dz=40        #+- 40 cm
r_min= 90
r_max= 130
#phi_min= -30 #+- 30 degrees
#phi_max= 30
phi_min= -18 #+- 30 degrees
phi_max= 18
h_Hit_B_r_z   = rt.TH2F('Hit_B_r_z'   , '', 2*dz, -1*dz, dz ,r_max-r_min    , r_min  , r_max  )
h_Hit_B_phi_z = rt.TH2F('Hit_B_phi_z' , '', 2*dz, -1*dz, dz ,phi_max-phi_min, phi_min, phi_max)

for entryNum in range(0, tree.GetEntries()):
    tree.GetEntry(entryNum)
    if entryNum>= maxEvent: break
    isBB   = getattr(tree, "isBB")
    str_e = 'ep'
    if for_em: str_e = 'em'
    str_mdc_mom        = "%s_mdc_mom"       %(str_e) 
    str_mdc_theta      = "%s_mdc_theta"     %(str_e) 
    str_mdc_phi        = "%s_mdc_phi"       %(str_e) 
    str_ext_x          = "%s_ext_x"         %(str_e) 
    str_ext_y          = "%s_ext_y"         %(str_e) 
    str_ext_z          = "%s_ext_z"         %(str_e) 
    str_ext_Px         = "%s_ext_Px"        %(str_e) 
    str_ext_Py         = "%s_ext_Py"        %(str_e) 
    str_ext_Pz         = "%s_ext_Pz"        %(str_e) 
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
    tmp_HitFirst_Px= getattr(tree, str_ext_Px         )
    tmp_HitFirst_Py= getattr(tree, str_ext_Py         )
    tmp_HitFirst_Pz= getattr(tree, str_ext_Pz         )
    tmp_Hit_x      = getattr(tree, str_emc_hit_x      )
    tmp_Hit_y      = getattr(tree, str_emc_hit_y      )
    tmp_Hit_z      = getattr(tree, str_emc_hit_z      )
    tmp_Hit_E      = getattr(tree, str_emc_hit_energy )
    MC_info[entryNum][0] = mdc_mom
    tmp_HitFirst_Pt = math.sqrt(tmp_HitFirst_Px*tmp_HitFirst_Px + tmp_HitFirst_Py*tmp_HitFirst_Py)
    HitFirstPtheta  = math.atan(tmp_HitFirst_Pt/tmp_HitFirst_Pz)*180/math.pi
    HitFirstPphi    = math.atan(tmp_HitFirst_Py/tmp_HitFirst_Px)*180/math.pi
    tmp_HitFirst_phi= math.atan(tmp_HitFirst_y/tmp_HitFirst_x)  *180/math.pi
    DeltaTheta      = HitFirstPtheta
    DeltaPhi        = HitFirstPphi - tmp_HitFirst_phi
    if DeltaPhi > 150 : DeltaPhi= -(180-DeltaPhi) # for ep it will < ~ -170 ,for em it will > ~ 170, it means B field is opposite to Z axis?
    if DeltaPhi < -150: DeltaPhi=   180+DeltaPhi
    MC_info[entryNum][1] = DeltaTheta if DeltaTheta > 0 else DeltaTheta + 180
    MC_info[entryNum][2] = DeltaPhi
    MC_info[entryNum][3] = tmp_HitFirst_z
    tmp_Hit_r_list = []
    for i in range(0, len(tmp_Hit_x)):
        tmp_Hit_r = math.sqrt(tmp_Hit_x[i]*tmp_Hit_x[i] + tmp_Hit_y[i]*tmp_Hit_y[i])
        if tmp_Hit_r < Depth[0] or tmp_Hit_r > Depth[-1]: continue
        tmp_Hit_r_list.append(tmp_Hit_r)
        tmp_Hit_phi = math.atan(tmp_Hit_y[i]/tmp_Hit_x[i])*180/math.pi
        h_Hit_B_r_z  .Fill(tmp_Hit_z[i]-tmp_HitFirst_z, tmp_Hit_r                   , tmp_Hit_E[i])
        h_Hit_B_phi_z.Fill(tmp_Hit_z[i]-tmp_HitFirst_z, tmp_Hit_phi-tmp_HitFirst_phi, tmp_Hit_E[i])
        '''
        for dp in  range(len(Depth)):
            if Depth[dp] <= tmp_Hit_r and tmp_Hit_r < Depth[dp+1] :
                index_dep = dp
                break
        '''
        if tmp_Hit_z[i] > tmp_HitFirst_z: index_col = int((tmp_Hit_z[i]-tmp_HitFirst_z)/cell_x) + int(0.5*nBin_col)
        else : index_col = int((tmp_Hit_z[i]-tmp_HitFirst_z)/cell_x) + int(0.5*nBin_col) -1
        if tmp_Hit_phi > tmp_HitFirst_phi: index_row = int((tmp_Hit_phi-tmp_HitFirst_phi)/cell_y) + int(0.5*nBin_row)
        else : index_row = int((tmp_Hit_phi-tmp_HitFirst_phi)/cell_y) + int(0.5*nBin_row) -1
        if index_col >= nBin_col or index_col <0 or index_row >= nBin_row or index_row<0: continue; ##skip this hit now, maybe can merge it?
        #index_row = int(index_row)
        #index_col = int(index_col)
        index_dep = 0
        Barrel_Hit[entryNum, index_row, index_col, index_dep] = Barrel_Hit[entryNum, index_row, index_col, index_dep] + tmp_Hit_E[i]
    Hit_Depth[entryNum][0] = sum(tmp_Hit_r_list)/len(tmp_Hit_r_list) if len(tmp_Hit_r_list)!=0 else 0

if True:
    dele_list = []
    for i in range(MC_info.shape[0]):
        #if abs(MC_info[i,2])>150:# for ep it will < ~ -170 ,for em it will > ~ 170, it means B field is opposite to Z axis?
        #    dele_list.append(i) ## remove the event has large dphi, which means this event emc first hit around 90 or 270 degree
        if Hit_Depth[i][0]==0 and i not in dele_list:
            dele_list.append(i) ## remove the event has no hits
    MC_info    = np.delete(MC_info   , dele_list, axis = 0)
    Hit_Depth  = np.delete(Hit_Depth , dele_list, axis = 0)
    Barrel_Hit = np.delete(Barrel_Hit, dele_list, axis = 0)

str_e = 'e+'        
if for_em: str_e = 'e-'        
str_hit = 'Hit'
if for_Fhit:str_hit = 'FHit'
plot_hist(h_Hit_B_r_z   ,'%s_barrel_r_z_plane_%s'  %(str_hit, str_e), 'Bhabha %s'%str_e)
plot_hist(h_Hit_B_phi_z ,'%s_barrel_phi_z_plane_%s'%(str_hit, str_e), 'Bhabha %s'%str_e)

hf = h5py.File(outFileName, 'w')
hf.create_dataset('Barrel_Hit', data=Barrel_Hit)
hf.create_dataset('Hit_Depth' , data=Hit_Depth )
hf.create_dataset('MC_info'   , data=MC_info)
hf.close()
print ('Done')

'''
g2D =  rt.TGraph2D()
g2D.SetPoint(n, tmp_Hit_x[i], tmp_Hit_y[i], tmp_Hit_z[i])
n=n+1
do_plot("",g2D, "total","e- 50GeV")
'''

