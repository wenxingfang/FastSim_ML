#!/usr/bin/env python
"""
Parameter Base
"""

__author__ = "JING Maoqiang <jingmq@ihep.ac.cn>"
__copyright__ = "Copyright (c) JING Maoqiang"
__created__ = "[2020-03-29 Sun 17:08]" 

import sys 
import os
import ROOT 


# ---------------------------------------------
# Parameters
# ---------------------------------------------

def pid_eff(var):
    if var == 'costheta':
        xmin = -1.0
        xmax = 1.0
        xbins = 20
    if var == 'ptrk':
        xmin = 0
        xmax = 2.5
        xbins = 25
    if var == 'phi':
        xmin = -3.14
        xmax = 3.14
        xbins = 20
    return xmin, xmax, xbins
def data_path(particle):
    if particle == 'e':
        data_path = '/junofs/users/wxfang/FastSim/GAN/BES/Dedx/raw_data/electron/electron.root'
        #data_path = '/junofs/users/wxfang/FastSim/GAN/BES/Dedx/pid_check/mc/mc_e-_NN.root'
        #data_path = '/junofs/users/wxfang/FastSim/GAN/BES/Dedx/pid_check/mc/mc_e-_off.root'
        mc_path_1 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/e-/output_off/*.root'
        #mc_path_2 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/e-/output_NN/*.root'
        mc_path_2 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output/NN/e-_twoSP/*.root'
        return (data_path,mc_path_1,mc_path_2)
    elif particle == 'p':
        data_path = '/junofs/users/wxfang/FastSim/GAN/BES/Dedx/raw_data/proton/proton.root'
        mc_path_1 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output_off/*.root'
        mc_path_2 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output_NN/*.root'
        #return (data_path,mc_path_1,mc_path_2)
        return (data_path, data_path, data_path)
    elif particle == 'K':
        data_path = '/junofs/users/wxfang/FastSim/GAN/BES/Dedx/raw_data/kaon/kaon.root'
        #mc_path_1 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output_off/*.root'
        #mc_path_2 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output_NN//*.root'
        mc_path_1 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output/off/k+_noDecay/*.root'
        mc_path_2 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output/NN//k+_noDecay/*.root'
        #mc_path_1 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output/off/k+/*.root'
        #mc_path_2 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output//NN/k+/*.root'
        return (data_path,mc_path_1,mc_path_2)
    elif particle == 'pi':
        data_path = '/junofs/users/wxfang/FastSim/GAN/BES/Dedx/raw_data/pion/pion.root'
        #mc_path_1 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output/off/pi+/*.root'
        #mc_path_2 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output//NN/pi+/*.root'
        #mc_path_1 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output/off/pi+_noDecay/*.root'
        #mc_path_2 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output/NN//pi+_noDecay/*.root'
        mc_path_1 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output/off/pi+_noDecay_mcma/*.root'
        mc_path_2 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output/NN//pi+_noDecay_mcma/*.root'
        return (data_path,mc_path_1,mc_path_2)
    elif particle == 'mu':
        data_path = '/junofs/users/wxfang/FastSim/GAN/BES/Dedx/raw_data/muon/muon.root'
        mc_path_1 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output/off/mu+/*.root'
        mc_path_2 = '/junofs/users/wxfang/FastSim/bes3/workarea/Mdc/DedxTrack/output//NN/mu+/*.root'
        return (data_path,mc_path_1,mc_path_2)
    else:
        print('wrong particle name, exit()')
        os.exit()
def com_patches(var, particle):
    file_list = []
    file_list.append('/besfs/groups/cal/dedx/jingmq/bes/pid_eff_check/python/files/dedx_pideff_' + var + '_' + particle + '_705_data.root')
    # file_list.append('/besfs/groups/cal/dedx/jingmq/bes/pid_eff_check/python/files/test_dedx_pideff_' + var + '_' + particle + '_705_data.root')
    file_list.append('/besfs/groups/cal/dedx/jingmq/bes/pid_eff_check/python/files/test_dedx_pideff_' + var + '_K_705_data.root')
    file_list.append('/besfs/groups/cal/dedx/jingmq/bes/pid_eff_check/python/files/test_dedx_pideff_' + var + '_p_705_data.root')
    return file_list
