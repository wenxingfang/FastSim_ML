#!/usr/bin/env python
"""
Get useful info from raw root files
"""

__author__ = "Maoqiang JING <jingmq@ihep.ac.cn>"
__copyright__ = "Copyright (c) Maoqiang JING"
__created__ = "[2019-09-03 Tue 05:41]"

from math import *
from array import array
import ROOT
from ROOT import TCanvas, gStyle, TLorentzVector, TTree
from ROOT import TFile, TH1F, TLegend, TArrow, TChain, TVector3
import sys, os
import logging
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s- %(message)s')

def usage():
    sys.stdout.write('''
NAME
    get_info.py

SYNOPSIS
    ./get_info.py [file_in] [file_out] [process]

AUTHOR
    Maoqiang JING <jingmq@ihep.ac.cn>

DATE
    August 2019
\n''')

def save(f_in, t, process):
    m_runNo = array('i', [0])
    m_evtNo = array('i', [0])
    m_E_beam = array('d', [999.])
    m_mode = array('i', [0])
    m_n_pi = array('i', [0])
    m_n_K = array('i', [0])
    m_prob_K = array('d', 100*[999.])
    m_prob_pi = array('d', 100*[999.])
    m_chi_K = array('d', 100*[999.])
    m_chi_pi = array('d', 100*[999.])
    m_charge_K = array('d', 100*[999.])
    m_charge_pi = array('d', 100*[999.])
    m_p_K = array('d', 100*[999.])
    m_p_pi = array('d', 100*[999.])
    m_cos_K = array('d', 100*[999.])
    m_cos_pi = array('d', 100*[999.])
    m_phi_K = array('d', 100*[999.])
    m_phi_pi = array('d', 100*[999.])
    m_probPH_K = array('d', 100*[999.])
    m_probPH_pi = array('d', 100*[999.])
    m_goodHits_K = array('d', 100*[999.])
    m_goodHits_pi = array('d', 100*[999.])
    m_chi2_vf = array('d', [999.])
    m_chi2_kf = array('d', [999.])
    t.Branch('runNo', m_runNo, 'm_runNo/I')
    t.Branch('evtNo', m_evtNo, 'm_evtNo/I')
    t.Branch('E_beam', m_E_beam, 'm_E_beam/I')
    t.Branch('mode', m_mode, 'm_mode/I')
    t.Branch('n_pi', m_n_pi, 'm_n_pi/I')
    t.Branch('n_K', m_n_K, 'm_n_K/I')
    t.Branch('prob_pi', m_prob_pi, 'm_prob_pi/D')
    t.Branch('prob_K', m_prob_K, 'm_prob_K/D')
    t.Branch('chi_pi', m_chi_pi, 'm_chi_pi/D')
    t.Branch('chi_K', m_chi_K, 'm_chi_K/D')
    t.Branch('charge_pi', m_charge_pi, 'm_charge_pi/D')
    t.Branch('charge_K', m_charge_K, 'm_charge_K/D')
    t.Branch('p_pi', m_p_pi, 'm_p_pi/D')
    t.Branch('p_K', m_p_K, 'm_p_K/D')
    t.Branch('cos_pi', m_cos_pi, 'm_cos_pi/D')
    t.Branch('cos_K', m_cos_K, 'm_cos_K/D')
    t.Branch('phi_pi', m_phi_pi, 'm_phi_pi/D')
    t.Branch('phi_K', m_phi_K, 'm_phi_K/D')
    t.Branch('probPH_pi', m_probPH_pi, 'm_probPH_pi/D')
    t.Branch('probPH_K', m_probPH_K, 'm_probPH_K/D')
    t.Branch('goodHits_pi', m_goodHits_pi, 'm_goodHits_pi/D')
    t.Branch('goodHits_K', m_goodHits_K, 'm_goodHits_K/D')
    t.Branch('chi2_vf', m_chi2_vf, 'm_chi2_vf/D')
    t.Branch('chi2_kf', m_chi2_kf, 'm_chi2_kf/D')
    t_track = f_in.Get('track')
    nentries = t_track.GetEntries()
    for ientry in range(nentries):
        t_track.GetEntry(ientry)
        if t_track.chi2_vf > 100 and t_track.chi2_kf > 200 and t_track.chi2_kf < 0:
            continue

        if t_track.mode == 1 and process == 'PiPiPiPi':
            m_runNo[0] = t_track.runNo
            m_evtNo[0] = t_track.evtNo
            m_E_beam[0] = t_track.beamE
            m_mode[0] = t_track.mode
            m_n_pi[0] = t_track.n_pi
            for i in xrange(t_track.n_pi):
                for j in xrange(5):
                    m_prob_pi[i * 5 + j] = t_track.trk_pi[i * 16 + j];
                for j in xrange(5):
                    m_chi_pi[i * 5 + j] = t_track.trk_pi[i * 16 + j + 5];
                m_charge_pi[i] = t_track.trk_pi[i * 16 + 10]
                m_p_pi[i] = t_track.trk_pi[i * 16 + 11]
                m_cos_pi[i] = t_track.trk_pi[i * 16 + 12]
                m_phi_pi[i] = t_track.trk_pi[i * 16 + 13]
                m_probPH_pi[i] = t_track.trk_pi[i * 16 + 14]
                m_goodHits_pi[i] = t_track.trk_pi[i * 16 + 15]
            m_chi2_vf[0] = t_track.chi2_vf
            m_chi2_kf[0] = t_track.chi2_kf
            t.Fill()

        if t_track.mode == 2 and process == 'PiPiPi0Pi0':
            m_runNo[0] = t_track.runNo
            m_evtNo[0] = t_track.evtNo
            m_E_beam[0] = t_track.beamE
            m_mode[0] = t_track.mode
            m_n_pi[0] = t_track.n_pi
            for i in xrange(t_track.n_pi):
                for j in xrange(5):
                    m_prob_pi[i * 5 + j] = t_track.trk_pi[i * 16 + j];
                for j in xrange(5):
                    m_chi_pi[i * 5 + j] = t_track.trk_pi[i * 16 + j + 5];
                m_charge_pi[i] = t_track.trk_pi[i * 16 + 10]
                m_p_pi[i] = t_track.trk_pi[i * 16 + 11]
                m_cos_pi[i] = t_track.trk_pi[i * 16 + 12]
                m_phi_pi[i] = t_track.trk_pi[i * 16 + 13]
                m_probPH_pi[i] = t_track.trk_pi[i * 16 + 14]
                m_goodHits_pi[i] = t_track.trk_pi[i * 16 + 15]
            m_chi2_vf[0] = t_track.chi2_vf
            m_chi2_kf[0] = t_track.chi2_kf
            t.Fill()

        if t_track.mode == 3 and process == 'PiPiPiPiPiPi':
            m_runNo[0] = t_track.runNo
            m_evtNo[0] = t_track.evtNo
            m_E_beam[0] = t_track.beamE
            m_mode[0] = t_track.mode
            m_n_pi[0] = t_track.n_pi
            for i in xrange(t_track.n_pi):
                for j in xrange(5):
                    m_prob_pi[i * 5 + j] = t_track.trk_pi[i * 16 + j];
                for j in xrange(5):
                    m_chi_pi[i * 5 + j] = t_track.trk_pi[i * 16 + j + 5];
                m_charge_pi[i] = t_track.trk_pi[i * 16 + 10]
                m_p_pi[i] = t_track.trk_pi[i * 16 + 11]
                m_cos_pi[i] = t_track.trk_pi[i * 16 + 12]
                m_phi_pi[i] = t_track.trk_pi[i * 16 + 13]
                m_probPH_pi[i] = t_track.trk_pi[i * 16 + 14]
                m_goodHits_pi[i] = t_track.trk_pi[i * 16 + 15]
            m_chi2_vf[0] = t_track.chi2_vf
            m_chi2_kf[0] = t_track.chi2_kf
            t.Fill()

        if t_track.mode == 4 and process == 'PiPiPiPiPi0Pi0':
            m_runNo[0] = t_track.runNo
            m_evtNo[0] = t_track.evtNo
            m_E_beam[0] = t_track.beamE
            m_mode[0] = t_track.mode
            m_n_pi[0] = t_track.n_pi
            for i in xrange(t_track.n_pi):
                for j in xrange(5):
                    m_prob_pi[i * 5 + j] = t_track.trk_pi[i * 16 + j];
                for j in xrange(5):
                    m_chi_pi[i * 5 + j] = t_track.trk_pi[i * 16 + j + 5];
                m_charge_pi[i] = t_track.trk_pi[i * 16 + 10]
                m_p_pi[i] = t_track.trk_pi[i * 16 + 11]
                m_cos_pi[i] = t_track.trk_pi[i * 16 + 12]
                m_phi_pi[i] = t_track.trk_pi[i * 16 + 13]
                m_probPH_pi[i] = t_track.trk_pi[i * 16 + 14]
                m_goodHits_pi[i] = t_track.trk_pi[i * 16 + 15]
            m_chi2_vf[0] = t_track.chi2_vf
            m_chi2_kf[0] = t_track.chi2_kf
            t.Fill()

        if t_track.mode == 5 and process == 'KKKK':
            m_runNo[0] = t_track.runNo
            m_evtNo[0] = t_track.evtNo
            m_E_beam[0] = t_track.beamE
            m_mode[0] = t_track.mode
            m_n_K[0] = t_track.n_K
            for i in xrange(t_track.n_K):
                for j in xrange(5):
                    m_prob_K[i * 5 + j] = t_track.trk_K[i * 16 + j];
                for j in xrange(5):
                    m_chi_K[i * 5 + j] = t_track.trk_K[i * 16 + j + 5];
                m_charge_K[i] = t_track.trk_K[i * 16 + 10]
                m_p_K[i] = t_track.trk_K[i * 16 + 11]
                m_cos_K[i] = t_track.trk_K[i * 16 + 12]
                m_phi_K[i] = t_track.trk_K[i * 16 + 13]
                m_probPH_K[i] = t_track.trk_K[i * 16 + 14]
                m_goodHits_K[i] = t_track.trk_K[i * 16 + 15]
            m_chi2_vf[0] = t_track.chi2_vf
            m_chi2_kf[0] = t_track.chi2_kf
            t.Fill()

        if t_track.mode == 6 and process == 'KKPi0Pi0':
            m_runNo[0] = t_track.runNo
            m_evtNo[0] = t_track.evtNo
            m_E_beam[0] = t_track.beamE
            m_mode[0] = t_track.mode
            m_n_K[0] = t_track.n_K
            for i in xrange(t_track.n_K):
                for j in xrange(5):
                    m_prob_K[i * 5 + j] = t_track.trk_K[i * 16 + j];
                for j in xrange(5):
                    m_chi_K[i * 5 + j] = t_track.trk_K[i * 16 + j + 5];
                m_charge_K[i] = t_track.trk_K[i * 16 + 10]
                m_p_K[i] = t_track.trk_K[i * 16 + 11]
                m_cos_K[i] = t_track.trk_K[i * 16 + 12]
                m_phi_K[i] = t_track.trk_K[i * 16 + 13]
                m_probPH_K[i] = t_track.trk_K[i * 16 + 14]
                m_goodHits_K[i] = t_track.trk_K[i * 16 + 15]
            m_chi2_vf[0] = t_track.chi2_vf
            m_chi2_kf[0] = t_track.chi2_kf
            t.Fill()

        if t_track.mode == 7 and process == 'KKKKKK':
            m_runNo[0] = t_track.runNo
            m_evtNo[0] = t_track.evtNo
            m_E_beam[0] = t_track.beamE
            m_mode[0] = t_track.mode
            m_n_K[0] = t_track.n_K
            for i in xrange(t_track.n_K):
                for j in xrange(5):
                    m_prob_K[i * 5 + j] = t_track.trk_K[i * 16 + j];
                for j in xrange(5):
                    m_chi_K[i * 5 + j] = t_track.trk_K[i * 16 + j + 5];
                m_charge_K[i] = t_track.trk_K[i * 16 + 10]
                m_p_K[i] = t_track.trk_K[i * 16 + 11]
                m_cos_K[i] = t_track.trk_K[i * 16 + 12]
                m_phi_K[i] = t_track.trk_K[i * 16 + 13]
                m_probPH_K[i] = t_track.trk_K[i * 16 + 14]
                m_goodHits_K[i] = t_track.trk_K[i * 16 + 15]
            m_chi2_vf[0] = t_track.chi2_vf
            m_chi2_kf[0] = t_track.chi2_kf
            t.Fill()

        if t_track.mode == 8 and process == 'KKKKPi0Pi0':
            m_runNo[0] = t_track.runNo
            m_evtNo[0] = t_track.evtNo
            m_E_beam[0] = t_track.beamE
            m_mode[0] = t_track.mode
            m_n_K[0] = t_track.n_K
            for i in xrange(t_track.n_K):
                for j in xrange(5):
                    m_prob_K[i * 5 + j] = t_track.trk_K[i * 16 + j];
                for j in xrange(5):
                    m_chi_K[i * 5 + j] = t_track.trk_K[i * 16 + j + 5];
                m_charge_K[i] = t_track.trk_K[i * 16 + 10]
                m_p_K[i] = t_track.trk_K[i * 16 + 11]
                m_cos_K[i] = t_track.trk_K[i * 16 + 12]
                m_phi_K[i] = t_track.trk_K[i * 16 + 13]
                m_probPH_K[i] = t_track.trk_K[i * 16 + 14]
                m_goodHits_K[i] = t_track.trk_K[i * 16 + 15]
            m_chi2_vf[0] = t_track.chi2_vf
            m_chi2_kf[0] = t_track.chi2_kf
            t.Fill()

def main():
    args = sys.argv[1:]
    if len(args)<3:
        return usage()
    file_in = args[0]
    file_out = args[1]
    process = args[2]

    f_in = TFile(file_in)
    f_out = TFile(file_out, 'recreate')
    t_out = TTree('save', 'save')
    save(f_in, t_out, process)

    f_out.cd()
    t_out.Write()
    f_out.Close()

if __name__ == '__main__':
    main()
