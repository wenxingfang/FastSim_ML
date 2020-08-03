#ifndef MuonFastSimVoxel_hh
#define MuonFastSimVoxel_hh

/*
 * Description:
 *     This class is used for the Voxel Method of Muon Fast Simulation.
 *     The distribution of hit time and nPE are generated from
 *     * anajuno/muonsim/share/dispatch.C
 *     * anajuno/muonsim/share/dispatch2.C
 *
 *     We only consider the vertex \vec{r} and position of PMT \vec{PMT}:
 *     * r**3
 *     * theta, <\vec{r}, \vec{PMT}>
 *
 *     A validation code can also be found in:
 *     * anajuno/muonsim/share/validateNPE.C
 *
 *     Three necessary helpers:
 *     1. Geometry Helper, supply the position of PMT
 *     2. Hit Time Helper, generate TOF (time of flight)
 *     3. NPE Helper, generate NPE.
 *
 *     The generated hits are put into hit collection. 
 * Author: Tao Lin <lintao@ihep.ac.cn>
 */

#include "SniperKernel/ToolBase.h"
#include "DetSimAlg/IAnalysisElement.h"

// retrieve the hit collection
#include "dywHit_PMT.hh"
#include "G4HCofThisEvent.hh"
#include "G4VHitsCollection.hh"
#include "G4SDManager.hh"
#include "MuonFastSimVoxelHelper.hh"
#include "PMTHitMerger.hh"


#include <map>
#include <vector>

#include <TGraph2D.h>
#include <TH2D.h>
#include <random>


#include "../NN/NNPred.h"

class G4Event;

class MuonFastSimVoxel: public IAnalysisElement,
                        public ToolBase {
public:
    MuonFastSimVoxel(const std::string& name);
    ~MuonFastSimVoxel();

    // Run Action
    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);
    // Event Action
    virtual void BeginOfEventAction(const G4Event*);
    virtual void EndOfEventAction(const G4Event*);
    // Stepping Action
    virtual void UserSteppingAction(const G4Step*); 

private:
    // bool generate_hits(int binx, int biny, double fractionPart, int pmtid, double start_time);
    bool generate_hits   (float r, float theta, double ratio, int pmtid, double start_time);
    bool generate_hits_v1(float r, float theta, double depe , int pmtid, double start_time); 
    bool generate_hits_v2(float r, float theta, double ratio, int pmtid, double start_time); 
    bool generate_hits_v3(float r , const std::vector<float>& vec_theta, const std::vector<int>& vec_id, float frac, double start_time);
    float get_lambda(float r, float theta, bool doInterP);
    bool save_hits(int pmtid, double hittime);
    bool save_hits_v(const std::vector<int>& pmtids, const std::vector<float>& hittimes, float s_time) ;

    double quenching(const G4Step* step);

private:
    // helper classes and variables
    VoxelMethodHelper::Geom* m_helper_geom;
    VoxelMethodHelper::HitTimeLoader* m_helper_hittime;
    VoxelMethodHelper::NPELoader* m_helper_npe;

    // input
    std::string geom_file;
    std::string npe_loader_file;
    std::string hittime_mean_file;
    std::string hittime_single_file;

    std::string m_hittime_low_input_op;
    std::string m_hittime_low_output_op;
    std::string hittime_low_bp_file;
    std::string m_hittime_high_input_op;
    std::string m_hittime_high_output_op;
    std::string hittime_high_bp_file;
    std::string m_npe_low_input_op;
    std::string m_npe_low_output_op;
    std::string npe_low_bp_file;
    std::string m_npe_high_input_op;
    std::string m_npe_high_output_op;
    std::string npe_high_bp_file;
    // merger
    // copy from dywSD_PMT_v2
    // keep the flag and time window exist
    bool m_merge_flag;
    double m_time_window;
    PMTHitMerger* m_pmthitmerger;

    // debug
    bool m_fill_tuple;
    TTree* m_evt_tree;
    int m_evtid;
    std::vector<int> m_pdgid;
    std::vector<float> m_edep;
    std::vector<float> m_Qedep;
    std::vector<float> m_steplength;

    bool m_quenching;
    double m_quenching_scale;
    double m_birksConst1;
    double m_birksConst2;

    // lazy loading
    bool m_lazy_loading;

    bool m_flag_npe;
    bool m_flag_time;
    bool m_flag_savehits;

    // wxfang
    bool m_use_hittime_dl;
    bool m_use_npe_dl;
    NNPred* m_NNpred_time_low;
    NNPred* m_NNpred_time_high;
    NNPred* m_NNpred_npe_low;
    NNPred* m_NNpred_npe_high;
    int m_PMTs;    
    std::string npe_mean_file;
    bool m_use_npe_mean;
    //TGraph2D* m_gr2d_npe_mean;
    TH2D* m_gr2d_npe_mean;
    int n_bins_x;
    int n_bins_y;
    float gr_x_min;
    float gr_x_max;
    float gr_y_min;
    float gr_y_max;
    std::mt19937 generator;
    G4ThreeVector m_pos0;
    int m_track_id;
};

#endif
