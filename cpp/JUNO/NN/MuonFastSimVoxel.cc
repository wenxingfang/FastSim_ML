#include "MuonFastSimVoxel.hh"

#include "SniperKernel/SniperPtr.h"
#include "SniperKernel/ToolFactory.h"
#include "SniperKernel/SniperLog.h"
#include "SniperKernel/SniperException.h"
#include "RootWriter/RootWriter.h"

#include "G4Run.hh"
#include "G4Event.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
// force load the dict for vector<float>
#include "TROOT.h"

#include "DetSimAlg/DetSimAlg.h"
#include "PMTSDMgr.hh"

#include "G4LossTableManager.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4Gamma.hh"
#include "G4Electron.hh"


#include <math.h>


DECLARE_TOOL(MuonFastSimVoxel);

MuonFastSimVoxel::MuonFastSimVoxel(const std::string& name)
    : ToolBase(name)
{
    m_gr2d_npe_mean = 0;    

    m_helper_geom = 0;
    m_helper_hittime = 0;
    m_helper_npe = 0;

    geom_file = "geom-geom-20pmt.root";
    npe_loader_file = "npehist3d_single.root"; 
    hittime_mean_file = "hist3d.root";
    hittime_single_file = "dist_tres_single.root";
    //npe_mean_file = "r_theta_graph.root";
    npe_mean_file = "r_theta_TH2.root";
    hittime_low_bp_file  = "torch_model/hit_time_low_MM.pt";
    hittime_high_bp_file = "torch_model/hit_time_high_MM.pt";
    npe_low_bp_file      = "torch_model/model_npe_low_MM.pt";
    npe_high_bp_file     = "torch_model/model_npe_high_MM.pt";

    m_merge_flag = false;
    m_time_window = 1;
    m_pmthitmerger = 0;

    m_fill_tuple = false;
    m_evt_tree = 0;

    m_PMTs = 17613;
    //m_PMTs = 100;

    declProp("GeomFile", geom_file);
    declProp("NPEFile", npe_loader_file);
    declProp("HitTimeMean", hittime_mean_file);
    declProp("HitTimeRes", hittime_single_file);
    declProp("NPEMeanFile", npe_mean_file);
    declProp("HitTimeLowBp", hittime_low_bp_file);
    declProp("HitTimeHighBp", hittime_high_bp_file);
    declProp("NpeLowBp" , npe_low_bp_file);
    declProp("NpeHighBp", npe_high_bp_file);

    declProp("MergeFlag", m_merge_flag);
    declProp("MergeTimeWindow", m_time_window);

    declProp("EnableNtuple", m_fill_tuple);

    declProp("EnableQuenching", m_quenching=true);
    declProp("QuenchingFactor", m_quenching_scale=0.93);
    declProp("BirksConst1", m_birksConst1 = 6.5e-3*(g/cm2/MeV));
    declProp("BirksConst2", m_birksConst2 = 1.5e-6*(g/cm2/MeV)*(g/cm2/MeV));

    // if lazy loading is true, the histograms will not load in initialization.
    declProp("LazyLoading", m_lazy_loading=false);

    // enable/disable nPE sampling
    declProp("SampleNPE", m_flag_npe=true);
    // enable/disable hit time sampling
    declProp("SampleTime", m_flag_time=true);
    // enable/disable save hits
    declProp("SaveHits", m_flag_savehits=true);

    // if using npe mean from graph.
    declProp("UseNPEMean", m_use_npe_mean=false);
    declProp("UseNPEDL"  , m_use_npe_dl  =false);
    declProp("UseHitTimeDL", m_use_hittime_dl=false);

}

MuonFastSimVoxel::~MuonFastSimVoxel() 
{

}

void
MuonFastSimVoxel::BeginOfRunAction(const G4Run* /*aRun*/) {
   

   char* cstr_0 = new char[hittime_low_bp_file.size() + 1];
            strcpy(cstr_0, hittime_low_bp_file.c_str()); 
   char* cstr_1 = new char[hittime_high_bp_file.size() + 1];
            strcpy(cstr_1, hittime_high_bp_file.c_str()); 
   char* cstr_2 = new char[npe_low_bp_file.size() + 1];
            strcpy(cstr_2, npe_low_bp_file.c_str()); 
   char* cstr_3 = new char[npe_high_bp_file.size() + 1];
            strcpy(cstr_3, npe_high_bp_file.c_str()); 
   m_NNpred_time_low  = new NNPred(cstr_0);
   m_NNpred_time_high = new NNPred(cstr_1);
   m_NNpred_npe_low   = new NNPred(cstr_2);
   m_NNpred_npe_high  = new NNPred(cstr_3);

   //m_NNpred_time_low->get(3, 10.0, 10.0, -0.5, 100);
   //m_NNpred_time_high->get(4, 10.0, 10.0, -0.5, 100);
   //m_NNpred_npe_low->get(1, 10.0, 1.0, 0.5, 1);
   //m_NNpred_npe_low->get(1, 10.0, 1.0, 0.3, 1);
   //m_NNpred_npe_low->get(1, 10.0, 1.0, 0.1, 1);
   //m_NNpred_npe_high->get(2, 16.0, 2.0, 0.4, 1);
   //m_NNpred_npe_high->get(2, 16.0, 2.0, 0.2, 1);
   //int tmp_npe = round(m_NNpred_npe_high->get(2, 16.0, 2.0, 0.6, 1));
   //std::cout <<"tmp_npe="<<tmp_npe<< std::endl;

   /*
   std::cout <<"hi 1"<< std::endl;
   //m_NNpred = new NNPred();
   std::string s_name = "/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/job_sub/pyTorch_job/old/my_module_model.pt";
   char* cstr = new char[s_name.size() + 1];
   //char cstr[s_name.size() + 1];
   strcpy(cstr, s_name.c_str()); 
   std::cout <<"cstr="<<cstr<< std::endl;
   m_NNpred = new NNPred(cstr);
   std::cout <<"hi 2"<< std::endl;
   m_NNpred->get(1, 0.5);
   std::cout <<"hi 3"<< std::endl;
   m_NNpred->get(0.5, 0.5);
   std::cout <<"hi 4"<< std::endl;
   */
   /*
            torch::jit::script::Module module;
            try { module = torch::jit::load("/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/job_sub/pyTorch_job/my_module_model.pt");
                }
            catch (const c10::Error& e) {
                 std::cerr << "error loading the model\n";
                       }

            std::cout << "ok\n";
            std::vector<torch::jit::IValue> inputs;
            std::vector<at::Tensor> inputs_vec;
            inputs_vec.push_back(torch::ones({1, 2})); // 1xCxHxW
            inputs_vec.push_back(torch::ones({1, 2})); // 1xCxHxW
            at::Tensor input_ = torch::cat(inputs_vec);
            inputs.push_back(input_);
            at::Tensor output = module.forward(inputs).toTensor();
            std::cout << output << '\n';
   */ 

    // load the magic distributions
    if (!m_helper_geom) {
        LogInfo << "Load Geometry File for Voxel Method: "
                << geom_file << std::endl;
        m_helper_geom = new VoxelMethodHelper::Geom();
        m_helper_geom->m_geom_file = geom_file;
        m_helper_geom->load();
        assert(m_helper_geom->PMT_pos.size());
    }
    if (m_use_hittime_dl==false)
    {
        if (!m_helper_hittime) {
            LogInfo << "Load Hit Time File for Voxel Method: "
                    << hittime_mean_file << " / " 
                    << hittime_single_file << std::endl;
            m_helper_hittime = new VoxelMethodHelper::HitTimeLoader();
            m_helper_hittime->m_single_filename = hittime_single_file;
            m_helper_hittime->m_lazy_loading=m_lazy_loading;
            m_helper_hittime->load();
            assert(m_helper_hittime->f_single);
        }
    }
    else
    {
            LogInfo << "Load Hit Time File for Voxel Method from DL: "
                    << hittime_low_bp_file <<","<< hittime_high_bp_file << std::endl; 


            //m_hittime_low_pred = new TFPrediction(hittime_low_bp_file, m_hittime_low_input_op, m_hittime_low_output_op);
            //m_hittime_low_pred = new TFPrediction(hittime_low_bp_file);
            //m_hittime_low_pred->m_input_op_name   = m_hittime_low_input_op;
            //m_hittime_low_pred->m_output_op_name  = m_hittime_low_output_op;
            //m_hittime_high_pred = new TFPrediction(hittime_high_bp_file, m_hittime_high_input_op, m_hittime_high_output_op);
            //m_hittime_high_pred = new TFPrediction(hittime_high_bp_file);
            //m_hittime_high_pred->m_input_op_name   = m_hittime_high_input_op;
            //m_hittime_high_pred->m_output_op_name  = m_hittime_high_output_op;
    }
    if(m_use_npe_mean==true){
        const char* preDir = gDirectory->GetPath();
        TFile* f_npe_mean = TFile::Open(npe_mean_file.c_str());
        gDirectory->cd(preDir);
        //m_gr2d_npe_mean = dynamic_cast<TGraph2D*>(f_npe_mean->Get("r_theta_meanNpe"));
        m_gr2d_npe_mean = dynamic_cast<TH2D*>(f_npe_mean->Get("r_theta_meanNpe"));
        LogInfo << "Load NPE Mean File for Voxel Method: "
                << npe_mean_file << std::endl;
        assert(m_gr2d_npe_mean);
        gr_x_min = m_gr2d_npe_mean->GetXaxis()->GetXmin();
        gr_x_max = m_gr2d_npe_mean->GetXaxis()->GetXmax();
        gr_y_min = m_gr2d_npe_mean->GetYaxis()->GetXmin();
        gr_y_max = m_gr2d_npe_mean->GetYaxis()->GetXmax();
        n_bins_x = m_gr2d_npe_mean->GetXaxis()->GetNbins();
        n_bins_y = m_gr2d_npe_mean->GetYaxis()->GetNbins();
        LogInfo << "Load NPE Mean for Voxel Method: gr_x_min= "
                << gr_x_min << ",gr_x_max=" <<gr_x_max<< ",gr_y_min=" <<gr_y_min<<",gr_y_max="<<gr_y_max<< std::endl;
    }
    else if(m_use_npe_dl==true)
    {
            LogInfo << "Load NPE bp File for Voxel Method from DL: "
                    << npe_low_bp_file <<"," << npe_high_bp_file << std::endl; 
            //m_npe_low_pred = new TFPrediction(npe_low_bp_file, m_npe_low_input_op, m_npe_low_output_op);
            //m_npe_low_pred = new TFPrediction(npe_low_bp_file);
            //m_npe_low_pred->m_input_op_name   = m_npe_low_input_op;
            //m_npe_low_pred->m_output_op_name  = m_npe_low_output_op;
            //m_npe_high_pred = new TFPrediction(npe_high_bp_file, m_npe_high_input_op, m_npe_high_output_op);
            //m_npe_high_pred = new TFPrediction(npe_high_bp_file);
            //m_npe_high_pred->m_input_op_name   = m_npe_high_input_op;
            //m_npe_high_pred->m_output_op_name  = m_npe_high_output_op;

    }
    else{

        if (!m_helper_npe) {
            LogInfo << "Load NPE File for Voxel Method: "
                    << npe_loader_file << std::endl;
            m_helper_npe = new VoxelMethodHelper::NPELoader();
            m_helper_npe->m_filename_npe = npe_loader_file;
            m_helper_npe->m_filename_npe_single = npe_loader_file;
            m_helper_npe->m_lazy_loading=m_lazy_loading;
            m_helper_npe->load();
        }
    }


    if (m_fill_tuple) {
        SniperPtr<RootWriter> svc(*getParent(), "RootWriter");
        if (svc.invalid()) {
            LogError << "Can't Locate RootWriter. If you want to use it, please "
                     << "enalbe it in your job option file."
                     << std::endl;
            return;
        }
        gROOT->ProcessLine("#include <vector>");

        m_evt_tree = svc->bookTree("SIMEVT/voxelevt", "evt");
        m_evt_tree->Branch("evtID", &m_evtid, "evtID/I");
        m_evt_tree->Branch("pdgid", &m_pdgid);
        m_evt_tree->Branch("edep", &m_edep);
        m_evt_tree->Branch("Qedep", &m_Qedep);
        m_evt_tree->Branch("steplength", &m_steplength);

    }

    if (!m_pmthitmerger) {
        // get the merger from other tool
        SniperPtr<DetSimAlg> detsimalg(*getParent(), "DetSimAlg");
        if (detsimalg.invalid()) {
            std::cout << "Can't Load DetSimAlg" << std::endl;
            assert(0);
        }

        ToolBase* t = 0;
        // find the tool first
        // create the tool if not exist
        std::string name = "PMTSDMgr";
        t = detsimalg->findTool(name);
        if (not t) {
            LogError << "Can't find tool " << name << std::endl;
            throw SniperException("Make sure you have load the PMTSDMgr.");
        }
        PMTSDMgr* pmtsd = dynamic_cast<PMTSDMgr*>(t);
        if (not pmtsd) {
            LogError << "Can't cast to " << name << std::endl;
            throw SniperException("Make sure PMTSDMgr is OK.");
        }
        m_pmthitmerger = pmtsd->getPMTMerger();
        // check the configure
        if (m_merge_flag and (not m_pmthitmerger->getMergeFlag())) {
            LogInfo << "change PMTHitMerger flag to: " << m_merge_flag << std::endl;
            m_pmthitmerger->setMergeFlag(m_merge_flag);
        }
        if (m_pmthitmerger->getMergeFlag() and (m_time_window!=m_pmthitmerger->getTimeWindow())) {
            LogInfo << "change PMTHitMerger merge time window to: " << m_time_window << std::endl;
            m_pmthitmerger->setTimeWindow(m_time_window);
        }
    }

    std::random_device rd;
    generator = std::mt19937(rd());
    
}

void
MuonFastSimVoxel::EndOfRunAction(const G4Run* /*aRun*/) {

}

void
MuonFastSimVoxel::BeginOfEventAction(const G4Event* evt) {
    // get the hit collection
    // NOTE: because in PMT SD v2, every event will create a new hit collection,
    //       so we need to reset the pointer in end of event action.


    // clear ntuple
    if (m_fill_tuple and m_evt_tree) {
        m_evtid = evt->GetEventID();
        m_pdgid.clear();
        m_edep.clear();
        m_Qedep.clear();
        m_steplength.clear();
    }
}

void
MuonFastSimVoxel::EndOfEventAction(const G4Event* /*evt*/) {
    if (m_fill_tuple and m_evt_tree) {
        m_evt_tree->Fill();
    }
}

void
MuonFastSimVoxel::UserSteppingAction(const G4Step* step) {
    G4Track* track = step->GetTrack();
    G4double edep = step->GetTotalEnergyDeposit();
    G4double steplength = step->GetStepLength();
    bool needToSim = false;
    if (edep > 0 and track->GetDefinition()->GetParticleName()!= "opticalphoton"
                 and track->GetMaterial()->GetName() == "LS") {
        needToSim = true;
    }
    if (not needToSim) {
        return;
    }
    // fill the ntuple
    if (m_fill_tuple and m_evt_tree) {
        m_pdgid.push_back(track->GetDefinition()->GetPDGEncoding());
        m_edep.push_back(edep);
        m_steplength.push_back(steplength);
    }
    // TODO MAGIC HERE
    // Need to apply the non-linearity correction
    if (m_quenching) {
        edep = quenching(step);
    }
    if (m_fill_tuple and m_evt_tree) {
        m_Qedep.push_back(edep);
    }
    // scale the Qedep back to edep.
    // In the input distribution, we assume 1MeV (edep) gamma or electron -> nPE
    // However, in the simulation, the actual Qedep is 0.93 (or 0.97) MeV.
    // edep /= m_quenching_scale; // For 1MeV gamma, the Qedep is 0.93MeV
                               // For 1MeV electron, the Qedep is 0.97MeV
    //wxfang scale 
    edep = edep*1.015; // photon to gamma or electron 
    int intPart = static_cast<int>(edep);
    double fractionPart = edep - intPart;
    // TODO: the position can be (pre+post)/2
    G4ThreeVector pos = step -> GetPreStepPoint() -> GetPosition();
    double start_time = step -> GetPreStepPoint() -> GetGlobalTime();

    G4ThreeVector pos_v1 = (step -> GetPreStepPoint() -> GetPosition() + step -> GetPostStepPoint() -> GetPosition())/2;

    // r3 and cos(theta)
    // TAxis *xaxis = m_helper_hittime->prof_mean->GetXaxis();
    // TAxis *yaxis = m_helper_hittime->prof_mean->GetYaxis();

    TVector3 pos_src(pos.x(), pos.y(), pos.z());
    TVector3 pos_src_v1(pos_v1.x(), pos_v1.y(), pos_v1.z());
    Double_t r = pos_src.Mag()/1e3; // mm -> m
    Double_t r_v1 = pos_src_v1.Mag()/1e3; // mm -> m
    std::vector<float> theta_vec ;
    std::vector<int> id_vec ;
    for (int i = 0; i < m_PMTs; ++i) {  // new Geom
        const TVector3& pos_pmt = m_helper_geom->PMT_pos[i];
        float theta_v1 = pos_pmt.Angle(pos_src_v1);
        theta_vec.push_back(theta_v1);
        id_vec.push_back(i);
    }
    //std::cout<<"theta_vec_intPart="<<theta_vec_intPart.size()<<",theta_vec_fracPart="<<theta_vec_fracPart.size()<<std::endl;
    
    if(m_use_npe_dl==true && m_use_hittime_dl==true ){
        
        for (int j = 0; j < intPart; ++j) {
            generate_hits_v3(r_v1, theta_vec, id_vec, 1, start_time);
        }
        // fraction part
            generate_hits_v3(r_v1, theta_vec, id_vec, fractionPart, start_time);
    }
    
}

bool
MuonFastSimVoxel::generate_hits(float r, float theta, double ratio, int pmtid, double start_time) 
{
    // if don't sample npe, just return
    if (!m_flag_npe) {
        return true;
    }

    Int_t npe_from_single = m_helper_npe->get_npe(r, theta);
    if (npe_from_single>0) {
        for (int hitj = 0; hitj < npe_from_single; ++hitj) {
            // skip the photon according to the energy deposit
            if (ratio<1 and gRandom->Uniform()>ratio) { 
                continue; 
            }
            Double_t hittime = start_time;
            if (m_flag_time) {
                hittime += m_helper_hittime->get_hittime(r, theta, 0);
            }
            // generated hit
            if (m_flag_savehits) {
                save_hits(pmtid, hittime);
            }
        }
    }
    return true;
}

bool MuonFastSimVoxel::generate_hits_v1(float r, float theta, double depe, int pmtid, double start_time) 
{
    // if don't sample npe, just return
    if (!m_flag_npe) {
        return true;
    }

    float lambda = depe*get_lambda(r, theta, false);
    //float lambda = depe*m_gr2d_npe_mean->GetBinContent(m_gr2d_npe_mean->FindFixBin(r,theta1));
    //std::cout<<"lambda="<<lambda<<",pmtid="<<pmtid<<std::endl;
    std::poisson_distribution<> d(lambda);
    Int_t npe_from_single = d(generator);
    //std::cout<<"npe_from_single="<<npe_from_single<<std::endl;

    if (npe_from_single>0) {
        for (int hitj = 0; hitj < npe_from_single; ++hitj) {
            // skip the photon according to the energy deposit
            //if (ratio<1 and gRandom->Uniform()>ratio) { 
            //    continue; 
            //}
            Double_t hittime = start_time;
            if (m_flag_time) {
                if(m_use_hittime_dl==false) hittime += m_helper_hittime->get_hittime(r, theta, 0);
                else                        {//Double_t hittime_1 = r<15 ? m_hittime_low_pred->predict(r, theta, 100) : m_hittime_high_pred->predict(r, theta, 100) ;
                                             hittime = hittime + 1 ;
                                            }
            }
            // generated hit
            if (m_flag_savehits) {
                save_hits(pmtid, hittime);
            }
        }
    }
    return true;
}


bool
MuonFastSimVoxel::generate_hits_v3(float r , const std::vector<float>& vec_theta, const std::vector<int>& vec_id, float frac, double start_time)
{
    // if don't sample npe, just return
    if (!m_flag_npe) {
        return true;
    }
    //for(unsigned i=0; i< vec_theta.size(); i++) std::cout<<"r="<<r<<",id="<<vec_id.at(i)<<",theta="<<vec_theta.at(i)<<std::endl;
    
    if(vec_theta.size()==0) return true;
    std::vector<float>* npe_from_pred  = r < 15 ? m_NNpred_npe_low->get(1, r, vec_theta, 1) : m_NNpred_npe_high->get(2, r, vec_theta, 1) ;
    int npe_size = npe_from_pred->size();
    //for(unsigned i=0; i< npe_from_pred->size(); i++) std::cout<<"id="<<vec_id.at(i)<<",pred npe="<<npe_from_pred->at(i)<<std::endl;
    assert(vec_id.size() == npe_size);
    std::vector<int> pmt_ids;
    std::vector<float> pmt_thetas;
    for(unsigned i=0; i<npe_size; i++)
    {
        int tmp_npe = round(npe_from_pred->at(i));
        int tmp_id  = vec_id.at(i);
        float tmp_theta  = vec_theta.at(i);
        for(int j=0; j<tmp_npe; j++)
        {
            if( gRandom->Uniform() > frac ) continue;
            pmt_ids.push_back(tmp_id);
            pmt_thetas.push_back(tmp_theta);
        }
    }
    //for(unsigned i=0; i<pmt_ids.size(); i++) std::cout<<"id="<<pmt_ids.at(i)<<",pmt_thetas="<<pmt_thetas.at(i)<<std::endl;
    if(pmt_thetas.size()==0) {delete npe_from_pred; return true;}
    std::vector<float>* hittimes = r<15 ? m_NNpred_time_low->get(3, r, pmt_thetas, 100) : m_NNpred_time_high->get(4, r, pmt_thetas, 100) ;
    //for(unsigned i=0; i< hittimes->size(); i++) std::cout<<"id="<<pmt_ids.at(i)<<",pred time="<<hittimes->at(i)<<std::endl;
    //std::cout<<"pred time size="<<hittimes->size()<<std::endl;
    if (m_flag_savehits) save_hits_v(pmt_ids, *hittimes, start_time);
    delete hittimes; 
    delete npe_from_pred; 
    return true;
    
}

bool MuonFastSimVoxel::save_hits_v(const std::vector<int>& pmtids, const std::vector<float>& hittimes, float s_time) {
    int N_hittimes = hittimes.size();
    for(unsigned i=0; i<N_hittimes; i++)
    {
        int pmtid = pmtids.at(i); 
        float hittime = hittimes.at(i) + s_time;
        if (m_pmthitmerger->getMergeFlag()) {
            // == if merged, just return true. That means just update the hit
            // NOTE: only the time and count will be update here, the others 
            //       will not filled.
            bool ok = m_pmthitmerger->doMerge(pmtid, hittime);
            if (ok) {
                return true;
            }
        }
        if (m_pmthitmerger->hasNormalHitType()) {
            dywHit_PMT* hit_photon = new dywHit_PMT();
            hit_photon->SetPMTID(pmtid);
            hit_photon->SetTime(hittime);
            hit_photon->SetCount(1); // FIXME
            // == insert
            m_pmthitmerger->saveHit(hit_photon);
        } else if (m_pmthitmerger->hasMuonHitType()) {
            dywHit_PMT_muon* hit_photon = new dywHit_PMT_muon();
            hit_photon->SetPMTID(pmtid);
            hit_photon->SetTime(hittime);
            hit_photon->SetCount(1); // FIXME
            // == insert
            m_pmthitmerger->saveHit(hit_photon);
        }

    }
    return true;
}



bool
MuonFastSimVoxel::save_hits(int pmtid, double hittime) {

    if (m_pmthitmerger->getMergeFlag()) {
        // == if merged, just return true. That means just update the hit
        // NOTE: only the time and count will be update here, the others 
        //       will not filled.
        bool ok = m_pmthitmerger->doMerge(pmtid, hittime);
        if (ok) {
            return true;
        }
    }
    if (m_pmthitmerger->hasNormalHitType()) {
        dywHit_PMT* hit_photon = new dywHit_PMT();
        hit_photon->SetPMTID(pmtid);
        hit_photon->SetTime(hittime);
        hit_photon->SetCount(1); // FIXME
        // == insert
        m_pmthitmerger->saveHit(hit_photon);
    } else if (m_pmthitmerger->hasMuonHitType()) {
        dywHit_PMT_muon* hit_photon = new dywHit_PMT_muon();
        hit_photon->SetPMTID(pmtid);
        hit_photon->SetTime(hittime);
        hit_photon->SetCount(1); // FIXME
        // == insert
        m_pmthitmerger->saveHit(hit_photon);
    }
    return true;
}

double 
MuonFastSimVoxel::quenching(const G4Step* step) {
    double QuenchedTotalEnergyDeposit = 0.0;

    G4Track* track = step->GetTrack();
    const G4DynamicParticle* aParticle = track->GetDynamicParticle();
    const G4Material* material = track->GetMaterial();

    G4double dE = step->GetTotalEnergyDeposit();
    G4double dx = step->GetStepLength();
    G4double dE_dx = dE/dx;

    if(track->GetDefinition() == G4Gamma::Gamma() && dE > 0)
    { 
      G4LossTableManager* manager = G4LossTableManager::Instance();
      dE_dx = dE/manager->GetRange(G4Electron::Electron(), dE, track->GetMaterialCutsCouple());
      //G4cout<<"gamma dE_dx = "<<dE_dx/(MeV/mm)<<"MeV/mm"<<G4endl;
    }

    G4double delta = dE_dx/material->GetDensity();//get scintillator density 
    //G4double birk1 = 0.0125*g/cm2/MeV;
    G4double birk1 = m_birksConst1;
    if(abs(aParticle->GetCharge())>1.5)//for particle charge greater than 1.
        birk1 = 0.57*birk1;
    
    G4double birk2 = 0;
    //birk2 = (0.0031*g/MeV/cm2)*(0.0031*g/MeV/cm2);
    birk2 = m_birksConst2;
    
    QuenchedTotalEnergyDeposit 
        = dE/(1+birk1*delta+birk2*delta*delta);


    return QuenchedTotalEnergyDeposit;
}


float MuonFastSimVoxel::get_lambda(float r, float theta, bool doInterP)
{

    float theta1 = theta*180/3.1415926535898 ;
    if(r<gr_x_min) r=gr_x_min+1e-5;
    else if(r>gr_x_max) r=gr_x_max-1e-5;
    if(theta1<gr_y_min) theta1=gr_y_min+1e-4;
    else if(theta1>gr_y_max) theta1=gr_y_max-1e-4;
    /////////// do interpolate //////////
    int bin_x =  m_gr2d_npe_mean->GetXaxis()->FindBin(r);
    int bin_y =  m_gr2d_npe_mean->GetYaxis()->FindBin(theta1);
    float lambda = m_gr2d_npe_mean->GetBinContent(bin_x, bin_y); 
    if(doInterP==false) return lambda;

    float bin_x_cen = m_gr2d_npe_mean->GetXaxis()->GetBinCenter(bin_x); 
    int a_bin_x = 0;
    if(r>bin_x_cen && bin_x<n_bins_x) 
    {
        a_bin_x = bin_x + 1;
    }
    else if(r<bin_x_cen && bin_x>0) 
    {
        a_bin_x = bin_x - 1;
    }
    if (a_bin_x != 0)
    {
        float a_bin_x_cen = m_gr2d_npe_mean->GetXaxis()->GetBinCenter(a_bin_x);
        float tmp = m_gr2d_npe_mean->GetBinContent(a_bin_x, bin_y); 
        float slope = (tmp-lambda)/(a_bin_x_cen-bin_x_cen);
        lambda = lambda + slope*(r-bin_x_cen);
    }

    float bin_y_cen = m_gr2d_npe_mean->GetYaxis()->GetBinCenter(bin_y); 
    int a_bin_y = 0;
    if(theta1>bin_y_cen && bin_y<n_bins_y) 
    {
        a_bin_y = bin_y + 1;
    }
    else if(theta1<bin_y_cen && bin_y>0) 
    {
        a_bin_y = bin_y - 1;
    }
    if (a_bin_y != 0)
    {
        float a_bin_y_cen = m_gr2d_npe_mean->GetYaxis()->GetBinCenter(a_bin_y);
        float tmp = m_gr2d_npe_mean->GetBinContent(bin_x, a_bin_y); 
        float slope = (tmp-lambda)/(a_bin_y_cen-bin_y_cen);
        lambda = lambda + slope*(theta1-bin_y_cen);
    }
    return lambda;
}
/*
double gaussrand()
{
    static double V1, V2, S;
    static int phase = 0;
    double X;
     
    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
             
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
         
        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
         
    phase = 1 - phase;
 
    return X;
}
*/
