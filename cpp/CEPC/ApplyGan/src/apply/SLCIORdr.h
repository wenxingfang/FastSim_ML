#ifndef SLCIORdr_h
#define SLCIORdr_h 1


#include "lcio.h"
#include "LCIOSTLTypes.h"
#include "IOIMPL/LCFactory.h"
#include "EVENT/LCIO.h"
#include "EVENT/LCEvent.h"
#include "EVENT/LCCollection.h"
#include "EVENT/SimCalorimeterHit.h"
#include "IO/LCReader.h"
#include "IO/LCWriter.h"
#include "IMPL/CalorimeterHitImpl.h"
#include "IMPL/LCCollectionVec.h"
#include "UTIL/CellIDDecoder.h"


//#include "TROOT.h"
//#include "TTree.h"
//#include "TFile.h"
#include <map>
#include "tensorflow/c/c_api.h"

using namespace lcio;
using namespace std;

class SLCIORdr {

    public:
        SLCIORdr(string name, string output, string mapTxt);
        ~SLCIORdr();
        bool configure();               
        bool mutate();    
        bool getMCinfo(LCCollection* lcCol, LCCollection* HitCol, CellIDDecoder<SimCalorimeterHit> & Hit_idDecoder);
        bool getHitInfo(LCCollection* Col, std::vector<int>& vec_ID0, std::vector<int>& vec_ID1, std::vector<double>& vec_Hit_x, std::vector<double>& vec_Hit_y, std::vector<double>& vec_Hit_z, std::vector<double>& vec_Hit_E , double& Hit_cm_x, double& Hit_cm_y, double& Hit_cm_z, double& Hit_tot_e);
        bool getDigiHitInfo(CellIDDecoder<CalorimeterHit> & Hit_idDecoder, LCCollection* Col, std::vector<int>& vec_ID0, std::vector<int>& vec_ID1, std::vector<double>& vec_Hit_x, std::vector<double>& vec_Hit_y, std::vector<double>& vec_Hit_z, std::vector<double>& vec_Hit_E , double& Hit_cm_x, double& Hit_cm_y, double& Hit_cm_z, double& Hit_tot_e, std::vector<int>& vec_ID_S, std::vector<int>& vec_ID_M, std::vector<int>& vec_ID_I, std::vector<int>& vec_ID_J, std::vector<int>& vec_ID_K);
        bool getHits(LCCollection* Col, CellIDDecoder<SimCalorimeterHit>& idDecoder, float x, float y, float z, std::vector<int>& Hits_ID, std::vector<float>& Hits_x, std::vector<float>& Hits_y, std::vector<float>& Hits_z, float& cell_x, float& cell_y, float& cell_z);
        void getID_x_y_z(const int& S, const int& M, const int& I, const int& J, const int& K, int& id, float& x, float& y, float& z);
        int  predict(TF_Session* session, TF_Status* status, TF_Graph* graph, const vector<float>& mc_vector, const string& input_op_name, const string& output_op_name, vector<float>& hit_vec) ;
        void MakeMap(const vector<float>& hit_e, const vector<int>& hit_id, const vector<float>& hit_x, const vector<float>& hit_y, const vector<float>& hit_z);
        bool finish();
        bool isEnd();
        bool clear();
    private:
        IO::LCReader* m_slcio_rdr;
        LCWriter* _lcWrt ;
        std::string sim_digi;
        long m_total_event;
        long m_processed_event;
        int m_noFound;
        int m_toBeSim;
        int m_toFar;
        double m_HitCm_x   ;
        double m_HitCm_y   ;
        double m_HitCm_z   ;
        double m_HitEn_tot ;
        string m_input_op_name;
        string m_output_op_name;

        std::vector<double> m_HitFirst_x ;
        std::vector<double> m_HitFirst_y ;
        std::vector<double> m_HitFirst_z ;
        std::vector<double> m_HitFirst_vtheta ;
        std::vector<double> m_HitFirst_vphi   ;
        std::vector<double> m_phi_rotated     ;

        std::vector<double> m_mc_pHitx        ;
        std::vector<double> m_mc_pHity        ;
        std::vector<double> m_mc_pHitz        ;
        std::vector<double> m_mc_pHit_theta   ;
        std::vector<double> m_mc_pHit_phi     ;
        std::vector<double> m_mc_pHit_rotated ;
        std::vector<double> m_mc_pHit_dz ;
        std::vector<double> m_mc_pHit_dy ;


        std::vector<double> m_mc_vertexX;
        std::vector<double> m_mc_vertexY;
        std::vector<double> m_mc_vertexZ;
        std::vector<double> m_mc_Px;
        std::vector<double> m_mc_Py;
        std::vector<double> m_mc_Pz;
        std::vector<double> m_mc_M;
        std::vector<double> m_mc_Charge; 
        std::vector<double> m_Hit_x;
        std::vector<double> m_Hit_y;
        std::vector<double> m_Hit_z;
        std::vector<double> m_Hit_E; 
        std::vector< std::vector<double> > m_Hits; 
        std::vector<int>   m_mc_Pdg, m_mc_genStatus, m_mc_simStatus, m_mc_np, m_mc_nd;
        std::vector<int>  m_Hit_ID0,m_Hit_ID1, m_ID_S, m_ID_M, m_ID_I, m_ID_J, m_ID_K;
//        TFile* file_out;
//        TTree* tree_out;
        int m_pid;
        bool m_is_gun;
        map<int, map<int, map<int, map<int, map<int, string> > > > >  SMIJK_map_ID_x_y_z;
        map<int, vector<float> > m_Hit_map ;

        TF_Buffer*                m_em_graph_def ;
        TF_Graph*                 m_em_graph ;
        TF_Status*                m_em_status;
        TF_ImportGraphDefOptions* m_em_graph_opts ;
        TF_SessionOptions*        m_em_sess_opts ;
        TF_Session*               m_em_session   ;

        TF_Buffer*                m_gamma_graph_def ;
        TF_Graph*                 m_gamma_graph ;
        TF_Status*                m_gamma_status;
        TF_ImportGraphDefOptions* m_gamma_graph_opts ;
        TF_SessionOptions*        m_gamma_sess_opts ;
        TF_Session*               m_gamma_session   ;

};

#endif

