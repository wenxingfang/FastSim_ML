#include "SLCIORdr.h"

#include "lcio.h"  //LCIO
#include "LCIOSTLTypes.h"
#include "IOIMPL/LCFactory.h"
#include "EVENT/LCIO.h"
#include "EVENT/LCEvent.h"
#include "EVENT/LCCollection.h"
#include "EVENT/SimCalorimeterHit.h"
#include "IO/LCReader.h"
#include "IMPL/MCParticleImpl.h"
#include "IMPL/LCCollectionVec.h"
#include "IMPL/CalorimeterHitImpl.h"
#include <IMPL/LCFlagImpl.h>
#include "UTIL/CellIDDecoder.h"

//#include "TROOT.h"
//#include "TTree.h"
//#include "TFile.h"

#include <map>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>
#include<cstring>
#include <typeinfo>   // operator typei


#include "my_utils.h"
#include "read_SMIJK_ID_x_y_z.cpp"
#include "tensorflow/c/c_api.h"


#ifndef PI
#define PI acos(-1)
#endif
#ifndef DEBUG
#define DEBUG true
#endif
using namespace lcio;
using namespace IMPL;
using namespace std;

template <class Type>
Type stringToNum(const string& str)
{
        istringstream iss(str);
        Type num;
        iss >> num;
        return num;
}

double gaussrand();
TF_Buffer* read_file(const char* file);

void free_buffer(void* data, size_t length) {
        free(data);
}

static void Deallocator(void* data, size_t length, void* arg) {
        free(data);
}

static void DeallocateBuffer(void* data, size_t) {
    std::free(data);
}


SLCIORdr::SLCIORdr(string name, string output, string mapTxt, string ref_name){


    printf("Hello from TensorFlow C library version %s\n", TF_Version());
    // Use read_file to get graph_def as TF_Buffer*
    string em_pb_name = "/junofs/users/wxfang/CEPC/CEPCOFF/ApplyGan/src/apply/model_em.pb";
    //TF_Buffer* graph_def = read_file("/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/model_em_low.pb");
    m_em_graph_def = read_file(em_pb_name.c_str());
    m_em_graph     = TF_NewGraph();
    // Import graph_def into graph
    m_em_status = TF_NewStatus();
    m_em_graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(m_em_graph, m_em_graph_def, m_em_graph_opts, m_em_status);

    string gamma_pb_name="/junofs/users/wxfang/CEPC/CEPCOFF/ApplyGan/src/apply/model_gamma.pb";
    m_gamma_graph_def  = read_file(gamma_pb_name.c_str());
    m_gamma_graph      = TF_NewGraph();
    m_gamma_status     = TF_NewStatus();
    m_gamma_graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(m_gamma_graph, m_gamma_graph_def, m_gamma_graph_opts, m_gamma_status);


    if (TF_GetCode(m_em_status) != TF_OK || TF_GetCode(m_gamma_status) != TF_OK ) 
    {
        fprintf(stderr, "ERROR: Unable to import graph");
        //fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(status));
    }
    else 
    {
        fprintf(stdout, "Successfully imported graph\n");
    }


    m_em_sess_opts = TF_NewSessionOptions();
    m_em_session   = TF_NewSession(m_em_graph, m_em_sess_opts, m_em_status);
    assert(TF_GetCode(m_em_status) == TF_OK);

    m_gamma_sess_opts = TF_NewSessionOptions();
    m_gamma_session   = TF_NewSession(m_gamma_graph, m_gamma_sess_opts, m_gamma_status);
    assert(TF_GetCode(m_gamma_status) == TF_OK);


_lcWrt = LCFactory::getInstance()->createLCWriter() ;
_lcWrt->open( output , LCIO::WRITE_NEW ) ;



read_SMIJK_ID_x_y_z(mapTxt, SMIJK_map_ID_x_y_z);


m_slcio_rdr = IOIMPL::LCFactory::getInstance()->createLCReader();
m_slcio_rdr->open(name.c_str());

m_slcio_rdr_ref = IOIMPL::LCFactory::getInstance()->createLCReader();
m_slcio_rdr_ref->open(ref_name.c_str());


m_processed_event=0;
m_noFound = 0;
m_toBeSim = 0;
m_toFar = 0;
m_input_op_name = "gen_input"; 
m_output_op_name = "cropping3d_1/strided_slice"; 
//m_is_gun = is_gun;
//m_pid    = pid   ;
/*
file_out = new TFile(output.c_str(),"RECREATE"); 
tree_out = new TTree("evt","test tree");
m_HitCm_x   = 0;
m_HitCm_y   = 0;
m_HitCm_z   = 0;
m_HitEn_tot = 0;
tree_out->Branch("evt_id",&m_processed_event,"evt_id/I");
tree_out->Branch("m_HitCm_x"  , &m_HitCm_x   ,"m_HitCm_x/D"     );
tree_out->Branch("m_HitCm_y"  , &m_HitCm_y   ,"m_HitCm_y/D"     );
tree_out->Branch("m_HitCm_z"  , &m_HitCm_z   ,"m_HitCm_z/D"     );
tree_out->Branch("m_HitEn_tot", &m_HitEn_tot ,"m_HitEn_tot/D"   );
tree_out->Branch("m_HitFirst_x"     , &m_HitFirst_x      );
tree_out->Branch("m_HitFirst_y"     , &m_HitFirst_y      );
tree_out->Branch("m_HitFirst_z"     , &m_HitFirst_z      );
tree_out->Branch("m_phi_rotated"    , &m_phi_rotated     );
tree_out->Branch("m_HitFirst_vtheta", &m_HitFirst_vtheta );
tree_out->Branch("m_HitFirst_vphi"  , &m_HitFirst_vphi   );
tree_out->Branch("m_mc_vertexX", &m_mc_vertexX  );
tree_out->Branch("m_mc_vertexY", &m_mc_vertexY  );
tree_out->Branch("m_mc_vertexZ", &m_mc_vertexZ  );
tree_out->Branch("m_mc_Px", &m_mc_Px       );
tree_out->Branch("m_mc_Py", &m_mc_Py       );
tree_out->Branch("m_mc_Pz", &m_mc_Pz       );
tree_out->Branch("m_mc_M", &m_mc_M        );
tree_out->Branch("m_Hit_x", &m_Hit_x       );
tree_out->Branch("m_Hit_y", &m_Hit_y       );
tree_out->Branch("m_Hit_z", &m_Hit_z       );
tree_out->Branch("m_Hit_E", &m_Hit_E       ); 
tree_out->Branch("m_mc_Pdg", &m_mc_Pdg      );
tree_out->Branch("m_mc_Charge", &m_mc_Charge      );
tree_out->Branch("m_mc_genStatus", &m_mc_genStatus); 
tree_out->Branch("m_mc_simStatus", &m_mc_simStatus); 
tree_out->Branch("m_mc_np", &m_mc_np       ); 
tree_out->Branch("m_mc_nd", &m_mc_nd       ); 
tree_out->Branch("m_Hit_ID0", &m_Hit_ID0      );
tree_out->Branch("m_Hit_ID1", &m_Hit_ID1      );
tree_out->Branch("m_ID_S", &m_ID_S      );
tree_out->Branch("m_ID_M", &m_ID_M      );
tree_out->Branch("m_ID_I", &m_ID_I      );
tree_out->Branch("m_ID_J", &m_ID_J      );
tree_out->Branch("m_ID_K", &m_ID_K      );
tree_out->Branch("m_mc_pHitx"       , &m_mc_pHitx       );
tree_out->Branch("m_mc_pHity"       , &m_mc_pHity       );
tree_out->Branch("m_mc_pHitz"       , &m_mc_pHitz       );
tree_out->Branch("m_mc_pHit_theta"  , &m_mc_pHit_theta  );
tree_out->Branch("m_mc_pHit_phi"    , &m_mc_pHit_phi    );
tree_out->Branch("m_mc_pHit_rotated", &m_mc_pHit_rotated);
tree_out->Branch("m_Hits", &m_Hits      ); 
tree_out->Branch("m_mc_pHit_dz", &m_mc_pHit_dz      ); 
tree_out->Branch("m_mc_pHit_dy", &m_mc_pHit_dy      ); 
std::cout<<"pi="<< PI << std::endl;
*/

}

SLCIORdr::~SLCIORdr(){
delete m_slcio_rdr;
delete m_slcio_rdr_ref;
}

bool SLCIORdr::mutate(){

    clear();
    EVENT::LCEvent *lcEvent     = m_slcio_rdr->readNextEvent(LCIO::UPDATE);
    EVENT::LCEvent *lcEvent_ref = m_slcio_rdr_ref->readNextEvent(LCIO::UPDATE);
    LCCollection *Col = NULL;
    LCCollection *ColVec = NULL;

    if(lcEvent && lcEvent_ref) {  //cout<<"det name="<< lcEvent->getDetectorName() <<endl;
                   const std::vector<std::string>* colVec = lcEvent->getCollectionNames();
                   const std::vector<std::string>* refcolVec = lcEvent_ref->getCollectionNames();
                   /*
                   for(unsigned int i=0; i< refcolVec->size(); i++){
                    std::cout<<refcolVec->at(i)<<std::endl;
                   }
                   */
                   for(unsigned int i=0; i< colVec->size(); i++){
                       if (colVec->at(i)=="MCParticle")
                       {
                           
                           //Col = lcEvent->getCollection("MCParticle");
                           //CellIDDecoder<SimCalorimeterHit> Hit_idDecoder(HitCol);
                           //string ecalinitString = HitCol->getParameters().getStringVal(LCIO::CellIDEncoding);
                           //m_Hit_map.clear();
                           //getMCinfo(Col,  HitCol, Hit_idDecoder);
                           
                           LCCollectionVec* calHits = new LCCollectionVec( LCIO::CALORIMETERHIT )  ;
                           LCCollection* refCol  = lcEvent_ref->getCollection("ECALBarrel");
                           //LCCollectionVec *ecalcol = new LCCollectionVec(LCIO::CALORIMETERHIT);
                           
                           LCCollection* refSimCol  = lcEvent_ref->getCollection("EcalBarrelSiliconCollection");
                           for( int i=0; i<refSimCol->getNumberOfElements(); i++ ){
                               EVENT::SimCalorimeterHit* in = dynamic_cast<EVENT::SimCalorimeterHit*>(refSimCol->getElementAt(i));
                               std::cout<<"sim hit ID="<<in->getCellID0()<<",x="<<in->getPosition()[0]<<",y="<<in->getPosition()[1]<<",z="<<in->getPosition()[2]<<std::endl;
                           }

                           LCFlagImpl flag;
                           flag.setBit(LCIO::CHBIT_LONG);                  //To set position & ID1
                           flag.setBit(LCIO::CHBIT_ID1);
                           flag.setBit(LCIO::RCHBIT_ENERGY_ERROR); //In order to use an additional FLOAT
                           flag.setBit(LCIO::RCHBIT_TIME);
                           //ecalcol->setFlag(flag.getFlag());
                           //string EcalinitString = refCol->getParameters().getStringVal(LCIO::CellIDEncoding);
                           string EcalinitString = "M:3,S-1:3,I:9,J:9,K-1:6";
                           //ecalcol->parameters().setValue(LCIO::CellIDEncoding, EcalinitString);
                           //CellIDDecoder<CalorimeterHit> Hit_idDecoder(refCol);
                           calHits->parameters().setValue(LCIO::CellIDEncoding, EcalinitString);
                           //calHits->setFlag(-1409286144); // need to be varified
                           calHits->setFlag(flag.getFlag()); // need to be varified
                           //std::cout<<"flag="<<refCol->getFlag()<<std::endl;
                           int NHit = refCol->getNumberOfElements();
                           for( int i=0; i<NHit; i++ ){
                               IMPL::CalorimeterHitImpl* in = dynamic_cast<IMPL::CalorimeterHitImpl*>(refCol->getElementAt(i));
                               CalorimeterHitImpl* calHit = new CalorimeterHitImpl ;
                               //calHit->setTime(in->getTime());
                               std::cout<<"digi hit ID="<<in->getCellID0()<<",x="<<in->getPosition()[0]<<",y="<<in->getPosition()[1]<<",z="<<in->getPosition()[2]<<std::endl;
                               //std::cout<<"time="<<in->getTime()<<std::endl;
                               //1.0/300*CluHitPos.Mag()
                               calHit->setEnergy( in->getEnergy() ) ;
                               calHit->setCellID0(in->getCellID0()) ;
                               calHit->setCellID1(0) ;
                               //calHit->setCellID1(in->getCellID0()) ;
	                       float pos[3] = { in->getPosition()[0] , in->getPosition()[1], in->getPosition()[2] } ;
                               float time = 0.01+(1.0/300)*sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
                               float time0 =0.0001+(1.0/300)*sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
                               //calHit->setTime(time);
                               calHit->setTime(time0);
                               /*
                               int StaveNum = Hit_idDecoder(in)["S-1"];// from 0 - 7
                               int MNum     = Hit_idDecoder(in)["M"  ];// from 1 - 5
                               int ZNum     = Hit_idDecoder(in)["J"  ];// from 0 - 89
                               int INum     = Hit_idDecoder(in)["I"  ];// from 0 - 169 for layer 0, then decrease for higher layer
                               int LayerNum = Hit_idDecoder(in)["K-1"];// from 0 - 28
                               if(LayerNum==0 && StaveNum==2 && MNum==4 ){
                                   std::cout<<"La="<<LayerNum<<",S="<<StaveNum<<",M="<<MNum<<",J="<<ZNum<<",I="<<INum<<",time="<<in->getTime()<<",tau="<<time0<<",e="<<in->getEnergy()<<std::endl;
                               }
                               else if(LayerNum==10 && StaveNum==2 && MNum==4 ){
                                   std::cout<<"La="<<LayerNum<<",S="<<StaveNum<<",M="<<MNum<<",J="<<ZNum<<",I="<<INum<<",time="<<in->getTime()<<",tau="<<time0<<",e="<<in->getEnergy()<<std::endl;
                               }
                               else if(LayerNum==20 && StaveNum==2 && MNum==4 ){
                                   std::cout<<"La="<<LayerNum<<",S="<<StaveNum<<",M="<<MNum<<",J="<<ZNum<<",I="<<INum<<",time="<<in->getTime()<<",tau="<<time0<<",e="<<in->getEnergy()<<std::endl;
                               }
                               */
                               calHit->setPosition(pos) ;
                               //calHit->setPosition(in->getPosition()) ;
                               calHits->addElement( calHit ) ;
                               //ecalcol->addElement( calHit ) ;
                           }
                           lcEvent->addCollection( calHits, "ECALBarrel") ; 
                           //lcEvent->addCollection( ecalcol, "ECALBarrel") ; 
                           _lcWrt->writeEvent(lcEvent) ;
                           break;
                       }// if mc particle
                   }
                  }
    else {cout<<"end file, total event="<<m_processed_event<<",N2beSim="<<m_toBeSim<<",NnoFound="<<m_noFound<<",NtoFar="<<m_toFar<<"\n"; return false;}
    m_processed_event ++;
    //tree_out->Fill();
    //if(m_processed_event%1000==0)cout << "Done for event "<<m_processed_event<< endl;
    if(m_processed_event%100==0)cout << "Done for event "<<m_processed_event<< endl;
    return true;
}

bool SLCIORdr::getMCinfo(LCCollection* lcCol, LCCollection* HitCol, CellIDDecoder<SimCalorimeterHit> & Hit_idDecoder){
      
    if(lcCol){
        //if(DEBUG)std::cout<<"Hi:"<<std::endl;
	int NHEP = lcCol->getNumberOfElements();
	for( int IHEP=0; IHEP<NHEP; IHEP++ ){
	  EVENT::MCParticle* in = dynamic_cast<EVENT::MCParticle*>(lcCol->getElementAt(IHEP));
          if(fabs(in->getPDG())!=11 && fabs(in->getPDG())!=22) continue;
          float mom_tmp = sqrt(in->getMomentum()[0]*in->getMomentum()[0] + in->getMomentum()[1]*in->getMomentum()[1] + in->getMomentum()[2]*in->getMomentum()[2]);
          if(mom_tmp < 1) continue;
          //if(fabs(in->getPDG())!=m_pid ) continue;
	  //EVENT::MCParticleVec parents = (dynamic_cast<EVENT::MCParticle*>(lcCol->getElementAt(IHEP)))->getParents();
	  //EVENT::MCParticleVec daughters = (dynamic_cast<EVENT::MCParticle*>(lcCol->getElementAt(IHEP)))->getDaughters();
	  //int np = parents.size();
	  //int nd = daughters.size();
          //if(m_is_gun && np!=0) continue;
          //float Distance = 1850; //mm
          float Distance = 1840; //mm, remove some back scatter particles
          float BField = 3.0; //T
	  //if (np != 0 ) continue; // only select original particle 
	  //if (in->isCreatedInSimulation() == true) continue; // only select gen particle 
	  //if (in->getGeneratorStatus() != 1) continue; // only select stable gen particle 
	  //if (fabs(in->getPDG()) != 11) continue; // only select e+- 
          bool vertex_beforeEcal   = beforeECAL(in->getVertex()[0]  , in->getVertex()[1]  , Distance);
          bool endpoint_beforeEcal = beforeECAL(in->getEndpoint()[0], in->getEndpoint()[1], Distance);
          if(vertex_beforeEcal!=true || endpoint_beforeEcal!=false) continue;
          //if(in->getCharge()==0) continue;           
          float pHitx=1850;// mm the x place for ECAL
          float pHity=0;
          float pHitz=0;
          float pHit_theta=0;
          float pHit_phi=0;
          float pHit_rotated=0;
          int getHit = getHitPoint(in->getCharge(), in->getVertex()[0], in->getVertex()[1], in->getVertex()[2], in->getMomentum()[0], in->getMomentum()[1], in->getMomentum()[2], BField, in->getEndpoint()[0], in->getEndpoint()[1], pHitx, pHity, pHitz, pHit_theta, pHit_phi, pHit_rotated);
          if(fabs(pHity) > 600) continue;//remove high Y region now
          if(fabs(pHitz) > 2300)continue;//remove endcap now
	  //cout << "DEBUG: PDG= " << in->getPDG() <<",pHitX="<<pHitx<<",pHitY="<<pHity<<",pHitZ="<<pHitz<<",rotated="<<pHit_rotated<<",EndX="<<in->getEndpoint()[0]<<",EndY="<<in->getEndpoint()[1]<<",EndZ="<<in->getEndpoint()[2]<< ", Mx= " << in->getMomentum()[0]<<", My= "<<in->getMomentum()[1]<<",Mz= " <<in->getMomentum()[2]<< "\n ";
          float real_z = pHitz ;
          float real_r = sqrt(pHitx*pHitx + pHity*pHity);
          float real_phi = getPhi(pHitx, pHity)+ pHit_rotated; 
          if(real_phi>=360) real_phi = real_phi - 360 ;
          float real_x = real_r*cos(real_phi*PI/180); 
          float real_y = real_r*sin(real_phi*PI/180); 
          std::vector<int>   Hit_id;
          std::vector<float>  Hit_x;
          std::vector<float>  Hit_y;
          std::vector<float>  Hit_z;
          float cell_x = 0 ;
          float cell_y = 0 ;
          float cell_z = 0 ;
	  //cout << "DEBUG: PDG= " << in->getPDG() <<",real_x="<<real_x<<",real_y="<<real_y<<",real_z="<<real_z<<",rotated="<<pHit_rotated<<",EndX="<<in->getEndpoint()[0]<<",EndY="<<in->getEndpoint()[1]<<",EndZ="<<in->getEndpoint()[2]<< ", Mx= " << in->getMomentum()[0]<<", My= "<<in->getMomentum()[1]<<",Mz= " <<in->getMomentum()[2]<< "\n ";
          bool access = getHits(HitCol, Hit_idDecoder, real_x, real_y, real_z, Hit_id, Hit_x, Hit_y, Hit_z, cell_x, cell_y, cell_z);
          int n_size = 31*31*29;
          m_toBeSim++;
          if(Hit_id.size()!=n_size) {m_noFound++; continue; }
           
          double tmp_dz = pHitz - cell_z;
          float cell_phi = getPhi(cell_x, cell_y) - pHit_rotated;
          double tmp_dy = pHity - sqrt(cell_x*cell_x + cell_y*cell_y)*sin(cell_phi*PI/180);
          if(sqrt(tmp_dz*tmp_dz + tmp_dy*tmp_dy)>5*sqrt(2)) {m_toFar++ ; continue;}
          if (pHit_theta < 0) pHit_theta = 180 + pHit_theta ;
          ////// do predict /////
          vector<float>  Hit_e;
          vector<float>  mc_info_input;
          float mom = sqrt( in->getMomentum()[0]*in->getMomentum()[0] + in->getMomentum()[1]*in->getMomentum()[1] + in->getMomentum()[2]*in->getMomentum()[2] );
          mc_info_input.push_back((mom-50.5)/49.5);
          mc_info_input.push_back((pHit_theta-90)/50);
          mc_info_input.push_back(pHit_phi/20);
          mc_info_input.push_back(tmp_dz/10);
          mc_info_input.push_back(tmp_dy/10);
          mc_info_input.push_back(pHitz/2000);
          if(fabs(in->getPDG())==11 )
          {
              int result = predict( m_em_session, m_em_status, m_em_graph, mc_info_input, m_input_op_name, m_output_op_name, Hit_e) ;
          }
          else if(fabs(in->getPDG())==22 )
          {
              int result = predict( m_gamma_session, m_gamma_status, m_gamma_graph, mc_info_input, m_input_op_name, m_output_op_name, Hit_e) ;
          }
          //cout<<"pHit_rotated="<<pHit_rotated<<",hit_x="<<pHitx<<",hit_y="<<pHity<<",hit_z="<<pHitz<<",real_x="<<real_x<<",real_y="<<real_y<<",real_z="<<real_z<<"cell_x="<<cell_x<<",cell_y="<<cell_y<<",cell_z="<<cell_z<<endl;
          //cout<<"tmp_dz="<<tmp_dz<<",tmp_dy="<<tmp_dy<<endl;
          if(Hit_e.size()!=n_size) continue;
          MakeMap(Hit_e, Hit_id, Hit_x, Hit_y, Hit_z);
          
	}
      }
      else {
          return false;
      }
      return true;
}

void SLCIORdr::MakeMap(const vector<float>& hit_e, const vector<int>& hit_id, const vector<float>& hit_x, const vector<float>& hit_y, const vector<float>& hit_z)
{
    for(int i=0; i< hit_e.size();i++)
    {
        vector<float> tmp;
        if(hit_id.at(i)==0 || hit_e.at(i) < 0.1) continue;
        int id = hit_id.at(i);

        map<int, vector<float> >::const_iterator got = m_Hit_map.find (id);
        if( got == m_Hit_map.end() )
        {
        //    std::cout<<"new hit"<<std::endl;
            tmp.push_back(hit_e.at(i));
            tmp.push_back(hit_x.at(i));
            tmp.push_back(hit_y.at(i));
            tmp.push_back(hit_z.at(i));
            m_Hit_map[id]= tmp ;
        }
        else
        {
        //    std::cout<<"same hit"<<", before en="<<(m_Hit_map[id])[0]<<std::endl;
            (m_Hit_map[id])[0]= (m_Hit_map[id])[0] + hit_e.at(i) ;
        //    std::cout<<"new en="<<(m_Hit_map[id])[0]<<std::endl;
        }
    }

}


bool SLCIORdr::getHitInfo(LCCollection* Col, std::vector<int>& vec_ID0, std::vector<int>& vec_ID1, std::vector<double>& vec_Hit_x, std::vector<double>& vec_Hit_y, std::vector<double>& vec_Hit_z, std::vector<double>& vec_Hit_E , double& Hit_cm_x, double& Hit_cm_y, double& Hit_cm_z, double& Hit_tot_e){
      if(Col)
      { 
         double tot_E = 0;
         double sum_x = 0;
         double sum_y = 0;
         double sum_z = 0;
         int NHit = Col->getNumberOfElements();
         for( int i=0; i<NHit; i++ ){
           EVENT::SimCalorimeterHit* in = dynamic_cast<EVENT::SimCalorimeterHit*>(Col->getElementAt(i));
          //cout<<"Hit "<<i <<",CellID0="<<in->getCellID0()<<", CellID1="<<in->getCellID1()<<", E= "<<in->getEnergy()<<" GeV, x="<<in->getPosition()[0]<<",y="<<in->getPosition()[1]<<",z="<<in->getPosition()[2]<<"\n";
          vec_ID0.push_back(in->getCellID0());
          vec_ID1.push_back(in->getCellID1());
          vec_Hit_x.push_back(in->getPosition()[0]);
          vec_Hit_y.push_back(in->getPosition()[1]);
          vec_Hit_z.push_back(in->getPosition()[2]);
          vec_Hit_E.push_back(in->getEnergy());
          tot_E = tot_E + in->getEnergy() ;
          sum_x = sum_x + (in->getEnergy())*(in->getPosition()[0]);
          sum_y = sum_y + (in->getEnergy())*(in->getPosition()[1]);
          sum_z = sum_z + (in->getEnergy())*(in->getPosition()[2]);
          }
         Hit_cm_x = Hit_cm_x + sum_x; 
         Hit_cm_y = Hit_cm_y + sum_y; 
         Hit_cm_z = Hit_cm_z + sum_z; 
         Hit_tot_e = Hit_tot_e + tot_E; 
         //std::cout<<"tot_e="<<Hit_tot_e<<std::endl;
      }
      else {
	  //cout << "Debug: no "<<Col->getTypeName()<<" Collection is found!" << endl;
          return false;
      }
    return true;

}

bool SLCIORdr::getDigiHitInfo(CellIDDecoder<CalorimeterHit> & Hit_idDecoder, LCCollection* Col, std::vector<int>& vec_ID0, std::vector<int>& vec_ID1, std::vector<double>& vec_Hit_x, std::vector<double>& vec_Hit_y, std::vector<double>& vec_Hit_z, std::vector<double>& vec_Hit_E , double& Hit_cm_x, double& Hit_cm_y, double& Hit_cm_z, double& Hit_tot_e, std::vector<int>& vec_ID_S, std::vector<int>& vec_ID_M, std::vector<int>& vec_ID_I, std::vector<int>& vec_ID_J, std::vector<int>& vec_ID_K){
      if(Col)
      { 
         double tot_E = 0;
         double sum_x = 0;
         double sum_y = 0;
         double sum_z = 0;
         int NHit = Col->getNumberOfElements();
         for( int i=0; i<NHit; i++ ){
           IMPL::CalorimeterHitImpl* in = dynamic_cast<IMPL::CalorimeterHitImpl*>(Col->getElementAt(i));
          //cout<<"Hit "<<i <<",CellID0="<<in->getCellID0()<<", CellID1="<<in->getCellID1()<<", E= "<<in->getEnergy()<<" GeV, x="<<in->getPosition()[0]<<",y="<<in->getPosition()[1]<<",z="<<in->getPosition()[2]<<"\n";
          int StaveNum = Hit_idDecoder(in)["S-1"];// from 0 - 7
          int MNum     = Hit_idDecoder(in)["M"  ];// from 1 - 5
          int ZNum     = Hit_idDecoder(in)["J"  ];// from 0 - 89
          int INum     = Hit_idDecoder(in)["I"  ];// from 0 - 169 for layer 0, then decrease for higher layer
          int LayerNum = Hit_idDecoder(in)["K-1"];// from 0 - 28
          
          vec_ID_S.push_back(StaveNum);
          vec_ID_M.push_back(MNum    );
          vec_ID_I.push_back(ZNum    );
          vec_ID_J.push_back(INum    );
          vec_ID_K.push_back(LayerNum);

          vec_ID0.push_back(in->getCellID0());
          vec_ID1.push_back(in->getCellID1());
          vec_Hit_x.push_back(in->getPosition()[0]);
          vec_Hit_y.push_back(in->getPosition()[1]);
          vec_Hit_z.push_back(in->getPosition()[2]);
          vec_Hit_E.push_back(in->getEnergy());
          tot_E = tot_E + in->getEnergy() ;
          sum_x = sum_x + (in->getEnergy())*(in->getPosition()[0]);
          sum_y = sum_y + (in->getEnergy())*(in->getPosition()[1]);
          sum_z = sum_z + (in->getEnergy())*(in->getPosition()[2]);
          }
         Hit_cm_x = Hit_cm_x + sum_x; 
         Hit_cm_y = Hit_cm_y + sum_y; 
         Hit_cm_z = Hit_cm_z + sum_z; 
         Hit_tot_e = Hit_tot_e + tot_E; 
         //std::cout<<"digi tot_e="<<Hit_tot_e<<std::endl;
      }
      else {
	  //cout << "Debug: no "<<Col->getTypeName()<<" Collection is found!" << endl;
          return false;
      }
    return true;

}
/*
        string initString = "M:3,S-1:3,I:9,J:9,K-1:6";          //Need to verify
        isohitcoll->parameters().setValue(LCIO::CellIDEncoding, initString);

        LCFlagImpl flag;
        flag.setBit(LCIO::CHBIT_LONG);
        flag.setBit(LCIO::CHBIT_ID1);
        flag.setBit(LCIO::RCHBIT_ENERGY_ERROR);
        isohitcoll->setFlag(flag.getFlag());
*/


bool SLCIORdr::getHits(LCCollection* Col, CellIDDecoder<SimCalorimeterHit>& idDecoder, float x, float y, float z, std::vector<int>& Hits_ID, std::vector<float>& Hits_x, std::vector<float>& Hits_y, std::vector<float>& Hits_z, float& cell_x, float& cell_y, float& cell_z){
      if(Col)
      { 
         float min_dist = 1e7;
         int first_index = -1;
         int StaveNum = 0 ;
         int MNum     = 0 ;
         int INum     = 0 ;
         int ZNum     = 0 ;
         int LayerNum = 0 ;
         int aStaveNum = 0 ;
         int aMNum     = 0 ;
         int aINum     = 0 ;
         int aZNum     = 0 ;
         int aLayerNum = 0 ;
         int NHit = Col->getNumberOfElements();
         //std::cout<<"NHit="<<NHit<<std::endl;
         int tmp_id;
         float tmp_x;
         float tmp_y;
         float tmp_z;
         int cell_id;
         for( int i=0; i<NHit; i++ )
         {
             EVENT::SimCalorimeterHit* in = dynamic_cast<EVENT::SimCalorimeterHit*>(Col->getElementAt(i));
             //LayerNum = idDecoder(in)["K-1"];
             //if(LayerNum!=0) continue;
             float tmp_dist =  (in->getPosition()[0]-x)*(in->getPosition()[0]-x)+(in->getPosition()[1]-y)*(in->getPosition()[1]-y)+(in->getPosition()[2]-z)*(in->getPosition()[2]-z); 
             if( tmp_dist < min_dist )
             {
                 min_dist = tmp_dist;
                 first_index = i ;
             }
         }
        
         if (first_index!=-1)
         {
             EVENT::SimCalorimeterHit* in = dynamic_cast<EVENT::SimCalorimeterHit*>(Col->getElementAt(first_index));
             StaveNum = idDecoder(in)["S-1"];// from 0 - 7
             MNum     = idDecoder(in)["M"  ];// from 1 - 5
             ZNum     = idDecoder(in)["J"  ];// from 0 - 89
             INum     = idDecoder(in)["I"  ];// from 0 - 169 for layer 0, then decrease for higher layer
             LayerNum = idDecoder(in)["K-1"];// from 0 - 28
             //std::cout<<"min_dist="<<min_dist<<",S="<<StaveNum<<",M="<<MNum<<",I="<<INum<<",J="<<ZNum<<",K="<<LayerNum<<std::endl; 
             // go to layer 0
             int Ishift = LayerNum != 0 ? int((LayerNum-0.1)/2.0)+1 : 0 ; // due to the I is shifted for each layer: shift 1 for L 1 and 2, shift 2 for L 3 and 4, shift 3 for L 5 and 6 and so on 
             INum = INum + Ishift;// INum in layer 0
             LayerNum = 0; 
             // search in 10x10 region   
             min_dist = 1e4;
             //want_dist = 25*2;
             int final_ZNum = 0;
             int final_INum = 0;
             int final_MNum = 0;
             for(int ii=-40;ii<41;ii++)
             {
                 for(int jj=-40;jj<41;jj++)
                 {
                     int tmp_INum = INum + ii;
                     int tmp_ZNum = ZNum + jj;
                     int tmp_MNum = MNum;
                     if(tmp_ZNum<0)      {tmp_ZNum = tmp_ZNum + 90; tmp_MNum = tmp_MNum-1;}
                     else if(tmp_ZNum>89){tmp_ZNum = tmp_ZNum - 90; tmp_MNum = tmp_MNum+1;}
                     getID_x_y_z(StaveNum, tmp_MNum, tmp_INum, tmp_ZNum, LayerNum, tmp_id, tmp_x, tmp_y, tmp_z);
                     float tmp_dist =  (tmp_x-x)*(tmp_x-x)+(tmp_y-y)*(tmp_y-y)+(tmp_z-z)*(tmp_z-z); 
                     if( tmp_dist < min_dist )
                     {
                         min_dist = tmp_dist;
                         final_ZNum = tmp_ZNum;
                         final_INum = tmp_INum;
                         final_MNum = tmp_MNum;
                     }

                 }
             }

             getID_x_y_z(StaveNum, final_MNum, final_INum, final_ZNum, LayerNum, cell_id, cell_x, cell_y, cell_z);
             for( int row=0; row<31; row++ )
             {
                 for( int col=0; col<31; col++ )
                 {
                     for( int dep=0; dep<29; dep++ )
                     {
                         aStaveNum = StaveNum;
                         aLayerNum = dep ; 
                         aMNum = final_MNum;
                         aZNum = (col-15)+final_ZNum;
                         if (aZNum < 0)       {aZNum = aZNum + 90; aMNum = aMNum-1;}
                         else if (aZNum > 89) {aZNum = aZNum - 90; aMNum = aMNum+1;}
                         int shift = aLayerNum != 0 ? int((aLayerNum-0.1)/2.0)+1 : 0 ; // due to the I is shifted for each layer: shift 1 for L 1 and 2, shift 2 for L 3 and 4, shift 3 for L 5 and 6 and so on 
                         aINum = (row-15)+final_INum-shift;
                         getID_x_y_z(aStaveNum, aMNum, aINum, aZNum, aLayerNum, tmp_id, tmp_x, tmp_y, tmp_z);
                         Hits_ID.push_back(tmp_id);
                         Hits_x .push_back(tmp_x );
                         Hits_y .push_back(tmp_y );
                         Hits_z .push_back(tmp_z );
                     }
                 }

             }
         }
      }
      else {
          return false;
      }
    return true;

}

void SLCIORdr::getID_x_y_z(const int& S, const int& M, const int& I, const int& J, const int& K, int& id, float& x, float& y, float& z)
{
    vector<string> str_result;
    int result = split(str_result, SMIJK_map_ID_x_y_z[S][M][I][J][K] , "_");
    if(str_result.size()==4)
    {
        id = stringToNum <int> (str_result.at(0));
        x  = stringToNum<float>(str_result.at(1));
        y  = stringToNum<float>(str_result.at(2));
        z  = stringToNum<float>(str_result.at(3));
    }
    else
    {
        id = 0;
        x  = 0;
        y  = 0;
        z  = 0;
    }
}

bool SLCIORdr::clear(){
    m_HitCm_x = 0;
    m_HitCm_y = 0;
    m_HitCm_z = 0;
    vector<double>().swap(m_HitFirst_x      );
    vector<double>().swap(m_HitFirst_y      );
    vector<double>().swap(m_HitFirst_z      );
    vector<double>().swap(m_phi_rotated     );
    vector<double>().swap(m_HitFirst_vtheta );
    vector<double>().swap(m_HitFirst_vphi   );

    m_HitEn_tot = 0;
    vector<double>().swap(m_mc_vertexX);
    vector<double>().swap(m_mc_vertexY);
    vector<double>().swap(m_mc_vertexZ);
    vector<double>().swap(m_mc_Px     );
    vector<double>().swap(m_mc_Py     );
    vector<double>().swap(m_mc_Pz     );
    vector<double>().swap(m_mc_M      );
    vector<double>().swap(m_Hit_x     );
    vector<double>().swap(m_Hit_y     );
    vector<double>().swap(m_Hit_z     );
    vector<double>().swap(m_Hit_E     ); 
    vector<double>().swap(m_mc_Charge  );
    vector<int>  ().swap(m_mc_Pdg      );
    vector<int>  ().swap(m_mc_genStatus); 
    vector<int>  ().swap(m_mc_simStatus); 
    vector<int>  ().swap(m_mc_np       ); 
    vector<int>  ().swap(m_mc_nd       ); 
    vector<int>  ().swap(m_Hit_ID0     );
    vector<int>  ().swap(m_Hit_ID1     );
    vector<int>  ().swap(m_ID_S     );
    vector<int>  ().swap(m_ID_M     );
    vector<int>  ().swap(m_ID_I     );
    vector<int>  ().swap(m_ID_J     );
    vector<int>  ().swap(m_ID_K     );
    vector<double>().swap(m_mc_pHitx       );
    vector<double>().swap(m_mc_pHity       );
    vector<double>().swap(m_mc_pHitz       );
    vector<double>().swap(m_mc_pHit_theta  );
    vector<double>().swap(m_mc_pHit_phi    );
    vector<double>().swap(m_mc_pHit_rotated);
    vector<double>().swap(m_mc_pHit_dz);
    vector<double>().swap(m_mc_pHit_dy);
    vector< vector<double> >().swap(m_Hits);


    return true;
}

bool SLCIORdr::isEnd(){
return false;
}

bool SLCIORdr::configure(){
return true;
}

bool SLCIORdr::finish(){
//file_out->cd();
//tree_out->Write();
return true;
}

int SLCIORdr::predict(TF_Session* session, TF_Status* status, TF_Graph* graph, const vector<float>& mc_vector, const string& input_op_name, const string& output_op_name, vector<float>& hit_vec) {
    
    //if(DEBUG)std::cout<<"TF input mom= "<<mc_vector.at(0)<<", M_dtheta="<<mc_vector.at(1)<<", M_dphi="<<mc_vector.at(2)<<", P_dz="<<mc_vector.at(3)<<",P_dphi="<<mc_vector.at(4)<<std::endl;
    /*
    // Use read_file to get graph_def as TF_Buffer*
    //TF_Buffer* graph_def = read_file("/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/model_em_low.pb");
    TF_Buffer* graph_def = read_file(pb_name.c_str());
    TF_Graph* graph = TF_NewGraph();
    // Import graph_def into graph
    TF_Status* status = TF_NewStatus();
    TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, graph_def, graph_opts, status);
    if (TF_GetCode(status) != TF_OK) 
    {
        fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(status));
        return 1;
    }
    else 
    {
        fprintf(stdout, "Successfully imported graph\n");
    }
    */
    // Create variables to store the size of the input and output variables
    const int n_mc = mc_vector.size(); // normalized mc info
    //const int n_mc = 5;// normalized mc info
    const int input_size = 512 + n_mc ;
    const int ouput_size = 31*31*29 ;
    const std::size_t num_bytes_in  = input_size * sizeof(float);
    const int num_bytes_out         = ouput_size * sizeof(float);
    // Set input dimensions - this should match the dimensionality of the input in
    //  the loaded graph, in this case it's  dimensional.
    //int64_t in_dims[]  = {1, input_size};
    const std::vector<std::int64_t> input_dims = {1, input_size};
    int64_t out_dims[] = {1, 31, 31, 29, 1};
    // ######################
    //  Set up graph inputs
    // ######################
    // Create a variable containing your values, in this case the input is a
    //  float
    //float  values[input_size] ;
    float* values = new float[input_size] ;
    for(int i=0; i<(input_size-n_mc); i++) values[i]=gaussrand();
    for(int i=0; i<n_mc ; i++) values[input_size-n_mc+i] = mc_vector.at(i);

    // Create vectors to store graph input operations and input tensors
    std::vector<TF_Output> inputs;
    std::vector<TF_Tensor*> input_values;
   // Pass the graph and a string name of your input operation
   // (make sure the operation name is correct)
    //TF_Operation* input_op = TF_GraphOperationByName(graph, "gen_input");
    TF_Operation* input_op = TF_GraphOperationByName(graph, input_op_name.c_str());
    //TF_Operation* input_op = TF_GraphOperationByName(graph, "gen_input");
    TF_Output input_opout = {input_op, 0};
    inputs.push_back(input_opout);
    // Create the input tensor using the dimension (in_dims) and size (num_bytes_in)
    // variables created earlier
    const std::int64_t* in_dims=input_dims.data();
    std::size_t in_num_dims=input_dims.size();
    TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, in_dims, static_cast<int>(in_num_dims), num_bytes_in);
    void* tensor_data = TF_TensorData(input_tensor);
    void* in_values=static_cast<void*>(values);
    std::memcpy(tensor_data, in_values, std::min(num_bytes_in, TF_TensorByteSize(input_tensor)));

    // Create the input tensor using the dimension (in_dims) and size (num_bytes_in)
    input_values.push_back(input_tensor);

    // Optionally, you can check that your input_op and input tensors are correct
    // by using some of the functions provided by the C API.
    //std::cout << "Input op info: " << TF_OperationNumOutputs(input_op) << "\n";
    //std::cout << "Input data info: " << TF_Dim(input_tensor, 0) << "\n";
    // ######################
    // Set up graph outputs (similar to setting up graph inputs)
    // ######################
    // Create vector to store graph output operations
    std::vector<TF_Output> outputs;
    //TF_Operation* output_op = TF_GraphOperationByName(graph, "re_lu_5/Relu");
    TF_Operation* output_op = TF_GraphOperationByName(graph, output_op_name.c_str());
    TF_Output output_opout = {output_op, 0};
    outputs.push_back(output_opout); 

    // Create TF_Tensor* vector
    std::vector<TF_Tensor*> output_values(outputs.size(), nullptr);
    // Similar to creating the input tensor, however here we don't yet have the
    // output values, so we use TF_AllocateTensor()
    TF_Tensor* output_value = TF_AllocateTensor(TF_FLOAT, out_dims, 5, num_bytes_out);
    output_values.push_back(output_value);
    // As with inputs, check the values for the output operation and output tensor
    //std::cout << "Output: " << TF_OperationName(output_op) << "\n";
    //std::cout << "Output info: " << TF_Dim(output_value, 0) << "\n";
    // ######################
    // Run graph
    // ######################
    //fprintf(stdout, "Running session...\n");
    /*
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, sess_opts, status);
    assert(TF_GetCode(status) == TF_OK);
    // Call TF_SessionRun
    */
    TF_SessionRun(session, nullptr,
                &inputs[0], &input_values[0], inputs.size(),
                &outputs[0], &output_values[0], outputs.size(),
                nullptr, 0, nullptr, status);

    // Assign the values from the output tensor to a variable and iterate over them
    float* out_vals = static_cast<float*>(TF_TensorData(output_values[0]));
    for (int i = 0; i < ouput_size; i++)
    {
        //std::cout <<"i="<<i <<", Output values info: " << *(out_vals++) << "\n";
        float dum = out_vals[i];
        //std::cout <<"i="<<i<<", Output values info: " << dum << "\n";
        hit_vec.push_back(dum);// the order is first row 0 ( col from 0 to 11), then row 1 ( col from 0 to 11) and so on.
    }


    delete[] values;
    //if(DEBUG) std::cout<<"end predict"<<std::endl;
  return 1;
}


TF_Buffer* read_file(const char* file) {
  FILE *f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);  //same as rewind(f);

  void* data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);

  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  //buf->data_deallocator = free_buffer;
  buf->data_deallocator = DeallocateBuffer;
  return buf;
}


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
