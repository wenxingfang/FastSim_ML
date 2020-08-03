//////////////////////////
// minus Tof energy  /
//////////////////////////
#include <cstdint>

#include "TF_Prediction/Prediction.h"

#include "GaudiKernel/MsgStream.h"
#include "GaudiKernel/IDataProviderSvc.h"
#include "GaudiKernel/Bootstrap.h"
#include "GaudiKernel/IMessageSvc.h"
#include "GaudiKernel/StatusCode.h"
#include "GaudiKernel/AlgFactory.h"
#include "GaudiKernel/ISvcLocator.h"
#include "GaudiKernel/SmartDataPtr.h"
#include "GaudiKernel/PropertyMgr.h"
#include "GaudiKernel/IJobOptionsSvc.h"
#include "GaudiKernel/Service.h"
#include "GaudiKernel/ThreadGaudi.h"

#include "ReconEvent/ReconEvent.h"
#include "EmcRawEvent/EmcDigi.h"
#include "EmcRecEventModel/RecEmcEventModel.h"
#include "EmcRecGeoSvc/EmcRecGeoSvc.h"
#include "EmcRecGeoSvc/EmcRecBarrelGeo.h"
#include "EmcRec/EmcRecTDS.h"
#include "EmcRec/EmcRecCluster2Shower.h"
#include "EmcRec/EmcRecParameter.h"
#include "RawEvent/RawDataUtil.h"
#include "RawDataProviderSvc/RawDataProviderSvc.h"
#include "RawDataProviderSvc/EmcRawDataProvider.h"
#include "EventModel/EventModel.h"
#include "EventModel/Event.h"
#include "EvtRecEvent/EvtRecEvent.h"
#include "EvtRecEvent/EvtRecTrack.h"
#include "EventModel/EventHeader.h"
#include "EventModel/EventHeader.h"

#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Vector/TwoVector.h"
#include "CLHEP/Geometry/Point3D.h"
using CLHEP::Hep3Vector;
using CLHEP::Hep2Vector;
using CLHEP::HepLorentzVector;
using namespace CLHEP;
#ifndef ENABLE_BACKWARDS_COMPATIBILITY
   typedef HepGeom::Point3D<double> HepPoint3D;
#endif




#include "tensorflow/c/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <math.h>
#include<cstring>
#include <algorithm>

#ifndef PI 
#define PI acos(-1)
#endif
#ifndef DEBUG
#define DEBUG false
#endif
using namespace std;

void BeforeParity(const vector<float>& vec_in,  vector<float>& vec_out);
void Decoder(const vector<float>& tf_output, vector<float>& output);
double getPhi(const double x, const double y);
double getTheta(const double x, const double y, const double z);
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
/////////////////////////////////////////////////////////////////////////////

Prediction::Prediction(const std::string& name, ISvcLocator* pSvcLocator) :
  Algorithm(name, pSvcLocator), m_myInt(0), m_myBool(0), m_myDouble(0),fCluster2Shower(0)
{
  // Part 1: Declare the properties
  declareProperty("MyInt", m_myInt);
  declareProperty("MyBool", m_myBool);
  declareProperty("MyDouble", m_myDouble);
  declareProperty("MyStringVec",m_myStringVec);
  declareProperty("EnergyThreshold", m_energyThreshold=0.04);
  declareProperty("em_Low_pb_name" , m_em_Low_pb_name      );
  declareProperty("em_High_pb_name", m_em_High_pb_name     );
  declareProperty("ep_Low_pb_name" , m_ep_Low_pb_name      );
  declareProperty("ep_High_pb_name", m_ep_High_pb_name     );
  declareProperty("input_op_name"  , m_input_op_name       );
  declareProperty("output_op_name_em" , m_output_op_name_em );
  declareProperty("output_op_name_ep" , m_output_op_name_ep );
  declareProperty("ElectronicsNoiseLevel" , m_ElectronicsNoiseLevel);
  declareProperty("Compensate"     , m_compensate=1);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

StatusCode Prediction::initialize(){
  m_event = 0;
  // Part 1: Get the messaging service, print where you are
  MsgStream log(msgSvc(), name());
  log << MSG::INFO << " Prediction initialize()" << endreq;
  // Part 2: Print out the property values
  /*
  log << MSG::INFO << "  MyInt =    " << m_myInt << endreq;
  log << MSG::INFO << "  MyBool =   " << (int)m_myBool << endreq;
  log << MSG::INFO << "  MyDouble = " << m_myDouble << endreq;
  for (unsigned int i=0; i<m_myStringVec.size(); i++) {
    log << MSG::INFO << "  MyStringVec[" << i << "] = " << m_myStringVec[i] 
	<< endreq;
  }
  */

  ////////////// needed part to make the code finish successfully /////
  EmcRecParameter::lock();
  EmcRecParameter& Para=EmcRecParameter::GetInstance();
  Para.SetDigiCalib(true);
  Para.SetTimeMin(0);
  Para.SetTimeMax(35);
  Para.SetMethodMode(1);
  Para.SetPosCorr(1);
  vector<string> list ;
  list.push_back("log");
  list.push_back("5x5");
  Para.SetPositionMode(list);
  Para.SetElecSaturation(1);

  EmcRecParameter::unlock();
  Para.SetDataMode(0);
  /////////////////////////////////////////////////////////////////



    printf("Hello from TensorFlow C library version %s\n", TF_Version());
    // Use read_file to get graph_def as TF_Buffer*
    //TF_Buffer* graph_def = read_file("/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/model_em_low.pb");
    m_em_Low_graph_def = read_file(m_em_Low_pb_name.c_str());
    m_em_Low_graph     = TF_NewGraph();
    // Import graph_def into graph
    m_em_Low_status = TF_NewStatus();
    m_em_Low_graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(m_em_Low_graph, m_em_Low_graph_def, m_em_Low_graph_opts, m_em_Low_status);
    if (TF_GetCode(m_em_Low_status) != TF_OK ) 
    {
        fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(m_em_Low_status));
        return StatusCode::FAILURE;
    }

    m_em_High_graph_def  = read_file(m_em_High_pb_name.c_str());
    m_em_High_graph      = TF_NewGraph();
    m_em_High_status     = TF_NewStatus();
    m_em_High_graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(m_em_High_graph, m_em_High_graph_def, m_em_High_graph_opts, m_em_High_status);

    m_ep_Low_graph_def = read_file(m_ep_Low_pb_name.c_str());
    m_ep_Low_graph     = TF_NewGraph();
    m_ep_Low_status = TF_NewStatus();
    m_ep_Low_graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(m_ep_Low_graph, m_ep_Low_graph_def, m_ep_Low_graph_opts, m_ep_Low_status);
    m_ep_High_graph_def  = read_file(m_ep_High_pb_name.c_str());
    m_ep_High_graph      = TF_NewGraph();
    m_ep_High_status     = TF_NewStatus();
    m_ep_High_graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(m_ep_High_graph, m_ep_High_graph_def, m_ep_High_graph_opts, m_ep_High_status);


    if (TF_GetCode(m_em_Low_status) != TF_OK || TF_GetCode(m_em_High_status) != TF_OK || TF_GetCode(m_ep_Low_status) != TF_OK || TF_GetCode(m_ep_High_status) != TF_OK ) 
    {
        fprintf(stderr, "ERROR: Unable to import graph");
        //fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(status));
        return StatusCode::FAILURE;
    }
    else 
    {
        fprintf(stdout, "Successfully imported graph\n");
    }


    m_em_Low_sess_opts = TF_NewSessionOptions();
    m_em_Low_session   = TF_NewSession(m_em_Low_graph, m_em_Low_sess_opts, m_em_Low_status);
    assert(TF_GetCode(m_em_Low_status) == TF_OK);
    m_em_High_sess_opts = TF_NewSessionOptions();
    m_em_High_session   = TF_NewSession(m_em_High_graph, m_em_High_sess_opts, m_em_High_status);
    assert(TF_GetCode(m_em_High_status) == TF_OK);

    m_ep_Low_sess_opts = TF_NewSessionOptions();
    m_ep_Low_session   = TF_NewSession(m_ep_Low_graph, m_ep_Low_sess_opts, m_ep_Low_status);
    assert(TF_GetCode(m_ep_Low_status) == TF_OK);
    m_ep_High_sess_opts = TF_NewSessionOptions();
    m_ep_High_session   = TF_NewSession(m_ep_High_graph, m_ep_High_sess_opts, m_ep_High_status);
    assert(TF_GetCode(m_ep_High_status) == TF_OK);
  

    ////////////////////////////////////////////
    m_tuple1 = ntupleSvc()->book ("FILE301/tf", CLID_ColumnWiseTuple, "N-Tuple");
    StatusCode sstatus;
    sstatus = m_tuple1->addItem ("EnSumHit",           m_EnSumHit );








  return StatusCode::SUCCESS;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

StatusCode Prediction::execute() {

  m_EnSumHit = 0;  

  m_event ++;
  // Part 1: Get the messaging service, print where you are
  MsgStream log(msgSvc(), name());
  log << MSG::INFO << "Prediction execute()" << endreq;
  // Part 2: Print out the different levels of messages
  /*
  log << MSG::DEBUG << "A DEBUG message" << endreq;
  log << MSG::INFO << "An INFO message" << endreq;
  log << MSG::WARNING << "A WARNING message" << endreq;
  log << MSG::ERROR << "An ERROR message" << endreq;
  log << MSG::FATAL << "A FATAL error message" << endreq;
  */
//**********************************************************************************************

    SmartDataPtr<EvtRecEvent>     evtRecEvent(eventSvc(),  EventModel::EvtRec::EvtRecEvent);
    SmartDataPtr<EvtRecTrackCol> evtRecTrkCol(eventSvc(),  EventModel::EvtRec::EvtRecTrackCol);
    IEmcRecGeoSvc* iGeoSvc;
    ISvcLocator* svcLocator = Gaudi::svcLocator();
    StatusCode sc = svcLocator->service("EmcRecGeoSvc",iGeoSvc);
    if(sc!=StatusCode::SUCCESS)  cout<<"Error: Can't get EmcRecGeoSvc"<<endl;
    fHitMap.clear(); 
    fClusterMap.clear();
    fShowerMap.clear();
//**********************************************************************************************
    float e_ext_x  = 0;   
    float e_ext_y  = 0;
    float e_ext_z  = 0;
    float e_ext_Px = 0;
    float e_ext_Py = 0;
    float e_ext_Pz = 0;
    float e_charge = 0;
    float e_p      = 0;
    float e_ext_z_new  = 0;
    float e_ext_Pz_new = 0;

    if(DEBUG)std::cout<<"for event "<<m_event<<std::endl;
    for(int i = 0; i < evtRecEvent->totalCharged(); i++)
    {
    	EvtRecTrackIterator itTrk=evtRecTrkCol->begin() + i;
    	if(!(*itTrk)->isMdcTrackValid()) continue;
    	RecMdcTrack *mdcTrk = (*itTrk)->mdcTrack();
    	if(!(*itTrk)->isEmcShowerValid()) continue;
    	RecEmcShower *emcTrk = (*itTrk)->emcShower();
        if(emcTrk->energy() < m_energyThreshold) continue;
        RecEmcCluster *emcCls = emcTrk->getCluster();
        //std::cout<<"emcCls="<<emcCls<<std::endl;
        if(emcCls==0)continue; 
    	if(!(*itTrk)->isExtTrackValid()) continue;
        RecExtTrack *extTrk = (*itTrk)->extTrack(); 
        e_ext_x  = (extTrk->emcPosition()).x();
        e_ext_y  = (extTrk->emcPosition()).y();
        e_ext_z  = (extTrk->emcPosition()).z();
        e_ext_Px = (extTrk->emcMomentum()).x();
        e_ext_Py = (extTrk->emcMomentum()).y();
        e_ext_Pz = (extTrk->emcMomentum()).z();
        e_charge = mdcTrk->charge() ;
        e_p      = mdcTrk->p() ;
        float etof = 0;
        if((*itTrk)->isTofTrackValid()){
           SmartRefVector<RecTofTrack> recTofTrackVec=(*itTrk)->tofTrack();
           if(!recTofTrackVec.empty()) etof=recTofTrackVec[0]->energy();
           if(etof>100.)etof=0;
         }
        e_p = (e_p > etof) ? (e_p - etof) : 0;

        if(fabs(e_ext_z)>137) continue;//barrel only now
        double M_dtheta = 0;
        double M_dphi   = 0;
        double P_dz     = 0;
        double P_dphi   = 0;
        vector<RecEmcID> vec_ID;
        bool do_parity = false;
        if( (e_charge > 0 && e_ext_z<0) || (e_charge < 0 && e_ext_z>0) )// to parity transfor
        {
            do_parity = true;
        }
        if(DEBUG)std::cout<<"for track "<<i<<std::endl;
        int Find = getInfo(iGeoSvc, e_ext_x, e_ext_y, e_ext_z, e_ext_Px, e_ext_Py, e_ext_Pz, emcCls, M_dtheta, M_dphi, P_dz, P_dphi, vec_ID, do_parity);
        if(Find==0) continue;
        vector<float> mc_vec;
        vector<float> vec_Hit_E;//check the order
        vector<float> vec_Hit_E_reform(121);
        if(e_charge > 0)
        {
            if(DEBUG)std::cout<<"i =" <<i<<",for ep, mom= "<<e_p<<", M_dtheta="<<M_dtheta<<", M_dphi="<<M_dphi<<", P_dz="<<P_dz<<",P_dphi="<<P_dphi<<",e_ext_z="<<e_ext_z<<std::endl;
            if(e_ext_z>=125)
            {
                mc_vec.push_back(e_p);
                mc_vec.push_back(M_dtheta/5);
                mc_vec.push_back(M_dphi  /10);
                mc_vec.push_back(P_dz    /2);
                mc_vec.push_back(P_dphi  /2);
                mc_vec.push_back(e_ext_z /150);
                predict(m_ep_High_session, m_ep_High_status, m_ep_High_graph, mc_vec, m_input_op_name, m_output_op_name_ep, vec_Hit_E) ;
                Decoder(vec_Hit_E, vec_Hit_E_reform);
            }
            else if(e_ext_z>=0)
            {
                mc_vec.push_back(e_p);
                mc_vec.push_back(M_dtheta/5);
                mc_vec.push_back(M_dphi  /10);
                mc_vec.push_back(P_dz    /2);
                mc_vec.push_back(P_dphi  /2);
                mc_vec.push_back(e_ext_z /150);
                predict(m_ep_Low_session, m_ep_Low_status, m_ep_Low_graph, mc_vec, m_input_op_name, m_output_op_name_ep, vec_Hit_E) ;
                Decoder(vec_Hit_E, vec_Hit_E_reform);
            }
            else if(e_ext_z>=-125)
            {
                mc_vec.push_back(e_p);
                mc_vec.push_back(M_dtheta/5);
                mc_vec.push_back(M_dphi  /10);
                mc_vec.push_back(P_dz    /2);
                mc_vec.push_back(P_dphi  /2);
                mc_vec.push_back(-e_ext_z/150);
                predict(m_ep_Low_session, m_ep_Low_status, m_ep_Low_graph, mc_vec, m_input_op_name, m_output_op_name_ep, vec_Hit_E) ;
                vector<float> vec_Hit_E_parity(121);
                BeforeParity(vec_Hit_E,  vec_Hit_E_parity);
                Decoder(vec_Hit_E_parity, vec_Hit_E_reform);
            } 
            else
            {
                mc_vec.push_back(e_p);
                mc_vec.push_back(M_dtheta/5);
                mc_vec.push_back(M_dphi  /10);
                mc_vec.push_back(P_dz    /2);
                mc_vec.push_back(P_dphi  /2);
                mc_vec.push_back(-e_ext_z/150);
                predict(m_ep_High_session, m_ep_High_status, m_ep_High_graph, mc_vec, m_input_op_name, m_output_op_name_ep, vec_Hit_E) ;
                vector<float> vec_Hit_E_parity(121);
                BeforeParity(vec_Hit_E,  vec_Hit_E_parity);
                Decoder(vec_Hit_E_parity, vec_Hit_E_reform);
            }
        }
        else if(e_charge < 0)
        {
            if(DEBUG)std::cout<<"i =" <<i<<",for em, mom= "<<e_p<<", M_dtheta="<<M_dtheta<<", M_dphi="<<M_dphi<<", P_dz="<<P_dz<<",P_dphi="<<P_dphi<<",e_ext_z="<<e_ext_z<<std::endl;
            if(e_ext_z>=125)
            {
                mc_vec.push_back(e_p);
                mc_vec.push_back(M_dtheta/5);
                mc_vec.push_back(M_dphi  /10);
                mc_vec.push_back(P_dz    /2);
                mc_vec.push_back(P_dphi  /2);
                mc_vec.push_back(-e_ext_z/150);
                predict(m_em_High_session, m_em_High_status, m_em_High_graph, mc_vec, m_input_op_name, m_output_op_name_em, vec_Hit_E) ;
                vector<float> vec_Hit_E_parity(121);
                BeforeParity(vec_Hit_E,  vec_Hit_E_parity);
                Decoder(vec_Hit_E_parity, vec_Hit_E_reform);
            }
            else if(e_ext_z>=0)
            {
                mc_vec.push_back(e_p);
                mc_vec.push_back(M_dtheta/5);
                mc_vec.push_back(M_dphi  /10);
                mc_vec.push_back(P_dz    /2);
                mc_vec.push_back(P_dphi  /2);
                mc_vec.push_back(-e_ext_z/150);
                predict(m_em_Low_session, m_em_Low_status, m_em_Low_graph, mc_vec, m_input_op_name, m_output_op_name_em, vec_Hit_E) ;
                vector<float> vec_Hit_E_parity(121);
                BeforeParity(vec_Hit_E,  vec_Hit_E_parity);
                Decoder(vec_Hit_E_parity, vec_Hit_E_reform);
            } 
            else if(e_ext_z>=-125)
            {
                mc_vec.push_back(e_p);
                mc_vec.push_back(M_dtheta /5);
                mc_vec.push_back(M_dphi   /10);
                mc_vec.push_back(P_dz     /2);
                mc_vec.push_back(P_dphi   /2);
                mc_vec.push_back(e_ext_z  /150);
                predict(m_em_Low_session, m_em_Low_status, m_em_Low_graph, mc_vec, m_input_op_name, m_output_op_name_em, vec_Hit_E) ;
                Decoder(vec_Hit_E, vec_Hit_E_reform);
            }
            else
            {   
                mc_vec.push_back(e_p);
                mc_vec.push_back(M_dtheta /5);
                mc_vec.push_back(M_dphi   /10);
                mc_vec.push_back(P_dz     /2);
                mc_vec.push_back(P_dphi   /2);
                mc_vec.push_back(e_ext_z  /150);
                predict(m_em_High_session, m_em_High_status, m_em_High_graph, mc_vec, m_input_op_name, m_output_op_name_em, vec_Hit_E) ;
                Decoder(vec_Hit_E, vec_Hit_E_reform);
            } 
            
        }
        MakerHitMap(vec_ID, vec_Hit_E_reform, fHitMap);

    }

    if(m_compensate==1)
    {
        SmartDataPtr<RecEmcHitCol> emcHitCol(eventSvc(),"/Event/Recon/RecEmcHitCol");
        if (!emcHitCol)
        {
            log << MSG::FATAL << "Could not find EMC emcHitCol" << endreq;
            return( StatusCode::FAILURE);
        }
        map<RecEmcID, RecEmcHit>::iterator iter;
        for(RecEmcHitCol::iterator it=emcHitCol->begin();it!= emcHitCol->end();it++)
        {
            RecEmcID id = (*it)->getCellId();
            iter = fHitMap.find(id);
            if(iter == fHitMap.end())
            {
                RecEmcHit aHit;
                aHit.CellId(id);
                aHit.Energy((*it)->getEnergy());
                aHit.Time(0);
                fHitMap[id]=aHit;    
                //fHitMap[id]=(*it);    
            }

        }
    }

    RemoveNoise(fHitMap, m_ElectronicsNoiseLevel);
    //if(DEBUG)std::cout<<"Removed Noise"<<std::endl;
   
    //std::cout << __FILE__ <<__LINE__ << std::endl;
    fHit2Cluster.Convert(fHitMap,fClusterMap);
    //std::cout << __FILE__ <<__LINE__ << std::endl;
    //if(DEBUG)std::cout<<"fHit2Cluster.Convert"<<std::endl;
    fCluster2Shower = new EmcRecCluster2Shower;
    fCluster2Shower->Convert(fClusterMap,fShowerMap);
    //std::cout << __FILE__ <<__LINE__ << std::endl;
    //if(DEBUG)std::cout<<"fCluster2Shower.Convert"<<std::endl;
 
    EmcRecTDS tds;
    tds.RegisterToTDS(fHitMap,fClusterMap,fShowerMap);
    //std::cout << __FILE__ <<__LINE__ << std::endl;
    //if(DEBUG)std::cout<<"RegisterToTDS"<<std::endl;
    
    
    SmartDataPtr<RecEmcHitCol> emcHitCol(eventSvc(),"/Event/Recon/RecEmcHitCol");
    if (!emcHitCol)
    {
        log << MSG::FATAL << "Could not find EMC emcHitCol" << endreq;
        return( StatusCode::FAILURE);
    }
    RecEmcHitCol::iterator iter3;
    int ii=0;
    for (iter3=emcHitCol->begin();iter3!= emcHitCol->end();iter3++)
    {
        /*
        RecEmcID id = (*iter3)->getCellId();
        RecEmcHit aHit;
        aHit.CellId(id);
        aHit.Energy((*iter3)->getEnergy()/GeV);
        aHit.Time(0);
        std::cout<<"my tf hit "<<ii<<", energy="<<(*iter3)->getEnergy()<<std::endl;
        ii++;
        */
        m_EnSumHit = m_EnSumHit + (*iter3)->getEnergy() ;
    }
    m_tuple1 -> write();
     
    if(DEBUG)std::cout<<"Done Prediction"<<std::endl;
    
    delete fCluster2Shower;
    return StatusCode::SUCCESS;
}



void Prediction::MakerHitMap(const vector<RecEmcID>& vec_id, const vector<float>& vec_e, RecEmcHitMap& aHitMap)
{
    
    //if(DEBUG)std::cout<<"vec id size="<<vec_id.size()<<", vec_e size="<<vec_e.size()<<std::endl;
    for(int i=0; i<vec_id.size(); i++)
    {
       RecEmcID id = vec_id.at(i);
       //double energy= vec_e.at(i);
       int npart   = EmcID::barrel_ec   (id);// 1 for barrel
       int ntheta  = EmcID::theta_module(id);// within 0-43
       int nphi    = EmcID::phi_module  (id);  //within 0-119
       if(npart!=1 || (ntheta<0 ||  ntheta>43) || (nphi<0 ||  nphi>119) ) continue;
       float energy= vec_e.at(i);

       //RecEmcHit aHit;
       //aHit.CellId(id);
       ////aHit.Energy(energy/GeV);
       //aHit.Energy(energy);
       //aHit.Time(0);
       //aHitMap[aHit.getCellId()]=aHit;
       //if(DEBUG)std::cout<<"MakerHitMap:i="<<i<<", energy="<<energy<<std::endl;
       
       map<RecEmcID, RecEmcHit>::iterator iter;
       iter = aHitMap.find(id);
       if(iter != aHitMap.end())
       {
           float new_e = (iter->second).getEnergy() + energy;
           //aHitMap.erase(iter->first);
           //RecEmcHit aHit;
           //aHit.CellId(id);
           //aHit.Energy(new_e/GeV);
           //aHit.Time(0);
           //aHitMap[aHit.getCellId()]=aHit;
           //std::cout<<"before e="<< (iter->second).getEnergy() << ", add e="<< energy<<std::endl;
           (iter->second).Energy(new_e);
           //(iter->second).Energy(new_e*1.e+3/GeV);
           //std::cout<<"new e="<< (iter->second).getEnergy()<<std::endl;
           //(iter->second).Energy(new_e/megaelectronvolt);
           //(iter->second).Energy(new_e);
       }
       else
       {  
           RecEmcHit aHit;
           aHit.CellId(id);
           aHit.Energy(energy);
           //aHit.Energy(energy*1.e+3/GeV);
           //aHit.Energy(energy/megaelectronvolt);
           //aHit.Energy(energy);
           aHit.Time(0);
           aHitMap[aHit.getCellId()]=aHit;
       }
       
    }

}

void Prediction::RemoveNoise(RecEmcHitMap& aHitMap, const float& ElectronicsNoiseLevel)
{
    int i =0;
    for(RecEmcHitMap::const_iterator it = aHitMap.begin(); it != aHitMap.end(); it++)
    {
        //std::cout<<"Hit map i="<<i<<",energy  ="<<(it->second).getEnergy()<<std::endl;
        //if((it->second).getEnergy()<ElectronicsNoiseLevel) aHitMap.erase(it->first);
        i++;
    }

}


int Prediction::predict(TF_Session* session, TF_Status* status, TF_Graph* graph, const vector<float>& mc_vector, const string& input_op_name, const string& output_op_name, vector<float>& hit_vec) {
    
    if(DEBUG)std::cout<<"TF input mom= "<<mc_vector.at(0)<<", M_dtheta="<<mc_vector.at(1)<<", M_dphi="<<mc_vector.at(2)<<", P_dz="<<mc_vector.at(3)<<",P_dphi="<<mc_vector.at(4)<<std::endl;
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
    const int ouput_size = 11*11 ;
    const std::size_t num_bytes_in  = input_size * sizeof(float);
    const int num_bytes_out         = ouput_size * sizeof(float);
    // Set input dimensions - this should match the dimensionality of the input in
    //  the loaded graph, in this case it's  dimensional.
    //int64_t in_dims[]  = {1, input_size};
    const std::vector<std::int64_t> input_dims = {1, input_size};
    int64_t out_dims[] = {1, 11, 11, 1};
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
    TF_Tensor* output_value = TF_AllocateTensor(TF_FLOAT, out_dims, 4, num_bytes_out);
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
    if(DEBUG) std::cout<<"end predict"<<std::endl;
  return 1;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

StatusCode Prediction::finalize() {

  //if(fCluster2Shower) delete fCluster2Shower;
  // Part 1: Get the messaging service, print where you are
  MsgStream log(msgSvc(), name());
  log << MSG::INFO << "Prediction finalize()" << endreq;
  
  // Delete variables
  TF_CloseSession               (m_em_Low_session, m_em_Low_status);
  TF_DeleteSession              (m_em_Low_session, m_em_Low_status);
  TF_DeleteSessionOptions       (m_em_Low_sess_opts);
  TF_DeleteImportGraphDefOptions(m_em_Low_graph_opts);
  TF_DeleteGraph                (m_em_Low_graph);
  TF_DeleteStatus               (m_em_Low_status);

  TF_CloseSession               (m_em_High_session, m_em_High_status);
  TF_DeleteSession              (m_em_High_session, m_em_High_status);
  TF_DeleteSessionOptions       (m_em_High_sess_opts);
  TF_DeleteImportGraphDefOptions(m_em_High_graph_opts);
  TF_DeleteGraph                (m_em_High_graph);
  TF_DeleteStatus               (m_em_High_status);

  TF_CloseSession               (m_ep_Low_session, m_ep_Low_status);
  TF_DeleteSession              (m_ep_Low_session, m_ep_Low_status);
  TF_DeleteSessionOptions       (m_ep_Low_sess_opts);
  TF_DeleteImportGraphDefOptions(m_ep_Low_graph_opts);
  TF_DeleteGraph                (m_ep_Low_graph);
  TF_DeleteStatus               (m_ep_Low_status);

  TF_CloseSession               (m_ep_High_session, m_ep_High_status);
  TF_DeleteSession              (m_ep_High_session, m_ep_High_status);
  TF_DeleteSessionOptions       (m_ep_High_sess_opts);
  TF_DeleteImportGraphDefOptions(m_ep_High_graph_opts);
  TF_DeleteGraph                (m_ep_High_graph);
  TF_DeleteStatus               (m_ep_High_status);

  return StatusCode::SUCCESS;
}


/*
StatusCode Prediction::RegisterReconEvent()
{
  if(DEBUG)std::cout<<"Prediction::RegisterReconEvent"<<std::endl;

  IDataProviderSvc* eventSvc;
  Gaudi::svcLocator()->service("EventDataSvc", eventSvc);
       
  //check whether the Recon has been already registered
  DataObject *aReconEvent;
  eventSvc->findObject("/Event/Recon",aReconEvent);
  if(aReconEvent==NULL) {
    //then register Recon
    aReconEvent = new ReconEvent();
    StatusCode sc = eventSvc->registerObject("/Event/Recon",aReconEvent);
    if(sc!=StatusCode::SUCCESS) {
      std::cout<< "Prediction:: Could not register ReconEvent" <<std::endl;
      return StatusCode::FAILURE;
    }
  }

  return StatusCode::SUCCESS;
}
StatusCode Prediction::RegisterHit(RecEmcHitMap& aHitMap)
{
  RegisterReconEvent();
    
  if(DEBUG)std::cout<<"Prediction::RegisterHit()"<<std::endl;

  IDataProviderSvc* eventSvc;
  Gaudi::svcLocator()->service("EventDataSvc", eventSvc);
  
  RecEmcHitCol *aRecEmcHitCol = new RecEmcHitCol;
  RecEmcHitMap::iterator iHitMap;
  for(iHitMap=aHitMap.begin();
      iHitMap!=aHitMap.end();
      iHitMap++){
    aRecEmcHitCol->add(new RecEmcHit(iHitMap->second));
    std::cout<<"from Hit map energy ="<<(iHitMap->second).getEnergy()<<std::endl;
  }

  //check whether the RecEmcHitCol has been already registered
  StatusCode sc;
  DataObject *aRecEmcHitEvent;
  eventSvc->findObject("/Event/Recon/RecEmcHitCol", aRecEmcHitEvent);
  if(aRecEmcHitEvent!=NULL) {
    if(DEBUG)std::cout<<"Prediction:: remove RegisterHit()"<<std::endl;
    //then unregister RecEmcHitCol
    sc = eventSvc->unregisterObject("/Event/Recon/RecEmcHitCol");
    delete aRecEmcHitEvent;   //it must be delete to avoid memory leakage
    if(sc!=StatusCode::SUCCESS) {
      std::cout<< "Prediction:: Could not unregister RecEmcHitCol" <<std::endl;
      return( StatusCode::FAILURE);
    }
  }

  if(DEBUG)std::cout<<"Prediction:: before register RegisterHit()"<<std::endl;
  sc = eventSvc->registerObject("/Event/Recon/RecEmcHitCol", aRecEmcHitCol);
  if(DEBUG)std::cout<<"Prediction:: after register RegisterHit()"<<std::endl;
  if(sc!=StatusCode::SUCCESS) {
    std::cout<< "Prediction:: Could not register RecEmcHitCol" <<std::endl;
    return( StatusCode::FAILURE);
  }

  return StatusCode::SUCCESS;
}
*/

int Prediction::getInfo(const IEmcRecGeoSvc* iGeoSvc, const double& x, const double& y, const double& z, const double& px, const double& py, const double& pz, const RecEmcCluster* clus, double& M_dtheta, double& M_dphi, double& P_dz, double& P_dphi, vector<RecEmcID>& cell_ID, bool& do_parity) {
    float dist_min = 999;
    int center_it=-1;
    int counter = 0;    
    RecEmcHitMap::const_iterator cen_it; 
    unsigned int npart,ntheta,nphi;
    for(RecEmcHitMap::const_iterator it = clus->Begin(); it != clus->End(); it++)
    {
        const std::pair<const Identifier, RecEmcHit> tmp_HitMap = *it;
        RecEmcHit tmp_EmcHit   = tmp_HitMap.second;
        HepPoint3D cellFCenter = tmp_EmcHit.getFrontCenter();
        float dist = sqrt( pow(cellFCenter.x()-x,2) + pow(cellFCenter.y()-y,2) + pow(cellFCenter.z()-z,2));
        if(dist < dist_min)
        {
            dist_min = dist;
            center_it = counter;       
            cen_it = it;
        } 
        counter ++;      
    }
    if(dist_min <= 2.5*sqrt(2) && center_it !=-1)
    {
        if(DEBUG)std::cout<<"find center"<<std::endl;
        //RecEmcHitMap::const_iterator it = clus->Begin() + center_it;
        RecEmcHitMap::const_iterator it = cen_it;
        const std::pair<const Identifier, RecEmcHit> tmp_HitMap = *it;
        RecEmcHit tmp_EmcHit   = tmp_HitMap.second;
        RecEmcID tmp_cellID    = tmp_EmcHit.getCellId();
        HepPoint3D cellFCenter = tmp_EmcHit.getFrontCenter();
        HepPoint3D cellCenter  = tmp_EmcHit.getCenter();
        HepPoint3D cellDirection = cellCenter - cellFCenter ;

        double M_phi = getPhi(px, py);
        double M_theta = do_parity==false ? getTheta(px, py, pz) : getTheta(px, py, -pz) ;
        double cellDir_phi   = getPhi(cellDirection.x(), cellDirection.y());
        double cellDir_theta = do_parity==false ? getTheta(cellDirection.x(), cellDirection.y(), cellDirection.z()) : getTheta(cellDirection.x(), cellDirection.y(), -cellDirection.z()) ;
        M_dphi   = M_phi - cellDir_phi ;
        if(M_dphi< -180) M_dphi = M_dphi + 360;
        else if(M_dphi > 180) M_dphi = M_dphi - 360;
        M_dtheta = M_theta - cellDir_theta ;
        P_dz = do_parity==false ? (z - cellFCenter.z()) : (-z + cellFCenter.z());
        P_dphi = getPhi(x, y) - getPhi(cellFCenter.x(),cellFCenter.y());
        if(P_dphi< -180) P_dphi = P_dphi + 360;
        else if(P_dphi > 180) P_dphi = P_dphi - 360;

        npart  = EmcID::barrel_ec(tmp_cellID);
        ntheta = EmcID::theta_module(tmp_cellID);// within 0-43
        nphi   = EmcID::phi_module(tmp_cellID);  //within 0-119
        //int id_theta[121]={-1};
        //int id_phi  [121]={-1};
        for(int i=0; i<11; i++)
        {
            unsigned int itheta = (i-5)+ntheta ;
            for(int j=0; j<11; j++)
            {
                unsigned iphi = (5-j)+nphi ;
                if(iphi> 119) iphi = iphi - 120 ;
                else if(iphi< 0) iphi = iphi + 120 ;
                //id_theta[i*11+j]=itheta   ;
                //id_phi  [i*11+j]=iphi    ;
                RecEmcID id=EmcID::crystal_id(1, itheta, iphi);
                cell_ID.push_back(id);
            }
        }
        /*
        for(int i=0; i<121; i++)
        {
            cell_E[i] = 0;
        }
        for(RecEmcHitMap::const_iterator it = clus->Begin(); it != clus->End(); it++)
        {
            const std::pair<const Identifier, RecEmcHit> tmp = *it;
            RecEmcHit tmp_Hit   = tmp.second;
            RecEmcID tmp_ID    = tmp_Hit.getCellId();
            double   tmp_E     = tmp_Hit.getEnergy();
            unsigned int ipart  = EmcID::barrel_ec(tmp_ID);
            unsigned int itheta = EmcID::theta_module(tmp_ID);// within 0-43 from positive Z to negative Z
            unsigned int iphi   = EmcID::phi_module(tmp_ID);  //within 0-119
            if(ipart !=1)continue;
            for(int i=0; i<121; i++)
            {
                if(id_theta[i]==itheta && id_phi[i]==iphi)
                {
                    cell_E[i] = tmp_E;
                    break;
                }
            }
        }
        */
        
    }
    else return 0;
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

double getPhi(const double x, const double y)
{
    if     (x==0 && y>0) return 90;
    else if(x==0 && y<0) return 270;
    else if(x==0 && y==0) return 0;
    double phi = atan(y/x)*180/PI;
    if                 (x<0) phi = phi + 180;
    else if     (x>0 && y<0) phi = phi + 360;
    return phi;
}


double getTheta(const double x, const double y, const double z)
{
    double pre = sqrt(x*x + y*y);
    double theta = z != 0 ? atan(pre/z)*180/PI : 90;
    if(theta<0) theta = 180 + theta;  
    return theta; 
}

void BeforeParity(const vector<float>& vec_in,  vector<float>& vec_out)
{
    for(int i=0; i<vec_in.size(); i++)
    {
        int nRow = int((i+1-0.1)/11);
        int nCol = i%11 ;
        int nCol_new = 10-nCol;
        int index = nRow*11 + nCol_new;
        vec_out[index] = vec_in.at(i);
    }
}


void Decoder(const vector<float>& tf_output, vector<float>& output)
{
    //std::cout<<"tf size="<<tf_output.size()<<std::endl;
    for(int i =0 ; i<tf_output.size(); i++)
    {
        int nRow = int((i+1-0.1)/11);
        int nCol = i%11 ;
        int real_i = nRow + 11*(10-nCol);
        //std::cout<<"tf i="<<i<<",value="<<tf_output.at(i)<<std::endl;
        output[real_i] = tf_output.at(i);
    }   
        /* according to the root2hdf5 code
        for i in range(121):
            nRow = i - 11*int(i/11.0)
            nCol =10 - int(i/11.0)
            Barrel_Hit[entryNum, nRow, nCol, 0] = tmp_Hit_E[i]
        */
}




/*
int tf_test(TF_Session* session, TF_Status* status, TF_Graph* graph){
    printf("tf test TensorFlow C library version %s\n", TF_Version());
    // Create variables to store the size of the input and output variables
    const int input_size = 512+5 ;
    const int ouput_size = 11*11 ;
    //const int num_bytes_in  = input_size * sizeof(float);
    const std::size_t num_bytes_in  = input_size * sizeof(float);
    const int num_bytes_out = ouput_size * sizeof(float);
    // Set input dimensions - this should match the dimensionality of the input in
    //  the loaded graph, in this case it's  dimensional.
    //int64_t in_dims[]  = {1, input_size};
    const std::vector<std::int64_t> input_dims = {1, input_size};
    int64_t out_dims[] = {1, 11, 11,1};
    // ######################
    //  Set up graph inputs
    // ######################
    // Create a variable containing your values, in this case the input is a
    //  float
    float values[input_size] ;
    //for(int i=0; i<(input_size-5); i++) values[i]=gaussrand();
    for(int i=0; i<(input_size-5); i++) values[i]=0.5;
    values[input_size-5] =  1.8;
    values[input_size-4] =  0.5;
    values[input_size-3] = -5.0/10;
    values[input_size-2] =  0.6;
    values[input_size-1] =  0.6;
    // Create vectors to store graph input operations and input tensors
    std::vector<TF_Output> inputs;
    std::vector<TF_Tensor*> input_values;

   // Pass the graph and a string name of your input operation
   // (make sure the operation name is correct)
    std::cout << "Hi 1" << "\n";

    TF_Operation* input_op = TF_GraphOperationByName(graph, "gen_input");
    TF_Output input_opout = {input_op, 0};
    inputs.push_back(input_opout);
    std::cout << "Hi 2" << "\n";
    // Create the input tensor using the dimension (in_dims) and size (num_bytes_in)
    // variables created earlier
    //TF_Tensor* input = TF_NewTensor(TF_FLOAT, in_dims, 2, values, num_bytes_in, &Deallocator, 0);
    //TF_Tensor* input = TF_NewTensor(TF_FLOAT, in_dims, 2, values, input_size, &Deallocator, 0);
    const std::int64_t* in_dims=input_dims.data();
    std::size_t in_num_dims=input_dims.size();
    TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, in_dims, static_cast<int>(in_num_dims), num_bytes_in);
    void* tensor_data = TF_TensorData(input_tensor);
    void* in_values=static_cast<void*>(values);
    std::memcpy(tensor_data, in_values, std::min(num_bytes_in, TF_TensorByteSize(input_tensor)));

    std::cout << "Hi 3" << "\n";
    // Create the input tensor using the dimension (in_dims) and size (num_bytes_in)
    input_values.push_back(input_tensor);

    // Optionally, you can check that your input_op and input tensors are correct
    // by using some of the functions provided by the C API.
    std::cout << "Input op info: " << TF_OperationNumOutputs(input_op) << "\n";
    std::cout << "Input data info: " << TF_Dim(input_tensor, 0) << "\n";
    // ######################
    // Set up graph outputs (similar to setting up graph inputs)
    // ######################
    // Create vector to store graph output operations
    std::vector<TF_Output> outputs;
    TF_Operation* output_op = TF_GraphOperationByName(graph, "re_lu_5/Relu");
    TF_Output output_opout = {output_op, 0};
    outputs.push_back(output_opout); 

    // Create TF_Tensor* vector
    std::vector<TF_Tensor*> output_values(outputs.size(), nullptr);
    // Similar to creating the input tensor, however here we don't yet have the
    // output values, so we use TF_AllocateTensor()
    TF_Tensor* output_value = TF_AllocateTensor(TF_FLOAT, out_dims, 4, num_bytes_out);
    output_values.push_back(output_value);
    // As with inputs, check the values for the output operation and output tensor
    std::cout << "Output: " << TF_OperationName(output_op) << "\n";
    std::cout << "Output info: " << TF_Dim(output_value, 0) << "\n";
    // ######################
    // Run graph
    // ######################
    fprintf(stdout, "Running session...\n");
//    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
//    TF_Session* session = TF_NewSession(graph, sess_opts, status);
//    assert(TF_GetCode(status) == TF_OK);
    // Call TF_SessionRun
    TF_SessionRun(session, nullptr,
                &inputs[0], &input_values[0], inputs.size(),
                &outputs[0], &output_values[0], outputs.size(),
                nullptr, 0, nullptr, status);

    // Assign the values from the output tensor to a variable and iterate over them
    float* out_vals = static_cast<float*>(TF_TensorData(output_values[0]));
    for (int i = 0; i < ouput_size; ++i)
    {
        if((i+1)%11==0)std::cout << "i+1=" << i+1 << "\n";
        std::cout << "Output values info: " << *out_vals++ << "\n";
    }

    fprintf(stdout, "Successfully run session\n");

    // Delete variables
    //TF_CloseSession(session, status);
    //TF_DeleteSession(session, status);
    //TF_DeleteSessionOptions(sess_opts);
    //TF_DeleteImportGraphDefOptions(graph_opts);
    //TF_DeleteGraph(graph);
    //TF_DeleteStatus(status);
    return 0;
}
*/
/*
int Predict(TF_Session* session, TF_Status* status, TF_Graph* graph, const vector<float>& mc_vector, const string& input_op_name, const string& output_op_name, vector<float>& hit_vec){
    printf("Predict TensorFlow C library version %s\n", TF_Version());
    // Create variables to store the size of the input and output variables
    //const int n_mc = 5;// normalized mc info
    const int input_size = 512 + 5 ;
    const int ouput_size = 11*11 ;
    const std::size_t num_bytes_in  = input_size * sizeof(float);
    const int num_bytes_out = ouput_size * sizeof(float);
    const std::vector<std::int64_t> input_dims = {1, input_size};
    int64_t out_dims[] = {1, 11, 11,1};
    // ######################
    //  Set up graph inputs
    // ######################
    // Create a variable containing your values, in this case the input is a
    //  float
    float values[input_size] ;
    //for(int i=0; i<(input_size-5); i++) values[i]=gaussrand();
    for(int i=0; i<(input_size-5); i++) values[i]=0.5;
    values[input_size-5] =  1.8;
    values[input_size-4] =  0.5;
    values[input_size-3] = -5.0/10;
    values[input_size-2] =  0.6;
    values[input_size-1] =  0.6;
    // Create vectors to store graph input operations and input tensors
    std::vector<TF_Output> inputs;
    std::vector<TF_Tensor*> input_values;

    TF_Operation* input_op = TF_GraphOperationByName(graph, "gen_input");
    //TF_Operation* input_op = TF_GraphOperationByName(graph, input_op_name.c_str());
    TF_Output input_opout = {input_op, 0};
    inputs.push_back(input_opout);
    const std::int64_t* in_dims=input_dims.data();
    std::size_t in_num_dims=input_dims.size();
    TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, in_dims, static_cast<int>(in_num_dims), num_bytes_in);
    void* tensor_data = TF_TensorData(input_tensor);
    void* in_values=static_cast<void*>(values);
    std::memcpy(tensor_data, in_values, std::min(num_bytes_in, TF_TensorByteSize(input_tensor)));
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
    TF_Operation* output_op = TF_GraphOperationByName(graph, "re_lu_5/Relu");
    //TF_Operation* output_op = TF_GraphOperationByName(graph, output_op_name.c_str());
    TF_Output output_opout = {output_op, 0};
    outputs.push_back(output_opout); 

    // Create TF_Tensor* vector
    std::vector<TF_Tensor*> output_values(outputs.size(), nullptr);
    // Similar to creating the input tensor, however here we don't yet have the
    // output values, so we use TF_AllocateTensor()
    TF_Tensor* output_value = TF_AllocateTensor(TF_FLOAT, out_dims, 4, num_bytes_out);
    output_values.push_back(output_value);
    // As with inputs, check the values for the output operation and output tensor
    //std::cout << "Output: " << TF_OperationName(output_op) << "\n";
    //std::cout << "Output info: " << TF_Dim(output_value, 0) << "\n";
    // ######################
    // Run graph
    // ######################
//    fprintf(stdout, "Running session...\n");
//    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
//    TF_Session* session = TF_NewSession(graph, sess_opts, status);
//    assert(TF_GetCode(status) == TF_OK);
    // Call TF_SessionRun
    TF_SessionRun(session, nullptr,
                &inputs[0], &input_values[0], inputs.size(),
                &outputs[0], &output_values[0], outputs.size(),
                nullptr, 0, nullptr, status);

    // Assign the values from the output tensor to a variable and iterate over them
    float* out_vals = static_cast<float*>(TF_TensorData(output_values[0]));
    for (int i = 0; i < ouput_size; ++i)
    {
        if((i+1)%11==0)std::cout << "i+1=" << i+1 << "\n";
        float dum = out_vals[i] ;
        std::cout << "Output values info: " << dum << "\n";
        hit_vec.push_back(dum);
    }

    //fprintf(stdout, "Successfully run session\n");

    // Delete variables
    //TF_CloseSession(session, status);
    //TF_DeleteSession(session, status);
    //TF_DeleteSessionOptions(sess_opts);
    //TF_DeleteImportGraphDefOptions(graph_opts);
    //TF_DeleteGraph(graph);
    //TF_DeleteStatus(status);
    return 1;
}
*/
