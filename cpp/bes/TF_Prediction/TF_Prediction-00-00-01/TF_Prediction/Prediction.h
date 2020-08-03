#ifndef P_Prediction
#define P_Prediction 1
#include "GaudiKernel/Algorithm.h"
#include "GaudiKernel/NTuple.h"
#include "ReconEvent/ReconEvent.h"
#include "EmcRawEvent/EmcDigi.h"
#include "EmcRecEventModel/RecEmcEventModel.h"
#include "EmcRecGeoSvc/EmcRecGeoSvc.h"
#include "EmcRecGeoSvc/EmcRecBarrelGeo.h"

#include "EmcRec/EmcRecHit2Cluster.h"
#include "EmcRec/EmcRecCluster2ShowerAbs.h"

#include "RawEvent/RawDataUtil.h"
#include "RawDataProviderSvc/RawDataProviderSvc.h"
#include "RawDataProviderSvc/EmcRawDataProvider.h"
#include "EventModel/EventModel.h"
#include "EventModel/Event.h"
#include "EvtRecEvent/EvtRecEvent.h"
#include "EvtRecEvent/EvtRecTrack.h"
#include "EventModel/EventHeader.h"
#include "EventModel/EventHeader.h"

#include "tensorflow/c/c_api.h"
#include <vector>
using namespace std;
/////////////////////////////////////////////////////////////////////////////

class Prediction:public Algorithm {
public:
  Prediction (const std::string& name, ISvcLocator* pSvcLocator);
  StatusCode initialize();
  StatusCode execute();
  StatusCode finalize();
  void MakerHitMap(const vector<RecEmcID>& vec_id, const vector<float>& vec_e, RecEmcHitMap& aHitMap);
  void RemoveNoise(RecEmcHitMap& aHitMap, const float& ElectronicsNoiseLevel);
  int  predict(TF_Session* session, TF_Status* status, TF_Graph* graph, const vector<float>& mc_vector, const string& input_op_name, const string& output_op_name, vector<float>& hit_vec) ;
  StatusCode RegisterReconEvent();
  StatusCode RegisterHit(RecEmcHitMap& aHitMap);
  int getInfo(const IEmcRecGeoSvc* iGeoSvc, const double& x, const double& y, const double& z, const double& px, const double& py, const double& pz, const RecEmcCluster* clus, double& M_dtheta, double& M_dphi, double& P_dz, double& P_dphi, vector<RecEmcID>& vec_ID, bool& do_parity) ;
  
private:

  int m_event;
  int m_myInt;
  bool m_myBool;
  double m_myDouble;
  std::vector<std::string> m_myStringVec;


  TF_Buffer*                m_em_Low_graph_def ;
  TF_Graph*                 m_em_Low_graph ;
  TF_Status*                m_em_Low_status;
  TF_ImportGraphDefOptions* m_em_Low_graph_opts ;
  TF_Buffer*                m_em_High_graph_def ;
  TF_Graph*                 m_em_High_graph ;
  TF_Status*                m_em_High_status;
  TF_ImportGraphDefOptions* m_em_High_graph_opts ;
  TF_Buffer*                m_ep_Low_graph_def ;
  TF_Graph*                 m_ep_Low_graph ;
  TF_Status*                m_ep_Low_status;
  TF_ImportGraphDefOptions* m_ep_Low_graph_opts ;
  TF_Buffer*                m_ep_High_graph_def ;
  TF_Graph*                 m_ep_High_graph ;
  TF_Status*                m_ep_High_status;
  TF_ImportGraphDefOptions* m_ep_High_graph_opts ;

  TF_SessionOptions* m_em_Low_sess_opts ;
  TF_Session*        m_em_Low_session   ;
  TF_SessionOptions* m_em_High_sess_opts ;
  TF_Session*        m_em_High_session   ;
  TF_SessionOptions* m_ep_Low_sess_opts ;
  TF_Session*        m_ep_Low_session   ;
  TF_SessionOptions* m_ep_High_sess_opts ;
  TF_Session*        m_ep_High_session   ;

  RecEmcHitMap     fHitMap;
  RecEmcClusterMap fClusterMap;
  RecEmcShowerMap  fShowerMap;

   EmcRecHit2Cluster    fHit2Cluster;
   EmcRecCluster2ShowerAbs *fCluster2Shower;



  float m_ElectronicsNoiseLevel;
  string m_energyThreshold ;
  string m_em_Low_pb_name  ;
  string m_em_High_pb_name ;
  string m_ep_Low_pb_name  ;
  string m_ep_High_pb_name ;
  string m_input_op_name   ;
  string m_output_op_name_em  ;
  string m_output_op_name_ep  ;
  int m_compensate;

  NTuple::Tuple*  m_tuple1;
  NTuple::Item<double> m_EnSumHit ;


};
#endif
