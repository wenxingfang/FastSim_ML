
#ifndef GAN_EMC
#define GAN_EMC 1

#include "GaudiKernel/MsgStream.h"
#include "GaudiKernel/AlgFactory.h"
#include "GaudiKernel/ISvcLocator.h"
#include "GaudiKernel/SmartDataPtr.h"
#include "GaudiKernel/IJobOptionsSvc.h"
#include "GaudiKernel/Service.h"
#include "GaudiKernel/ThreadGaudi.h"
#include "GaudiKernel/Algorithm.h"
#include "GaudiKernel/PropertyMgr.h"

#include "EmcRecEventModel/RecEmcEventModel.h"
#include "EmcRec/EmcRecDigit2Hit.h"
#include "EmcRec/EmcRecHit2Cluster.h"
#include "EmcRec/EmcRecCluster2ShowerAbs.h"

using namespace Event;

void ProduceHitMap(SmartDataPtr<EmcMcHitCol> emcMcHitCol,  RecEmcHitMap& fHitMap);

#endif
