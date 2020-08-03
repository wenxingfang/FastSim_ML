//
//  Convert from Cluster Map to Shower Map
//
//  Created by Zhe Wang 2004, 3, 29
//
#include <iostream>

#include "EmcRec/EmcRecCluster2Shower.h"
//#include "EmcRec/EmcRecSeedLocalMax.h"
#include "EmcRec/EmcRecSeedEThreshold.h"
#include "EmcRec/EmcRecSplitWeighted.h"

// Constructors and destructors
EmcRecCluster2Shower::EmcRecCluster2Shower()
{   
  //cout<<"====== EmcRec: Offline Mode ======"<<endl;
  //   fSeedFinder=new EmcRecSeedLocalMax;
  fSeedFinder=new EmcRecSeedEThreshold;
  fSplitter=new EmcRecSplitWeighted;
}

EmcRecCluster2Shower:: ~EmcRecCluster2Shower()
{
  delete fSeedFinder;
  delete fSplitter;
}

void EmcRecCluster2Shower::Convert(RecEmcClusterMap& aClusterMap,
    RecEmcShowerMap& aShowerMap)
{
  RecEmcClusterMap::iterator ciClusterMap;

  RecEmcIDVector aMaxVec;
//std::cout << "Convert start"<<" aClusterMap size="<<aClusterMap.size() << std::endl;
  for(ciClusterMap=aClusterMap.begin();
      ciClusterMap!=aClusterMap.end();
      ++ciClusterMap)
  {
//std::cout <<"Convert 1" << std::endl;
    //++++++++++++++++++++++++++
    //get its local maximum list
    fSeedFinder->Seed(ciClusterMap->second,aMaxVec);
//std::cout <<"Convert 2" << std::endl;
    //++++++++++++++++++++++++++++++++++++++++++++++
    //put seeds to cluster
    if(!aMaxVec.empty()) {
//std::cout <<"Convert 21" << std::endl;
      ci_RecEmcIDVector ciMax;
      for(ciMax=aMaxVec.begin();
          ciMax!=aMaxVec.end();
          ++ciMax) {
//std::cout <<"Convert 211" << std::endl;
        ciClusterMap->second.InsertSeed(ciClusterMap->second.Find(*ciMax)->second);
      }
//std::cout <<"Convert 22" << std::endl;
    }
    //++++++++++++++++++++++++++++++++++++++++++++++
    //split it into showers and push into shower map
//std::cout <<"Convert 3, aMaxVec.size()="<<aMaxVec.size()<<","<< std::endl;
    fSplitter->Split(ciClusterMap->second,aMaxVec,aShowerMap);
//std::cout <<"Convert 4"<< std::endl;
  }
//std::cout << "Convert end" << std::endl;
}

