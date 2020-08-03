#include "GaudiKernel/MsgStream.h"
#include "GaudiKernel/AlgFactory.h"
#include "GaudiKernel/ISvcLocator.h"
#include "GaudiKernel/SmartDataPtr.h"
#include "GaudiKernel/PropertyMgr.h"
#include "GaudiKernel/IJobOptionsSvc.h"
#include "EventModel/EventHeader.h"
#include "EmcRawEvent/EmcDigi.h"
#include "EmcRecEventModel/RecEmcEventModel.h"
#include "McTruth/McParticle.h"
#include "McTruth/EmcMcHit.h"
#include "RawEvent/RawDataUtil.h"

#include "EmcRec/EmcRec.h"
#include "EmcRecGeoSvc/EmcRecGeoSvc.h"
#include "EmcRec/EmcRecParameter.h"
#include "EmcRec/EmcRecFastCluster2Shower.h"
#include "EmcRec/EmcRecCluster2Shower.h"
#include "EmcRec/EmcRecTofMatch.h"
#include "EmcRec/EmcRecTofDigitCalib.h"
//#include "EmcRec/EmcRecFindTofShower.h"
#include "EmcRec/EmcRecTDS.h"
#include "RawDataProviderSvc/RawDataProviderSvc.h"
#include "RawDataProviderSvc/EmcRawDataProvider.h"
// tianhl for mt
#include "GaudiKernel/Service.h"
#include "GaudiKernel/ThreadGaudi.h"
// tianhl for mt


using CLHEP::Hep3Vector;
#ifndef ENABLE_BACKWARDS_COMPATIBILITY
   typedef HepGeom::Point3D<double> HepPoint3D;
#endif

#include "EmcRec/GAN.h"

#include "Rest/cURL.h"
#include "Rest/object.h"

#include <iostream>
#include <fstream>
#include <math.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <vector>


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


int rest(const std::string& gan_url, const std::string& output_op, const std::vector< std::vector<double> >& in_info, std::vector< std::vector<double> >& out_info) {
    std::cout << "start do rest.." << std::endl;
    int Np=in_info.size();
    cJSON *root;
    root = cJSON_CreateObject();
    cJSON *js_body;
    cJSON_AddItemToObject(root,"instances", js_body = cJSON_CreateArray());
    for(int i=0;i<Np;i++) 
    {
        cJSON *js_list;
        cJSON_AddItemToArray(js_body, js_list = cJSON_CreateArray());
        for(int j=0; j<512; j++)
        {
            cJSON_AddItemToArray(js_list,cJSON_CreateNumber(gaussrand()));
        }
        cJSON_AddItemToArray(js_list,cJSON_CreateNumber((in_info.at(i)).at(0)));
        cJSON_AddItemToArray(js_list,cJSON_CreateNumber((in_info.at(i)).at(1)));
        cJSON_AddItemToArray(js_list,cJSON_CreateNumber((in_info.at(i)).at(2)/10));
        cJSON_AddItemToArray(js_list,cJSON_CreateNumber((in_info.at(i)).at(3)));
        cJSON_AddItemToArray(js_list,cJSON_CreateNumber((in_info.at(i)).at(4)));
    }
    
    char *s = cJSON_PrintUnformatted(root);
    //std::string url = "http://10.10.6.229:12345/v1/models/em_Low:predict"; 
    std::string url = gan_url; 
    Rest::cURL curlobj;
    if (curlobj.request(url, s)) {
        //std::cout << curlobj.result() << std::endl;
        cJSON* json = cJSON_Parse(curlobj.result().c_str());
        /*
        bool is_array = false;
        if (json->type==cJSON_Array) {
            std::cout << "It's array!" << std::endl;
            is_array = true;
        } else {
            std::cout << "It's object!" << std::endl;
        }
        */
        cJSON *js_list = cJSON_GetObjectItem(json, "predictions");
        int array_size = cJSON_GetArraySize(js_list);
        //printf("array size is %d\n",array_size);
        //std::cout <<js_list->type << std::endl;
        //printf("js type %d\n",js_list->type);
        for(int i=0; i< array_size; i++) {
            std::vector<double> evt_vec; 
            cJSON *evt = cJSON_GetArrayItem(js_list, i);
            //cJSON *e_list = cJSON_GetObjectItem(evt, "re_lu_5/Relu:0");
            cJSON *e_list = cJSON_GetObjectItem(evt, output_op.c_str());
            int nRow = cJSON_GetArraySize(e_list);
            for(int j=0; j< nRow; j++)
            {
                cJSON *col_list = cJSON_GetArrayItem(e_list, j);
                int nCol = cJSON_GetArraySize(col_list);
                for(int k=0; k< nCol; k++)
                {
                    cJSON *hit = cJSON_GetArrayItem(col_list, k);
                    cJSON *hit1 = cJSON_GetArrayItem(hit, 0);
                    evt_vec.push_back(hit1->valuedouble);
                    //printf("%d | ",hit1->type);
                    //printf("%f | ",hit1->valuedouble);
                }
     
            }
            out_info.push_back(evt_vec);
        }
        
    }

    printf(" Free memory now \n");
    if(s){
         //   printf(" %s \n",s);
            free(s);
        }
    if(root) cJSON_Delete(root);
    return 1;
}


void ProduceHitMap(SmartDataPtr<EmcMcHitCol> emcMcHitCol,  RecEmcHitMap& aHitMap)
{
    if (!emcMcHitCol) {
          //log << MSG::WARNING << "Could not find EMC truth" << endreq;
          std::cout << "Could not find EMC truth" << std::endl;
        }

    EmcRecParameter& Para=EmcRecParameter::GetInstance();
    RecEmcID mcId;
    unsigned int mcTrackIndex;
    double mcPosX=0,mcPosY=0,mcPosZ=0;
    double Mom=0,mcPx=0,mcPy=0,mcPz=0;
    double mcEnergy=0;

    std::cout << "EmcMcHitCol size" <<emcMcHitCol->size()<< std::endl;
    EmcMcHitCol::iterator iterMc;
    for(iterMc=emcMcHitCol->begin();iterMc!=emcMcHitCol->end();iterMc++)
    {
        std::vector<double> mc_vec;
        mcId=(*iterMc)->identify();
        int npart  = EmcID::barrel_ec   (mcId);
        int ntheta = EmcID::theta_module(mcId);// within 0-43
        int nphi   = EmcID::phi_module  (mcId);  //within 0-119
        std::cout<<"EmcMcHit npart="<<npart<<",ntheta="<<ntheta<<",nphi="<<nphi<<std::endl;
        /*
        mcPosX=(*iterMc)->getPositionX();
        mcPosY=(*iterMc)->getPositionY();
        mcPosZ=(*iterMc)->getPositionZ();
        mcPx=(*iterMc)->getPx();
        mcPy=(*iterMc)->getPy();
        mcPz=(*iterMc)->getPz();
        Mom = sqrt(mcPx*mcPx + mcPy*mcPy + mcPz*mcPz); 
        mc_vec.push_back(Mom);
        //mcEnergy=(*iterMc)->getDepositEnergy();
        //mcTrackIndex=(*iterMc)->getTrackIndex();

        RecEmcHit aHit;
        //aHit.CellId(aDigit.CellId());
        aHit.CellId(mcId);
        aHit.Energy(1);
        aHit.Time(0);
        //aHit.Energy(eout/GeV);
        //aHit.Time(aDigit.TDC());
        if(aHit.getEnergy()>=Para.ElectronicsNoiseLevel())  aHitMap[aHit.getCellId()]=aHit;
        */
    }
    //rest(const std::string& gan_url, const std::string& output_op, const std::vector< std::vector<double> >& in_info, std::vector< std::vector<double> >& out_info);
}
