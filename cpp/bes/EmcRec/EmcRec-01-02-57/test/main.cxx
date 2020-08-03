
#include "EmcRec/EmcRecNeighbor.h"

int main()
{
   EmcRecID id;
   unsigned int module,theta,phi;
   module=EmcID::getENDCAP_WEST();
   theta=2;
   phi=24;
   
   id=EmcID::crystal_id(module,theta,phi);
   cout<<id<<"\t"
     <<EmcID::barrel_ec(id)<<"\t"
     <<EmcID::theta_module(id)<<"\t"
     <<EmcID::phi_module(id)<<endl;
   cout<<endl;
//   cout<<"aaaaaaaaaaaaaa"<<endl;
   
   EmcRecIDVector neighbor=EmcRecNeighbor::GetNeighbors(id);
//   cout<<"bbbbbbbbbbbbbb"<<endl;
   
   EmcRecIDVector nextNeighbor=EmcRecNeighbor::GetNextNeighbors(id);
//   cout<<"cccccccccccccc"<<endl;
   
   EmcRecIDVector::iterator iN;
   for(iN=neighbor.begin();
       iN!=neighbor.end();
       ++iN) {
      cout<<*iN<<"\t"
        <<EmcID::barrel_ec(*iN)<<"\t"
        <<EmcID::theta_module(*iN)<<"\t"
        <<EmcID::phi_module(*iN)<<endl;
      
   }
   cout<<endl;
   
   for(iN=nextNeighbor.begin();
       iN!=nextNeighbor.end();
       ++iN) {
       cout<<*iN<<"\t"
        <<EmcID::barrel_ec(*iN)<<"\t"
        <<EmcID::theta_module(*iN)<<"\t"
        <<EmcID::phi_module(*iN)<<endl;
      //cout<<*iN<<endl;
   }
   cout<<endl;
}

