#include "SLCIORdr.h"

#include <iostream>
#include <map>

using namespace std;

int main(int argc, char* argv[]){

for(int i=0;i<argc;i++) cout<<argv[i]<<endl;

std::string simFile = argv[1];
std::string outputFile    = argv[2];
std::string map_name    = argv[3];


SLCIORdr* reader  = new SLCIORdr(simFile, outputFile, map_name);

int N=0;
while(reader->mutate() && N<1000000){N++;}
//while(reader->mutate() && N<2){N++;cout<<"next event"<<endl;}

reader->finish();
cout<<"done"<<endl;
return 0;
}
