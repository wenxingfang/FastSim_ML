#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <map> 
using namespace std;


//void read_SMIJK_ID_x_y_z(const string& file, map<int, map<int, map<int, map<int, map<int, int [4]> > > > > &  Map)
//void read_SMIJK_ID_x_y_z(const string& file, map<int, map<int, map<int, map<int, map<int, IDXYZ> > > > > &  Map)
void read_SMIJK_ID_x_y_z(const string& file, map<int, map<int, map<int, map<int, map<int, string> > > > > &  Map)
{

cout<<"start read_SMIJK_ID_x_y_z"<<"\n";
ifstream inf;
inf.open(file);

string sline;//每一行
string out;
string s1,s2,s3,s4,s5,s6,s7,s8,s9;
int ID_S,ID_M,ID_I,ID_J,ID_K, ID, x, y, z;
while(getline(inf,sline))
{
istringstream sin(sline);
sin>>s1>>s2>>s3>>s4>>s5>>s6>>s7>>s8>>s9;
//cout<<s1<<" "<<s2<<" "<<s3<<" "<<s4<<" "<<s5<<" "<<s6<<" "<<s7<<" "<<s8<<" "<<s9<<"\n";
ID_S  = atoi( s1.c_str() );
ID_M  = atoi( s2.c_str() );
ID_I  = atoi( s3.c_str() );
ID_J  = atoi( s4.c_str() );
ID_K  = atoi( s5.c_str() );
//ID    = atoi( s6.c_str() );
//x     = atoi( s7.c_str() );
//y     = atoi( s8.c_str() );
//z     = atoi( s9.c_str() );
//cout<<x<<" "<<y<<" "<<z<<" "<<ID<<" "<<"\n";
//Map.insert(make_pair(x,make_pair(y,make_pair(z,ID))));
//int tmp[4]={ID, x, y, z};
//if(!Map[ID_S][ID_M][ID_I][ID_J][ID_K]) Map[ID_S][ID_M][ID_I][ID_J][ID_K] =tmp;
Map[ID_S][ID_M][ID_I][ID_J][ID_K] = (s6+"_"+s7+"_"+s8+"_"+s9);
}

}
