#ifndef myUTILs
#define myUTILs 1

#include <string>
#include <cstring>
#include <vector>
#include <sstream>
using namespace std;

float linear(float x, float x0, float y0, float x1, float y1);
int partition(float phi);
float getTheta(float x, float y);
float getPhi(float x, float y);
bool beforeECAL(float x, float y, float distance);

int getHitPoint(const int& charge, const float & x0, const float & y0, const float & z0, const float & px, const float & py, const float &pz, const float & B, const float & Endx, const float & Endy, const float & Hitx, float& Hity, float& Hitz, float& theta, float& phi, float& rotated);


int split(vector<string>& res, const string& str, const string& delim) ;
#endif
