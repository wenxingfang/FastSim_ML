#include "my_utils.h"
#include <iostream>
#include <math.h>


#ifndef PI 
#define PI acos(-1)
#endif

float linear(float x, float x1, float y1, float x2, float y2)
{
    
    if(x1==x2) throw "x1 == x2 !";
    return ((y2-y1)/(x2-x1))*x + (y1*x2-y2*x1)/(x2-x1);

}

int partition(float phi) // phi should < 360 and > 0
{   if(phi<0) throw "Wrong input phi!";
    if(phi<=22.5 || phi > (360-22.5)) return 1;
    else if(22.5       <phi && phi<=(22.5+1*45)) return 2;
    else if((22.5+1*45)<phi && phi<=(22.5+2*45)) return 3;
    else if((22.5+2*45)<phi && phi<=(22.5+3*45)) return 4;
    else if((22.5+3*45)<phi && phi<=(22.5+4*45)) return 5;
    else if((22.5+4*45)<phi && phi<=(22.5+5*45)) return 6;
    else if((22.5+5*45)<phi && phi<=(22.5+6*45)) return 7;
    else if((22.5+6*45)<phi && phi<=(22.5+7*45)) return 8;
    else{std::cout<<"something wrong"<<std::endl;return -1;}
}

float getPhi(float x, float y)
{
    if     (x==0 && y>0) return 90;
    else if(x==0 && y<0) return 270;
    else if(x==0 && y==0) return 0;
    float phi = atan(y/x)*180/PI;
    if                 (x<0) phi = phi + 180;
    else if     (x>0 && y<0) phi = phi + 360;
    return phi;
}



bool beforeECAL(float x, float y, float distance)
{
    if(x==0) 
    {
        if(fabs(y)<distance)return true;
        else return false;
    }
    float phi = getPhi(x,y);
    int part = partition(phi);
    if(part==1 || part==5)
    {
        if (fabs(x)<distance)return true;
        else return false;
    }
    else if(part==3 || part==7)
    {
        if (fabs(y)<distance)return true;
        else return false;
    }
    else if(part==2)
    {
        float x1 = distance;   
        float y1 = distance*tan(22.5*PI/180);   
        float x2 = distance*tan(22.5*PI/180);   
        float y2 = distance;   
        float yp = linear(x, x1, y1, x2, y2);
        if (y < yp) return true;
        else return false;
    }
    else if(part==4)
    {
        float x1 = -distance;   
        float y1 = distance*tan(22.5*PI/180);   
        float x2 = -distance*tan(22.5*PI/180);   
        float y2 = distance;   
        float yp = linear(x, x1, y1, x2, y2);
        if (y < yp) return true;
        else return false;
    }
    else if(part==6)
    {
        float x1 = -distance;   
        float y1 = -distance*tan(22.5*PI/180);   
        float x2 = -distance*tan(22.5*PI/180);   
        float y2 = -distance;   
        float yp = linear(x, x1, y1, x2, y2);
        if (y > yp) return true;
        else return false;
    }
    else if(part==8)
    {
        float x1 =  distance;   
        float y1 = -distance*tan(22.5*PI/180);   
        float x2 =  distance*tan(22.5*PI/180);   
        float y2 = -distance;   
        float yp = linear(x, x1, y1, x2, y2);
        if (y > yp) return true;
        else return false;
    }
    else throw "Wrong in beforeECAL!";

}

int getHitPoint(const int& charge, const float & x0, const float & y0, const float & z0, const float & px, const float & py, const float &pz, const float & B, const float & Endx, const float & Endy, const float & Hitx, float& Hity, float& Hitz, float& theta, float& phi, float& rotated)
{
  
    float end_phi = getPhi(Endx, Endy);
    float phi0    = getPhi(x0, y0);
    float phi_p0  = getPhi(px, py);
    int part = partition(end_phi); // phi should < 360 and > 0
    rotated = (part-1)*45;
    float r = sqrt(x0*x0 + y0*y0);
    float pt = sqrt(px*px + py*py);
    //std::cout<<"phi0="<<phi0<<",phi_p0="<<phi_p0<<std::endl;
    float new_x0 = r*cos((phi0-rotated)*PI/180);
    float new_y0 = r*sin((phi0-rotated)*PI/180);
    if(x0==0 && y0==0)
    {
        new_x0=x0;
        new_y0=y0;
    }
    float new_px = pt*cos((phi_p0-rotated)*PI/180);
    float new_py = pt*sin((phi_p0-rotated)*PI/180);
    //std::cout<<"px="<<px<<",py="<<py<<",new_px="<<new_px<<",new_py="<<new_py<<std::endl;
    if(charge!=0){
    float radius = 1000*pt/(fabs(charge)*B*0.3);// to mm
    float x_c = 0;
    float y_c = 0;
    theta = atan(pt/pz)*180/PI;
    if(theta<0) theta = theta + 180;
    if(charge > 0)
    {
        x_c = new_py>0 ? new_x0 + radius*fabs(new_py)/pt : new_x0 - radius*fabs(new_py)/pt ; 
        y_c = (new_px/new_py)*(new_x0-x_c) +  new_y0; 
        if(radius <= fabs(Hitx-x_c)) return 0;
        Hity = y_c + sqrt(radius*radius-(Hitx-x_c)*(Hitx-x_c));
        phi = atan((x_c-Hitx)/(Hity-y_c))*180/PI;
    }
    else 
    {
        x_c = new_py>0 ? new_x0 - radius*fabs(new_py)/pt : new_x0 + radius*fabs(new_py)/pt ; 
        y_c = (new_px/new_py)*(new_x0-x_c) +  new_y0; 
        if(radius <= fabs(Hitx-x_c)) return 0;
        Hity = y_c - sqrt(radius*radius-(Hitx-x_c)*(Hitx-x_c));
        //std::cout<<"Hity="<<Hity<<",x_c="<<x_c<<",y_c="<<y_c<<",rad="<<radius<<",new_x0="<<new_x0<<",new_y0="<<new_y0<<",new_px="<<new_px<<",new_py="<<new_py<<std::endl;
        phi = atan((Hitx-x_c)/(y_c-Hity))*180/PI;
    }
    float cosPhi = ((new_x0-x_c)*(Hitx-x_c)+(new_y0-y_c)*(Hity-y_c))/(radius*radius);
    Hitz = z0 + pz*acos(cosPhi)*radius/pt; 
    return 1;
    }// charge!=0
    else
    {
        Hity = y0 + new_py*(Hitx-x0)/new_px;
        Hitz = z0 + pz    *(Hitx-x0)/new_px;
        theta = atan(pt/pz)*180/PI;
        //phi   = atan(py/px)*180/PI;
        phi   = atan(new_py/new_px)*180/PI;
        return 1;
    }
}



int split(vector<string>& res, const string& str, const string& delim) {
    if("" == str) return 0;  
    //先将要切割的字符串从string类型转换为char*类型  
    char* strs = new char[str.length() + 1] ;
    strcpy(strs, str.c_str());   
    char* d = new char[delim.length() + 1];  
    strcpy(d, delim.c_str());  
    char* p = strtok(strs, d);  
    while(p) 
    {  
        string s = p; //分割得到的字符串转换为string类型  
        res.push_back(s); //存入结果数组 
        p = strtok(NULL, d);
    }
    delete [] strs;  
    delete [] d;  
    return 1;  
    
}
