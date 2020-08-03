#ifndef NNPred_h
#define NNPred_h

#include<string>
#include<vector>

struct Magic;

class NNPred
{
public: 
    NNPred(char s_model[]);
    ~NNPred();
    std::vector<float>* get(int type, float r, const std::vector<float>& v_theta, float scale);//r from 0-20 m, theta from 0-pi
    float get(int type, float mom, float costheta, float scale, float shift);//mom in GeV, theta from 0-pi
private:
    std::string m_name;
    Magic* m_magic;
};
#endif
