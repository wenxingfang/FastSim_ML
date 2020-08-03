#ifndef NNPred_h
#define NNPred_h

#include<string>
#include<vector>

struct Magic;

class NNPred
{
public: 
    //NNPred(std::string s_model) ;
    NNPred(char s_model[]);
    //NNPred() ;
    ~NNPred();
    float get(float pt, float theta);
    float get(int type, float r, float theta, float noise, float scale);
    std::vector<float>* get(int type, float r, const std::vector<float>& v_theta, float scale);//r from 0-20 m, theta from 0-pi
    //int get(int type, float r, const std::vector<float>& v_theta, float scale);//r from 0-20 m, theta from 0-pi
private:
    std::string m_name;
    // torch::jit::script::Module* m_module;
    Magic* m_magic;
};
#endif
