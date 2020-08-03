#include "NNPred.h"
#include <torch/script.h> // One-stop header.
#include <ATen/Parallel.h> // One-stop header.
#include <iostream>
#include <memory>

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

struct Magic {
    Magic(){
        module = torch::jit::load("/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/job_sub/pyTorch_job/old/my_module_model.pt");
           }
    torch::jit::script::Module module;
};


NNPred::NNPred(char s_model[])
{
  m_name = std::string(s_model);
  std::cout <<"model name="<<m_name<<std::endl;
 /*// the m_magic still has run time error
  try {
      m_magic = new Magic() ;
      }
  catch (...) {
        throw "error loading the model";
              }
  std::cout << "ok\n";
  */
}

NNPred::~NNPred()
{
}

float NNPred::get(int type, float mom, float costheta, float scale, float shift)//mom in GeV, theta from 0-pi
{

//clock_t t0, t1;
//t0 = clock();

//  at::init_num_threads(); // not good
auto options = torch::TensorOptions().dtype(torch::kFloat32);
//auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 1);//seems gives error
//std::vector<at::Tensor> inputs_vec;
std::vector<float> tmp_vec;
tmp_vec.push_back(mom);
tmp_vec.push_back(costheta);
tmp_vec.push_back(gaussrand());
at::Tensor input_ = torch::from_blob(tmp_vec.data(), {1,3}, options);
/*
float in_array[] = {16.0/20, (0.11*180/3.1415926535898)/180, 0.5, 16.0/20, (0.12*180/3.1415926535898)/180, 0.5, 16.0/20, (0.13*180/3.1415926535898)/180, 0.5};
std::vector<float> tmp_vec;
tmp_vec.push_back(16.0/20);
tmp_vec.push_back((0.11*180/3.1415926535898)/180);
tmp_vec.push_back(0.5);
tmp_vec.push_back(16.0/20);
tmp_vec.push_back((0.12*180/3.1415926535898)/180);
tmp_vec.push_back(0.5);
tmp_vec.push_back(16.0/20);
tmp_vec.push_back((0.13*180/3.1415926535898)/180);
tmp_vec.push_back(0.5);
*/
//t1 = clock();
//std::cout << "111 dt="<<(t1-t0) / (double) CLOCKS_PER_SEC<<",t0="<<t0<<",t1="<<t1<<std::endl;
//std::cout <<"input_ dim()"<<input_.dim()<<",itemsize()="<<input_.itemsize()<<",is_mkldnn="<<input_.is_mkldnn()<<","<< '\n';
//input_.print();
//std::cout <<"tensor="<< input_ << std::endl;

at::Tensor output;
if(type==1)
{   
    static torch::jit::script::Module   module_1 = torch::jit::load(m_name);
    torch::jit::IValue output0 = module_1.forward({input_});
    output = output0.toTensor()*scale + shift ;
}
else if(type==2) 
{
    static torch::jit::script::Module   module_2 = torch::jit::load(m_name);
    torch::jit::IValue output0 = module_2.forward({input_});
    output = output0.toTensor()*scale + shift ;
}
else if(type==3) 
{
    static torch::jit::script::Module   module_3 = torch::jit::load(m_name);
    torch::jit::IValue output0 = module_3.forward({input_});
    output = output0.toTensor()*scale + shift ;
}
else if(type==4) 
{
    static torch::jit::script::Module   module_4 = torch::jit::load(m_name);
    torch::jit::IValue output0 = module_4.forward({input_});
    output = output0.toTensor()*scale + shift ;
}
else if(type==5) 
{
    static torch::jit::script::Module   module_5 = torch::jit::load(m_name);
    torch::jit::IValue output0 = module_5.forward({input_});
    output = output0.toTensor()*scale + shift ;
}
else  
{
    static torch::jit::script::Module   module_6 = torch::jit::load(m_name);
    torch::jit::IValue output0 = module_6.forward({input_});
    output = output0.toTensor()*scale + shift ;
}
//std::cout<<"output tensor="<< output << std::endl;
std::vector<float> v ( output.data<float>(), output.data<float>() + output.numel() );
float res = v.at(0);
//std::cout<<"output="<<res<<std::endl;
return res;
}



std::vector<float>* NNPred::get(int type, float r, const std::vector<float>& v_theta, float scale)//r from 0-20 m, theta from 0-pi
{

//clock_t t0, t1;
//t0 = clock();

//  at::init_num_threads(); // not good
auto options = torch::TensorOptions().dtype(torch::kFloat32);
//auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 1);//seems gives error
int N_size = v_theta.size();
//std::vector<at::Tensor> inputs_vec;
std::vector<float> tmp_vec;
for(unsigned int i=0; i < N_size; i++)
{
    //float in_array[] = {r/20, (v_theta.at(i)*180/3.1415926535898)/180, gaussrand()};
    //inputs_vec.push_back(torch::from_blob(in_array, {3}, options).to(torch::Device("cuda:0")));
    tmp_vec.push_back(r/20.0);
    tmp_vec.push_back((v_theta.at(i)*180/3.1415926535898)/180);
    tmp_vec.push_back(gaussrand());
}
//at::Tensor input_ = torch::cat(inputs_vec);// not new dim
//at::Tensor input_ = torch::stack(inputs_vec);// new dim, works

/*
float in_array[] = {16.0/20, (0.11*180/3.1415926535898)/180, 0.5, 16.0/20, (0.12*180/3.1415926535898)/180, 0.5, 16.0/20, (0.13*180/3.1415926535898)/180, 0.5};
std::vector<float> tmp_vec;
tmp_vec.push_back(16.0/20);
tmp_vec.push_back((0.11*180/3.1415926535898)/180);
tmp_vec.push_back(0.5);
tmp_vec.push_back(16.0/20);
tmp_vec.push_back((0.12*180/3.1415926535898)/180);
tmp_vec.push_back(0.5);
tmp_vec.push_back(16.0/20);
tmp_vec.push_back((0.13*180/3.1415926535898)/180);
tmp_vec.push_back(0.5);
*/
at::Tensor input_ = torch::from_blob(tmp_vec.data(), {N_size,3}, options);
//t1 = clock();
//std::cout << "111 dt="<<(t1-t0) / (double) CLOCKS_PER_SEC<<",t0="<<t0<<",t1="<<t1<<std::endl;
//std::cout <<"input_ dim()"<<input_.dim()<<",itemsize()="<<input_.itemsize()<<",is_mkldnn="<<input_.is_mkldnn()<<","<< '\n';
//input_.print();
//std::cout <<"tensor="<< input_ << std::endl;

at::Tensor output;
if(type==1)
{   
    static torch::jit::script::Module   module_1 = torch::jit::load(m_name);
//t0 = clock();
//std::cout << "222 dt="<<(t0-t1) / (double) CLOCKS_PER_SEC<<",t0="<<t0<<",t1="<<t1<<std::endl;
    torch::jit::IValue output0 = module_1.forward({input_});
//t1 = clock();
//std::cout << "333 dt="<<(t1-t0) / (double) CLOCKS_PER_SEC<<",t0="<<t0<<",t1="<<t1<<std::endl;
    output = output0.toTensor()*scale;
//t0 = clock();
//std::cout << "444 dt="<<(t0-t1) / (double) CLOCKS_PER_SEC<<",t0="<<t0<<",t1="<<t1<<std::endl;
}
else if(type==2) 
{
    static torch::jit::script::Module   module_2 = torch::jit::load(m_name);
    torch::jit::IValue output0 = module_2.forward({input_});
    output = output0.toTensor()*scale;
}
else if(type==3) 
{
    static torch::jit::script::Module   module_3 = torch::jit::load(m_name);
    torch::jit::IValue output0 = module_3.forward({input_});
    output = output0.toTensor()*scale;
}
else 
{
    static torch::jit::script::Module   module_4 = torch::jit::load(m_name);
    torch::jit::IValue output0 = module_4.forward({input_});
    output = output0.toTensor()*scale;
}
std::vector<float>* v = new std::vector<float>( output.data<float>(), output.data<float>() + output.numel() );
//for(unsigned i=0; i < v->size(); i++){std::cout<<"i="<<i<<",v(i)="<<v->at(i)<<std::endl;}
return v;
}
