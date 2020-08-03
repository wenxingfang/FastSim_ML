#c++ -c -I"./" -I"/hpcfs/cepc/higgs/wxfang/JUNO/torch/include"  -pipe   -W -Wall -Wwrite-strings -Wpointer-arith -Woverloaded-virtual  -std=c++14  -fPIC  -D_GNU_SOURCE -o ./NNPred.o     ./NNPred.cc 
#c++ -fPIC -shared -Wl,--no-undefined -L/hpcfs/cepc/higgs/wxfang/JUNO/torch/lib -ltorch -lc10 -ltorch_cpu -o ./libNNPred.so  ./NNPred.o 

c++ -c -D_GLIBCXX_USE_CXX11_ABI=0 -I"./" -I"/hpcfs/juno/junogpu/fangwx/mini_conda/envs/pyTorch_v1p2_py37/lib/python3.7/site-packages/torch/include/" -I"/hpcfs/juno/junogpu/fangwx/mini_conda/envs/pyTorch_v1p2_py37/lib/python3.7/site-packages/torch/include/torch/csrc/api/include/"  -pipe   -W -Wall -Wwrite-strings -Wpointer-arith -Woverloaded-virtual  -std=c++14  -fPIC  -D_GNU_SOURCE -o ./NNPred.o     ./NNPred.cc 
c++ -fPIC -shared -Wl,--no-undefined -L/hpcfs/juno/junogpu/fangwx/mini_conda/envs/pyTorch_v1p2_py37/lib/python3.7/site-packages/torch/lib/ -ltorch -lc10 -lc10_cuda -o ./libNNPred.so  ./NNPred.o 

#c++ -c -I"./" -I"/hpcfs/cepc/higgs/wxfang/JUNO/libtorch/include"   -pipe   -W -Wall -Wwrite-strings -Wpointer-arith -Woverloaded-virtual  -std=c++14  -fPIC  -D_GNU_SOURCE -o ./NNPred.o     ./NNPred.cc 
#c++ -fPIC -shared -Wl,--no-undefined -L/hpcfs/cepc/higgs/wxfang/JUNO/libtorch/lib -ltorch -lc10 -ltorch_cpu -o ./libNNPred.so  ./NNPred.o 
