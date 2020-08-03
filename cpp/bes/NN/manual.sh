c++ -c -I"./" -I"/junofs/users/wxfang/pytorch/torch/include"   -pipe   -W -Wall -Wwrite-strings -Wpointer-arith -Woverloaded-virtual  -std=c++14  -fPIC  -D_GNU_SOURCE -o ./NNPred.o     ./NNPred.cc 
c++ -fPIC -shared -Wl,--no-undefined -L/junofs/users/wxfang/pytorch/torch/lib -ltorch -lc10 -ltorch_cpu -o ./libNNPred.so  ./NNPred.o 

#c++ -c -I"./" -I"/junofs/users/wxfang/libtorch-cxx11-abi-shared-with-deps-latest/libtorch/include"   -pipe   -W -Wall -Wwrite-strings -Wpointer-arith -Woverloaded-virtual  -std=c++14  -fPIC  -D_GNU_SOURCE -o ./NNPred.o     ./NNPred.cc 
#c++ -fPIC -shared -Wl,--no-undefined -L/junofs/users/wxfang/libtorch-cxx11-abi-shared-with-deps-latest/libtorch/lib -ltorch -lc10 -ltorch_cpu -o ./libNNPred.so  ./NNPred.o 
