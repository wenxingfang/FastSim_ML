from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())

class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mm(input)
          #output = self.weight.mv(input)
          #output = 2*input
        else:
          #output = self.weight + input
          output = 2*input
        return output

my_module = MyModule(10,2)
sm = torch.jit.script(my_module)
sm.save("/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/job_sub/pyTorch_job/my_module_model.pt")
#my_module.save("/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/job_sub/pyTorch_job/my_module_model.pt")
print('saved pt')
