import torch.nn as nn
import math
import torch
from torch.autograd import Variable
from scipy.linalg import hadamard

class HadamardProj(nn.Module):

    def __init__(self, input_size, output_size, bias=True, fixed_weights=True, fixed_stepsize=None):
        super(HadamardProj, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        sz = 2 ** int(math.ceil(math.log(max(input_size, output_size), 2)))
        mat = torch.from_numpy(hadamard(sz))
        if fixed_weights:
            self.proj = Variable(mat, requires_grad=False)
        else:
            self.proj = nn.Parameter(mat)

        init_stepsize = 1. / math.sqrt(self.output_size)

        if fixed_stepsize is not None:
            self.stepsize = Variable(torch.Tensor(
                [fixed_stepsize]), requires_grad=False)
        else:
            self.stepsize = nn.Parameter(torch.Tensor([init_stepsize]))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(
                output_size).uniform_(-init_stepsize, init_stepsize))
        else:
            self.register_parameter('bias', None)

        self.eps = 1e-8

    def forward(self, x):
        if not isinstance(self.stepsize, nn.Parameter):
            self.stepsize = self.stepsize.type_as(x)
        x = x / (x.norm(2, -1, keepdim=True) + self.eps)
        w = self.proj.type_as(x)

        out = -self.stepsize * \
            nn.functional.linear(x, w[:self.output_size, :self.input_size])
        if self.bias is not None:
            out = out + self.bias.view(1, -1)
        return out


class Proj(nn.Module):

    def __init__(self, input_size, output_size, bias=True, init_stepsize=10):
        super(Proj, self).__init__()
        if init_stepsize is not None:
            self.weight = nn.Parameter(torch.Tensor(1).fill_(init_stepsize))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_size).fill_(0))
        self.proj = Variable(torch.Tensor(
            output_size, input_size), requires_grad=False)
        torch.manual_seed(123)
        nn.init.orthogonal(self.proj)

    def forward(self, x):
        w = self.proj.type_as(x)
        x = x / x.norm(2, -1, keepdim=True)
        out = nn.functional.linear(x, w)
        if hasattr(self, 'weight'):
            out = out * self.weight
        if hasattr(self, 'bias'):
            out = out + self.bias.view(1, -1)
        return out

class LinearFixed(nn.Linear):

    def __init__(self, input_size, output_size, bias=True, init_stepsize=10):
        super(LinearFixed, self).__init__(input_size, output_size, bias)
        self.stepsize = nn.Parameter(torch.Tensor(1).fill_(init_stepsize))

    def forward(self, x):
        w = self.weight / self.weight.norm(2, -1, keepdim=True)
        x = x / x.norm(2, -1, keepdim=True)
        out = nn.functional.linear(x, w, self.bias)
        return out
