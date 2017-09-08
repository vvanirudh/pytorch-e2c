'''
Model script

Author: Anirudh Vemula
Date: 7th September, 2017
'''
import torch.nn as nn
from torch.autograd import Variable
import torch


class LinearO(nn.Linear):
    def __init__(self, inputSize, outputSize):
        super(LinearO, self).__init__(inputSize, outputSize)
        self.reset_parameters()

    def reset_parameters(self):
        initScale = 1.1

        M1 = Variable(torch.randn(self.weight.size(0), self.weight.size(0)))
        M2 = Variable(torch.randn(self.weight.size(1), self.weight.size(1)))

        n_min = min(self.weight.size(0), self.weight.size(1))

        # QR decompositions of random matrices ~ N(0, 1)
        Q1, R1 = torch.qr(M1)
        Q2, R2 = torch.qr(M2)

        self.weight.copy_(Q1.narrow(1, 0, n_min) * Q2.narrow(0, 0, n_min) * initScale)
        self.bias.zero_()


class AddCons(nn.Module):

    def __init__(self, constant_scalar, ip):
        super(AddCons, self).__init__()
        self.constant_scalar = constant_scalar

        self.inplace = ip or False

    def forward(self, inp):
        if self.inplace:
            inp.add_(self.constant_scalar)
            self.output = inp
        else:
            self.output.resizeAs_(inp)
            self.output.copy_(inp)
            self.output.add_(self.constant_scalar)

        return self.output

    
class Reparametrize(nn.Module):

    def __init__(self, dimension):
        super(Reparametrize, self).__init__()
        self.dimension = dimension

    def forward(self, inp):
        self.eps = torch.randn(inp[1].size(0), self.dimension)
        self.output = torch.exp(torch.mul(inp[1], 0.5)) * self.eps

        self.output = self.output + inp[0]

        return self.output
