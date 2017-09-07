'''
Criterion script

Author: Anirudh Vemula
Date: 7th September, 2017
'''
import torch
from torch.autograd import Variable


def KLDCriterion(inp, target):
    sizeAverage = True
    sigma_squared = torch.exp(inp[1])
    mu_squared = torch.exp(inp[0])
    numElements = torch.numel(target)

    output = 1 + inp[1]
    output = output - mu_squared
    output = output - sigma_squared
    output = output / numElements
    output = 0.5 * torch.sum(output)

    return output

def KLDistCriterion(inp, target):
    
