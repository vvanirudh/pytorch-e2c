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
    sizeAverage = False
    # Removed clone from torch7 code
    q_mu = inp[0, 0]
    q_sigma = torch.exp(inp[1, 0])
    p_mu = target[0, 0]
    p_sigma = torch.exp(target[1, 0])

    n_dim = torch.numel(q_mu)

    iqv = torch.ones(n_dim)
    iqv = iqv / q_sigma

    diff = q_mu - p_mu

    output = - (torch.sum(torch.log(q_sigma)) -
                torch.sum(torch.log(p_sigma)) +
                torch.sum(iqv * p_sigma) +
                torch.sum(torch.pow(diff, 2) * iqv) -
                n_dim) / 2

    return output
