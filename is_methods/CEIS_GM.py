"""
********************************************************************************
Cross entropy-based importance sampling with Gaussian Mixture
********************************************************************************
Inspired by the numpy implementation from
Sebastian Geyer (s.geyer@tum.de)
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
https://github.com/MatthiasWiller/ERA-software
********************************************************************************
Args:
    Ns          : number of samples
    rho         : sample quantile
    limit_func  : limit state function
    D           : problem dimension
    K           : number of components in the mixture model
********************************************************************************
Returns:
    Pf              : probability of failure
    (pi, mu, sig)   : converged EM responsibilities, means, and covariances
    samples         : object to store samples at each level
********************************************************************************
"""
import torch
from torch.distributions import (
    Categorical, MixtureSameFamily, MultivariateNormal
)

from is_methods.EMGM import EMGM

def CEIS_GM(Ns, rho, limit_func, D, K):
    # Initialize values
    l = 0       # initial level
    max_l = 100 # max number of levels
    
    # prior parameters
    mu0 = torch.zeros(D)
    sig0 = torch.eye(D).repeat(K,1,1)

    # structures to hold output data
    xi_hat = torch.zeros(max_l+1)
    samples  = list()

    # CE procedure
    # initial values
    xi_hat[l] = 1.0
    mu  = mu0
    sig = sig0
    pi  = torch.ones(K) / K

    I = torch.empty(Ns)
    X = torch.zeros((Ns, D))
    geval = torch.zeros(Ns)

    # iterate
    for l in range(max_l):
        # generate samples
        gm = MixtureSameFamily(
                mixture_distribution=Categorical(probs=pi),
                component_distribution=MultivariateNormal(mu, sig))

        X = gm.sample((Ns,))
        samples.append(X)

        # evaluate the limit state function
        geval = limit_func(X)

        # calculate h for the likelihood ratio
        h = pi * MultivariateNormal(mu, sig).log_prob(X[:,None,:]).exp()
        h = h.sum(dim=1)

        # check for convergence
        #if xi_hat[l] == 0:
        if xi_hat[l] == 0:
            break

        # compute xi
        xi_hat[l+1] = torch.maximum(torch.tensor(0), torch.quantile(geval, rho))

        # Indicator function
        I = (geval <= xi_hat[l+1])

        # calculate likelihood ratio
        W = MultivariateNormal(torch.zeros(D), torch.eye(D)).log_prob(X).exp()/h
        # parameter update: EM algorithm
        [mu, sig, pi] = EMGM(X[I,:], W[I], K)


    # compute probability of failure
    W_f = MultivariateNormal(torch.zeros(D), torch.eye(D)).log_prob(X).exp()/h
    I_f = (geval <= 0)
    Pf = 1/Ns*sum(I_f*W_f)
    
    return [Pf, (pi, mu, sig), samples]