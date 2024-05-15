"""
********************************************************************************
Sequential importance sampling
********************************************************************************
Inspired by the numpy implementation from
Iason Papaioannou (iason.papaioannou@tum.de)
Matthias Willer (matthias.willer@tum.de)
Implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
https://github.com/MatthiasWiller/ERA-software
********************************************************************************
Args:
    Ns          : Number of samples
    rho         : chain-to-seeds ratio
    limit_func  : limit state function
    D           : problem dimension
    K           : number of components in the Gaussian mixture model
********************************************************************************
Returns:
    Pf              : probability of failure
    (pi, mu, sig)   : converged EM responsibilities, means, and covariances
    samples         : object to store samples at each level
********************************************************************************
"""
import scipy as sp
import torch
from torch.distributions import (
    Categorical, MixtureSameFamily, MultivariateNormal, Normal
)

from is_methods.EMGM import EMGM

def SIS_GM(Ns, rho, limit_func, D, K):
    # initialize values
    max_m   = 100       # estimated number of iterations
    m       = 0         # counter for number of levels
    samples = list()    # space for samples in U-space

    # SIS parameters
    nc          = int(Ns*rho)   # number of Markov chains
    lenchain    = int(Ns/nc)    # number of samples per Markov chain
    burn_in     = 5             # burn-in period
    tolCOV      = 1.5           # tolerance of COV of weights  

    # initialize samples
    Sk      = torch.ones(max_m)           # space for expected weights
    sigmak  = torch.zeros(max_m)          # space for sigmak

    # Step 1
    # perform the first Monte Carlo simulation
    uk = MultivariateNormal(torch.zeros(D), torch.eye(D)).sample((Ns,))
    gk   = limit_func(uk)
    samples.append(uk.T)
    
    # set initial subset and failure level
    gmu = gk.mean()
    sigmak[m] = 50*gmu

    # iterate
    p = Normal(0.0, 1.0)
    for m in range(max_m):
        # Step 2 and 3
        # compute sigma and weights
        if m == 0:
            func = lambda x: torch.abs(
                (p.cdf(-gk/x)).std()/(p.cdf(-gk/x)).mean()-tolCOV)
            sigma2      = sp.optimize.fminbound(func, 0, float(10.0*gmu))
            sigmak[m+1] = sigma2
            wk          = p.cdf(-gk/sigmak[m+1])
        else:
            func = lambda x: torch.abs(
                (p.cdf(-gk/x)/p.cdf(-gk/sigmak[m])).std() / \
                    (p.cdf(-gk/x)/p.cdf(-gk/sigmak[m])).mean()-tolCOV)
            sigma2      = sp.optimize.fminbound(func, 0, float(sigmak[m]))
            sigmak[m+1] = sigma2
            wk          = p.cdf(-gk/sigmak[m+1])/p.cdf(-gk/sigmak[m])

        # Step 4
        # compute estimate of expected w 
        Sk[m] = wk.mean()
        # exit algorithm if no convergence is achieved
        if Sk[m] == 0: break
        # normalized weights
        wk = wk/Sk[m]/Ns
        # fit Gaussian Mixture
        [mu, sig, pi] = EMGM(uk, wk, K)
        gm = MixtureSameFamily(
                mixture_distribution=Categorical(probs=pi),
                component_distribution=MultivariateNormal(mu, sig)) 

        # Step 5
        # resample
        ind = torch.multinomial(wk, nc, replacement=True)
        # seeds for chains
        gk0 = gk[ind]
        uk0 = uk[ind,:]

        # Step 6
        # perform M-H
        count = 0
        # delete previous samples
        gk = torch.zeros(Ns)
        uk = torch.zeros((Ns,D))
      
        for k in range(nc):
            # set seed for chain
            u0 = uk0[k,:]
            g0 = gk0[k]

            for i in range(lenchain+burn_in):  
                if i == burn_in:
                    count = count-burn_in
                        
                # get candidate sample from conditional normal distribution
                v = gm.sample() 
                # Evaluate limit-state function              
                gv = limit_func(v[:,None].T) 

                # compute acceptance probability
                pdfn = gm.log_prob(u0).exp()
                pdfd = gm.log_prob(v).exp()

                ratio = p.cdf(-gv/sigmak[m+1])*torch.prod(p.cdf(v))*pdfn / \
                    p.cdf(-g0/sigmak[m+1])/torch.prod(p.log_prob(u0).exp())/pdfd
                alpha_t = torch.min(torch.tensor([1, ratio]))
                # check if sample is accepted
                unif_val = torch.rand(1)
                if unif_val <= alpha_t:
                    uk[count,:] = v
                    gk[count]   = gv
                    u0 = v
                    g0 = gv
                else:
                    uk[count,:] = u0
                    gk[count]   = g0

                count = count+1
        
        samples.append(uk)

        if sigmak[m+1] == 0:
            COV_Sl = torch.nan
        else:
            COV_Sl = ((gk < 0)/p.cdf(-gk/sigmak[m+1])).std() / \
                ((gk < 0)/p.cdf(-gk/sigmak[m+1])).mean()

        print('COV_Sl = {:2.4f}'.format(COV_Sl))
        if COV_Sl < 0.01: break

    # calculate probability of failure
    Prf = ((gk < 0)/p.cdf(-gk/sigmak[m+1])).mean()*torch.prod(Sk)
    M = m+1
    return [Prf, (pi, mu, sig), samples]