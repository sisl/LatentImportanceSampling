import numpy as np
import scipy as sp
from scipy.stats import norm
import torch
from torch.distributions import MultivariateNormal, Normal


def sequential_is(n_samples:int, rho:float, obj_func, model, flow, space):
    """
    Estimate the probability of failure using sequential importance sampling.
    Inspired by the numpy implementation from the Engineering Risk Analysis 
    Group at Technische Universitat Munchen.
    https://github.com/MatthiasWiller/ERA-software

    [1] Papaioannou, I., Papadimitriou, C., & Straub, D. (2016). Sequential 
        importance sampling for structural reliability analysis. Structural 
        Safety, 62, 66-75.

    Args:
        n_samples (int): number of samples to draw at each intermediate failure distribution.
        rho (float): rho-quantile of intermediate failure domains.
        obj_func: the problem-specific objective function.
        model: GMM object.
    
    Returns:
        Pf (float): the estimate of the probability of failure.
        model: GMM object with updated parameters.
        total_samples (int): the total number of samples drawn during the IS process.
    """

    d = model.n_features
    k = model.n_components

    max_m   = 50       # max number of intermediate failure domains
    m       = 0         # counter for number of levels

    # SIS parameters
    nc          = int(n_samples*rho)   # number of Markov chains
    lenchain    = int(n_samples/nc)    # number of samples per Markov chain
    burn_in     = 5             # burn-in period
    tolCOV      = 1.5           # tolerance of COV of weights  

    # initialize samples
    Sk      = torch.ones(max_m)     # expected weights
    sigmak  = torch.zeros(max_m)    # sigmak

    # Step 1
    # perform the first Monte Carlo simulation
    if space == "latent":
        uk = MultivariateNormal(torch.zeros(d), torch.eye(d)).sample((n_samples,))
    else:
        with torch.no_grad():
            uk = flow.sample(n_samples)

    gk = obj_func(uk)
    
    wk = torch.zeros(n_samples)
    gv = torch.zeros(n_samples)

    # set initial subset and failure level
    gmu = gk.mean()
    sigmak[m] = 50*gmu

    p = Normal(0.0, 1.0)
    N_samples = 0
    for m in range(max_m):
        # Step 2 and 3
        # compute sigma and weights
        if m == 0:
            func = lambda x: np.abs(
                np.exp(norm.logcdf(-gk/x)).std() / \
                    np.exp(norm.logcdf(-gk/x)).mean() - tolCOV)
            sigma2      = sp.optimize.fminbound(func, 1e-6, float(10.0*gmu))
            sigmak[m+1] = sigma2
            log_wk      = norm.logcdf(-gk/sigmak[m+1])
            wk          = torch.tensor(log_wk, dtype=torch.float32).exp()
        else:
            func = lambda x: np.abs(
                np.exp(norm.logcdf(-gk/x) - norm.logcdf(-gk/sigmak[m])).std()/ \
                    np.exp(norm.logcdf(-gk/x) - norm.logcdf(-gk/sigmak[m])).mean() - tolCOV)
            sigma2      = sp.optimize.fminbound(func, 1e-6, float(sigmak[m]))
            sigmak[m+1] = sigma2
            log_wk      = norm.logcdf(-gk/sigmak[m+1]) - norm.logcdf(-gk/sigmak[m])
            wk          = torch.tensor(log_wk, dtype=torch.float32).exp()

        # Step 4
        # compute estimate of expected w 
        Sk[m] = wk.mean()
        # normalized weights
        wnork = wk/Sk[m]/n_samples
        # parameter update: EM algorithm
        model.fit(uk, wnork, k)

        # Step 5
        # resample
        ind = torch.multinomial(wnork, nc, replacement=True)
        # seeds for chains
        g0 = gk[ind]
        u0 = uk[ind,:]

        # Step 6
        # perform M-H
        count = 0
        gk_chains = torch.zeros((lenchain,nc))
        uk_chains = torch.zeros((lenchain,nc,d))
        for i in range(lenchain+burn_in):  
            if i == burn_in:
                count = count-burn_in
                        
            # get candidate sample from conditional normal distribution
            v = model.sample(nc) 
            N_samples += nc

            # Evaluate limit-state function              
            gv = obj_func(v)

            # compute acceptance probability (calculations in log-space for 
            # numerical stability)
            logpdfn = model.log_prob(u0)
            logpdfd = model.log_prob(v)

            num1 = torch.tensor(norm.logcdf(-gv/sigmak[m+1]), dtype=torch.float32)
            num2 = torch.tensor(norm.logcdf(v), dtype=torch.float32).sum(dim=1)
            num3 = logpdfn

            den1 = torch.tensor(norm.logcdf(-g0/sigmak[m+1]), dtype=torch.float32)
            den2 = p.log_prob(u0).sum(dim=1)
            den3 = logpdfd

            log_ratio = num1 + num2 + num3 - den1 - den2 - den3

            ratio = log_ratio.exp()
                
            alpha_t = torch.minimum(torch.ones(nc), ratio)

            # check if sample is accepted
            unif_val = torch.rand(nc)
            accepted = unif_val <= alpha_t
            accepted_idx = accepted.nonzero(as_tuple=True)[0]

            uk_chains[count, accepted_idx, :] = v[accepted]
            gk_chains[count, accepted_idx] = gv[accepted]
            u0[accepted] = v[accepted_idx]
            g0[accepted] = gv[accepted_idx]

            uk_chains[count, ~accepted, :] = u0[~accepted]
            gk_chains[count, ~accepted] = g0[~accepted]

            count = count+1
        
        uk_chains = torch.permute(uk_chains, (1,0,2))
        uk = uk_chains.reshape(-1, d)
        gk_chains = torch.permute(gk_chains, (1,0))
        gk = gk_chains.flatten()

        log_cdf = torch.tensor(norm.logcdf(-gk/sigmak[m+1]), dtype=torch.float32)
        COV_Sl_std = ((gk < 0)/log_cdf.exp()).std()
        COV_Sl_mean = ((gk < 0)/log_cdf.exp()).mean()
        COV_Sl = COV_Sl_std / COV_Sl_mean

        print('COV_Sl = {:2.4f}'.format(COV_Sl))
        if COV_Sl < 0.01: 
            break
 
    x = model.sample(n_samples)
    geval = obj_func(x)
    log_h = model.log_prob(x)
    if space == "latent":
        log_p = MultivariateNormal(torch.zeros(d), torch.eye(d)).log_prob(x)
    else:
        with torch.no_grad():
            log_p = flow.log_prob(x)

    log_weights = log_p - log_h
    weights = log_weights.exp()
    if any(weights < 1e-6):
        weights += 1e-6
    I = (geval <= 0.0)
    Pf = 1/n_samples*sum(I*weights)

    return [Pf, model, N_samples]