# packages
import torch
from torch.distributions import MultivariateNormal


def cross_entropy_is(n_samples:int, rho:float, obj_func, model, flow, space):
    """
    Estimate the probability of failure using cross-entropy importance sampling.
    Inspired by the numpy implementation from the Engineering Risk Analysis 
    Group at Technische Universitat Munchen.
    https://github.com/MatthiasWiller/ERA-software

    [1] Geyer, S., Papaioannou, I., & Straub, D. (2019). Cross entropy-based 
        importance sampling using Gaussian densities revisited. Structural 
        Safety, 76, 15-27.

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

    # max number of intermediate failure domains
    max_steps = 50

    # rho-quantiles of intermediate failure domains; Eq. (14) [1]
    xi_hat = torch.zeros(max_steps+1)
    xi_hat[0] = 1.0

    # cross-entropy importance sampling
    geval = torch.zeros(n_samples)
    total_samples = 0
    for step in range(max_steps):
        # generate samples
        x = model.sample(n_samples)
        # evaluate the limit state function
        geval = obj_func(x)
        # calculate h for the likelihood ratio
        log_h = model.log_prob(x)
        # check for convergence
        if xi_hat[step] == 0:
            break
        # compute xi
        xi_hat[step+1] = torch.maximum(torch.tensor(0), torch.quantile(geval, rho))
        # indicator function
        I = (geval <= xi_hat[step+1])

        # calculate likelihood ratio
        if space == "latent":
            log_p = MultivariateNormal(torch.zeros(d), torch.eye(d)).log_prob(x)
        else:
            with torch.no_grad():
                log_p = flow.log_prob(x)

        log_weights = log_p - log_h
        weights = log_weights.exp()
        if any(weights < 1e-6):
            weights += 1e-6
        # parameter update: EM algorithm
        model.fit(x[I,:], weights[I], k)
        total_samples += n_samples

    if space == "latent":
        log_p = MultivariateNormal(torch.zeros(d), torch.eye(d)).log_prob(x)
    else:
        with torch.no_grad():
            log_p = flow.log_prob(x)

    log_weights = log_p - log_h
    weights = log_weights.exp()
    if any(weights < 1e-6):
        weights += 1e-6
    I = (geval <= 0)
    Pf = 1/n_samples*sum(I*weights)

    return [Pf, model, total_samples]