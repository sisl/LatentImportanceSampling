"""
********************************************************************************
Perform soft EM algorithm for fitting the Gaussian mixture model
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
    X   : [Ns x D] tensor of data
    W   : [Ns] tensor of importance weights
    K   : number of Gaussians in the Mixture
********************************************************************************
Returns:
    mu : [K x D] tensor of means of Gaussians in the mixture
    si : [K x D x D] tensor of covariance matrices of Gaussians in the mixture
    pi : [K] tensor of responsibilities 
********************************************************************************
"""
import torch
from torch.distributions import MultivariateNormal

#*******************************************************************************
# Expectation Maximization
#*******************************************************************************
def EMGM(X, W, K):

    W = W[:,None]
    # initialization probabilistic assignments
    R = initialization(X.T, K)

    tol       = 1e-5
    maxiter   = 500
    llh       = torch.full([maxiter],-torch.inf)
    converged = False
    t         = 0

    # soft EM algorithm
    while (not converged) and (t+1 < maxiter):
        t = t+1   
        label = torch.argmax(R, axis=1)
        u = torch.unique(label) # non-empty components
        if R.size(dim=1) != u.size(dim=0):
            R = R[:,u]          # remove empty components
        
        [mu, sig, pi] = maximization(X,W,R)
        [R, llh[t]]  = expectation(X, W, mu, sig, pi)

        if t > 1:
            diff = llh[t]-llh[t-1]
            eps = abs(diff)
            converged = ( eps < tol*abs(llh[t]) )

    if converged:
        print('Converged in', t,'steps.')
    else:
        print('Not converged in ', maxiter, ' steps.')

    return [mu, sig, pi]

#*******************************************************************************
# Initialization
#*******************************************************************************
def initialization(X, K):
    # Random initialization
    n = X.size(1)
    idx = torch.randint(n, (K,))
    m = X[:, idx]
    
    # Calculate labels using PyTorch operations
    similarity = torch.matmul(m.t(), X) - torch.sum(m * m, dim=0).reshape(-1, 1) / 2
    label = torch.argmax(similarity, dim=0)
    
    # Ensure unique labels
    u = torch.unique(label)
    while K != len(u):
        idx = torch.randint(n, (K,))
        m = X[:, idx]
        
        # Calculate labels using PyTorch operations
        similarity = torch.matmul(m.t(), X) - torch.sum(m * m, dim=0).reshape(-1, 1) / 2
        label = torch.argmax(similarity, dim=0)
        u = torch.unique(label)

    # Create one-hot encoding matrix R
    R = torch.zeros(n, K, dtype=torch.int)
    R[torch.arange(n), label] = 1

    return R

#*******************************************************************************
# Expectation
#*******************************************************************************
def expectation(X, W, mu, sig, pi):

    logpdf = MultivariateNormal(mu, sig).log_prob(X[:,None,:]) + torch.log(pi)
    T = torch.logsumexp(logpdf, dim=1, keepdim=True)
    R = torch.exp(logpdf - T)

    llh = torch.sum(W * T) / torch.sum(W)

    return [R, llh]

#*******************************************************************************
# Maximization
#*******************************************************************************
def maximization(X, W, R):
    R = W * R
    d = X.size(1)
    k = R.size(1)

    nk = torch.sum(R, dim=0)
    if any(nk == 0):  # prevent division by zero
        nk += 1e-6

    w = nk / torch.sum(W)

    mu = R.T @ X / nk[:,None]

    sig = torch.zeros(k, d, d)
    sqrtR = torch.sqrt(R)
    for i in range(k):
        Xo = X - mu[i,:]
        Xo = Xo * sqrtR[:, i][:,None]
        sig[i, :, :] = Xo.T @ Xo / nk[i]
        sig[i, :, :] = sig[i, :, :] + torch.eye(d) * (1e-6)  # add a prior for numerical stability

    return [mu, sig, w]
