# packages
import torch
from torch.distributions import (
    Categorical, MixtureSameFamily, MultivariateNormal
)
from typing import Tuple


class GMM():
    """
    Initialize a Gaussian mixture model.
        
    Args:
        n_components (int): number of mixture components (alias: k).
        n_features (int): number of input dimensions (alias: d).
    
    Learnable Parameters:
        mu (torch.Tensor): (k, d) tensor of component mean vectors.
        Cov (torch.Tensor): (k, d, d) tensor of covariance matrices.
        pi (torch.Tensor): (k,) tensor of mixing proportions.
    """

    def __init__(self, n_features, n_components):
        super(GMM, self).__init__()
        self.n_components = n_components
        self.n_features = n_features

        self.mu     = torch.zeros(n_components, n_features)
        self.Cov    = torch.eye(n_features).repeat(n_components,1,1)
        self.pi     = torch.ones(n_components) / n_components

    def sample(self, n:int) -> torch.Tensor:
        """
        Sample from the learned generative model.

        Args:
            n (int): total number of samples to draw.

        Returns:
            samples (torch.Tensor): (n, d) tensor of generated samples.
        """
        gmm = MixtureSameFamily(
            mixture_distribution=Categorical(probs=self.pi),
            component_distribution=MultivariateNormal(self.mu, self.Cov)
        )
        samples = gmm.sample((n,)).squeeze()

        return samples

    def component_log_prob(self, x:torch.Tensor) -> torch.Tensor:
        """
        Compute the log-likelihoods associated with each mixture component.

        Args:
            x (torch.Tensor): [N x D] tensor of input data

        Returns:
            component_log_probs (torch.Tensor): [N x K] tensor of per-component 
            log-likelihoods 
        """
        component_log_probs = MultivariateNormal(self.mu, self.Cov).log_prob(x[:,None,:]) + torch.log(self.pi)
        
        return component_log_probs
    
    def log_prob(self, x:torch.Tensor) -> torch.Tensor:
        """
        Computes the per-sample log-likelihoods.

        Args:
            x (torch.Tensor): (n, d) tensor of input data.

        Returns:
            log_probs (torch.Tensor): (n,) tensor of per-sample log-likelihoods. 
        """
        gmm = MixtureSameFamily(
            mixture_distribution=Categorical(probs=self.pi),
            component_distribution=MultivariateNormal(self.mu, self.Cov)
        )
        log_probs = gmm.log_prob(x)

        return log_probs
    
    def expectation(self, x:torch.Tensor, weights:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the expected log-likelihood given the current parameters.

        Args:
            x (torch.Tensor): (n, d) tensor of input data.
            weights (torch.Tensor): (n,) tensor of importance weights.

        Returns:
            r (torch.Tensor): (n, k) tensor of per-component responsibilities.
            llh (torch.Tensor): total log-likelihood of the input data.
        """

        component_log_probs = self.component_log_prob(x)
        log_probs = self.log_prob(x).reshape(-1, 1)

        r = torch.exp(component_log_probs - log_probs)
        llh = torch.sum(weights * log_probs) / torch.sum(weights)
        
        return [r, llh]

    def maximization(self, x:torch.Tensor, weights:torch.Tensor, r:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the parameters that maximize the expected log-likelihood.

        Args:
            x (torch.Tensor): (n, d) tensor of input data.
            weights (torch.Tensor): (n,) tensor of importance weights.
            r (torch.Tensor): (n, k) tensor of per-component responsibilities.

        Returns:
            mu (torch.Tensor): (k, d) tensor of component mean vectors.
            Cov (torch.Tensor): (k, d, d) tensor of covariance matrices.
            pi (torch.Tensor): (k,) tensor of mixing proportions.
        """

        d = x.shape[1]
        k = r.shape[1]
        r = weights * r
        r_sum = torch.sum(r, dim=0)
        if any(r_sum < 1e-6): # prevent division by zero
            r_sum += 1e-6

        mu = r.T @ x / r_sum[:,None]

        Cov = torch.zeros(k, d, d)
        sqrt_r = torch.sqrt(r)
        for i in range(k):
            Xo = x - mu[i,:]
            Xo = Xo * sqrt_r[:, i][:,None]
            Cov[i, :, :] = Xo.T @ Xo / r_sum[i]
            Cov[i, :, :] = Cov[i, :, :] + torch.eye(d) * (1e-5)  # add a prior for numerical stability

        pi = r_sum / torch.sum(weights)

        def clip_pi(pi, min_weight=1e-3):
            # Ensure no weight is below the minimum
            clipped_pi = torch.clamp(pi, min=min_weight)
            # Renormalize the weights to sum to 1
            normalized_pi = clipped_pi / clipped_pi.sum(dim=-1, keepdim=True)
            return normalized_pi
        
        pi = clip_pi(pi)

        return [mu, Cov, pi]

    def initialization(self, x:torch.Tensor, k:int) -> torch.Tensor:
        """
        Randomly initialize cluster assignments.

        Args:
            x (torch.Tensor): (n, d) tensor of input data.
            k (int): number of mixture components.

        Returns:
            r (torch.Tensor): tensor of per-component responsibilities.
        """

        # random initialization
        n = x.shape[0]
        idx = torch.randint(n, (k,))
        m = x[idx, :]
        
        # assign data to components
        similarity = x @ m.T - torch.sum(m * m, dim=1)[None, :] / 2
        label = torch.argmax(similarity, dim=1)
        # ensure unique labels
        u = torch.unique(label)
        while k != len(u):
            idx = torch.randint(n, (k,))
            m = x[idx, :]
            similarity = x @ m.T - torch.sum(m * m, dim=1)[None, :] / 2
            label = torch.argmax(similarity, dim=1)
            u = torch.unique(label)

        # create one-hot responsibility tensor
        r = torch.zeros(n, k, dtype=torch.int)
        r[torch.arange(n), label] = 1

        return r

    def fit(self, x:torch.Tensor, weights:torch.Tensor, k:int):
        """
        Fit the model parameters using expectation-maximization.

        Args:
            x (torch.Tensor): (n, d) tensor of input data.
            weights (torch.Tensor): (n,) tensor of importance weights.
            k (int): number of mixture components.
        """

        weights = weights[:,None]

        r = self.initialization(x, k)

        tol       = 1e-5
        maxiter   = 200
        llh       = torch.full([maxiter],-torch.inf)
        converged = False
        t         = 0

        # soft EM algorithm
        while (not converged) and (t+1 < maxiter):
            t = t+1   
            
            [mu, Cov, pi] = self.maximization(x, weights, r)
            self.mu = mu
            self.Cov = Cov
            self.pi = pi
            # E-step
            [r, llh[t]] = self.expectation(x, weights)

            if t > 1:
                diff = llh[t]-llh[t-1]
                eps = abs(diff)
                converged = ( eps < tol*abs(llh[t]) )

        if converged:
            print('Converged in', t,'steps.')
        else:
            print('Not converged in ', maxiter, ' steps.')

        return self
