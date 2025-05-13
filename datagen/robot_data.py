# packages and setup
import numpy as np
import torch
from torch.distributions import MultivariateNormal

N = 10000  # number of simulations to run

# set seed for reproducibility
torch.manual_seed(0)


def transition_model(mu_curr, u_curr):
    dt = 1  # time step
    s = u_curr[:,0]; alpha = u_curr[:,1]; theta = mu_curr[:,2]
    mu_next = mu_curr + \
        dt*torch.stack([s*torch.cos(theta), s*torch.sin(theta), alpha],dim=1)

    return mu_next


#*******************************************************************************
# generate data
#*******************************************************************************
mu0 = torch.zeros((N,3))                           # initial state
muW = torch.zeros(mu0.shape[1])                    # noise mean
Q = torch.diag(torch.tensor([0.1, 0.1, 0.01]))     # noise cov

s = 5.                                      # speed [m/s]
alpha = 0.1                                 # angular rate [rad/s]
u = torch.tensor([s, alpha]).repeat((N,1))  # control

T = 40                                      # simulation length
states = torch.zeros((N, T, mu0.shape[1]))  # array to hold simulated states
process_noise = MultivariateNormal(muW, Q)  # process noise matrix

# generate trajectories
mu = mu0
states[:, 0,:] = mu0
for i in range(T):
    mu = transition_model(mu, u) + \
            process_noise.sample((N,))
    if i == 15:
        unif = torch.rand(N)
        u[:,1][unif > 0.5] = -0.1
    states[:, i, :] = mu

# the target is the final state
targets = states[:, -1, :]

normalized_targets = (targets - targets.mean(dim=0))/targets.std(dim=0)

np.savetxt("robot-flow.csv", normalized_targets, delimiter=",")
