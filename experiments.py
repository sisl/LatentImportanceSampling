#*******************************************************************************
# Imports and Setup
#*******************************************************************************
# packages
import argparse
import json
import pandas as pd
import pickle
from prdc import compute_prdc
import yaml

# torch imports
import torch
from torch.distributions import (
    Categorical, MixtureSameFamily, MultivariateNormal
)

# file imports
from is_methods.CEIS_GM import CEIS_GM
from is_methods.SIS_GM import SIS_GM
from utils import outside_ellipsoid, min_enclosing_ellipsoid

# setup
parser = argparse.ArgumentParser()
parser.add_argument('--simulator',
                    choices=['robot', 'racecar', 'f16'],
                    default='robot',
                    help='Choose an autonomous systems simulator.')
simulator = parser.parse_args()

with open('configs/{}.yaml'.format(simulator.simulator), 'r') as file:
    args = yaml.safe_load(file)


#*******************************************************************************
# File IO
#*******************************************************************************
tb_key = args['base'] + '-' + args['linear'] + '-' + args['key']
flow_file = open("flows/{}".format(tb_key), "rb")
flow = pickle.load(flow_file)
flow.eval()

flow_df = pd.read_csv("data/{}-flow.csv".format(args['key']), header=None)
flow_data = torch.tensor(flow_df.values, dtype=torch.float32)

mcs_df = pd.read_csv("data/{}-mcs.csv".format(args['key']), header=None)
mcs_data = torch.tensor(mcs_df.values, dtype=torch.float32)


#*******************************************************************************
# Create Target Region
#*******************************************************************************
if args['key'] == 'robot': 
    
    def generate_cube_corners(x_lim, y_lim, z_lim):
        # generate all possible combinations of coordinates
        corners = torch.stack(torch.meshgrid(x_lim, y_lim, z_lim), dim=-1)
        # reshape to get the corners as rows
        corners = corners.reshape(-1, 3)

        return corners

    # generate the corners of the cube
    x_lim1 = torch.tensor([-1.0, -2.0])
    y_lim1 = torch.tensor([-2.25, -3.25])
    z_lim1 = torch.tensor([1.25, 2.25])
    x_in1 = generate_cube_corners(x_lim1, y_lim1, z_lim1)

    x_lim2 = torch.tensor([0.75, 1.75])
    y_lim2 = torch.tensor([-3.25, -4.25])
    z_lim2 = torch.tensor([-1.0, -2.0])
    x_in2 = generate_cube_corners(x_lim2, y_lim2, z_lim2)
    
elif args['key'] == 'racecar':
    x = flow_data[:args['subset']]
    region1 = lambda x : ((x[:,0] > 0.0) & (x[:,2] > 2.75))
    region2 = lambda x : ((x[:,6] < -2.25) & (x[:,0] > 1.5))

    mask1 = region1(x)
    mask2 = region2(x)
    x_in1 = x[mask1]
    x_in2 = x[mask2]

elif args['key'] == 'f16':
    x = flow_data[:args['subset']]
    region1 = lambda x : (x[:,3] > 1.45)
    region2 = lambda x : (x[:,10] < -2.45)

    mask1 = region1(x)
    mask2 = region2(x)
    x_in1 = x[mask1]
    x_in2 = x[mask2]

sigma1t, mu1t = min_enclosing_ellipsoid(x_in1)
sigma2t, mu2t = min_enclosing_ellipsoid(x_in2)

def target_limit_func(x):
    return torch.minimum(
        outside_ellipsoid(x, mu1t, sigma1t),
        outside_ellipsoid(x, mu2t, sigma2t)
    )


#*******************************************************************************
# Approximate Limit Function in Latent Space
#*******************************************************************************
if args['space'] == 'target':
    # define limit function
    def limit_func(x):
        return target_limit_func(x)
    
elif args['space'] == 'latent':
    with torch.no_grad():
        u_in1 = flow.transform_to_noise(x_in1)
        u_in2 = flow.transform_to_noise(x_in2)

    sigma1, mu1 = min_enclosing_ellipsoid(u_in1)
    sigma2, mu2 = min_enclosing_ellipsoid(u_in2)

    # define limit function
    def limit_func(x):
        return torch.minimum(
            outside_ellipsoid(x, mu1, sigma1),
            outside_ellipsoid(x, mu2, sigma2)
        )

else:
    raise RuntimeError('Incorrect reference frame.')


#*******************************************************************************
# Perform Importance Sampling
#*******************************************************************************
torch.manual_seed(0)

ref_Pf = (target_limit_func(mcs_data) < 0.).sum() / len(mcs_data)

print("\n**********")
print("Ref Pf:\t {:.4f}".format(ref_Pf))
print("**********\n")

# prep for coverage metric
real_samples = mcs_data[(target_limit_func(mcs_data) < 0.)]
if len(real_samples) > args['eval_size']:
    real_samples = real_samples[:args['eval_size']]

Pfs = torch.zeros(args['n_trials'])
rel_err = torch.zeros(args['n_trials'])
Ntots = torch.zeros(args['n_trials'])
avg_lps = torch.zeros(args['n_trials'])
densities = torch.zeros(args['n_trials'])
coverages = torch.zeros(args['n_trials'])

for i in range(args['n_trials']):
    # perform IS
    print('Trial {}'.format(i))
    if args['is_method'] == 'ce':
        [Pf, (pi, mu, sig), samples] = \
            CEIS_GM(args['N'], args['rho'], limit_func, args['features'], 
                    args['K'])
    elif args['is_method'] == 'sis':
        [Pf, (pi, mu, sig), samples] = \
            SIS_GM(args['N'], args['rho'], limit_func, args['features'], 
                   args['K'])

    proposal = MixtureSameFamily(
                mixture_distribution=Categorical(probs=pi),
                component_distribution=MultivariateNormal(mu, sig))
    
    q_samples = proposal.sample((args['eval_size'],))
    
    # compute metrics
    if args['space'] == 'latent':
        with torch.no_grad():
            fake_samples = flow._transform.inverse(q_samples)[0]
    else:
        fake_samples = q_samples

    with torch.no_grad():
        avg_lp = flow.log_prob(fake_samples).mean()

    Ntot = args['N'] * len(samples)

    fake = fake_samples[(target_limit_func(fake_samples) < 0.0)]
    prdc_metrics = compute_prdc(real_samples, fake, nearest_k=5) 
    Pfs[i] = Pf
    rel_err[i] = (Pf - ref_Pf) / ref_Pf
    avg_lps[i] = avg_lp
    densities[i] = prdc_metrics['density']
    coverages[i] = prdc_metrics['coverage']
    Ntots[i] = Ntot

print("\n********************")
print("Avg Failure Prob: {:.4f}".format(Pfs.mean()))
print("Avg Log Prob: {:.4f}".format(avg_lps.mean()))
print("Avg Density: {:.4f}".format(densities.mean()))
print("Avg Coverage: {:.4f}".format(coverages.mean()))
print("Avg Number of Samples: {:.1f}".format(Ntots.mean()))
print("********************\n")

metrics = {}
metrics['pf mean'] = float(Pfs.mean())
metrics['pf std'] = float(Pfs.std())
metrics['rel err mean'] = float(rel_err.mean())
metrics['rel err std'] = float(rel_err.std())
metrics['avg lp mean'] = float(avg_lps.mean())
metrics['avg lp std'] = float(avg_lps.std())
metrics['coverage mean'] = float(coverages.mean())
metrics['coverage std'] = float(coverages.std())
metrics['density mean'] = float(densities.mean())
metrics['density std'] = float(densities.std())
metrics['Ntot mean'] = float(Ntots.mean())
metrics['Ntot std'] = float(Ntots.std())
metrics['N'] = float(args['N'])
metrics['n_trials'] = float(args['n_trials'])

# save results
with open("results/{}-{}-{}test.json".format(
    args['key'], args['is_method'], args['space']), "w") as outfile: 
    json.dump(metrics, outfile)
