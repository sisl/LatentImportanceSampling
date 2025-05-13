#*******************************************************************************
# Imports and Setup
#*******************************************************************************
# packages
import argparse
import json
import numpy as np
import os
import pandas as pd
import pickle
from prdc import compute_prdc
import torch
import yaml

# file imports
from is_methods.cross_entropy import cross_entropy_is
from is_methods.sequential import sequential_is
from utils.ellipsoids import outside_ellipsoid, min_enclosing_ellipsoid
from utils.gmm import GMM


def save_metrics(dir, Pfs, rel_errs, log_probs, coverages, densities, total_samples, samples_per_level):
    metrics = {}
    metrics['Pfs'] = np.array(Pfs.numpy()).tolist()
    metrics['Pfs mean'] = float(Pfs.mean())
    metrics['Pfs std'] = float(Pfs.std())
    metrics['rel errs'] =  np.array(rel_errs.numpy()).tolist()
    metrics['rel errs mean'] = float(rel_errs.mean())
    metrics['rel errs std'] = float(rel_errs .std())
    metrics['log probs'] = np.array(log_probs.numpy()).tolist()
    metrics['log probs mean'] = float(log_probs.mean())
    metrics['log probs std'] = float(log_probs.std())
    metrics['coverages'] = np.array(coverages.numpy()).tolist()
    metrics['coverages mean'] = float(coverages.mean())
    metrics['coverages std'] = float(coverages.std())
    metrics['densities'] = np.array(densities.numpy()).tolist()
    metrics['densities mean'] = float(densities.mean())
    metrics['densities std'] = float(densities.std())
    metrics['total samples'] = np.array(total_samples.numpy()).tolist()
    metrics['total samples mean'] = float(total_samples.mean())
    metrics['total samples std'] = float(total_samples.std())
    metrics['samples_per_level'] = float(samples_per_level)

    # save results
    with open(os.path.join(dir, "results.json"), "w") as outfile:
        json.dump(metrics, outfile)


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--simulator', type=str, default='robot',
                    choices=['robot', 'racecar', 'f16'])
parser.add_argument('--space', type=str, default='latent',
                    choices=['latent', 'target'])
parser.add_argument('--is_method', type=str, default='ce',
                    choices=['ce', 'sis'])
args = parser.parse_args()

# read in configs
with open('configs/{}.yaml'.format(args.simulator), 'r') as file:
    config = yaml.safe_load(file)
config['simulator'] = args.simulator
config['space'] = args.space
config['is_method'] = args.is_method


results_dir = f"./results/{config['simulator']}/{config['space']}-{config['is_method']}"
os.makedirs(results_dir, exist_ok=True)


#*******************************************************************************
# file IO
#*******************************************************************************
flow_name = config['base'] + '-' + config['linear'] + '-' + config['key']
flow_file = open("flows/{}".format(flow_name), "rb")
flow = pickle.load(flow_file)
flow.eval()

flow_df = pd.read_csv("data/{}-flow.csv".format(config['key']), header=None)
flow_data = torch.tensor(flow_df.values, dtype=torch.float32)

mcs_df = pd.read_csv("data/{}-mcs.csv".format(config['key']), header=None)
mcs_data = torch.tensor(mcs_df.values, dtype=torch.float32)


#*******************************************************************************
# create target region
#*******************************************************************************
if config['simulator'] == 'robot': 
    
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
    
elif config['simulator'] == 'racecar':
    x = flow_data[:config['subset']]
    region1 = lambda x : ((x[:,0] > 0.0) & (x[:,2] > 2.75))
    region2 = lambda x : ((x[:,6] < -2.25) & (x[:,0] > 1.5))

    mask1 = region1(x)
    mask2 = region2(x)
    x_in1 = x[mask1]
    x_in2 = x[mask2]

elif config['simulator'] == 'f16':
    x = flow_data[:config['subset']]
    region1 = lambda x : (x[:,3] > 1.45)
    region2 = lambda x : (x[:,10] < -2.45)

    mask1 = region1(x)
    mask2 = region2(x)
    x_in1 = x[mask1]
    x_in2 = x[mask2]

Cov1t, mu1t = min_enclosing_ellipsoid(x_in1)
Cov2t, mu2t = min_enclosing_ellipsoid(x_in2)

def target_obj_func(x):
    return torch.minimum(
        outside_ellipsoid(x, mu1t, Cov1t),
        outside_ellipsoid(x, mu2t, Cov2t)
    )


#*******************************************************************************
# approximate limit function in latent space
#*******************************************************************************
if config['space'] == 'target':
    # define objective function
    def obj_func(x):
        return target_obj_func(x)
    
elif config['space'] == 'latent':
    with torch.no_grad():
        u_in1 = flow.transform_to_noise(x_in1)
        u_in2 = flow.transform_to_noise(x_in2)

    Cov1, mu1 = min_enclosing_ellipsoid(u_in1)
    Cov2, mu2 = min_enclosing_ellipsoid(u_in2)

    # define objective function
    def obj_func(x):
        return torch.minimum(
            outside_ellipsoid(x, mu1, Cov1),
            outside_ellipsoid(x, mu2, Cov2)
        )

else:
    raise RuntimeError('Incorrect reference frame.')


#*******************************************************************************
# perform importance sampling
#*******************************************************************************
ref_Pf = (target_obj_func(mcs_data) < 0.).sum() / len(mcs_data)

print("\n****************************************")
print(f"{config['is_method'].upper()} with {config['space']}-space proposals")
print(f"System: {config['simulator']} with {config['features']} dimensions")
print("Ref Pf:\t {:.6f}".format(ref_Pf))
print("****************************************\n")

# prep for coverage metric
real_samples = mcs_data[(target_obj_func(mcs_data) < 0.)]
if len(real_samples) > config['eval_size']:
    real_samples = real_samples[:config['eval_size']]

all_Pfs = torch.zeros(config['n_trials'])
all_rel_errs = torch.zeros(config['n_trials'])
all_log_probs = torch.zeros(config['n_trials'])
all_coverages = torch.zeros(config['n_trials'])
all_densities = torch.zeros(config['n_trials'])
all_total_samples = torch.zeros(config['n_trials'])


for i in range(config['n_trials']):
    torch.manual_seed(i)

    model = GMM(
        n_components=config['K'],
        n_features=config['features']
    )
    
    # importance sampling
    print('Trial {}'.format(i))
    if config['is_method'] == 'ce':
        [Pf, model, total_samples] = \
            cross_entropy_is(config['N'], config['rho'], obj_func, model, flow, config['space'])
    elif config['is_method'] == 'sis':
        [Pf, model, total_samples] = \
            sequential_is(config['N'], config['rho'], obj_func, model, flow, config['space'])

    q_samples = model.sample(config['eval_size'])
    
    # compute metrics
    if args.space == 'latent':
        with torch.no_grad():
            fake_samples = flow._transform.inverse(q_samples)[0]
    else:
        fake_samples = q_samples

    # log-likelihood
    with torch.no_grad():
        log_probs = flow.log_prob(fake_samples)

    fake = fake_samples[(target_obj_func(fake_samples) < 0.0)]
    # density and coverage metrics
    prdc_metrics = compute_prdc(real_samples, fake, nearest_k=5) 

    all_Pfs[i] = Pf
    all_rel_errs[i] = (Pf - ref_Pf) / ref_Pf
    all_log_probs[i] = log_probs.mean()
    all_coverages[i] = prdc_metrics['coverage']
    all_densities[i] = prdc_metrics['density']
    all_total_samples[i] = total_samples

    save_metrics(results_dir, all_Pfs, all_rel_errs, all_log_probs, all_coverages, all_densities, all_total_samples, config['N'])

print("\n----------------------------------------")
print("Avg Failure Prob: {:.6f}".format(all_Pfs.mean()))
print("Avg Log Prob: {:.2f}".format(all_log_probs.mean()))
print("Avg Coverage: {:.4f}".format(all_coverages.mean()))
print("Avg Density: {:.4f}".format(all_densities.mean()))
print("Avg Samples: {:.1f}".format(all_total_samples.mean()))
print("----------------------------------------\n")