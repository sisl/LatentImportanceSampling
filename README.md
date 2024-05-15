# Latent Importance Sampling

This repository contains code and experiments for the manuscript:

> L. A. Kruse, A. E. Tzikas, H. Delecki, M. M. Arief, M. J. Kochenderfer, _Enhanced Importance Sampling through Latent Space Exploration in Normalizing Flows_, IEEE Robotics and Automation Letters (Under Review).

## Dependencies

See `environment.yaml` for required packages, or use this to create a Conda environment with all dependencies:
```bash
conda env create -f environment.yaml
```

This repository was tested with Python 3.11 and PyTorch 2.1. The `racecar` and `f16` experiments require additional installations (discussed in the following section) if you want to retrain a flow model or generate a dataset of Monte Carlo simulations.

## Data Generation
To create a dataset (for flow training or Monte Carlo evaluations) for the nonholonomic `robot` experiments, run the following command:
```
python3 robot-data.py
```

Generating data for the `racecar` experiments requires a [Julia install](https://julialang.org/downloads/) and the MPOPIS (Model Predictive Optimized Path Integral Strategies) repository from the following paper:
```
@inproceedings{Asmar2023},
  title = {Model Predictive Optimized Path Integral Strategies},
  author = {Dylan M. Asmar and Rasalu Senanayake and Shawn Manuel and Mykel J. Kochenderfer},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year = {2023}
```
Specific installation instructions can be found in the [MPOPIS repository](https://github.com/sisl/MPOPIS).

Generating data for the `f16` experiments requires a [jax install](https://github.com/google/jax#installation) and the Jax F16 Dynamics repository from the following paper:
```
@inproceedings{So-RSS-23},
  title = {Solving Stabilize-Avoid Optimal Control via Epigraph Form and Deep Reinforcement Learning}, 
  author = {Oswin So AND Chuchu Fan}, 
  booktitle = {Proceedings of Robotics: Science and Systems}, 
  year = {2023}, 
} 
```
Specific installation instructions can be found in the [Jax F16 repository](https://github.com/MIT-REALM/jax-f16).

## Flow Training

Pre-trained flows are provided in the `flows\` folder. However, if you wish to train a flow from scratch, you can run the following command, replacing the `simulator` argument with
the simulator of choice (`robot`, `racecar`, `f16`):
```
python3 train.py --simulator 'robot'
```
Flow hyperparameters are stored in `yaml` files in the `configs\` folder.

## Latent Importance Sampling Experiments

The experiments expect a `{simulator}-flow.csv` and `{simulator}-mcs.csv` file for each simulator. `robot-flow.csv` and `robot-mcs.csv` are provided in the repository. The following command runs importance sampling and computes metrics:
```
python3 experiments.py --simulator 'robot'
```
Importance sampling hyperparameters are stored in `yaml` files in the `configs\` folder and can be adjusted for different runs. For example, to change between target space sampling and latent space sampling, edit the `space` entry. Results are stored in the `results\` folder.

