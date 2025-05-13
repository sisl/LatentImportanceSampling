# Latent Importance Sampling

This repository contains code and experiments for the paper:

> Kruse, L. A., Tzikas, A., Delecki, H., Arief, M., & Kochenderfer, M. J., _Enhanced Importance Sampling through Latent Space Exploration in Normalizing Flows_, AAAI 2025. [[arXiv]](https://arxiv.org/abs/2501.03394)


## Dependencies

See `environment.yaml` for required packages, or use this to create a Conda environment with all dependencies:
```bash
conda env create -f environment.yaml
```

This repository was tested with Python 3.11 and PyTorch 2.1. The `racecar` and `f16` experiments require additional installations (discussed in the following section) if you want to retrain a flow model or generate a dataset of Monte Carlo simulations.

## Data

The experiments expect a `{simulator}-flow.csv` and `{simulator}-mcs.csv` file for each simulator (`robot`, `racecar`, `f16`). Generated datasets can be downloaded at [this link](https://drive.google.com/drive/folders/12O6kFP5PHiMBGiqutJHAXnoL9OzmnIHx?usp=sharing), and the .csv files should be placed into a folder called `data/`.

New datasets (for additional flow training or Monte Carlo evaluations) can be created using the files in the `datagen/` folder. For the nonholonomic `robot` experiments, run the following command:
```
python3 datagen/robot-data.py
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

Pre-trained flows are provided in the `flows/` folder. However, if you wish to train a flow from scratch, you can run the following command, replacing the `simulator` argument with
the simulator of choice (`robot`, `racecar`, `f16`):
```
python3 train.py --simulator 'robot'
```
Flow hyperparameters are stored in `yaml` files in the `configs/` folder.

## Latent Importance Sampling Experiments

To reproduce the experimental results, run the `experiments.sh` file. Importance sampling hyperparameters are stored in `yaml` files in the `configs/` folder.

## How to Cite
If you find this code useful in your research, please cite the following publication:
```
@inproceedings{kruse2025enhanced,
  title={Enhanced Importance Sampling through Latent Space Exploration in Normalizing Flows},
  author={Kruse, Liam Anthony and Tzikas, Alexandros and Delecki, Harrison and Arief, Mansur and Kochenderfer, Mykel J},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={17},
  pages={17983--17989},
  year={2025}
}
```
