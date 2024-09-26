# QMIX modification for exploration and training stability enhancement
Code for the paper "Rethinking Exploration and Experience Exploitation in Value-Based Multi-Agent Reinforcement Learning".

## Installation

```bash
pip3 install -r docker/requirements.txt
```

Additionally it is necessary to install StarCraft II:

```bash
bash docker/install_sc2.sh
```

Alternatively, it is possible to build the Dockerfile:

```bash
cd docker
bash build.sh
```

## Training

For training, run the command:

```bash
python3 remix.py --map_name=corridor --envs_count=1 --adaptive_epsilon --tanh_coef=0.04 --continuous_buffer --burn_in --batch_size=64 --train_frequency=64 --omit_wandb
```

In order to save the results of training in wandb, you can remove the argument "--omit\_wandb" and set up the project name with "--wandb\_project=<project\_name>".
