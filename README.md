# TAEA

This repository contains the implementation of the **TAEA** framework, including benchmark data, pretrained models, source code, training modules, and demo scripts.

## Repository Structure

The project is organized as follows:

```text
TAEA/
├─ data/                         # Data and experiment resources
│  ├─ benchmark/                # Benchmark instances and datasets
│  ├─ pretrained/               # Pretrained models
│  ├─ rl/                       # RL-related data
│  └─ trans/                    # Transformer-related data
│
├─ scripts/                     # Entry scripts and runnable demos
│  ├─ run_demo.m
│  ├─ train_rl_demo.m
│  ├─ train_trans_aco_demo.m
│  └─ train_trans_seq_demo.m
│
├─ src/                         # Main source code of the TAEA framework
│  ├─ core/                     # Shared utilities and common components
│  ├─ rl/                       # Reinforcement learning modules
│  └─ trans/                    # Transformer-based modules
│
└─ training/                    # Training pipelines and training utilities
   ├─ rl/                       # RL training code
   └─ trans/                    # Transformer training code
```

## Folder Description

* **data/**
  Stores all data and experiment resources used in the project.

  * **benchmark/**: benchmark instances and datasets.
  * **pretrained/**: pretrained models.
  * **rl/**: RL-related data.
  * **trans/**: Transformer-related data.

* **scripts/**
  Contains runnable demo and training scripts for quick testing and reproduction.

* **src/**
  Contains the main implementation of the TAEA framework.

  * **core/**: common utilities and shared components.
  * **rl/**: reinforcement learning modules.
  * **trans/**: Transformer-based modules.

* **training/**
  Provides training pipelines and supporting functions for different learning modules.

  * **rl/**: RL training procedures.
  * **trans/**: Transformer training procedures.

## Getting Started

The scripts in `scripts/` provide the main entry points of the project:

* `run_demo.m`
  Runs a basic demonstration of the framework.

* `train_rl_demo.m`
  Trains the RL-based model.

* `train_trans_aco_demo.m`
  Trains the Transformer-based model under the ACO setting.

* `train_trans_seq_demo.m`
  Trains the Transformer-based sequential model.

## Notes

1. Make sure the required MATLAB environment and toolboxes are installed before running the scripts.
2. Check the paths to benchmark data, pretrained models, and output folders before training or testing.
3. Pretrained models can be placed under `data/pretrained/` for direct evaluation or fine-tuning.

## Citation

If you use this code in your research, please cite the corresponding paper of TAEA.
