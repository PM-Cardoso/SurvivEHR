[![medRxiv](https://img.shields.io/badge/medRxiv-10.1101%2F2025.08.04.25332916v1-blue)](https://www.medrxiv.org/content/10.1101/2025.08.04.25332916v1)

# SurvivEHR

SurvivEHR is a research project developing a foundation model for survival analysis on electronic health records (EHR). It introduces a decoder-only transformer trained on time-to-event prediction tasks, with a novel competing risks survival objective. This repository contains the code associated with the SurvivEHR model, as described in the medRxiv preprint "SurvivEHR: An EHR Foundation Model for Time-to-Event Prediction with Survival Objectives" (available on medRxiv: DOI: 10.1101/2025.08.04.25332916).

SurvivEHR is trained on a large corpus of EHR data -- over 7.6 billion coded events from 23 million patients in UK primary care. This large-scale pretraining enables the model to learn rich representations of patient histories which can be used in downstream fine-tuning tasks.

# Project Overview

SurvivEHR is a generative transformer-based foundation model trained on over 7.6 billion EHR events from 23 million patients in UK primary care. The model introduces a competing risks, time-to-event pretraining objective that enables forecasting of future clinical events (e.g. new diagnoses, lab investigations, medications) and patient mortality. By learning from longitudinal health records, SurvivEHR can capture complex patient trajectories across multiple long-term conditions. In our experiments, SurvivEHR demonstrated strong risk stratification performance and outperformed traditional survival models across multiple prediction tasks. It also showed effective transfer learning to specific prognostic tasks (especially in low-resource settings), highlighting its potential as a reusable EHR foundation model for clinical risk prediction

> Note: This repository provides the research code for training and evaluating SurvivEHR as described in the preprint. Due to data privacy and size constraints, pre-trained model weights are not included. Users can reproduce training or fine-tune SurvivEHR on their own data following the instructions below.

# Installation and Setup Instructions

The repository now supports a local macOS workflow using a standard `venv` and a sibling `FastEHR-main` checkout.

## Recommended local setup (macOS + `venv`)

SurvivEHR currently works best with Python `3.10` or `3.11`. If you already created `.venv` with Python `3.14`, recreate it with Python `3.11` before installing dependencies.

### Expected workspace layout

```text
Deep-learning_EHR/
├── SurvivEHR/
└── FastEHR-main/
```

### One-command setup

From the `SurvivEHR` repository root:

```bash
PYTHON_BIN="$(command -v python3.11)" ./scripts/setup_local_env.sh
```

If `python3.11` is not on your `PATH`, set `PYTHON_BIN` explicitly to a compatible interpreter.

### Manual setup

```bash
cd "/Users/p.cardoso/Library/CloudStorage/OneDrive-UniversityofExeter/Documents/Data Analysis/Deep-learning_EHR/SurvivEHR"
rm -rf .venv
/Users/p.cardoso/.pyenv/versions/3.11.9/bin/python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip "setuptools<81" wheel
python -m pip install -r requirements.txt
cd ../FastEHR-main
python -m pip install -r requirements-py311.txt
python -m pip install -e .
cd ../SurvivEHR
```

`FastEHR-main/requirements-py311.txt` contains `-e .`, so it must be installed while your shell is inside the `FastEHR-main` directory.

## Runtime configuration

The default config file is now `examples/modelling/SurvivEHR/confs/default.yaml`. It reads dataset paths from environment variables, which keeps the repo portable across machines.

Set these before running experiments:

```bash
export FASTEHR_ROOT="/Users/p.cardoso/Library/CloudStorage/OneDrive-UniversityofExeter/Documents/Data Analysis/Deep-learning_EHR/FastEHR-main"
export SURVIVEHR_DB_PATH="/absolute/path/to/cprd.db"
export SURVIVEHR_DS_PATH="/absolute/path/to/pretrain_dataset"
export SURVIVEHR_META_PATH="/absolute/path/to/meta_information.pickle"
```

`SURVIVEHR_META_PATH` is optional unless your dataset requires a custom metadata file.

## Smoke tests

After installation, verify imports:

```bash
source .venv/bin/activate
python -c "import FastEHR, torch, pytorch_lightning, transformers; print('imports ok')"
python -c "from examples.modelling.SurvivEHR.run_experiment import run; print('entrypoint ok')"
```

To inspect the resolved Hydra config without starting training:

```bash
source .venv/bin/activate
python examples/modelling/SurvivEHR/run_experiment.py --cfg job
```

To run a real experiment, provide valid dataset paths and optionally reduce worker count and batch size for a laptop:

```bash
source .venv/bin/activate
export FASTEHR_ROOT="/Users/p.cardoso/Library/CloudStorage/OneDrive-UniversityofExeter/Documents/Data Analysis/Deep-learning_EHR/FastEHR-main"
export SURVIVEHR_DB_PATH="/absolute/path/to/cprd.db"
export SURVIVEHR_DS_PATH="/absolute/path/to/pretrain_dataset"
python examples/modelling/SurvivEHR/run_experiment.py data.min_workers=0 data.batch_size=4
```

## Laptop notes

- On Apple Silicon, CPU-only PyTorch installs cleanly with the current dependency set.
- `wandb` logging is disabled by default in `examples/modelling/SurvivEHR/confs/default.yaml` for local runs.
- The experiment entrypoint now auto-detects a sibling `FastEHR-main` checkout, so `FASTEHR_ROOT` is optional when your folder layout matches the example above.
- If you see `data.path_to_db does not exist` or `data.path_to_ds does not exist`, export the dataset paths above or override them on the command line.

## Option 2: Create and run in an apptainer

Apptainer is used instead of Docker on high performance computing systems due to the administrative privileges that are required to run the latter.
Ensure you are not already running an apptainer (e.g. interactive nodes).

1) Build container image (saved to /rds, not repo)
```bash
bash containers/container_build.sh
```

3) Create/update venv on /rds and sync dependencies
```bash
bash containers/env_bootstrap.sh
```

5) Run commands in the container using the venv, e.g.
```bash
bash containers/run_in_container.sh python -V
```

## Install FastEHR (Data Pipeline)

Whilst custom data pipelines can be used, we recommend using FastEHR. SurvivEHR uses the FastEHR library for EHR data preprocessing (providing tools for data pre-processing, loading, dataset construction, etc.). In this workspace, a local checkout already exists at `../FastEHR-main`, and the experiment runner can use that location automatically.

# Examples Directory Overview

The repository includes an examples/ directory containing scripts and Jupyter notebooks demonstrating how to prepare data and run the SurvivEHR model on various tasks. Below is an overview of the contents and their usage:

- **Data Preparation Examples (examples/data/)**: This folder contains scripts and notebooks for building the model input datasets from raw EHR data (using FastEHR outputs).
    - **1_build_database**: How to build an SQLite database for your Electronic Health Records to enable fast data pipeline building.
    - **2_build_pre_training_dataset/**: How to construct the large-scale pre-training dataset.
        - Combining events from all patients into a sequential format
            - Splitting data into cross-validation splits (including creation or loading of splits used elsewhere in the pipeline to avoid practice leakage).
            - Pull and adapt meta information to be used elsewhere in the data pipeline (e.g. tokenisation, vocabulary truncation, outlier removal).
            - Additionally includes examples for experiments which stratify by region (e.g., separate cohorts for different geographic regions like North-East vs. London).
    - **3_benchmark_data/**: Convert supervised FastEHR datasets for cross-sectional benchmarks (e.g. DeepHit, DeSurv, Random Survival Forest)
    - **4_convert_BEHRT_data/**: Convert self-supervised and supervised FastEHR datasets for time series benchmarks using FastEHR's adapter framework (e.g. BEHRT)
- **SurvivEHR Examples (examples/modelling/)**: This folder contains scripts and notebooks for running SurvivEHR's experiments
    - **/**: The root directory contains wrapper scripts for the models given in `src/models/`.
        - `run_experiment.py` wraps all experiments and calls: `setup_causal_experiment.py` for pre-training; `set_fewshot_experiment.py` for supervised cases which do use the pre-training architecture, such as zero-shot experiments; and `setup_finetune_experiment.py` for supervised experiments which replace the pre-training head.
    - **notebooks**: Each folder in the notebooks contains the various experiments presented throughout the accompanying manuscript.
      


# Citation

If you use or reference SurvivEHR in your research or work, please cite the accompanying paper:

```Charles Gadd et al. (2025). SurvivEHR: a competing risks, time-to-event foundation model for multiple long-term conditions from primary care electronic health records. medRxiv. DOI: 10.1101/2025.08.04.25332916.```

# Copyright

© 2025 Charles Gadd, University of Oxford. All rights reserved where applicable.

Distributed under the GNU GPL v3.0 (or later). See LICENSE for details.
