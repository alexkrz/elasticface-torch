# Elasticface-torch

This is an updated version of the original [ElasticFace repository](https://github.com/fdbtrs/ElasticFace).

We make the following adjustments:

- Replace `mxnet` by Huggingface's `datasets` library
- Ensure compatibility with `torch > 2.0` and `numpy > 2.0`

## Setup

We recommend [miniforge](https://conda-forge.org/download/) to set up your python environment.
In case VSCode does not detect your conda environments, install [nb_conda](https://github.com/conda-forge/nb_conda-feedstock) in the base environment.

```bash
conda env create -n $YOUR_ENV_NAME -f environment.yml
conda activate $YOUR_ENV_NAME
pip install -r requirements.txt
pre-commit install
```

## Basic training

Run

```bash
python train.py
```

for training a standard ArcFace model.

## Todos

- [x] Add logging with `tensorboard`
- [x] Check that trainings can be executed deterministically
- [ ] Adjust `train.py` for multi-gpu training
- [ ] Add cases to config depending on dataset and loss
