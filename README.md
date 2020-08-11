# lfads-pytorch
PyTorch implementation of the LFADS architecture.

The [Latent Factor Analysis via Dynamical Systems (LFADS)](https://arxiv.org/abs/1608.06315) is a sequential autoencoder-based method designed with the purpose of [inferring single-trial dynamics of neural activity](https://rdcu.be/6Wji). The major part of the model operates on neural firing rates, not spikes and can be applied to various data modalities.  

For a basic application take a look at the toy example notebook.  

The code in this repository is inspired by [LFADS JAX implementation](https://github.com/google-research/computation-thru-dynamics). Only the basic architecture is implemented. The controller is not yet a part of the repository.  

A description of the method, its parameters and additional literature can be found in the above hyperlinks as well as [here](https://github.com/google-research/computation-thru-dynamics/blob/master/notebooks/LFADS%20Tutorial.ipynb).

### Contents
- lfads.py - LFADS implementation.
- lfads-toy-example.ipynb - a notebook with an easy to follow toy example.
- utils.py - two functions supporting the notebook code.
