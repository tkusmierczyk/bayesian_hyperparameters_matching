

# Bayesian models hyperparameters learning by matching moments of prior predictive distribution.

The code demonstrates how to choose hyperparameters for priors in Bayesian models. The priors are found using SGD to minimize a discrepancy between prior predictive distribution moments and requested values. For example, a user provides an expectation and a variance for Poisson Matrix Factorization model and the algorithm searches for matching hyperparameters.


### Paper

The code was used and is necessary to reproduce results from the paper:

E. de Souza da Silva, T. Kuśmierczyk (equal contribution), M. Hartmann, A. Klami: **Prior specification via prior predictive matching: Poisson matrix factorization and beyond**


### Abstract

Hyperparameter optimization for machine learning models is typically carried out by some sort of cross-validation procedure or global optimization, both of which require running the learning algorithm numerous times. We show that for Bayesian hierarchical models there is an appealing alternative that allows selecting good hyperparameters without learning the model parameters during the process at all, facilitated by the prior predictive distribution that marginalizes out the model parameters. We propose
an approach that matches suitable statistics of the prior predictive distribution with ones provided by an expert. We propose a model-independent optimization procedure. 


### Main files 

  * pmf_sgd_optimization.ipynb  - Jupter Notebook illustrating how priors matching requested values of prior predictive expectation and/or variance can be found for Poisson Matrix Factorization (PMF) model using SGD.
  * hpf_sgd_optimization.ipynb  - Jupter Notebook illustrating how priors matching requested values of prior predictive expectation and/or variance can be found for Hierarchical Poisson Matrix Factorization (HPF) model using SGD.
  * pmf_estimators_analysis.ipynb  - Jupter Notebook illustrating bias and variance of the estimators used in pmf_sgd_optimization.ipynb for PMF model.
  * pmf_surface_visualizations.ipynb  - Jupter Notebook illustrating 1D and 2D projections of optimization space for the problem of matching Poisson Matrix Factorization (PMF) prior predicitve distribution variance (minimization of the discrepancy=(Variance-100)^2 ). We consider two parametrizations: abcd vs mu-sgima. 


### Sampling code 

  * pmf_model.py  - Methods calculating E[Y] and E[Y^2] (and therefore also Var[Y]) over prior predictive distribution for Poisson Matrix Factorization.
  * hpf_model.py  - Methods calculating E[Y] and E[Y^2] (and therefore also Var[Y]) over prior predictive distribution for Hierarchical Poisson Matrix Factorization.


### Additional files 

  * aux.py, aux_plt.py, boplotting/*  - Auxiliary functions for tensor processing and plotting.


### Pre-installation Requirements

The code was tested using Python 3.6 from Anaconda 2018.*.
It requires TensorFlow 1.13.1 (in eager mode), TensorFlow Probability 0.6.0, numpy, pandas, seaborn, and matplotlib to be preinstalled.


