# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:16:07 2023

@author: Administrator
"""

import sys
import gdown
import torch
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import smt
import os
import pandas as pd


# Import GP and BoTorch functions
import gpytorch as gpt
from botorch.models import SingleTaskGP, ModelListGP
#f3rom botorch.models import gpytorch
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils import standardize
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.constraints import GreaterThan
from gpytorch.models import ExactGP
from mpl_toolkits.axes_grid1 import make_axes_locatable
from smt.sampling_methods import LHS
from torch.optim import SGD
from torch.optim import Adam
from scipy.stats import norm
import time

# Fitting of the GP model with the help of the base kernel

class SimpleCustomGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        #self.mean_module = LinearMean(train_X.shape[-1])
        self.covar_module = ScaleKernel(
            #base_kernel=MaternKernel(nu=1.5, ard_num_dims=train_X.shape[-1]),
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
            #base_kernel=PeriodicKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
# Optimization of the hyperparamter of the GP model

def optimize_hyperparam_trainGP(train_X, train_Y):
    # Gp model fit

    gp_surro = SimpleCustomGP(train_X, train_Y)
    gp_surro = gp_surro.double()
    gp_surro.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-1))
    mll1 = ExactMarginalLogLikelihood(gp_surro.likelihood, gp_surro)
    
    # fit_gpytorch_model(mll)
    mll1 = mll1.to(train_X)
    gp_surro.train()
    gp_surro.likelihood.train()
    ## Here we use Adam optimizer with learning rate =0.1, user can change here with different algorithm and/or learning rate for each GP
    optimizer1 = Adam([{'params': gp_surro.parameters()}], lr=0.1) #0.01 set for BEPFM data, recommended to check the lr for any new data
    #optimizer1 = SGD([{'params': gp_surro.parameters()}], lr=0.0001)

    NUM_EPOCHS = 150

    for epoch in range(NUM_EPOCHS):
        # clear gradients
        optimizer1.zero_grad()
        # forward pass through the model to obtain the output MultivariateNormal
        output1 = gp_surro(train_X)
        # Compute negative marginal log likelihood
        loss1 = - mll1(output1, gp_surro.train_targets)
        # back prop gradients
        loss1.backward(retain_graph=True)
        # print last iterations
        if (epoch + 1) > NUM_EPOCHS: #Stopping the print for now
            print("GP Model trained:")
            print("Iteration:" + str(epoch + 1))
            print("Loss:" + str(loss1.item()))
            # print("Length Scale:" +str(gp_PZO.covar_module.base_kernel.lengthscale.item()))
            print("noise:" + str(gp_surro.likelihood.noise.item()))

        optimizer1.step()

    gp_surro.eval()
    gp_surro.likelihood.eval()
    return gp_surro

def cal_posterior(gp_surro, test_X):
    y_pred_means = torch.empty(len(test_X), 1)
    y_pred_vars = torch.empty(len(test_X), 1)
    t_X = torch.empty(1, test_X.shape[1])
    for t in range(0, len(test_X)):
        with torch.no_grad(), gpt.settings.max_lanczos_quadrature_iterations(32), \
            gpt.settings.fast_computations(covar_root_decomposition=False, log_prob=False,
                                                      solves=True), \
            gpt.settings.max_cg_iterations(100), \
            gpt.settings.max_preconditioner_size(80), \
            gpt.settings.num_trace_samples(128):

                t_X[:, 0] = test_X[t, 0]
                t_X[:, 1] = test_X[t, 1]
                #t_X = test_X.double()
                y_pred_surro = gp_surro.posterior(t_X)
                y_pred_means[t, 0] = y_pred_surro.mean
                y_pred_vars[t, 0] = y_pred_surro.variance

    return y_pred_means, y_pred_vars

def acqmanEI(y_means, y_vars, train_Y, ieval):
    y_means = y_means.detach().numpy()
    y_vars = y_vars.detach().numpy()
    y_std = np.sqrt(y_vars)
    fmax = train_Y.max()
    fmax = fmax.detach().numpy()
    best_value = fmax
    EI_val = np.zeros(len(y_vars))
    Z = np.zeros(len(y_vars))
    eta = 0.01

    for i in range(0, len(y_std)):
        if (y_std[i] <= 0):
            EI_val[i] = 0
        else:
            Z[i] =  (y_means[i]-best_value-eta)/y_std[i]
            EI_val[i] = (y_means[i]-best_value-eta)*norm.cdf(Z[i]) + y_std[i]*norm.pdf(Z[i])

    # Eliminate evaluated samples from consideration to avoid repeatation in future sampling
    EI_val[ieval] = -1
    acq_val = np.max(EI_val)
    acq_cand = [k for k, j in enumerate(EI_val) if j == acq_val]
    #print(acq_val)
    return acq_cand, acq_val, EI_val

def estimate (train_X, train_Y, test_X, y_pred_means):
    #Best solution among the evaluated data
    loss = torch.max(train_Y)
    ind = torch.argmax(train_Y)
    X_opt = torch.empty((1,2))
    X_opt[0, 0] = train_X[ind, 0]
    X_opt[0, 1] = train_X[ind, 1]

    # Best estimated solution from GP model considering the non-evaluated solution
    loss = torch.max(y_pred_means)
    ind = torch.argmax(y_pred_means)
    X_opt_GP = torch.empty((1,2))
    X_opt_GP[0, 0] = test_X[ind, 0]
    X_opt_GP[0, 1] = test_X[ind, 1]

    return X_opt, X_opt_GP

#Plotting function wrt the GP generated funtion
def plot_results(train_X, train_Y, test_X, y_pred_means, y_pred_vars, X_opt, X_opt_GP, i):
    pen = 10**0
    #Objective map
    fig,ax=plt.subplots(ncols=2,figsize=(12,5))

    a = ax[0].scatter(test_X[:,0], test_X[:,1], c=y_pred_means, cmap='viridis', linewidth=0.2)
    ax[0].scatter(train_X[:,0], train_X[:,1], marker='s', c=train_Y, cmap='jet', linewidth=0.2)
    ax[0].scatter(X_opt[0, 0], X_opt[0, 1], marker='x', c='r')
    ax[0].scatter(X_opt_GP[0, 0], X_opt_GP[0, 1], marker='o', c='r')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(a, cax=cax, orientation='vertical')
    ax[0].set_title('Objective (GP mean) map', fontsize=10)
    ax[0].axes.xaxis.set_visible(False)
    ax[0].axes.yaxis.set_visible(False)
    #ax[0].colorbar(a)

    b = ax[1].scatter(test_X[:,0], test_X[:,1], c=y_pred_vars, cmap='viridis', linewidth=0.2)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(b, cax=cax, orientation='vertical')
    ax[1].set_title('Objective (GP var) map', fontsize=10)
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)
    #ax[1].colorbar(b)

    plt.savefig('acquisition_results_step=' + str(i) +'.png', dpi = 300, bbox_inches = 'tight', pad_inches = 1.0)
    #plt.show()


def parameter_prediction(file_string):
    
    os.chdir(r"D:\scripts for labview\files\other")
    file_name = str(file_string)
    data = pd.read_csv(file_name, delimiter = "\t", skiprows = 1, header = None)

    x_1t = data.loc[:,0]
    x_2t = data.loc[:,1]
    y_t = data.loc[:,2]
    
    x_1t_valid = []
    x_2t_valid = []
    y_t_valid = []
    
    #ignore_zero_element = True
    
    for k in range(0, len(y_t)):
        if y_t[k] > - 0.5:       # Data corresponding to "no scans" are ignored
            x_1t_valid.append(x_1t[k])
            x_2t_valid.append(x_2t[k])
            y_t_valid.append(y_t[k])
    
    x1 = torch.asarray(x_1t_valid)
    x2 = torch.asarray(x_2t_valid)
    
    train_X = torch.vstack((x1, x2))
    train_X = torch.transpose(train_X, 0, 1)
    
    Y = torch.asarray(y_t_valid)
    train_Y = Y
    
    #print(train_X)
    
    
    # Generate high resolution grid space for exploration
    B= torch.linspace(0.01, 0.11, 50)     # The bias space ranging from 10 mV to 110 mV
    SP= torch.linspace(400, 2000, 50)       # The setpoint space ranging from 400 pA to 2000 pA
    X= [B, SP]
    X_feas = torch.empty((X[0].shape[0]*X[1].shape[0], 2))
    k=0
    
    for t1 in range(0, X[0].shape[0]):
        for t2 in range(0, X[1].shape[0]):
            X_feas[k, 0] = X[0][t1]
            X_feas[k, 1] = X[1][t2]
            k=k+1
    
    X_feas_norm = torch.empty((X_feas.shape[0], X_feas.shape[1]))
    train_X_norm = torch.empty((train_X.shape[0], train_X.shape[1]))
    
    # Normalize X
    for i in range(0, X_feas.shape[1]):
        X_feas_norm[:, i] = (X_feas[:, i] - torch.min(X_feas[:, i])) / (torch.max(X_feas[:, i]) - torch.min(X_feas[:, i]))
    
    for i in range(0, train_X.shape[1]):
        train_X_norm[:, i] = (train_X[:, i] - torch.min(X_feas[:, i])) / (torch.max(X_feas[:, i]) - torch.min(X_feas[:, i]))
    
    test_X, test_X_norm = X_feas, X_feas_norm
    
    #print(len(train_X))    
    
    
    ## Gp model fit
    N = 1
    m = train_X.shape[0]
    gp_surro = optimize_hyperparam_trainGP(train_X_norm, train_Y)
    idx=[]
    #params = 1 # Here we will define all the other params fixed for scanning at STM
    for i in range(1, N + 1):
        print("step: ", i)
        print("Sample #" + str(m + 1)+ "\n\n")
        # Calculate posterior for analysis for intermidiate iterations
        y_pred_means, y_pred_vars = cal_posterior(gp_surro, test_X_norm)
        if ((i == 1) or ((i % 10) == 0)):
            # Plotting functions to check the current state exploration and Pareto fronts
            # X_eval, X_GP = play_varTBO.plot_iteration_results(train_X, train_Y, test_X, y_pred_means,
            #                                            y_pred_vars, new_spec_x, new_spec_y, img, i)
            X_eval, X_GP = estimate(train_X, train_Y, test_X, y_pred_means)
            plot_results(train_X, train_Y, test_X, y_pred_means, y_pred_vars, X_eval, X_GP, len(x1))
        
    
        acq_cand, acq_val, EI_val = acqmanEI(y_pred_means, y_pred_vars, train_Y, idx)
        val = acq_val
        ind = np.random.choice(acq_cand) # When multiple points have same acq values
        idx = np.hstack((idx, ind))
        print(idx)
        if i == 1:
            idx = int(idx)
        print(idx)  # Index in the exploration space
    
    
        ## Find next point which maximizes the learning through exploration-exploitation
        if (i == 1):
            val_ini = val
          # Check for convergence
        if ((val) < 0):  # Stop for negligible expected improvement
            print("Model converged due to sufficient learning over search space ")
            break
        else:
            nextX = torch.empty((1, 2))
            nextX_norm = torch.empty(1, 2)
            nextX[0,:] = test_X[ind, :]
            nextX_norm [0, :] = test_X_norm[ind, :]
            idx_x = int(nextX[0, 0])
            idx_y = int(nextX[0, 1])    
    
    controls_array = []
    controls_array_1 = []
    controls_array.append(round(nextX.numpy()[0][0], 3))
    controls_array.append(round(nextX.numpy()[0][1], 3))
    
    controls_array = np.asarray(controls_array)
    
    for element in controls_array:
        controls_array_1.append(element)
    
       
    return controls_array_1
    

    