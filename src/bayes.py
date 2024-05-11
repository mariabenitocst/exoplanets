import numpy as np
import sys
sys.path.append("/home/mariacst/software/environments/venv_pyro/lib/python3.10/site-packages")
import torch
sys.path.append("utils/")
from pyro_model import model
from functools import partial
import pyro
from scipy.stats import gaussian_kde
from pyro.distributions import Exponential, Pareto, Uniform, Normal
from tqdm.auto import trange, tqdm
from math import pi, log
from _torchutils import TorchInterpNd
from types import new_class
from clipppy.distributions.conundis import ConUnDisMixin
import os
#from pyro_model import T_int, power_law_rho, dm_profile, T_DM, temp, model

cPareto      = new_class('Pareto', (ConUnDisMixin[Pareto], Pareto), dict(register=Pareto))
cExponential = new_class('Exponential', (ConUnDisMixin[Exponential], Exponential), dict(register=Exponential))

def return_mode(samples):
    """Mode 1D distribution"""
    # kernel density estimation
    kde = gaussian_kde(samples)
    # evaluate KDE @ various points
    x       = np.linspace(samples.min(), samples.max(), 1000)
    density = kde(x)
    # return
    return x[np.argmax(density)]

def approx_logBE(include_DM=False,
                 ex="old_v0", 
                 ex_noDM="old_v0",
                 rank=20, 
                 N=100, 
                 sigma=0.01,
                 C=20., 
                 alpha=1.5,
                 path = "out/", 
                 random=False):
    """
    Approximate Bayesian Evidence
    """

    # load data
    file_name = path + "samps_{}_N{}_s{}_alpha{}_C{}_v100.0_r{}.pt".format(ex, 
                        N, sigma, alpha, C, rank)
    samples   = torch.load(file_name, map_location=torch.device('cpu'))
    file_name = path + "data_{}_N{}_s{}_alpha{}_C{}_v100.0_r{}.pt".format(ex_noDM, 
                        N, sigma, alpha, C, rank)
    data  = torch.load(file_name, map_location=torch.device('cpu'))
    file_name = path + "real_{}_N{}_s{}_alpha{}_C{}_v100.0_r{}.pt".format(ex_noDM, 
                        N, sigma, alpha, C, rank)
    real  = torch.load(file_name, map_location=torch.device('cpu'))
    
    if random:
        N     = len(real["R"])
        sigma = real["sigma"] 
        alpha = real["alpha"]
        C     = real["C"]
    
    # conditional probability (posterior) @ posterior mode
    samps      = []
    samps_mean = dict()
    for key in ['A', 'M', 'R']:
        samps_mean[key] = []
        for i in range(N):
            samps.append(samples[key][:, i]) 
            samps_mean[key].append(return_mode(samples[key][:, i]))
    for key in (['tau_R', 'gamma_M', 'alpha', 'C'] if include_DM else ['tau_R', 'gamma_M']):
        samps_mean[key] = []
        samps.append(samples[key][:])
        samps_mean[key].append(return_mode(samples[key][:]))

    cov          = torch.stack(samps).cov()
    lconditional = -0.5*(torch.stack(samps).shape[0]*np.log(2*np.pi) + cov.logdet())
    
    cmodel = partial(model, N=N, rel_err_R=sigma, rel_err_M=sigma, 
                     rel_err_A=sigma, rel_err_T=sigma, 
                     include_DM=include_DM)

    if include_DM==False:
        ccmodel = pyro.condition(cmodel, data=dict(Rhat=data['Rhat'], 
                                               Mhat=data['Mhat'], 
                                               Ahat=data['Ahat'], That=data['That'], 
                                               R=torch.tensor(samps_mean['R'], dtype=torch.float32), 
                                               M=torch.tensor(samps_mean['M'], dtype=torch.float32),
                                               A=torch.tensor(samps_mean['A'], dtype=torch.float32),
                                               gamma_M=torch.tensor(samps_mean['gamma_M'], dtype=torch.float32),
                                               tau_R=torch.tensor(samps_mean['tau_R'], dtype=torch.float32)
                                               ))
    else:
        ccmodel = pyro.condition(cmodel, data=dict(Rhat=data['Rhat'], 
                                               Mhat=data['Mhat'], 
                                               Ahat=data['Ahat'], That=data['That'], 
                                               R=torch.tensor(samps_mean['R'], dtype=torch.float32), 
                                               M=torch.tensor(samps_mean['M'], dtype=torch.float32),
                                               A=torch.tensor(samps_mean['A'], dtype=torch.float32),
                                               gamma_M=torch.tensor(samps_mean['gamma_M'], dtype=torch.float32),
                                               tau_R=torch.tensor(samps_mean['tau_R'], dtype=torch.float32), 
                                               alpha=torch.tensor(samps_mean['alpha'], dtype=torch.float32), 
                                               C=torch.tensor(samps_mean['C'], dtype=torch.float32)
                                               ))
    
    trace = pyro.poutine.trace(ccmodel).get_trace()
    logp  = trace.log_prob_sum() # log-joint probability
    # return
    return (N, sigma, alpha, logp-lconditional) if random else logp - lconditional
