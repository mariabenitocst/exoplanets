import sys
sys.path.append("/home/mariacst/software/environments/venv_pyro/lib/python3.10/site-packages")
sys.path.append("../src/")
sys.path.append("../src/utils/")
import time
import pyro
import torch
torch.set_num_threads(1)
from pyro.distributions import Exponential, Pareto, Uniform, Normal
from pyro.infer.autoguide.initialization import init_to_value, init_to_sample
from functools import partial
from types import new_class
from clipppy.distributions.conundis import ConUnDisMixin
from _torchutils import TorchInterpNd
from pyro.infer import NUTS, MCMC
from math import pi
import numpy as np
from pyro_model import T_int, power_law_rho, dm_profile, T_DM, temp, model

cPareto      = new_class('Pareto', (ConUnDisMixin[Pareto], Pareto), 
                         dict(register=Pareto))
cExponential = new_class('Exponential', (ConUnDisMixin[Exponential], Exponential), 
                         dict(register=Exponential))


######################
## Input parameters ##
######################
ex         = sys.argv[1]
rank       = int(sys.argv[2])
N          = int(sys.argv[3])
sigma      = float(sys.argv[4])
f_true     = 1.
alpha_true = float(sys.argv[5])
C_true     = float(sys.argv[6])
v          = torch.tensor(100., dtype=torch.float32) # km/s
Tcut       = None # Not implemented yet!

# condition model on true params
smodel = partial(model, N=N, rel_err_R=sigma, rel_err_M=sigma, 
                 rel_err_A=sigma, rel_err_T=sigma)

# generate mock obs
#pyro.set_rng_seed(rank)
mock = pyro.poutine.trace(
            pyro.condition(smodel, data=dict(
                alpha=torch.tensor(alpha_true),
                C=torch.tensor(C_true),
                gamma_M=0.6,
                tau_R=1.43
            ))
        ).get_trace()

data = {
            key: mock.nodes[key]['value']
            for key in ('That', 'Rhat', 'Mhat', 'Ahat')
        }
real = {
            key: mock.nodes[key]['value']
            for key in ('T', 'R', 'M', 'A', 'alpha', 'C')
        }
# save mock obs and latent
torch.save(data, 'out/data_{}_N{}_s{}_alpha{}_C{}_v{}_r{}.pt'.format(
           ex, N, sigma, alpha_true, C_true, v, rank))
torch.save(real, 'out/real_{}_N{}_s{}_alpha{}_C{}_v{}_r{}.pt'.format(
           ex, N, sigma, alpha_true, C_true, v, rank))

# condition model on generated data
cmodel = pyro.condition(smodel, data=data)

nuts_kernel = NUTS(cmodel, 
                   init_strategy=init_to_value(values={'R':real["R"],
                                                       'M':real["M"],
                                                       'A':real["A"],
                                                       'T':real["T"]}, 
                   fallback=init_to_sample()), full_mass=True,
                   adapt_step_size=True, adapt_mass_matrix=True,
                   jit_compile=True, ignore_jit_warnings=True)
start_time = time.time()
mcmc = MCMC(nuts_kernel, num_samples=3200, warmup_steps=1400)
mcmc.run()
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
mcmc.summary()

samples = mcmc.get_samples()
torch.save(samples, 'out/samps_{}_N{}_s{}_alpha{}_C{}_v{}_r{}.pt'.format(
           ex, N, sigma, alpha_true, C_true, v, rank))
