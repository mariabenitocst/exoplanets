import numpy as np
import sys
sys.path.append("/home/mariacst/software/environments/venv_pyro/lib/python3.10/site-packages")
import torch
sys.path.append("../src/")
sys.path.append("../src/utils/")
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
from bayes import approx_logBE
#from pyro_model import T_int, power_law_rho, dm_profile, T_DM, temp, model

ex    = sys.argv[1]
rank  = int(sys.argv[2])
N     = int(sys.argv[3])
sigma = float(sys.argv[4])
path  = "out/"
alpha = float(sys.argv[5])
C     = float(sys.argv[6])

logBE_noDM = approx_logBE(include_DM=False, N=N, C=C, alpha=alpha, sigma=sigma, 
                          ex=ex+"_noDM", ex_noDM=ex,
                          rank=rank, path=path)
logBE_DM   = approx_logBE(include_DM=True, 
                          ex=ex, ex_noDM=ex,
                          N=N, 
                          C=C, alpha=alpha, sigma=sigma, rank=rank, path=path)
print("log BR = ", logBE_DM - logBE_noDM)
