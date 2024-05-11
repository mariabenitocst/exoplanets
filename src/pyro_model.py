import sys
#sys.path.append("/home/mariacst/software/environments/venv_pyro/lib/python3.10/site-packages")
sys.path.append("utils/")
import torch
import pyro
from pyro.distributions import Exponential, Pareto, Uniform, Normal
from pyro.infer.autoguide.initialization import init_to_value, init_to_sample
from pyro.infer import NUTS
from functools import partial
from types import new_class
from clipppy.distributions.conundis import ConUnDisMixin
from _torchutils import TorchInterpNd
from math import pi
import numpy as np
import os

cPareto      = new_class('Pareto', (ConUnDisMixin[Pareto], Pareto), 
                         dict(register=Pareto))
cExponential = new_class('Exponential', (ConUnDisMixin[Exponential], Exponential), 
                         dict(register=Exponential))

# minimum latent mass brown dwarfs in simulation
M_min =torch.tensor(0.015, dtype=torch.float32) # M_sun
# maximum latent mass brown dwarfs in simulationI
M_max =torch.tensor(0.051, dtype=torch.float32) # M_sun

# load ATMO model
path = "../data/"
n_unique_age  = 100
n_unique_mass = 71
T_int_model = torch.tensor(np.loadtxt(path + 
                  "./ATMO_CEQ_vega_MIRI.txt").reshape((n_unique_mass, 
                                                       n_unique_age, 3))[..., -1], 
                            dtype=torch.get_default_dtype())
# velocity dispersion DM particles (sigma_DM)
v = torch.tensor(100., dtype=torch.float32) # km/s

##########################
## Probabilistic model  ##
##########################
def T_int(Mhat, Ahat, T, debugging=False):  
    """Intrinsic standard (no DM heating) temperature @ (Mhat, Ahat)"""
    age_min  = 8 # Gyr
    age_max  = 10 # Gyr
    mass_min = 0.005 # Msun
    mass_max = 0.075 # Msun

    if debugging:
        min_value = torch.min(Ahat)
        max_value = torch.max(Ahat)
        assert min_value.item() > age_min
        assert max_value.item() < age_max
    
        min_value = torch.min(Mhat)
        max_value = torch.max(Mhat)
        assert min_value.item() > mass_min
        assert max_value.item() < mass_max
    
    xgrid, ygrid = torch.meshgrid(Ahat, Mhat, indexing='xy')
    Teff = TorchInterpNd(T, (age_min, age_max),
                         (mass_min, mass_max))(xgrid, ygrid)[..., 0]
    # return
    return torch.diagonal(Teff)

def power_law_rho(R, parameters):
    """
    Power-law DM density profile @ distance R from Galactic centre
    Density has same units as DM normalisation C
    """ 
    # inner slope
    alpha = parameters[0]
    # normalization
    C     = parameters[1]
    # Return
    return C*torch.pow(R, -alpha)   

def dm_profile(r, Rsun=8.178, v=100):
    """
    Phase-space DM profile
    """
    ########################################################
    alpha   = pyro.sample('alpha', Uniform(0., 3.))
    C       = pyro.sample('C', Uniform(0.5, 100.)) # GeV/cm3
    ########################################################
    _rhoDM = power_law_rho(r, [alpha, C]) # GeV/cm3 
    _vD    = v
    # average DM velocity
    _vDM   = (8./(3*pi))**(0.5)*_vD # km/s
    # return
    return _rhoDM, _vD, _vDM

def T_DM(R, M, v, epsilon=1, Rsun=8.178):
    """Temperature [K] due to DM capture and annihilation"""  
    # gravitational constant [m3 / (kg s2)]
    _G                = 6.6743e-11
    # Stefan-Boltzmann constant [W / (m2 K4)]
    _sigma_sb         = 5.6704e-08
    conversion_into_w = 0.16021766 
    # Assuming Rjup radius
    radius = 71492000. # R_jup [m]
    # escape velocity 
    vesc   = torch.sqrt(2*_G*M/radius)*1e-3 # km/s  
    #####################################
    f = 1.
    #####################################
    _rhoDM, _vD, _vDM = dm_profile(R, Rsun, v)
    # return
    return torch.pow((f*_rhoDM*_vDM*(1+3./2.*(vesc/_vD)**2)*
                      conversion_into_w)/(4*_sigma_sb*epsilon), 1./4.)

def temp(T_int, T_DM):
    """Brown dwarf total (intrinsic + DM) temperature"""
    return ((T_int**4 + T_DM**4)**(0.25))

def model(
    N,
    rel_err_R=0.1,
    rel_err_M=0.1,
    rel_err_A=0.1,
    rel_err_T=0.1,
    Rhat=None, Mhat=None, Ahat=None, That=None,
    include_DM=True
):
    """
    Probabilistic model for observation of brown dwarfs

    Assumptions
    -----------
    -) latent Galactocentric distance BDs \in [0.1, 1] kpc
    -) latent BD massess within 14 -- 55 Mjup (~0.01 -- 0.05 Msun) 
    -) latent ages \in [8, 10] Gyr
    
    """
    tau_R   = pyro.sample('tau_R', Uniform(1, 2)) 
    gamma_M = pyro.sample('gamma_M', Normal(0.6, 0.1))
    
    with pyro.plate('For each BD', N): 
        R = pyro.sample('R', cExponential(tau_R, 
                                    constraint_lower=torch.tensor(0.1), 
                                    constraint_upper=torch.tensor(1.)))
        M = pyro.sample('M', cPareto(M_min, gamma_M, 
                                     constraint_upper=torch.tensor(M_max)))
        A = pyro.sample('A', Uniform(8, 10))

    T = pyro.deterministic('T', temp(
        T_int(M, A, T_int_model),
        T_DM(R, M, v) if include_DM else 0
    ), event_dim=1)
        
    return (
        pyro.sample('Rhat', Normal(R, rel_err_R*R), obs=Rhat), 
        pyro.sample('Mhat', Normal(M, rel_err_M*M), obs=Mhat),
        pyro.sample('Ahat', Normal(A, rel_err_A*A), obs=Ahat),
        pyro.sample('That', Normal(T, rel_err_T*T), obs=That)
    )

