import sys
sys.path.append("/home/mariacst/exoplanets/running/.env/lib/python3.6/site-packages")
sys.path.append("/home/sven/exoplanetenv/lib/python3.6/site-packages")
import numpy as np
from scipy.interpolate import griddata
import importlib
import mock_generation
importlib.reload(mock_generation)
from mock_generation import mock_population_all
from astropy.constants import R_jup
import glob
import pickle
from scipy.interpolate import interp1d
from utils import temperature_withDM
import matplotlib.pyplot as plt
import multinest_functions as solver
import time

# Constant parameters & conversions ==========================================
rho0                    = 0.42 # Local DM density [GeV/cm3]
epsilon                 = 1.
Rsun                    = 8.178 # Sun galactocentric distance [kpc]
conv_Msun_to_kg         = 1.98841e+30 # [kg/Msun] 
# ============================================================================
# Input parameters
ex         = sys.argv[1]
rank       = int(sys.argv[2])
nBDs       = int(sys.argv[3])
relTobs    = 0.1
sigma      = float(sys.argv[4])
f_true     = 1.
gamma_true = float(sys.argv[5])
rs_true    = float(sys.argv[6])
v          = 100.
Tcut       = 0.
# ------------------------------------------------------------------------
# Load theoretical cooling model
path = "/home/mariacst/exoplanets/running/data/"
data = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
points = np.transpose(data[0:2, :])
values = data[2]
# Mock observation
np.random.seed(rank)
(robs, sigmarobs, Tobs, sigmaTobs, Mobs,
     sigmaMobs, Aobs, sigmaAobs) = mock_population_all(nBDs, relTobs, sigma,
                                      sigma, sigma, f_true, gamma_true,
                                      rs_true, rho0_true=rho0, Tmin=Tcut, v=v)
## calculate predictic intrinsic temperature
xi       = np.transpose(np.asarray([Aobs, Mobs]))
Teff     = griddata(points, values, xi)
# Load variables analytical derivatives Tint
masses, a, b = np.genfromtxt(path + "derv_ana_wrt_A.dat", unpack=True)
ages, c = np.genfromtxt(path + "derv_ana_wrt_M.dat", unpack=True)
a_interp = interp1d(masses, a)
b_interp = interp1d(masses, b)
c_interp = interp1d(ages, c)

# ---------------------- multinest solver -------------------------------------

# number of dimensions our problem has
parameters = ["A", "alpha"]
n_params   = len(parameters)
nlive      = 9000
tol        = 0.1

t0 = time.time()
# run MultiNest
solution = solver.MyModelPyMultiNest(Tobs, robs, sigmaTobs, sigmarobs, Mobs,
    sigmaMobs, Aobs, sigmaAobs, Teff, points, values, a_interp, b_interp, 
    c_interp, v, rho0, n_dims=n_params, n_live_points=nlive, evidence_tolerance=tol,
    sampling_efficiency=0.5,
    outputfiles_basename="out/log/{}/".format(rank)+ex+ "_N{}_sigma{}_gamma{}_rs{}_v{}".format(nBDs, sigma, gamma_true, rs_true, rank), 
    resume=False, verbose=False)
t1 = time.time()
print("Time taken to run 'PyMultiNest' is {} seconds".format(t1-t0))
