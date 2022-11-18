import sys
sys.path.append("/home/mariacst/exoplanets/running/.env/lib/python3.6/site-packages")
import emcee
import numpy as np
from scipy.interpolate import griddata
import imp
import mock_generation
imp.reload(mock_generation)
from mock_generation import mock_population_all
from astropy.constants import R_jup
import glob
import pickle
from scipy.interpolate import interp1d
from emcee_functions import lnprob

#start = time.time()

# Constant parameters & conversions ==========================================
rho0                    = 0.42 # Local DM density [GeV/cm3]
epsilon                 = 1.
Rsun                    = 8.178 # Sun galactocentric distance [kpc]
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
Tcut       = 650.
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
# ------------------ RECONSTRUCTION --------------------------------------
# Load variables analytical derivatives Tint
masses, a, b = np.genfromtxt(path + "derv_ana_wrt_A.dat", unpack=True)
ages, c = np.genfromtxt(path + "derv_ana_wrt_M.dat", unpack=True)
a_interp = interp1d(masses, a)
b_interp = interp1d(masses, b)
c_interp = interp1d(ages, c)

ndim     = 3
nwalkers = 500
# first guess
p0 = [[0.9, 0.9, 20.] + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
              args=(robs, sigmarobs, Mobs, sigmaMobs, Aobs, sigmaAobs, Tobs,
                    sigmaTobs, Teff, points, values, a_interp, b_interp, 
                    c_interp, v), 
              moves=[(emcee.moves.DEMove(), 0.8),           
                     (emcee.moves.DESnookerMove(), 0.2)])
  
pos, prob, state  = sampler.run_mcmc(p0, 2000, progress=True)
sampler.reset()
pos, prob, state  = sampler.run_mcmc(pos, 12000, progress=True)


# Save likelihood
path = "/hdfs/local/mariacst/exoplanets/results/likelihood/v100/analytic_test/"#+ex+"/"
filepath    = (path + "like_" + ex)
file_object = open(filepath + ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"
                    %(nBDs, sigma, f_true, gamma_true, rs_true))            
                    + "v" + str(rank), "wb") 
pickle.dump(sampler.flatlnprobability, file_object, protocol=2)
file_object.close() 
# Save posterior
path = "/hdfs/local/mariacst/exoplanets/results/posterior/v100/analytic_test/"#+ex+"/"
filepath    = (path + "posterior_" + ex)       
file_object2 = open(filepath + ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"
                    %(nBDs, sigma, f_true, gamma_true, rs_true))            
                    + "v" + str(rank), "wb")                                  
pickle.dump(sampler.flatchain, file_object2, protocol=2)
file_object2.close()

print(sampler.get_autocorr_time()[0])
