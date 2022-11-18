import sys
sys.path.append("/home/mariacst/exoplanets/running/.env/lib/python3.6/site-packages")
import emcee
import numpy as np
from scipy.interpolate import griddata
#import imp
#import mock_generation
#imp.reload(mock_generation)
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
ex         = "Tcut_onlySigmaT" #sys.argv[1]
rank       = int(sys.argv[4])
nBDs       = int(sys.argv[1])
relTobs    = 0.1
sigma_r    = 0.#float(sys.argv[4])
sigma_M    = 0.#float(sys.argv[5])
sigma_A    = 0.#float(sys.argv[6])
f_true     = 1.
gamma_true = 1.2#float(sys.argv[7])
rs_true    = 10.#float(sys.argv[8])
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
     sigmaMobs, Aobs, sigmaAobs) = mock_population_all(nBDs, relTobs, sigma_M,
                                      sigma_r, sigma_A, f_true, gamma_true,
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
masses, b1 = np.genfromtxt(path + "dderv_ana_wrt_AM.dat", unpack=True)
b1_interp  = interp1d(masses, b1)
ages, c1  = np.genfromtxt(path + "dderv_ana_wrt_MA.dat", unpack=True)
c1_interp = interp1d(ages, c1)

ndim     = 3
nwalkers = int(sys.argv[2])#150
# first guess
p0 = [[0.9, 0.9, 20.] + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
              args=(robs, sigmarobs, Mobs, sigmaMobs, Aobs, sigmaAobs, Tobs,
                    sigmaTobs, Teff, a_interp, b_interp, b1_interp,
                    c_interp, c1_interp, v), 
              moves=[(emcee.moves.DEMove(), 0.8),           
                     (emcee.moves.DESnookerMove(), 0.2)])

steps = int(sys.argv[3])
pos, prob, state  = sampler.run_mcmc(p0, 200, progress=True)
sampler.reset()
pos, prob, state  = sampler.run_mcmc(pos, steps, progress=True)

# Save likelihood
path = "./"#/hdfs/local/mariacst/exoplanets/results/onlySigmaT/"
filepath    = (path + "like_" + ex)
file_object = open(filepath + ("_N%i_sigma0.1_gamma%.1frs%.1f_nwalkers%i_steps%i_"
                    %(nBDs, gamma_true, rs_true, nwalkers, steps))            
                    + "v" + str(rank), "wb") 
pickle.dump(sampler.flatlnprobability, file_object, protocol=2)
file_object.close() 
# Save posterior
#path = "./results/"
filepath    = (path + "posterior_" + ex)       
file_object2 = open(filepath + ("_N%i_sigma0.1_gamma%.1frs%.1f_nwalkers%i_steps%i_"
                    %(nBDs, gamma_true, rs_true, nwalkers, steps))            
                    + "v" + str(rank), "wb")                                  
pickle.dump(sampler.flatchain, file_object2, protocol=2)
file_object2.close()

print(sampler.get_autocorr_time()[0])
