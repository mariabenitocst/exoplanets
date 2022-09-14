# ===========================================================================
#
# This file contains functions for simulating a mock population of brown
# dwarfs (BDs)
#
# ===========================================================================
import _utils
import imp
imp.reload(_utils)
import numpy as np
from scipy.interpolate import griddata
from astropy.constants import L_sun, R_jup, M_jup, M_sun
from _utils import temperature_withDM
import sys

def rho_bulge(r, phi, theta, R0=8.178, x0=0.899, y0=0.386, z0=0.250, 
              alpha=0.415):
    """
    Density profile for Stanek + '97 (E2) bulge [arbitrary units]
    (all spatial coordiantes are given in kpc)
    """
    x0 = x0*R0/8. # rescale to adopted R0 value
    y0 = y0*R0/8. 
    # return
    return (np.exp(-np.sqrt(np.sin(theta)**2*((np.cos(phi+alpha)/x0)**2 +
                            (np.sin(phi+alpha)/y0)**2) + 
                            (np.cos(theta)/z0)**2)*r))
def rho_disc(r, theta, R0=8.178, Rd=2.15, zh=0.40):
    """
    Density profile for Bovy and Rix disc [arbitrary units]
    (all spatial coordiantes are given in kpc)
    """
    Rd = Rd*R0/8. # rescale to adopted R0 value
    # return
    return np.exp(-r*np.sin(theta)/Rd)*np.exp(-r*np.cos(theta)/zh)

def rho(r, phi, theta, R0=8.178):
    """
    Density profile [arbitrary units]
    """
    # continuity condition at r = 1 kpc
    C    = rho_disc(1., theta, R0)/rho_bulge(1., phi, theta, R0)
    _rho = C*rho_bulge(r, phi, theta, R0)
    # return
    return (np.heaviside(1.-r, 1.)*_rho + 
            np.heaviside(r-1., 0.)*rho_disc(r, theta, R0))

def spatial_sampling(nBDs, phi=0., theta=np.pi/2., R0=8.178):
    """
    Sampling nBDs points from density profile rho using Von Neumann 
    acceptance-rejection technique
    """
    ymin = 0.1; ymax = 1.0#R0
    #print("maximimum observed GC distance = ", ymax)
    umin = np.min([rho(ymin, phi, theta), rho(1., phi, theta), 
                   rho(R0, phi, theta)])
    umax = np.max([rho(ymin, phi, theta), rho(1., phi, theta), 
                   rho(R0, phi, theta)])
    i = 0
    r = np.ones(nBDs)*100
    while i<nBDs:
        yi = np.random.uniform(ymin, ymax)
        ui = np.random.uniform(umin, umax)
        if ui < rho(yi, phi, theta, R0):
            r[i] = yi
            i+=1
    # return 
    return r

def IMF_sampling(alpha, size, Mmin=14, Mmax=55):
    """
    Sampling from power-law distribution
    """
    y = np.random.uniform(0, 1, size=size)
    return ((Mmax**(alpha+1) - Mmin**(alpha+1))*y + Mmin**(alpha+1))**(1./(alpha+1))


def mock_population_all(N, relT, relM, relRobs, relA,
                        f_true, gamma_true, rs_true, rho0_true=0.42, 
                        Tmin=0., v=None):
    """
    Generate N observed exoplanets

    Assumptions
    -----------
    1) N observed exoplanets distributed according to E2 bulge + BR disc
    2) (All) exoplanets radius = Rjup
    3) BD evolution model taken from ATMO 2020
    4) BDs have masses chosen between 14-55 Mjup assuming power-law IMF and
       unifrom age distribution between 1-10 Gyr
    5) Tobs has relative uncertainty rel_unc_Tobs
    6) Estimated masses have an uncertainty of rel_mass
    """
    #np.random.seed(42)
    #print(Tmin)
    _N = int(8.5*N)
    # galactocentric radius of simulated exoplanets
    r_obs = spatial_sampling(_N)
    # Age
    ages = np.random.uniform(1., 10., _N) # [yr] / [1-10 Gyr]
    # Mass
    mass = IMF_sampling(-0.6, _N, Mmin=6, Mmax=75) # [Mjup]
    mass = mass*M_jup.value/M_sun.value # [Msun]
    # load theoretical BD cooling model - ATMO 2020
    path =  "/home/mariacst/exoplanets/running/data/"
    data = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
    points = np.transpose(data[0:2, :])
    values = data[2]
    xi = np.transpose(np.asarray([ages, mass]))

    Teff     = griddata(points, values, xi)
    # Observed velocity (internal heating + DM)
    Tobs = temperature_withDM(r_obs, Teff, R=R_jup.value,
                           M=mass*M_sun.value,
                           f=f_true, p=[gamma_true, rs_true, rho0_true], 
                           v=v)
    # add Gaussian noise
    Tobs_wn = Tobs + np.random.normal(loc=0, scale=(relT*Tobs), size=_N)
    mass_wn = mass + np.random.normal(loc=0, scale=(relM*mass), size=_N)
    robs_wn = r_obs + np.random.normal(loc=0, scale=(relRobs*r_obs), size=_N)
    ages_wn = ages + np.random.normal(loc=0, scale=(relA*ages), size=_N)
    # select only those objects with masses between 14 and 55 Mjup and T > Tmin
    pos  = np.where((mass_wn > 0.015) & (mass_wn < 0.051) & # 16 - 53 Mjup!
                    (Tobs_wn > Tmin) & 
                    (robs_wn > 0.1) & (robs_wn < 1.) & 
                    (ages_wn > 1.002) & (ages_wn < 9.998))
    #print("Tmin = ", Tmin, len(pos[0]))
    if len(pos[0]) < N:
        sys.exit("Less objects than required!")
    #return
    return (robs_wn[pos][:N], relRobs*r_obs[pos][:N],
            Tobs_wn[pos][:N], relT*Tobs[pos][:N],
            mass_wn[pos][:N], relM*mass[pos][:N],
            ages_wn[pos][:N], relA*ages[pos][:N])

def mock_population_all_Asimov(N, relT, relM, relR, relA,                          
                          f_true, gamma_true, rs_true, rho0_true=0.42,          
                          Tmin=0., v=None):                                     
    """                                                                         
    Generate N observed exoplanets                                              
                                                                                
    Assumptions                                                                 
    -----------                                                                 
    1) N observed exoplanets distributed according to E2 bulge + BR disc        
    2) (All) exoplanets radius = Rjup                                           
    3) BD evolution model taken from ATMO 2020                                  
    4) BDs have masses chosen between 14-55 Mjup assuming power-law IMF and     
       unifrom age distribution between 1-10 Gyr                                
    """                                                                            
    _N = int(7.*N)                                                                
    # galactocentric radius of simulated exoplanets                             
    robs = spatial_sampling(_N)                                                    
    # Age                                                                       
    ages = np.random.uniform(1., 10., _N) # [yr] / [1-10 Gyr]                   
    # Mass                                                                      
    mass = IMF_sampling(-0.6, _N, Mmin=6, Mmax=75) # [Mjup]                     
    mass = mass*M_jup.value/M_sun.value # [Msun]                                
    # load theoretical BD cooling model - ATMO 2020                             
    path   =  "/home/mariacst/exoplanets/running/data/"                            
    data   = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)         
    points = np.transpose(data[0:2, :])                                            
    values = data[2]                                                               
    xi     = np.transpose(np.asarray([ages, mass]))                                
    # Intrinsic/internal temperature [K]                                        
    Teff     = griddata(points, values, xi)                                        
    #print(Teff)                                                                
    # Observed velocity (internal heating + DM) [K]                             
    Tobs = temperature_withDM(robs, Teff, R=R_jup.value,                          
                           M=mass*M_sun.value,                                     
                           f=f_true, p=[gamma_true, rs_true, rho0_true],           
                           v=v)                                                    
    # add Gaussian noise                                                        
    Tobs_wn = Tobs + np.random.normal(loc=0, scale=(relT*Tobs), size=_N)           
    mass_wn = mass + np.random.normal(loc=0, scale=(relM*mass), size=_N)           
    robs_wn = robs + np.random.normal(loc=0, scale=(relR*robs), size=_N)           
    ages_wn = ages + np.random.normal(loc=0, scale=(relA*ages), size=_N)           
    # select only those objects with masses between 14 & 55 Mjup and T > Tmin   
    # actually from 16 - 52 Mjup not to cause problems w/ derivatives           
    pos  = np.where((mass_wn > 0.015) & (mass_wn < 0.051) &                        
                    (Tobs_wn > Tmin) &                                                
                    (robs_wn > 0.1) & (robs_wn < 1.) &                             
                    (ages_wn > 1.002) & (ages_wn < 9.998))                         
    if len(pos[0]) < N:                                                            
        sys.exit("Less objects than required!")                                    
    #return                                                                     
    return (robs[pos][:N], relR*robs[pos][:N],                                     
            Tobs[pos][:N], relT*Tobs[pos][:N],                                   
            mass[pos][:N], relM*mass[pos][:N],                                     
            ages[pos][:N], relA*ages[pos][:N])                                     
