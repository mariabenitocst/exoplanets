import sys
sys.path.append("/home/mariacst/exoplanets/.venv/lib/python3.6/site-packages")
import imp
import mock_generation
imp.reload(mock_generation)
from mock_generation import mock_population_all, mock_population_all_Asimov
import numpy as np
from scipy.interpolate import griddata
from utils import T_DM, temperature_withDM
from astropy.constants import R_jup, M_sun
from scipy.stats import percentileofscore
from derivatives import derivativeTintana_wrt_A
from derivatives import derivativeTDM_wrt_M, derivativeTDM_wrt_r
from scipy.interpolate import interp1d

# Constant parameters & conversions ========================================== 
conv_Msun_to_kg = 1.98841e+30 # [kg/Msun]                              
# ============================================================================ 


def sigma_Tmodel2(r, M, A, sigma_r, sigma_M, sigma_A,
                  Tint, _TDM, Ttot, a, b, c,
                  f, params, v, R=R_jup.value, Rsun=8.178, epsilon=1):
    """                                                                         
    Return squared uncertainty in model temperature [UNITS??]                   
                                                                                
    Input:                                                                      
        r : Galactocentric distance [kpc]                                       
        M : mass [Msun]                                                         
        A : age [Gyr]                                                              
        a : interpolation function                                                 
        b : interpolation function                                                 
        c : = derivativeTintana_wrt_M - interpolation function [K/Msun]            
                                                                                
    Assumption: uncertainties in age, mass and galactocentric distance          
        are independent                                                         
    """                                                                            
    dervT_M = ((Tint/Ttot)**3* c(A) +                                              
               (_TDM/Ttot)**3*derivativeTDM_wrt_M(r, f, params, M, v))
    
    # return                                                                    
    return (np.power((Tint/Ttot)**3*derivativeTintana_wrt_A(M, A, a, b)*sigma_A, 2)+ 
            np.power(dervT_M*sigma_M, 2)+                                          
            np.power((_TDM/Ttot)**3*derivativeTDM_wrt_r(r, f, params, M, v)*sigma_r, 2)) 

def lnL_sb(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,
           Tobs, sigma_Tobs, Tint, a, b, c,
           v=None, R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Return ln(L) assuming predicted temperature = DM + intrinsic
    """  
    # Calculate predicted temperature (DM + intrinsic)
    TDM = T_DM(robs, M=Mobs*conv_Msun_to_kg, f=f, params=[gamma, rs, rho0], v=v)
    Tmodel = np.power(np.power(Tint, 4) + np.power(TDM, 4), 0.25)
    
    _sigma_Tmodel2 = sigma_Tmodel2(robs, Mobs, Aobs, sigma_robs, sigma_Mobs,       
                                   sigma_Aobs, Tint, TDM, Tmodel, a, b, c, 
                                   f, [gamma, rs, rho0], v) 

    # return                                                                   
    return -0.5*np.sum(np.log(sigma_Tobs**2 + _sigma_Tmodel2) + 
                       (Tmodel-Tobs)**2/(sigma_Tobs**2 + _sigma_Tmodel2)) 


def lnL_b(Mobs, sigma_Mobs, Aobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, a, b, c):
    """
    Return ln(L) assuming predicted temperature = intrinsic
    """  
    
    sigma_Tint2 = (np.power(derivativeTintana_wrt_A(Mobs, Aobs, a, b)*sigma_Aobs, 2) + 
                   np.power(c(Aobs)*sigma_Mobs, 2))
    
    # return
    return -0.5*np.sum(np.log(sigma_Tobs**2 + sigma_Tint2) + 
                       (Tint-Tobs)**2/(sigma_Tobs**2 + sigma_Tint2)) 


def TS(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,  
       Tobs, sigma_Tobs, Tint, a, b, c,                         
       v=None, R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Test statistics
    """
    # return
    return (-2.*lnL_sb(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, 
                       Aobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, a, b, c, 
                       v=v, R=R, Rsun=Rsun, rho0=rho0, 
                       epsilon=epsilon)
            +2*lnL_b(Mobs, sigma_Mobs, Aobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, 
                     a, b, c)
            )

def sigma_Tmodel2_obs(r, M, A, sigma_r, sigma_M, sigma_A, Tint, dervTint_M, 
                  dervTint_A, f, params, v=None, R=R_jup.value, Rsun=8.178, 
                  epsilon=1):               
    """                                                                        
    Return squared uncertainty in model temperature [UNITS??]                  
                                                                               
    Input:                                                                     
        r : Galactocentric distance [kpc]                                      
        M : mass [Msun]                                                        
        A : age [Gyr]                                                          
                                                                               
    Assumption: uncertainties in age, mass and galactocentric distance         
        are independent                                                        
    """                                                                        
    M_in_kg = M*conv_Msun_to_kg                                                
                                                                               
    _TDM = T_DM(r, R=R, M=M_in_kg, Rsun=Rsun, f=f, params=params, v=v,         
                epsilon=epsilon)                                               
    Ttot = np.power(_TDM**4 + Tint**4, 0.25)                                   
                                                                               
    dervT_M = ((Tint/Ttot)**3*dervTint_M +                                     
               (_TDM/Ttot)**3*derivativeTDM_wrt_M(r, f, params, M, v=v, R=R,   
                                                  Rsun=Rsun,epsilon=epsilon))  
    # return                                                                   
    return (np.power((Tint/Ttot)**3*dervTint_A*sigma_A, 2)+                    
            np.power(dervT_M*sigma_M, 2)+                                      
            np.power((_TDM/Ttot)**3*derivativeTDM_wrt_r(r, f, params, M, v=v,  
                                  R=R, Rsun=Rsun, epsilon=epsilon)*sigma_r, 2))

def lnL_sb_obs(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,
           Tobs, sigma_Tobs, Tint, dervTint_M, dervTint_A,
           v=None, R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Return ln(L) assuming predicted temperature = DM + intrinsic
    """  
    # Calculate predicted temperature (DM + intrinsic)
    Tmodel = temperature_withDM(robs, Tint, M=Mobs*conv_Msun_to_kg, f=f,
                           p=[gamma, rs, rho0], v=v)
    
    _sigma_Tmodel2 = sigma_Tmodel2_obs(robs, Mobs, Aobs, sigma_robs, sigma_Mobs,   
                                   sigma_Aobs, Tint, dervTint_M, dervTint_A,   
                                   f, [gamma, rs, rho0], v=v, R=R, Rsun=Rsun,  
                                   epsilon=epsilon)                            
    # return                                                                   
    return -0.5*np.sum(np.log(sigma_Tobs**2 + _sigma_Tmodel2) + 
                       (Tmodel-Tobs)**2/(sigma_Tobs**2 + _sigma_Tmodel2)) 


def lnL_b_obs(sigma_Mobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, dervTint_M, 
          dervTint_A):
    """
    Return ln(L) assuming predicted temperature = intrinsic
    """  
    
    sigma_Tint2 = (np.power(dervTint_A*sigma_Aobs, 2) + 
                   np.power(dervTint_M*sigma_Mobs, 2))
    
    # return
    return -0.5*np.sum(np.log(sigma_Tobs**2 + sigma_Tint2) + 
                       (Tint-Tobs)**2/(sigma_Tobs**2 + sigma_Tint2)) 

def _TS_obs(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,  
       Tobs, sigma_Tobs, Tint, dervTint_M, dervTint_A,                         
       v=None, R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Test statistics
    """
    # return
    return (-2.*lnL_sb_obs(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, 
                       Aobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, dervTint_M, 
                       dervTint_A, v=v, R=R, Rsun=Rsun, rho0=rho0, 
                       epsilon=epsilon)
            +2*lnL_b_obs(sigma_Mobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, 
                     dervTint_M, dervTint_A)
            )


def UL_at_rs(rs, f, nBDs, relT, relM, relR, relA, 
             robs, sigmarobs, Mobs, sigmaMobs, Aobs, sigmaAobs, Tobs, 
             sigmaTobs, Teff, a, b, c, dervTint_A, dervTint_M,
             rho0=0.42, v=None, gamma_min=0.01, gamma_max=2.):
    # Grid in gamma
    gamma_k = np.linspace(gamma_min, gamma_max, 100) # change this?
    #print(gamma_k)
    for g in gamma_k:                                                          
        #print(g)
        # Observed TS                                                          
        TS_obs = _TS_obs(g, f, rs, robs, sigmarobs, Mobs, sigmaMobs, Aobs,     
                         sigmaAobs, Tobs, sigmaTobs, Teff, dervTint_M, dervTint_A, 
                         v=v)
        if TS_obs > 3.84:
            gamma_up = g
            break   
    #return
    return gamma_up

if __name__=="__main__":
    
    f         = 1.
    nBDs      = int(sys.argv[2])
    sigma     = float(sys.argv[3])
    rs        = float(sys.argv[1])

    gamma_min = -0.5
    gamma_max = 1.2

    relT = 0.1;
    ex   = "baseline"
    if ex=="baseline":
        Tcut = 0.
    elif ex=="T650":
        Tcut = 650.
    print(ex, Tcut, sigma)
    v    = 100. # km/s
    # Load ATMO2020 model
    path     = "/home/mariacst/exoplanets/running/data/"
    data     = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
    points   = np.transpose(data[0:2, :])
    values   = data[2]
    # Generate real observation
    
    rank=300
    _rs = np.ones(rank)*rs
    _g  = np.ones(rank)*100
    for i in range(rank):
        print(i)
        seed     = i #+ 350
        np.random.seed(seed)
        (robs, sigmarobs, Tobs, sigmaTobs, Mobs,                                  
        sigmaMobs, Aobs, sigmaAobs) = mock_population_all(nBDs, relT, 
                                      sigma, sigma, sigma, 0., 1., 1., 
                                      Tmin=Tcut, v=v)                        
        xi   = np.transpose(np.asarray([Aobs, Mobs]))
        Teff = griddata(points, values, xi)
        # Calculate derivatives Tint wrt Age and Mass
        masses, a, b = np.genfromtxt(path + "derv_ana_wrt_A.dat", unpack=True)
        ages, c = np.genfromtxt(path + "derv_ana_wrt_M.dat", unpack=True)
        a_interp = interp1d(masses, a)
        b_interp = interp1d(masses, b)
        c_interp = interp1d(ages, c)

        dervTint_A = derivativeTintana_wrt_A(Mobs, Aobs, a_interp, b_interp)
        dervTint_M = c_interp(Aobs) 

        # UL
        gamma_up = UL_at_rs(rs, f, nBDs, relT, sigma, sigma, sigma,
                        robs, sigmarobs, Mobs, sigmaMobs, Aobs,
                        sigmaAobs, Tobs, sigmaTobs, Teff, a_interp, b_interp,
                        c_interp, dervTint_A, dervTint_M, v=v, 
                        gamma_min=gamma_min, gamma_max=gamma_max)
        _g[i] = gamma_up

    # save results
    np.savetxt("UL_" + ex + "_nBDs%i_f%.1f_sigma%.1f_rs%.1f_gNFW.dat" 
               %(nBDs, f, sigma, rs), np.array((_rs, _g)).T, fmt="%.1f  %.4f")
