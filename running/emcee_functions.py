import numpy as np
from astropy.constants import R_jup
from derivatives import derivativeTDM_wrt_M, derivativeTDM_wrt_r, derivativeTintana_wrt_A
from utils import T_DM, temperature_withDM

# Constant parameters & conversions ==========================================
conversion_into_K_vs_kg = 1.60217e-7                                          
conversion_into_w       = 0.16021766                                          
conv_Msun_to_kg         = 1.98841e+30 # [kg/Msun]                             
rho0                    = 0.42 # Local DM density [GeV/cm3]
epsilon                 = 1.
Rsun                    = 8.178 # Sun galactocentric distance [kpc]
# ============================================================================

def lnprior(p):
    f, gamma, rs = p
    if ( 0. < gamma < 3. and 0.01 < f < 2. and 0.01 < rs < 70.):
        return 0.
    return -np.inf


def sigma_Tmodel2(r, M, A, sigma_r, sigma_M, sigma_A,                         
                  Tint, points, values, f, params, a, b, c,
                  v, R=R_jup.value, Rsun=8.178, epsilon=1):              
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
    M_in_kg = M*conv_Msun_to_kg                                               
                                                                              
    _TDM = T_DM(r, M=M_in_kg, f=f, params=params, v=v)
    Ttot = np.power(_TDM**4 + Tint**4, 0.25)                                  
                                                                              
    dervT_M = ((Tint/Ttot)**3* c(A) + 
               (_TDM/Ttot)**3*derivativeTDM_wrt_M(r, f, params, M, v))
    # return                                                                  
    return (np.power((Tint/Ttot)**3*derivativeTintana_wrt_A(M, A, a, b)*sigma_A, 2)+
            np.power(dervT_M*sigma_M, 2)+                                     
            np.power((_TDM/Ttot)**3*derivativeTDM_wrt_r(r, f, params, M, v)*sigma_r, 2))

def residual(p, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,         
             Tobs, sigma_Tobs, Tint, points, values, a, b, c, v):
    """                                                                       
    Log likelihood function (without normalization!)                          
    """                                                                       
    # unroll free parameters                                                  
    f, gamma, rs = p                                                          
    # model temperature [K]                                                   
    Tmodel = temperature_withDM(robs, Tint, M=Mobs*conv_Msun_to_kg, f=f,      
                                p=[gamma, rs, rho0], v=v)                     
    _sigma_Tmodel2 = sigma_Tmodel2(robs, Mobs, Aobs, sigma_robs, sigma_Mobs,  
                                  sigma_Aobs, Tint, points, values,
                                  f, [gamma, rs, rho0], a, b, c, v)
    # return                                                                  
    return (-0.5*np.sum(np.log(sigma_Tobs**2 + _sigma_Tmodel2) + 
                        (Tmodel-Tobs)**2/(sigma_Tobs**2 + _sigma_Tmodel2))) 


def lnprob(p, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,            
           Tobs, sigma_Tobs, Tint, points, values, a, b, c, v):
    lp = lnprior(p)
    if not np.isfinite(lp):
        # Return
        return -np.inf
    # Return
    return lp + residual(p, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,
             Tobs, sigma_Tobs, Tint, points, values, a, b, c, v)


