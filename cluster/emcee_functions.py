import numpy as np
from astropy.constants import R_jup
from _utils import T_DM, delta_temperature_withDM
from _utils import sigma_Tmodel2, delta_sigma_Tmodel2
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


def residual(p, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,         
             Tobs, sigma_Tobs, Tint, a, b, b1, c, c1, v):
    """                                                                       
    Log likelihood function (without normalization!)                          
    """                                                                       
    # unroll free parameters                                                  
    f, gamma, rs = p
    # DM temperature [K]
    TDM = T_DM(robs, M=Mobs*conv_Msun_to_kg, f=f, params=[gamma, rs, rho0], v=v)
    # total temperature [K]
    Ttot = np.power(np.power(Tint, 4) + np.power(TDM, 4), 0.25)
    
    Ttot_corr = Ttot + delta_temperature_withDM(robs, Mobs, Aobs, sigma_robs, 
                                  sigma_Mobs, sigma_Aobs, Tint, TDM,          
                                  f, [gamma, rs, rho0], a, b, c, v)
    
    sigmaT2_corr = (sigma_Tmodel2(robs, Mobs, Aobs, sigma_robs, sigma_Mobs,  
                                  sigma_Aobs, Tint, TDM, Ttot,
                                  f, [gamma, rs, rho0], a, b, c, v) +
                    delta_sigma_Tmodel2(robs, Mobs, Aobs, sigma_robs, sigma_Mobs,
                                  sigma_Aobs, Tint, TDM, Ttot,                
                                  f, [gamma, rs, rho0], a, b, b1, c, c1, v))
    # return                                                                  
    return (-0.5*np.sum(np.log(sigma_Tobs**2 + sigmaT2_corr) + 
                        (Ttot_corr-Tobs)**2/(sigma_Tobs**2 + sigmaT2_corr))) 


def lnprob(p, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,            
           Tobs, sigma_Tobs, Tint, a, b, b1, c, c1, v):
    lp = lnprior(p)
    if not np.isfinite(lp):
        # Return
        return -np.inf
    # Return
    return lp + residual(p, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,
             Tobs, sigma_Tobs, Tint, a, b, b1, c, c1, v)



