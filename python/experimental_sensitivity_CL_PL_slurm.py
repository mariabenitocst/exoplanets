import sys
sys.path.append("/home/mariacst/exoplanets/.venv/lib/python3.6/site-packages")
sys.path.append("/home/mariacst/exoplanets/exoplanets/python/")
import imp
import mock_generation
imp.reload(mock_generation)
from mock_generation import mock_population_all, mock_population_all_Asimov
import numpy as np
from scipy.interpolate import griddata
from utils import PL_T_DM
from astropy.constants import R_jup, M_sun
from scipy.stats import percentileofscore
from derivatives import derivativeTintana_wrt_A
from derivatives import derivativeTDM_wrt_M, derivativeTDM_wrt_r
from scipy.interpolate import interp1d

# Constant parameters & conversions ========================================== 
conv_Msun_to_kg = 1.98841e+30 # [kg/Msun]                              
# ============================================================================ 

def sigma_Tmodel2(r, M, A, sigma_r, sigma_M, sigma_A,                           
                  Tint, TDM, Ttot, A_cte, alpha, a, b, c, v):                   
                                                                                
    dervT_M = ((Tint/Ttot)**3* c(A) +                                           
               (TDM/Ttot)**3*derivativeTDM_wrt_M(r, M, A_cte, alpha, v))        
    # return                                                                    
    return (np.power((Tint/Ttot)**3*derivativeTintana_wrt_A(M, A, a, b)*sigma_A, 2)+
            np.power(dervT_M*sigma_M, 2)+                                       
            np.power((TDM/Ttot)**3*derivativeTDM_wrt_r(r, M, A_cte, alpha, A)*sigma_r, 2))


def lnL_sb(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,
           Tobs, sigma_Tobs, Tint, a, b, c,
           v=None, R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Return ln(L) assuming predicted temperature = DM + intrinsic
    """  
    # Calculate predicted temperature (DM + intrinsic)
    A_cte  = rho0*np.power(Rsun, gamma)*np.power(1+Rsun/rs, 3.-gamma)*f         
    TDM    = PL_T_DM(robs, Mobs*conv_Msun_to_kg, A_cte, gamma, v)               
    Tmodel = np.power(np.power(Tint, 4) + np.power(TDM, 4), 0.25)               
                                                                                
    _sigma_Tmodel2 = sigma_Tmodel2(robs, Mobs, Aobs, sigma_robs, sigma_Mobs,
                                   sigma_Aobs, Tint, TDM, Tmodel, A_cte, gamma, 
                                   a, b, c, v)   
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

def p_value_sb(gamma_k, f, rs, nBDs, relT, relM, relR, relA, points, values,    
               a, b, c, TS_obs, steps=300, Tmin=0., v=None, ex="baseline"):
    """                                                                         
    Return p-value for gamma_k @ (f, rs) under s+b hypothesis                   
    """                                                                         
    # Compute TS pdf                                                            
    TS_k = np.zeros(steps) 
    for i in range(steps):                                                      
        # Generate experiments under s+b hypothesis  
        np.random.seed(i+1)
        (robs, sigmarobs, Tobs, sigmaTobs, Mobs, sigmaMobs, Aobs,               
        sigmaAobs) = mock_population_all(nBDs, relT, relM, relR, relA, 
                                           f, gamma_k, rs, Tmin=Tmin, v=v)     
        # Predicted intrinsic temperatures                                         
        xi       = np.transpose(np.asarray([Aobs, Mobs]))                       
        Teff     = griddata(points, values, xi)                                 
        # TS                                                                    
        TS_k[i] = TS(gamma_k, f, rs, robs, sigmarobs, Mobs, sigmaMobs, Aobs,    
                     sigmaAobs, Tobs, sigmaTobs, Teff, a, b, c, v=v)  
    # return                                                                    
    return 100-percentileofscore(TS_k, TS_obs, kind="strict") 

def p_value_b(gamma_k, f, rs, nBDs, relT, relM, relR, relA, points, values,   
              a, b, c, TS_obs, steps=300, Tmin=0., v=None, ex="baseline"):
    """                                                                         
    Return p-value for gamma_k @ (f, rs) under b hypothesis                     
    """                                                                         
    # Compute TS pdf                                                            
    TS_k = np.zeros(steps)
    for i in range(steps): 
        # Generate experiments under s+b hypothesis 
        np.random.seed(i+1)
        (robs, sigmarobs, Tobs, sigmaTobs, Mobs, sigmaMobs, Aobs,               
        sigmaAobs) = mock_population_all(nBDs, relT, relM, relR, relA, 
                                           0., 1., 1., Tmin=Tmin)
        # Predicted intrinsic temperatures                                      
        xi       = np.transpose(np.asarray([Aobs, Mobs]))                       
        Teff     = griddata(points, values, xi)                                 
        # TS                                                                    
        TS_k[i] = TS(gamma_k, f, rs, robs, sigmarobs, Mobs, sigmaMobs, Aobs,    
                     sigmaAobs, Tobs, sigmaTobs, Teff, a, b, c, v=v)  
    # return
    return 100-percentileofscore(TS_k, TS_obs, kind="strict") 


def sigma_Tmodel2_obs(r, M, A, sigma_r, sigma_M, sigma_A,
                  Tint, TDM, Ttot, A_cte, alpha, dervTint_M, dervTint_A, v): 
    
    dervT_M = ((Tint/Ttot)**3*dervTint_M + 
               (TDM/Ttot)**3*derivativeTDM_wrt_M(r, M, A_cte, alpha, v))
    # return                                                                    
    return (np.power((Tint/Ttot)**3*dervTint_A*sigma_A, 2)+ 
            np.power(dervT_M*sigma_M, 2)+                                          
            np.power((TDM/Ttot)**3*derivativeTDM_wrt_r(r, M, A_cte, alpha, A)*sigma_r, 2)) 


def lnL_sb_obs(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,
           Tobs, sigma_Tobs, Tint, dervTint_M, dervTint_A,
           v=None, R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Return ln(L) assuming predicted temperature = DM + intrinsic
    """  
    # Calculate predicted temperature (DM + intrinsic)
    A_cte  = rho0*np.power(Rsun, gamma)*np.power(1+Rsun/rs, 3.-gamma)*f
    TDM    = PL_T_DM(robs, Mobs*conv_Msun_to_kg, A_cte, gamma, v)
    Tmodel = np.power(np.power(Tint, 4) + np.power(TDM, 4), 0.25)
    
    _sigma_Tmodel2 = sigma_Tmodel2_obs(robs, Mobs, Aobs, sigma_robs, sigma_Mobs,   
                                   sigma_Aobs, Tint, TDM, Tmodel, A_cte, gamma, 
                                   dervTint_M, dervTint_A, v)
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


def UL_at_rs(rs, f, nBDs, relT, relM, relR, relA, points, values,
             robs, sigmarobs, Mobs, sigmaMobs, Aobs, sigmaAobs, Tobs, 
             sigmaTobs, Teff, a, b, c, dervTint_A, dervTint_M,
             steps=300, rho0=0.42, Tmin=0., v=None, gamma_min=0.01, gamma_max=2.95):
    # Grid in gamma
    gamma_k = np.linspace(gamma_min, gamma_max, 30) # change this?
    for g in gamma_k:
        # Observed TS                                                          
        TS_obs = _TS_obs(g, f, rs, robs, sigmarobs, Mobs, sigmaMobs, Aobs,     
                         sigmaAobs, Tobs, sigmaTobs, Teff, dervTint_M, dervTint_A, 
                         v=v)                
        # s + b hypothesis                                                     
        _p_sb = p_value_sb(g, f, rs, nBDs, relT, relM, relR, relA,          
                           points, values, a, b, c, TS_obs, steps=steps, 
                           Tmin=Tmin, v=v)
        #b hypothesis                                                          
        _p_b = p_value_b(g, f, rs, nBDs, relT, relM, relR, relA,            
                         points, values, a, b, c, TS_obs, steps=steps, 
                         Tmin=Tmin, v=v)             
        try:                                                                   
            CL = _p_sb / _p_b                                                  
        except ZeroDivisionError:                                              
            CL = 200.                                                          
        if CL < 0.05:                                                          
            gamma_up = g                                                    
            break   
    #return
    return gamma_up

if __name__=="__main__":
    
    f         = 1.#float(sys.argv[1])
    nBDs      = 100
    sigma     = 0.1
    gamma_max = float(sys.argv[4])

    rs        = float(sys.argv[1])
    gamma_min = float(sys.argv[3])
    steps     = 200 # Need to vary
    
    relT = 0.1;
    ex   = "T650"
    Tcut = 650.
    v    = 100. # km/s
    # Load ATMO2020 model
    path     = "/home/mariacst/exoplanets/running/data/"
    data     = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
    points   = np.transpose(data[0:2, :])
    values   = data[2]

    # Calculate derivatives Tint wrt Age and Mass                               
    masses, a, b = np.genfromtxt(path + "derv_ana_wrt_A.dat", unpack=True)         
    ages, c = np.genfromtxt(path + "derv_ana_wrt_M.dat", unpack=True)              
    a_interp = interp1d(masses, a)                                                 
    b_interp = interp1d(masses, b)                                                 
    c_interp = interp1d(ages, c)

    seed = int(sys.argv[2]) + 350

    # Generate real observation
    np.random.seed(seed)
    (robs, sigmarobs, Tobs, sigmaTobs, Mobs,
        sigmaMobs, Aobs, sigmaAobs) = mock_population_all(nBDs, relT, 
                                      sigma, sigma, sigma, 0., 1., 1., 
                                      Tmin=Tcut, v=v)
    xi   = np.transpose(np.asarray([Aobs, Mobs]))
    Teff = griddata(points, values, xi) 
    # Calculate derivatives Tint wrt Age and Mass
    dervTint_A = derivativeTintana_wrt_A(Mobs, Aobs, a_interp, b_interp)
    dervTint_M = c_interp(Aobs)

    # UL
    gamma_up = UL_at_rs(rs, f, nBDs, relT, sigma, sigma, sigma, points, 
                               values, robs, sigmarobs, Mobs, sigmaMobs, Aobs,
                               sigmaAobs, Tobs, sigmaTobs, Teff, a_interp, b_interp,
                               c_interp, dervTint_A, dervTint_M, steps=steps, 
                               Tmin=Tcut, v=v, 
                               gamma_min=gamma_min, gamma_max=gamma_max)
    print("%.1f  %.4f" %(rs, gamma_up))
