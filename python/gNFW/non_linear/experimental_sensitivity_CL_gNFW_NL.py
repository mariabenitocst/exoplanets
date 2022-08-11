import sys
sys.path.append("/home/mariacst/exoplanets/.venv/lib/python3.6/site-packages")
sys.path.append("/home/mariacst/exoplanets/exoplanets/python/")
import imp
import mock_generation
imp.reload(mock_generation)
from mock_generation import mock_population_all
import numpy as np
from scipy.interpolate import griddata
from _utils import T_DM
from astropy.constants import R_jup, M_sun
from scipy.stats import percentileofscore
from scipy.interpolate import interp1d
from _utils import delta_temperature_withDM, sigma_Tmodel2, delta_sigma_Tmodel2 
from _utils import derivativeTintana_wrt_A, derivativeTDM_wrt_M
from _utils import derivativeTDM_wrt_r
from _utils import dderivativeT_wrt_A, dderivativeT_wrt_AM, dderivativeT_wrt_MA

# Constant parameters & conversions ========================================== 
conv_Msun_to_kg = 1.98841e+30 # [kg/Msun]                              
# ============================================================================ 

def delta_temperature_int(M, A, sigma_A, Tint, a, b):                                  
    """                                                                            
    2nd/3rd-order correction to expected temperature due to non-linear relation 
    between temperature and mass, age, galactocentric distance variables.          
                                                                                   
    Correction = 0.5*Tr(H_0*C)=0.5*(delta_MM T*sigma_M^2 +                         
                                    delta_AA T*sigma_A^2 +                         
                                    delta_RR T*sigma_R^2)                          
    """                                                                            
    # return                                                                       
    return 0.5*(dderivativeT_wrt_A(M, A, Tint, Tint, a, b)*np.power(sigma_A, 2)   
            )

def delta_sigma_Tint2(r, M, A, sigma_M, sigma_A, Tint, a, b, b1, c, c1):     
    """                                                                         
    2nd-order correction to linear propagion of uncertainties in Tint                   
    """                                                                         
    MA = (                                                                      
    dderivativeT_wrt_AM(r, M, A, a, b, b1, c, Tint, 0., Tint, 1., [1., 1., 1.], 1.)*sigma_A**2*
    dderivativeT_wrt_MA(r, M, A, a, b, c, c1, Tint, 0., Tint, 1., [1., 1., 1.], 1.)*sigma_M**2
            )                                                                   
    # return                                                                    
    return (0.5*(                                                               
    np.power(dderivativeT_wrt_A(M, A, Tint, Tint, a, b)*sigma_A**2, 2) + MA))


def lnL_sb(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,
           Tobs, sigma_Tobs, Tint, a, b, b1, c, c1, v,
           R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Return ln(L) assuming predicted temperature = DM + intrinsic
    """  
    # Calculate predicted temperature (DM + intrinsic)
    TDM    = T_DM(robs, M=Mobs*conv_Msun_to_kg, f=f, params=[gamma, rs, rho0], v=v)
    Ttot   = np.power(np.power(Tint, 4) + np.power(TDM, 4), 0.25)               
    
    Tmodel = Ttot + delta_temperature_withDM(robs, Mobs, Aobs,     
                        sigma_robs, sigma_Mobs, sigma_Aobs, Tint, TDM,  
                        f, [gamma, rs, rho0], a, b, c, v) 
                                                                      
    _sigma_Tmodel2 = (sigma_Tmodel2(robs, Mobs, Aobs, sigma_robs, sigma_Mobs,   
                                   sigma_Aobs, Tint, TDM, Ttot, f, 
                                   [gamma, rs, rho0], a, b, c, v) +
                      delta_sigma_Tmodel2(robs, Mobs, Aobs, sigma_robs,    
                        sigma_Mobs, sigma_Aobs, Tint, TDM, Ttot,         
                        f, [gamma, rs, rho0], a, b, b1, c, c1, v)) 

    # return                                                                   
    return -0.5*np.sum(np.log(sigma_Tobs**2 + _sigma_Tmodel2) + 
                       (Tmodel-Tobs)**2/(sigma_Tobs**2 + _sigma_Tmodel2)) 


def lnL_b(robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs, Tobs, 
          sigma_Tobs, Tint, a, b, b1, c, c1):
    """
    Return ln(L) assuming predicted temperature = intrinsic
    """  
    
    Tint_corr = Tint + delta_temperature_int(Mobs, Aobs, sigma_Aobs, Tint, a, b)
    
    sigma_Tint2 = (sigma_Tmodel2(robs, Mobs, Aobs, sigma_robs, sigma_Mobs,
                                   sigma_Aobs, Tint, 0., Tint, 1., 1., 
                                   a, b, c, 1.) +
             delta_sigma_Tint2(robs, Mobs, Aobs, sigma_Mobs, sigma_Aobs, Tint,
             a, b, b1, c, c1))
    
    # return
    return -0.5*np.sum(np.log(sigma_Tobs**2 + sigma_Tint2) + 
                       (Tint_corr-Tobs)**2/(sigma_Tobs**2 + sigma_Tint2)) 


def TS(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,  
       Tobs, sigma_Tobs, Tint, a, b, b1, c, c1, v,
       R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Test statistics
    """
    # return
    return (-2.*lnL_sb(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, 
                       Aobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, a, b, b1, c, 
                       c1, v)
            +2*lnL_b(robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs, Tobs, 
                     sigma_Tobs, Tint, a, b, b1, c, c1)
            )

def p_value_sb(gamma_k, f, rs, nBDs, relT, relM, relR, relA, points, values,    
               a, b, b1, c, c1, v, TS_obs, 
               steps=300, Tmin=0., ex="baseline"):
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
                     sigmaAobs, Tobs, sigmaTobs, Teff, a, b, b1, c, c1, v)
    # return                                                                    
    return 100-percentileofscore(TS_k, TS_obs, kind="strict") 

def p_value_b(gamma_k, f, rs, nBDs, relT, relM, relR, relA, points, values,   
              a, b, b1, c, c1, TS_obs, 
              steps=300, Tmin=0., ex="baseline"):
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
                     sigmaAobs, Tobs, sigmaTobs, Teff, a, b, b1, c, c1, v)
    # return
    return 100-percentileofscore(TS_k, TS_obs, kind="strict") 


def sigma_Tmodel2_obs(r, M, A, sigma_r, sigma_M, sigma_A,
                  Tint, TDM, Ttot, f, params, dervTint_M, dervTint_A, v): 
    
    dervT_M = ((Tint/Ttot)**3*dervTint_M + 
               (TDM/Ttot)**3*derivativeTDM_wrt_M(r, f, params, M, v))
    # return                                                                    
    return (np.power((Tint/Ttot)**3*dervTint_A*sigma_A, 2)+ 
            np.power(dervT_M*sigma_M, 2)+                                          
            np.power((TDM/Ttot)**3*derivativeTDM_wrt_r(r, f, params, M, A)*sigma_r, 2)) 


def lnL_sb_obs(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,
           Tobs, sigma_Tobs, Tint, dervTint_M, dervTint_A, a, b, b1, c, c1, v,
           R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Return ln(L) assuming predicted temperature = DM + intrinsic
    """  
    # Calculate predicted temperature (DM + intrinsic)
    TDM    = T_DM(robs, M=Mobs*conv_Msun_to_kg, f=f, params=[gamma, rs, rho0], v=v)
    Ttot   = np.power(np.power(Tint, 4) + np.power(TDM, 4), 0.25)
    
    Tmodel = Ttot + delta_temperature_withDM(robs, Mobs, Aobs,     
                        sigma_robs, sigma_Mobs, sigma_Aobs, Tint, TDM,
                        f, [gamma, rs, rho0], a, b, c, v)

    _sigma_Tmodel2 = (sigma_Tmodel2_obs(robs, Mobs, Aobs, sigma_robs, sigma_Mobs,   
                                   sigma_Aobs, Tint, TDM, Ttot, f, [gamma, rs, rho0], 
                                   dervTint_M, dervTint_A, v) +
                      delta_sigma_Tmodel2(robs, Mobs, Aobs, sigma_robs,    
                        sigma_Mobs, sigma_Aobs, Tint, TDM, Ttot,         
                        f, [gamma, rs, rho0], a, b, b1, c, c1, v)) 
    # return                                                                   
    return -0.5*np.sum(np.log(sigma_Tobs**2 + _sigma_Tmodel2) + 
                       (Tmodel-Tobs)**2/(sigma_Tobs**2 + _sigma_Tmodel2)) 


def lnL_b_obs(robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,
              Tobs, sigma_Tobs, Tint, dervTint_M, dervTint_A, a, b, b1, c, c1,
              R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):

    """
    Return ln(L) assuming predicted temperature = intrinsic
    """  
    
    Tint_corr = Tint + delta_temperature_int(Mobs, Aobs, sigma_Aobs, Tint, a, b)

    _sigma_Tmodel2 = (sigma_Tmodel2_obs(robs, Mobs, Aobs, sigma_robs, sigma_Mobs,
                                   sigma_Aobs, Tint, 0., Tint, 1., 1., 
                                   dervTint_M, dervTint_A, 1.) +
             delta_sigma_Tint2(robs, Mobs, Aobs, sigma_Mobs, sigma_Aobs, Tint,
             a, b, b1, c, c1))
    
    # return
    return -0.5*np.sum(np.log(sigma_Tobs**2 + _sigma_Tmodel2) + 
                       (Tint_corr-Tobs)**2/(sigma_Tobs**2 + _sigma_Tmodel2)) 

def _TS_obs(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,  
       Tobs, sigma_Tobs, Tint, dervTint_M, dervTint_A, a, b, b1, c, c1, v,
       _lnL_b, R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Test statistics
    """
    # return
    return (-2.*lnL_sb_obs(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, 
                       Aobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, dervTint_M, 
                       dervTint_A, a, b, b1, c, c1, v)
            +2*_lnL_b
            )


def UL_at_rs(rs, f, nBDs, relT, relM, relR, relA, points, values,
             robs, sigmarobs, Mobs, sigmaMobs, Aobs, sigmaAobs, Tobs, 
             sigmaTobs, Teff, a, b, c, dervTint_A, dervTint_M, b1, c1, 
             steps=300, rho0=0.42, v=None, Tmin=0., gamma_min=0.01, gamma_max=2.95):
    # Grid in gamma
    gamma_k = np.linspace(gamma_min, gamma_max, 30) # change this?             
    print(gamma_k)
    
    _lnL_b_obs = lnL_b_obs(robs, sigmarobs, Mobs, sigmaMobs,
                       Aobs, sigmaAobs, Tobs, sigmaTobs, Teff, dervTint_M,
                       dervTint_A, a, b, b1, c, c1)
    for g in gamma_k:
        print(g)
        # Observed TS                                                          
        TS_obs = _TS_obs(g, f, rs, robs, sigmarobs, Mobs, sigmaMobs, Aobs,     
                         sigmaAobs, Tobs, sigmaTobs, Teff, dervTint_M, dervTint_A, 
                         a, b, b1, c, c1, v, _lnL_b_obs)
        # s + b hypothesis                                                     
        _p_sb = p_value_sb(g, f, rs, nBDs, relT, relM, relR, relA, points, 
                           values, a, b, b1, c, c1, v, TS_obs, 
                           steps=steps, Tmin=Tmin)
        #b hypothesis                                                          
        _p_b = p_value_b(g, f, rs, nBDs, relT, relM, relR, relA, points,
                         values, a, b, b1, c, c1, TS_obs, 
                         steps=steps, Tmin=Tmin)
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
    
    f         = 1.
    nBDs      = 100
    sigma     = 0.1
    Tcut      = 0. # K
    gamma_max = [2.]

    rs        = [10.]
    gamma_min = [0.01]
    steps     = 200 # Need to vary
    
    relT = 0.1;
    ex   = "baseline"
    v    = 100. # km/s
    # Load ATMO2020 model
    path     = "/home/mariacst/exoplanets/running/data/"
    data     = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
    points   = np.transpose(data[0:2, :])
    values   = data[2]
    gamma_up = np.ones(len(rs))*10.
    # Generate real observation
    seed     = int(sys.argv[1]) + 350
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

    masses, b1 = np.genfromtxt(path + "dderv_ana_wrt_AM.dat", unpack=True)
    b1_interp  = interp1d(masses, b1)
    ages, c1  = np.genfromtxt(path + "dderv_ana_wrt_MA.dat", unpack=True)
    c1_interp = interp1d(ages, c1) 

    i = 0
    for _rs in rs:
        # UL
        gamma_up[i] = UL_at_rs(_rs, f, nBDs, relT, sigma, sigma, sigma, points, 
                               values, robs, sigmarobs, Mobs, sigmaMobs, Aobs,
                               sigmaAobs, Tobs, sigmaTobs, Teff, a_interp, b_interp,
                               c_interp, dervTint_A, dervTint_M, b1_interp, 
                               c1_interp, steps=steps, Tmin=Tcut, v=v, 
                               gamma_min=gamma_min[i], gamma_max=gamma_max[i])
        print("%.1f  %.4f" %(_rs, gamma_up[i]))
        i+=1

    #print(gamma_up)
    output = np.array((np.array(rs), gamma_up))
    # save results
    np.savetxt("UL_" + ex + "_nBDs%i_sigma%.1f_f%.1f_steps%i_Asimov_PL.dat" 
               %(nBDs, sigma, f, steps), 
               output.T, fmt="%.4f  %.4f")
