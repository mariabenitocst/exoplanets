#TODO: En realidad no he hecho un chequado chequeado, sino que he tirado
#para adelante como los de Alicante, esperando que todo este bien!

import sys
sys.path.append("/home/mariacst/exoplanets/.venv/lib/python3.6/site-packages")
import imp
import mock_generation
imp.reload(mock_generation)
from mock_generation import mock_population_all, mock_population_all_Asimov
import numpy as np
from scipy.interpolate import griddata
from astropy.constants import R_jup, M_sun
from scipy.stats import percentileofscore
from scipy.interpolate import interp1d
from _utils import dderivativeT_wrt_AM, dderivativeT_wrt_MA, dderivativeT_wrt_A
from _utils import T_DM, delta_temperature_withDM, sigma_Tmodel2
from _utils import delta_sigma_Tmodel2

# Constant parameters & conversions ========================================== 
conv_Msun_to_kg = 1.98841e+30 # [kg/Msun]                              
# ============================================================================ 
def lnL_sb_obs(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,
           Tobs, sigma_Tobs, Tint, a, b, b1, c, c1,
           v=100., R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Return ln(L) assuming predicted temperature = DM + intrinsic
    """  
    # DM temperature [K]
    TDM = T_DM(robs, M=Mobs*conv_Msun_to_kg, f=f, params=[gamma, rs, rho0], v=v)
    # Calculate predicted temperature (DM + intrinsic)
    Ttot = np.power(np.power(Tint, 4) + np.power(TDM, 4), 0.25) 
    
    Ttot_corr = Ttot + delta_temperature_withDM(robs, Mobs, Aobs,
                        sigma_robs, sigma_Mobs, sigma_Aobs, Tint, TDM,
                        f, [gamma, rs, rho0], a, b, c, v)

    sigmaT2_corr = (sigma_Tmodel2(robs, Mobs, Aobs, sigma_robs, sigma_Mobs,
                            sigma_Aobs, Tint, TDM, Ttot, f,
                            [gamma, rs, rho0], a, b, c, v) +
                    delta_sigma_Tmodel2(robs, Mobs, Aobs, sigma_robs,
                        sigma_Mobs, sigma_Aobs, Tint, TDM, Ttot, f,
                        [gamma, rs, rho0], a, b, b1, c, c1, v)
                    )

    # return
    return -0.5*np.sum(np.log(sigma_Tobs**2 + sigmaT2_corr) + 
                       (Ttot_corr-Tobs)**2/(sigma_Tobs**2 + sigmaT2_corr)) 

def delta_temperature_Tint(M, A, sigma_A, Tint, a, b):
    """                                                                            
    2nd/3rd-order correction to expected temperature due to non-linear relation 
    between temperature and mass, age, galactocentric distance variables.          
                                                                                   
    Correction = 0.5*Tr(H_0*C)=0.5*(delta_MM T*sigma_M^2 +
                                    delta_AA T*sigma_A^2)
    """
    # return
    return 0.5*(dderivativeT_wrt_A(M, A, Tint, Tint, a, b)*np.power(sigma_A, 2))  

def delta_sigma_Tint2(M, A, sigma_M, sigma_A, Ttot, a, b, b1, c, c1):        
    """                                                                         
    2nd-order correction to linear propagion of uncertainties                   
    """
    MA = (                                                                      
    dderivativeT_wrt_AM(1., M, A, a, b, b1, c, Ttot, 0., Ttot, 1., [1., 1., 1.], None)*sigma_A**2*
    dderivativeT_wrt_MA(1., M, A, a, b, c, c1, Ttot, 0., Ttot, 1., [1., 1., 1.], None)*sigma_M**2
            )
    # return                                                                    
    return (0.5*(np.power(dderivativeT_wrt_A(M, A, Ttot, Ttot, a, b)*sigma_A**2, 2))  
            + MA)


def lnL_b_obs(Mobs, Aobs, sigma_Mobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, 
              a, b, b1, c, c1):
    """
    Return ln(L) assuming predicted temperature = intrinsic
    """  
    Tint_corr = Tint + delta_temperature_Tint(Mobs, Aobs, sigma_Aobs, Tint, a, b) 

    # TODO: check that delta_sigma_Tint2 is doing correct thing!!!!
    sigmaT2_corr = (sigma_Tmodel2(1., Mobs, Aobs, 0., sigma_Mobs, sigma_Aobs,
                            Tint, 0., Tint, 1., [1., 1., 1.], a, b, c, None) + 
                    delta_sigma_Tint2(Mobs, Aobs,
                        sigma_Mobs, sigma_Aobs, Tint, a, b, b1, c, c1)
                    )  
    # return
    return -0.5*np.sum(np.log(sigma_Tobs**2 + sigmaT2_corr) + 
                       (Tint_corr-Tobs)**2/(sigma_Tobs**2 + sigmaT2_corr)) 

def _TS_obs(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,
            Tobs, sigma_Tobs, Tint, a, b, b1, c, c1, v=None):
    """
    Test statistics
    """
    # return
    return (-2.*lnL_sb_obs(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, 
                       Aobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, a, b, b1,
                       c, c1, v=v)
            +2*lnL_b_obs(Mobs, Aobs, sigma_Mobs, sigma_Aobs, Tobs, sigma_Tobs, 
                       Tint, a, b, b1, c, c1)
            )


def UL_at_rs(rs, f, nBDs, robs, sigmarobs, Mobs, sigmaMobs, Aobs, sigmaAobs, 
             Tobs, sigmaTobs, Teff, a, b, b1, c, c1,
             rho0=0.42, v=None, gamma_min=0.01, gamma_max=2.95):
    # Grid in gamma
    gamma_k = np.linspace(gamma_min, gamma_max, 100) # change this?
    #print(gamma_k)
    for g in gamma_k:
        #print(g)
        # Observed TS
        TS_obs = _TS_obs(g, f, rs, robs, sigmarobs, Mobs, sigmaMobs, Aobs,
                         sigmaAobs, Tobs, sigmaTobs, Teff, a, b, b1, c, c1, 
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
    
    gamma_min = 0.
    gamma_max = 2.
    
    relT = 0.1;
    ex   = "baseline"
    if ex=="baseline":
        Tcut = 0.
    elif ex=="T650":
        Tcut=650.
    print(Tcut, nBDs, sigma)
    v    = 100. # km/s
    # Load ATMO2020 model
    path     = "/home/mariacst/exoplanets/running/data/"
    data     = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
    points   = np.transpose(data[0:2, :])
    values   = data[2]

    rank=100
    _rs = np.ones(rank)*rs
    _g  = np.ones(rank)*100
    for i in range(rank):
        print(i)
        # Generate real observation
        seed = i #+ 350
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
        masses, b1 = np.genfromtxt(path + "dderv_ana_wrt_AM.dat", unpack=True)
        b1_interp  = interp1d(masses, b1)
        ages, c1  = np.genfromtxt(path + "dderv_ana_wrt_MA.dat", unpack=True)
        c1_interp = interp1d(ages, c1)   

        gamma_up = UL_at_rs(rs, f, nBDs, robs, sigmarobs, Mobs, sigmaMobs, Aobs,
                        sigmaAobs, Tobs, sigmaTobs, Teff, a_interp, b_interp,
                        b1_interp, c_interp, c1_interp, 
                        v=v, gamma_min=gamma_min, gamma_max=gamma_max)
        _g[i] = gamma_up

    # save results
    np.savetxt("UL_" + ex + "_nBDs%i_f%.1f_sigma%.1f_rs%.1f_gNFW.dat" 
            %(nBDs, f, sigma, rs), np.array((_rs, _g)).T, fmt="%.1f  %.4f")
