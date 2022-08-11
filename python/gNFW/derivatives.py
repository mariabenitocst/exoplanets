import numpy as np
from scipy.interpolate import griddata, interp1d
from scipy.misc import derivative
from astropy.constants import R_jup, M_jup, G, sigma_sb
from utils import gNFW_rho, vc

# Constant parameters & conversions ========================================== 
_sigma_sb = sigma_sb.value
_G        = G.value
conversion_into_K_vs_kg = 1.60217e-7
conversion_into_w       = 0.16021766
conv_Msun_to_kg         = 1.98841e+30 # [kg/Msun]
# ============================================================================

def derivativeTDM_wrt_M(r, f, params, M, v, R=R_jup.value, Rsun=8.178,
                        epsilon=1):
    """
    Return (analytical) derivative of DM temperature wrt mass @ 
    (f, gamma, rs, rho0, r, M, R) [K/Msun]

    Input
    -----
        r      : Galactocentric distance [kpc]
        params : DM parameters [f, gamma, rs]
        M      : exoplanet mass [Msun]
    """
    # escape velocity
    vesc   = np.sqrt(2*_G*M*conv_Msun_to_kg/R)*1e-3 # km/s
    if v:
        _vD = v
        #print(_vD, "here i am")
    else:
        _vD    = np.sqrt(3/2.)*vc(Rsun, r, params) # km/s
        
    _vDM   =  np.sqrt(8./(3*np.pi))*_vD # km/s
    _rhoDM = gNFW_rho(Rsun, r, params) # GeV/cm3

    # DM temperature^-3 [1/K^3]
    T_DM3 = np.power((f*_rhoDM*_vDM*(1+3./2.*np.power(vesc/_vD, 2))*
                     conversion_into_w)/(4*_sigma_sb*epsilon), -3./4.)
    # return 
    return (T_DM3*3./16.*np.sqrt(8./3./np.pi)*f/_sigma_sb/
            epsilon*_rhoDM*_G/_vD/R*
            conversion_into_K_vs_kg*conv_Msun_to_kg
           )


def derivativeTDM_wrt_r(r, f, params, M, v, R=R_jup.value, Rsun=8.178,
                        epsilon=1):
    """
    Return (analytical) derivative of DM temperature wrt r @ 
    (f, gamma, rs, rho0, r, M, R) [K/kpc]
    
    Assumption: DM velocity and velocity dispersion constant!
    
    Input
    -----
        r      : Galactocentric distance [kpc]
        params : DM parameters [f, gamma, rs]
        M      : exoplanet mass [Msun]
    """
    # escape velocity
    vesc   = np.sqrt(2*_G*M*conv_Msun_to_kg/R)*1e-3 # km/s
    if v:
        _vD = v
        #print(_vD, "here i am")
    else:
        _vD    = np.sqrt(3/2.)*vc(Rsun, r, params) # km/s
        
    _vDM   =  np.sqrt(8./(3*np.pi))*_vD # km/s
    _rhoDM = gNFW_rho(Rsun, r, params) # GeV/cm3

    # DM temperature [K]
    T_DM = np.power((f*_rhoDM*_vDM*(1+3./2.*np.power(vesc/_vD, 2))*
                     conversion_into_w)/(4*_sigma_sb*epsilon), 1./4.)
    
    return(0.25*T_DM*(-params[0]/r - (3-params[0])/(params[1] + r)))

def derivativeTintana_wrt_A(M, A, a, b):
    """
    Return (analytical) derivative of interinsic temperature wrt age [K/Gyr]
    (ATMO temperatures are fitted by a/A^b)
    
    Input
    -----
        M : mass [Msun]
        A : age [Gyr]
        a : =f(M) - interpolation function
        b : =f(M) - interpolation function
    """
    return (-a(M)*b(M)*np.power(A, -b(M)-1))


def derivativeTint_wrt_A(M, A, points, values, size=7000, h=0.001):
    """
    Return (numerical) derivative of intrinsic temperature wrt Age [K/Gyr]
    
    Input
    -----
        M : mass [Msun]
        A : age [Gyr]
    """   
    ages   = np.linspace(1., 10., size)
    mass   = np.ones(size)*M
    xi     = np.transpose(np.asarray([ages, mass]))
    Teff   = griddata(points, values, xi)
    # return
    return derivative(interp1d(ages, Teff), A, dx=h)

def derivativeTint_wrt_M(M, A, points, values, size=7000, h=0.001):
    """
    Return (numerical) derivative of intrinsic temperature wrt mass [K/Msun]
    
    Input
    -----
        M : mass [Msun]
        A : age [Gyr]
    """   
    ages   = np.ones(size)*A
    mass   = np.linspace(0.013, 0.060, size)
    xi     = np.transpose(np.asarray([ages, mass]))
    Teff   = griddata(points, values, xi)
    # return
    return derivative(interp1d(mass, Teff), M, dx=h)


def derivativeT_wrt_M(r, M, A, Tint, TDM, points, values, f, params,
                      size=7000, h=0.001, v=None,                              
                      R=R_jup.value, Rsun=8.178, epsilon=1):                   
    """                                                                        
    Return derivatite of (intrinsic + DM) temperature wrt mass [K/Msun]        
                                                                               
    Input                                                                      
    -----                                                                      
        r : Galactocentric distance [kpc]                                      
        M : mass [Msun]                                                        
        A : age [Gyr]                                                          
    """   
    Ttot = np.power(TDM**4 + Tint**4, 0.25)
    # return 
    return ((Tint/Ttot)**3*derivativeTint_wrt_M(M, A, points, values, size=size, 
                                                h=h) +
            (TDM/Ttot)**3*derivativeTDM_wrt_M(r, f, params, M, v=v, R=R, Rsun=Rsun,
                                              epsilon=epsilon)
           )
