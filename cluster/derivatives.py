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
    mass   = np.linspace(0.013, 0.053, size)
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

def dderivativeTDM_wrt_M(r, f, params, M, v, TDM, dervTDM_M,
                         R=R_jup.value, Rsun=8.178, epsilon=1):
    """
    Return second derivative DM-heated temperature wrt mass [K/Msun^2]
    """
    #TODO: Are the formula + conversion units correct? 
    if v:                                                                          
        _vD = v                                                                    
    else:                                                                          
        _vD    = np.sqrt(3/2.)*vc(Rsun, r, params) # km/s                          
                                                                                   
    _vDM   =  np.sqrt(8./(3*np.pi))*_vD # km/s

    _rhoDM = gNFW_rho(Rsun, r, params) # GeV/cm3
    
    conversion = 3.18578e23 #1.6021766e-7*(M_sun.value)

    # return
    return (-9./16.*_G*f*_rhoDM/(R*_sigma_sb*epsilon) *
            np.sqrt(8./(3*np.pi))/_vD * np.power(TDM, -4)*dervTDM_M*conversion)


def dderivativeT_wrt_M(r, M, A, Tint, TDM, Ttot, c, f, params, v, 
                       R=R_jup.value, Rsun=8.178, epsilon=1):
    """
    Return second derivative of temperature wrt mass [K/Msun^2]
    """
    dervTDM_M = derivativeTDM_wrt_M(r, f, params, M, v)
    dervT_M   = ((Tint/Ttot)**3* c(A) + (TDM/Ttot)**3*dervTDM_M) 

    #TODO: may be missing factors (Tint/Ttot) & (TDM/Ttot)!!
    # return
    return (-3/Ttot * np.power(dervT_M, 2) + np.power(Ttot, -3)*(
             3*Tint*Tint*np.power(c(A), 2) + 
             3*TDM*TDM*np.power(dervTDM_M, 2) +
             np.power(TDM, 3)*dderivativeTDM_wrt_M(r, f, params, M, v, TDM, dervTDM_M)))

def dderivativeTint_wrt_A(M, A, a, b):
    """
    Return second derivative of intrinsic temperature wrt age [K/Gyr^2]
    """
    # return
    return (a(M)*b(M)*(1+b(M))*np.power(A, -b(M)-2))

def dderivativeT_wrt_A(M, A, Tint, Ttot, a, b):
    """
    Return second derivative of temperature wrt age [K/Gyr^2]
    """
    dervT_A = (Tint/Ttot)**3*derivativeTintana_wrt_A(M, A, a, b)
    # return
    return (3*(-1./Ttot+1./Tint)*dervT_A*dervT_A +
            np.power(Tint/Ttot, 3)*dderivativeTint_wrt_A(M, A, a, b)
            )

def dderivativeTDM_wrt_r(r, f, params, M, v, TDM):
    # return
    return 0.25*(derivativeTDM_wrt_r(r, f, params, M, v)*(-params[0]/r-(3-params[0])/(params[1]+r))
                 + TDM*(params[0]/np.power(r, 2)+(3-params[0])/np.power(params[1]+r, 2)))

def dderivativeT_wrt_r(r, f, params, M, v, TDM, Ttot):
    """
    Return second derivative of temperature wrt Galactocenctric distance
    [K/kpc^2]
    """
    # return
    return np.power(TDM/Ttot, 3)*(-3*(1./Ttot-1./TDM)*derivativeTDM_wrt_r(r, f, params, M, v)
            + dderivativeTDM_wrt_r(r, f, params, M, v, TDM)
            )

def dderivativeTint_wrt_AM(M, A, a, b, b1, a1=25918.3):
    """
    delta_{M, A}^2(intrinsic temperature)

    Input
    -----
        M    : mass [Msun]
        A    : age [Gyr]
        a, b : Tint = a*mass + b - interpolation functions
        a1   : da/dM [K/Gyr/Msun]
        b1   : db/dM [K/Msun] - interpolation function
    """
    # return
    return (np.power(A, -b(M)-1)*(-b(M)*a1 - a(M)*b1(M)*(1-b(M))))

def dderivativeT_wrt_AM(r, M, A, a, b, b1, c, Tint, TDM, Ttot, f, params, v):
    """
    delta_{M, A}^2 (temperature)
    """
    dervTint_A = derivativeTintana_wrt_A(M, A, a, b)
    dervT_M    = ((Tint/Ttot)**3* c(A) + 
                   (TDM/Ttot)**3*derivativeTDM_wrt_M(r, f, params, M, v))
    # return
    return (-3./Ttot*(dervT_M)*((Tint/Ttot)**3*dervTint_A) + 
            3.*Tint**2/Ttot**3*c(A)*dervTint_A + 
            (Tint/Ttot)**3*dderivativeTint_wrt_AM(M, A, a, b, b1)
            )

def dderivativeT_wrt_MA(r, M, A, a, b, c, c1, Tint, TDM, Ttot, f, params, v):
    """
    delta_{A, M}^2(temperature)
    """
    dervTint_A = derivativeTintana_wrt_A(M, A, a, b)                               
    dervT_M    = ((Tint/Ttot)**3* c(A) +                                           
                   (TDM/Ttot)**3*derivativeTDM_wrt_M(r, f, params, M, v))          
    # return                                                                       
    return (-3./Ttot*(dervT_M)*((Tint/Ttot)**3*dervTint_A) +                       
            3.*Tint**2/Ttot**3*c(A)*dervTint_A +                                   
            (Tint/Ttot)**3*c1(A)
            ) 

def dderivativeTDM_wrt_Mr(r, f, params, M, v):
    """
    delta_{r, M}^2(DM temperature)
    """
    #TODO: Are the formula + conversion units correct?                          
    if v:                                                                       
        _vD = v                                                                 
    else:                                                                       
        _vD    = np.sqrt(3/2.)*vc(Rsun, r, params) # km/s                          
                                                                                
    _vDM   =  np.sqrt(8./(3*np.pi))*_vD # km/s                                  
                                                                                
    _rhoDM = gNFW_rho(Rsun, r, params) # GeV/cm3     

    dervTDM_r   = derivativeTDM_wrt_r(r, f, params, M, v)
    dervRhoDM_r = -rhoDM*(params[0]/r + (3.-params[0]/(params[1]+r)))
                                                                            
    # return                                                                    
    return (-3./16.*_G*f/(_sigma_sb*epsilon)*np.sqrt(8./(3*np.pi))/_vD/TDM**3*
            (-rhoDM/r**2 - 3./r*_rhoDM/TDM*dervTDM_r + 1./r*dervRhoDM_r)
        )

def dderivativeT_wrt_Mr(r, M, A, c, Tint, TDM, Ttot, f, params, v):
    """
    delta_{r, M}^2(temperature)
    """
    dervTDM_r = derivativeTDM_wrt_r(r, f, params, M, v)
    dervTDM_M = derivativeTDM_wrt_M(r, f, params, M, v)
    dervT_M   = ((Tint/Ttot)**3* c(A) + (TDM/Ttot)**3*dervTDM_M)
    # return                                           
    return (-3./Ttot*(dervT_M)*(TDM/Ttot)**3*dervTDM_r +                       
            3.*TDM**2/Ttot**3*dervTDM_M*dervTDM_r +                          
            (TDM/Ttot)**3*dderivativeTDM_wrt_Mr(r, f, params, M, v)
            ) 

def dderivativeTDM_wrt_rM(r, f, params, M, v):
    """
    delta_{M, r}^2(DM temperature)
    """
    # return
    return(0.25*derivativeTDM_wrt_M(r, f, params, M, v)*
                (-params[0]/r - (3-params[0])/(params[1] + r)))


def dderivativeT_wrt_rM(r, M, A, c, Tint, TDM, Ttot, f, params, v):
    """
    delta_{M, r}^2(temperature)
    """
    dervTDM_r = derivativeTDM_wrt_r(r, f, params, M, v) 
    dervTDM_M = derivativeTDM_wrt_M(r, f, params, M, v)
    dervT_M   = ((Tint/Ttot)**3* c(A) +                                       
                (TDM/Ttot)**3*dervTDM_M)          
    # return                                                                  
    return (-3./Ttot*(dervT_M)*((Tint/Ttot)**3*dervTDM_r) +                   
            3.*TDM**2/Ttot**3*dervTDM_M*dervTDM_r +                           
            (TDM/Ttot)**3*dderivativeTDM_wrt_rM(r, f, params, M, v)           
            ) 


def dderivativeT_wrt_rA(r, M, A, a, b, Tint, TDM, Ttot, f, params, v):
    """
    delta_{A, r}^2(temperature) = delta_{r, A}^2(temperature)
    """
    # return
    return (-3./Ttot*(TDM/Ttot)**3*derivativeTDM_wrt_r(r, f, params, M, v)*
        (Tint/Ttot)**3*derivativeTintana_wrt_A(M, A, a, b))


