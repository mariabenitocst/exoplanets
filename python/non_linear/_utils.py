import numpy as np
from astropy.constants import R_jup, M_jup, G, sigma_sb
from scipy.special import hyp2f1
from scipy.interpolate import interp1d
import astropy.units as u


# Constant parameters & conversions ==========================================  
_sigma_sb = sigma_sb.value                                                      
_G        = G.value                                                             
conversion_into_K_vs_kg = 1.60217e-7                                            
conversion_into_w       = 0.16021766                                            
conv_Msun_to_kg         = 1.98841e+30 # [kg/Msun]                               
# ============================================================================ 

def delta_temperature_withDM(r, M, A, sigma_r, sigma_M, sigma_A, Tint, TDM, A_cte,
                             alpha, a, b, c, v):                                  
    """                                                                            
    2nd/3rd-order correction to expected temperature due to non-linear relation 
    between temperature and mass, age, galactocentric distance variables.          
                                                                                   
    Correction = 0.5*Tr(H_0*C)=0.5*(delta_MM T*sigma_M^2 +                         
                                    delta_AA T*sigma_A^2 +                         
                                    delta_RR T*sigma_R^2)                          
    """                                                                            
    Ttot = np.power(TDM**4 + Tint**4, 0.25)                                        
    # return                                                                       
    return 0.5*(dderivativeT_wrt_M(r, M, A, Tint, TDM, Ttot, c, A_cte, alpha, v)*np.power(sigma_M, 2)
            + dderivativeT_wrt_A(M, A, Tint, Ttot, a, b)*np.power(sigma_A, 2)   
            + dderivativeT_wrt_r(r, A_cte, alpha, M, v, TDM, Ttot)*np.power(sigma_r, 2)
            )  

def vc(Rsun, Rint, parameters):
    data = np.genfromtxt("../data/rc_e2bulge_R08.178_J_corr.dat", unpack=True)
    r = data[0]
    vB = data[1]
    data = np.genfromtxt("../data/rc_hgdisc_R08.178_corr.dat", unpack=True)
    vD = data[1]
    vDM = vgNFW(Rsun, r, parameters)
    vtot = np.sqrt(np.power(vB, 2) + np.power(vD, 2)+ np.power(vDM, 2))
    vtot_intp = interp1d(r, vtot)
    return vtot_intp(Rint)

def vgNFW(Rsun, R, parameters):
    """
    Rotation velocity for gNFW dark matter density profile
    """
    # gNFW parameters
    gamma = parameters[0]
    Rs    = parameters[1]
    rho0  = parameters[2] 
    v     = []; 
    for Rint in R:
        hyp=np.float(hyp2f1(3-gamma,3-gamma,4-gamma,-Rint/Rs))
        Integral=(-2**(2+3*gamma)*np.pi*Rint**(3-gamma)*(1+
                  Rsun*(1./Rs))**(3-gamma)*rho0*hyp)/(-3+gamma)
        v.append(np.sqrt(1.18997*10.**(-31.)*Integral/Rint)*3.08567758*10.**(16.))
    v = np.array(v,dtype=np.float64)      
    # Return
    return v

def gNFW_rho(Rsun, R, parameters):
    """
    Return gNFW density profile at r distance from the GC
    Denstiy has same units as local DM density rho0
    """
    # gNFW parameters
    gamma = parameters[0] 
    Rs    = parameters[1]
    rho0  = parameters[2]
    # Density profile
    rho   = rho0*np.power(Rsun/R, gamma)*np.power((Rs+Rsun)/(Rs+R), 3-gamma)    
    # Return
    return rho

def heat_DM(r, f=1, R=R_jup.value, M=M_jup.value, Rsun=8.178, 
            parameters=[1., 20., 0.42], v=None):
    """
    Heat flow due to DM capture and annihilation
    """
    vesc   = (np.sqrt(2*_G*M/R))*1e-3 # km/s 
    if v:
        _vD = v
        #print(_vD, "here i am")
    else:
        _vD    = np.sqrt(3/2.)*vc(Rsun, r, parameters) # km/s
        #print("rC")
    _vDM   =  np.sqrt(8./(3*np.pi))*_vD # km/s
    _rhoDM = gNFW_rho(Rsun, r, parameters) # GeV/cm3

    # return
    return (f*np.pi*R**2*_rhoDM*_vDM*(1+3./2.*np.power(vesc/_vD, 2))*
            conversion_into_w) # W

def PL_T_DM(r, M, A, alpha, v, R=R_jup.value, epsilon=1.):
    """                                                                         
    DM temperature                                                              
    """                                                                            
    # escape velocity                                                              
    vesc   = np.sqrt(2*_G*M/R)*1e-3 # km/s                                      
    _vD = v                                                                 
                                                                                
    _vDM   =  np.sqrt(8./(3*np.pi))*_vD # km/s                                  
    _rhoDM = A*np.power(r, -alpha)  # GeV/cm3!!!!                                
    # return                                                                    
    return np.power((_rhoDM*_vDM*(1+3./2.*np.power(vesc/_vD, 2))*             
                    conversion_into_w)/(4*_sigma_sb*epsilon), 1./4.) 

def T_DM(r, R=R_jup.value, M=M_jup.value, Rsun=8.178, f=1., 
         params=[1., 20., 0.42], v=None, epsilon=1.):                                       
    """                                                                        
    DM temperature                                                             
    """   
    # escape velocity
    vesc   = np.sqrt(2*_G*M/R)*1e-3 # km/s                      
    if v:                                                                      
        _vD = v                                                                
    else:                                                                      
        _vD    = np.sqrt(3/2.)*vc(Rsun, r, params) # km/s                      
                                                                               
    _vDM   =  np.sqrt(8./(3*np.pi))*_vD # km/s                                 
    _rhoDM = gNFW_rho(Rsun, r, params) # GeV/cm3                               
    # return                                                                   
    return np.power((f*_rhoDM*_vDM*(1+3./2.*np.power(vesc/_vD, 2))*    
                    conversion_into_w)/(4*_sigma_sb*epsilon), 1./4.)

def temperature_withDM(r, Tint, R=R_jup.value, M=M_jup.value, 
                       f=1., p=[1., 20., 0.42], v=None, Rsun=8.178, epsilon=1):
    """
    Exoplanet temperature : internal heating + DM heating
    """
    return (np.power(np.power(Tint, 4) + 
                     np.power(T_DM(r, R=R, M=M, Rsun=Rsun, f=f, params=p, v=v, 
                                   epsilon=epsilon), 4)
                     , 0.25))

def temperature(heat, R, epsilon=1):
    return np.power(heat/(4*np.pi*R**2*sigma_sb*epsilon), 0.25)

def heat(temp, R, epsilon=1):
        return (4*np.pi*R**2*sigma_sb.value*temp**4*epsilon)

# ============================================================================
# DERIVATIVES
# ============================================================================

def derivativeTDM_wrt_M(r, M, A, alpha, v, R=R_jup.value, Rsun=8.178,
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
    _vD    = v
    _vDM   = np.sqrt(8./(3*np.pi))*_vD # km/s
    _rhoDM = A*np.power(r, -alpha)  # GeV/cm3!!!!

    # DM temperature^-3 [1/K^3]
    T_DM3 = np.power((_rhoDM*_vDM*(1+3./2.*np.power(vesc/_vD, 2))*
                     conversion_into_w)/(4*_sigma_sb*epsilon), -3./4.)
    # return 
    return (T_DM3*3./16.*np.sqrt(8./3./np.pi)/_sigma_sb/
            epsilon*_rhoDM*_G/_vD/R*
            conversion_into_K_vs_kg*conv_Msun_to_kg
           )


def derivativeTDM_wrt_r(r, M, A, alpha, v, R=R_jup.value, Rsun=8.178,
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
    _vD = v
    _vDM   =  np.sqrt(8./(3*np.pi))*_vD # km/s
    
    _rhoDM = A*np.power(r, -alpha)  # GeV/cm3!!!!

    # DM temperature [K]
    T_DM = np.power((_rhoDM*_vDM*(1+3./2.*np.power(vesc/_vD, 2))*
                     conversion_into_w)/(4*_sigma_sb*epsilon), 1./4.)
    
    return(0.25*T_DM*(-alpha/r))

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


# ===========================================================================   
# Second-order derivative                                                     
# =========================================================================== 

def dderivativeTDM_wrt_M(TDM, dervTDM_M):                                          
    """                                                                            
    Return second derivative DM-heated temperature wrt mass [K/Msun^2]             
    """                                                                            
    # return                                                                       
    return -3./TDM*dervTDM_M**2  

def dderivativeT_wrt_M(r, M, A, Tint, TDM, Ttot, c, A_cte, alpha, v):
    """                                                                            
    Return second derivative of temperature wrt mass [K/Msun^2]                    
    """                                                                            
    #TODO: sacar fuera der la funci?n dervTDM_M --> input                          
    dervTDM_M = derivativeTDM_wrt_M(r, M, A_cte, alpha, v)
    dervT_M   = ((Tint/Ttot)**3* c(A) + (TDM/Ttot)**3*dervTDM_M)
    # return
    return (-3./Ttot*dervT_M**2 + np.power(Ttot, -3)*(                             
             3.*Tint**2*c(A)**2 +                                                  
             3.*TDM**2*dervTDM_M**2 +                                              
             TDM**3*dderivativeTDM_wrt_M(TDM, dervTDM_M))                          
    ) 

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

def dderivativeTDM_wrt_r(r, A_cte, alpha, M, v, TDM, dervTDM_r):
    # return                                                                    
    return 0.25*(dervTDM_r*(-alpha/r)
       + TDM*(alpha/np.power(r, 2))) 

def dderivativeT_wrt_r(r, A_cte, alpha, M, v, TDM, Ttot):
    """                                                                         
    Return second derivative of temperature wrt Galactocenctric distance        
    [K/kpc^2]                                                                   
    """     
    dervTDM_r = derivativeTDM_wrt_r(r, M, A_cte, alpha, v)
    # return
    return np.power(TDM/Ttot, 3)*(-3*(1./Ttot-1./TDM)*dervTDM_r
            + dderivativeTDM_wrt_r(r, A_cte, alpha, M, v, TDM, dervTDM_r)                     
            )  

#TODO:esta derivada es mas smooth que la derivada numerica, esto es consecuencia
# de la interpolacion --> las diferencias son menores que un 10%                
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


def dderivativeT_wrt_AM(r, M, A, a, b, b1, c, Tint, TDM, Ttot, A_cte, alpha, v):
    """                                                                         
    delta_{M, A}^2 (temperature)                                                
    """                                                                         
    dervTint_A = derivativeTintana_wrt_A(M, A, a, b)                            
    dervT_M    = ((Tint/Ttot)**3* c(A) +                                        
                   (TDM/Ttot)**3*derivativeTDM_wrt_M(r, M, A_cte, alpha, v))
    # return                                                                    
    return (-3./Ttot*(dervT_M)*((Tint/Ttot)**3*dervTint_A) +                    
            3.*Tint**2/Ttot**3*c(A)*dervTint_A +                                
            (Tint/Ttot)**3*dderivativeTint_wrt_AM(M, A, a, b, b1)               
            )  

#TODO:sierra behavior de ddTint/dAdM --> como consecuencia de que solo          
# contamos con Tint para valores discretos de A, y esta grid es coarse!         
# las diferencias entre la derivada numerica y la analitica son hasta de        
# un factor 2!                                                                  
#Afecta esto a los resultados? Smooth the curve?                                
def dderivativeT_wrt_MA(r, M, A, a, b, c, c1, Tint, TDM, Ttot, A_cte, alpha, v):
    """                                                                         
    delta_{A, M}^2(temperature)                                                 
    """                                                                         
    dervTint_A = derivativeTintana_wrt_A(M, A, a, b)                            
    dervT_M    = ((Tint/Ttot)**3* c(A) +                                        
                   (TDM/Ttot)**3*derivativeTDM_wrt_M(r, M, A_cte, alpha, v))
    # return                                                                    
    return (-3./Ttot*(dervT_M)*((Tint/Ttot)**3*dervTint_A) +                    
            3.*Tint**2/Ttot**3*c(A)*dervTint_A +                                
            (Tint/Ttot)**3*c1(A)                                                
            )  

def dderivativeTDM_wrt_rM(r, M, A_cte, alpha, v):
    """                                                                         
    delta_{M, r}^2(DM temperature) = delta_{r, M}^2(DM temperatur)
    """                                                                         
    # return                                                                    
    return -0.25*derivativeTDM_wrt_M(r, M, A_cte, alpha, v)*alpha/r

def dderivativeT_wrt_rM(r, M, A, c, Tint, TDM, Ttot, A_cte, alpha, v):
    """                                                                         
    delta_{M, r}^2(temperature) = delta_{r, M}^2(temperature)
    """                                                                         
    dervTDM_r = derivativeTDM_wrt_r(r, M, A_cte, alpha, v)
    dervTDM_M = derivativeTDM_wrt_M(r, M, A_cte, alpha, v)
    dervT_M   = ((Tint/Ttot)**3* c(A) +                                         
                (TDM/Ttot)**3*dervTDM_M)                                        
    # return                                                                    
    return (-3./Ttot*(dervT_M)*((TDM/Ttot)**3*dervTDM_r) +                      
            3.*TDM**2/Ttot**3*dervTDM_M*dervTDM_r +                             
            (TDM/Ttot)**3*dderivativeTDM_wrt_rM(r, M, A_cte, alpha, v)
            )  

def dderivativeT_wrt_rA(r, M, A, a, b, Tint, TDM, Ttot, A_cte, alpha, v):
    """                                                                         
    delta_{A, r}^2(temperature) = delta_{r, A}^2(temperature)                   
    """                                                                         
    # return                                                                    
    return (-3./Ttot*(TDM/Ttot)**3*derivativeTDM_wrt_r(r, M, A_cte, alpha, v)*
        (Tint/Ttot)**3*derivativeTintana_wrt_A(M, A, a, b)) 


# ============================================================================  
# UNCERTAINTIES                                                                 
# This script implements the propagation of mass, age, galactocentric           
# distance (r) uncertainties into an uncertainty in BD temperature              
#                                                                               
# Notice that temperature is not a linear function of mass, age and r.          
# Therefore, the temperature function can not be approximated by the 1st-order  
# Taylor expansion around the means of mass, age and r.                         
# This script implements uncertainty propagation assuming 2nd-order Taylor      
# expansion (see e.g. Mana & Pennecchi 2007)                                    
# ===========================================================================   
                                                                                
def sigma_Tmodel2(r, M, A, sigma_r, sigma_M, sigma_A,                           
                  Tint, TDM, Ttot, A_cte, alpha, a, b, c, v):
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
               (TDM/Ttot)**3*derivativeTDM_wrt_M(r, M, A_cte, alpha, v))
    # return                                                                    
    return (np.power((Tint/Ttot)**3*derivativeTintana_wrt_A(M, A, a, b)*sigma_A, 2)+
            np.power(dervT_M*sigma_M, 2)+                                       
            np.power((TDM/Ttot)**3*derivativeTDM_wrt_r(r, M, A_cte, alpha, v)*sigma_r, 2))
                                                                                
                                                                                
def delta_sigma_Tmodel2(r, M, A, sigma_r, sigma_M, sigma_A,                     
                        Tint, TDM, Ttot, A_cte, alpha, a, b, b1, c, c1, v):        
    """                                                                         
    2nd-order correction to linear propagion of uncertainties                   
    """                                                                         
    Mr = (                                                                      
    2*dderivativeT_wrt_rM(r, M, A, c, Tint, TDM, Ttot, A_cte, alpha, v)*sigma_r**2   
        )                                                                       
    MA = (                                                                      
    dderivativeT_wrt_AM(r, M, A, a, b, b1, c, Tint, TDM, Ttot, A_cte, alpha, v)*sigma_A**2*
    dderivativeT_wrt_MA(r, M, A, a, b, c, c1, Tint, TDM, Ttot, A_cte, alpha, v)*sigma_M**2
            )                                                                   
    Ar = (dderivativeT_wrt_rA(r, M, A, a, b, Tint, TDM, Ttot, A_cte, alpha, v)*    
          (sigma_r**2 + sigma_A**2))                                            
    # return                                                                    
    return (0.5*(                                                               
    np.power(dderivativeT_wrt_M(r, M, A, Tint, TDM, Ttot, c, A_cte, alpha, v)*sigma_M**2, 2)
    +                                                                           
    np.power(dderivativeT_wrt_A(M, A, Tint, Ttot, a, b)*sigma_A**2, 2)          
    +                                                                           
    np.power(dderivativeT_wrt_r(r, A_cte, alpha, M, v, TDM, Ttot)*sigma_r**2, 2))  
    + MA + Mr + Ar) 
