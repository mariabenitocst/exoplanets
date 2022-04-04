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

