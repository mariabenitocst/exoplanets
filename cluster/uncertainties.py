# This script implements the propagation of mass, age, galactocentric
# uncertainties into an uncertainty in BD temperature
# (see e.g. Mana & Pennecchi 2007)
# =========================================================================== 

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
    #TODO: Include Ttot as argument to reduce internal calculations
    Ttot = np.power(_TDM**4 + Tint**4, 0.25)                                      
                                                                                  
    dervT_M = ((Tint/Ttot)**3* c(A) +                                              
               (_TDM/Ttot)**3*derivativeTDM_wrt_M(r, f, params, M, v))             
    # return                                                                    
    return (np.power((Tint/Ttot)**3*derivativeTintana_wrt_A(M, A, a, b)*sigma_A, 2)+
            np.power(dervT_M*sigma_M, 2)+                                         
            np.power((_TDM/Ttot)**3*derivativeTDM_wrt_r(r, f, params, M, v)*sigma_r, 2))


def delta_sigma_Tmodel(r, sigma_r, M, sigma_M, A, sigma_A, 
                       Tint, TDM, Ttot, a, b, b1, c, f, params, v):
    """
    Correction to linear propagation of uncertainties up to XXX
    """
    Mr = (
    dderivativeT_wrt_Mr(r, M, A, c, Tint, TDM, Ttot, f, params, v)*sigma_M**2 + 
    dderivativeT_wrt_rM(r, M, A, c, Tint, TDM, Ttot, f, params, v)*sigma_r**2
        )
    MA = (
    dderivativeT_wrt_AM(r, M, A, a, b, b1, c, Tint, TDM, Ttot, f, params, v)*sigma_A**2*
    dderivativeT_wrt_MA(r, M, a, b, c, c1, Tint, TDM, Ttot, f, params, v)*sigma_M**2
            )
    _dervT_wrt_rA = dderivativeT_wrt_rA(r, M, A, a, b, Tint, TDM, Ttot, f, params, v)
    Ar = (_dervT_wrt_rA*sigma_r**2 + _dervT_wrt_rA*sigma_A**2)
    #TODO: in description, up to XXX - which order?
    tr_11 = (
    np.power(dderivativeT_wrt_M(r, M, A, Tint, TDM, Ttot, c, f, params, v)*sigma_M**2, 2)
    + MA + Mr)
    tr_22 = (
    np.power(dderivativeT_wrt_A(M, A, Tint, Ttot, a, b)*sigma_A**2, 2)
    + MA + Ar)
    tr_33 = (
    np.power(dderivativeT_wrt_r(r, f, params, M, v, TDM, Ttot)*sigma_r**2, 2)
    + Mr + Ar)
    # return
    return (tr_11 + tr_22 + tr_33)
