import numpy as np                                                                 
from astropy.constants import R_jup                                                
from utils import PL_T_DM
from pymultinest.solve import Solver
import sys
from derivatives import derivativeTDM_wrt_M, derivativeTDM_wrt_r
from derivatives import derivativeTintana_wrt_A

# Constant parameters & conversions ==========================================
conversion_into_K_vs_kg = 1.60217e-7                                          
conversion_into_w       = 0.16021766                                          
conv_Msun_to_kg         = 1.98841e+30 # [kg/Msun]                             
rho0                    = 0.42 # Local DM density [GeV/cm3]
epsilon                 = 1.
Rsun                    = 8.178 # Sun galactocentric distance [kpc]
# ============================================================================

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
            np.power((TDM/Ttot)**3*derivativeTDM_wrt_r(r, M, A_cte, alpha, A)*sigma_r, 2)) 


class MyModelPyMultiNest(Solver):
    """

    Args:
        data (:class:`numpy.ndarray`): an array containing the observed data
        abscissa (:class:`numpy.ndarray`): an array containing the points 
                                           at which the data were taken
        modelfunc (function): a function defining the model
        sigma (float): the standard deviation of the noise in the data
        **kwargs: keyword arguments for the run method
    """
    def __init__(self, data, abcissa, sigma, 
                sigmar, M, sigmaM, A, sigmaA, 
                Teff, points, values, a, b, c, v, rho0,
                **kwargs):
        self.T       = data # observed temperatures [K]
        self.Terr    = sigma # uncertainty observed temperatures [K]
        self.r       = abcissa # observed Galactocentric distance [kpc]
        #self.logTerr = np.log(sigma)
        self.ndata   = len(data) # number observed BDs
        
        self.sigmar  = sigmar
        self.M       = M
        self.M_in_kg = self.M*conv_Msun_to_kg
        self.sigmaM  = sigmaM
        self.A       = A
        self.sigmaA  = sigmaA
        self.Teff    = Teff
        self.points  = points
        self.values  = values
        self.a       = a
        self.b       = b
        self.c       = c
        self.v       = v
        self.rho0    = rho0

        Solver.__init__(self, **kwargs)
    
    def Prior(self, cube):                                                 
        # wider:
        #A_cte = cube[0]*(6.-1e-4) + 1e-4 # x 100 --> factor include in LogLike!
        #alpha = cube[1]*(3.-(-2.)) + (-2.)
        # extra_wider:
        #A_cte = cube[0]*(9.-1e-4) + 1e-4
        alpha = cube[1]*(3.-(-3.)) + (-3.) 
        # log:
        A_cte = cube[0]*(3.-(-2.)) + (-2.)
        # return
        return np.array([A_cte, alpha])

    def LogLikelihood(self, cube):
        # Extract params
        A_cte, alpha = cube[0], cube[1]
        #print("Inside LogLike")

        A_cte = 10**A_cte
        #A_cte = A_cte*10 # factor from prior!

        TDM = PL_T_DM(self.r, self.M_in_kg, A_cte, alpha, self.v)
        # model temperature [K]
        Tmodel = np.power(np.power(self.Teff, 4) + np.power(TDM, 4), 0.25)
        #print("Tmodel = ", Tmodel)                      
        _sigma_Tmodel2 = sigma_Tmodel2(self.r, self.M, self.A, self.sigmar,     
                                   self.sigmaM, self.sigmaA, self.Teff,            
                                   TDM, Tmodel, A_cte, alpha,
                                   self.a, self.b, self.c, self.v)
        #print("sigma_Tmodel2", _sigma_Tmodel2)
        norm = (-0.5*self.ndata*np.log(2*np.pi)
                - 0.5*np.sum(np.log(self.Terr**2 + _sigma_Tmodel2)))
        chi2 = np.sum((Tmodel-self.T)**2/(self.Terr**2 + _sigma_Tmodel2)) 
        #print(norm-0.5*chi2)
        # return
        return (norm-0.5*chi2) 
