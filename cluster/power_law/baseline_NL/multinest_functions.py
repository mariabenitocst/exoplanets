import numpy as np                                                                 
from astropy.constants import R_jup                                                
from _utils import PL_T_DM, delta_temperature_withDM
from pymultinest.solve import Solver
import sys
from _utils import sigma_Tmodel2, delta_sigma_Tmodel2 

# Constant parameters & conversions ==========================================
conversion_into_K_vs_kg = 1.60217e-7                                          
conversion_into_w       = 0.16021766                                          
conv_Msun_to_kg         = 1.98841e+30 # [kg/Msun]                             
rho0                    = 0.42 # Local DM density [GeV/cm3]
epsilon                 = 1.
Rsun                    = 8.178 # Sun galactocentric distance [kpc]
# ============================================================================

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
                Teff, a, b, b1, c, c1, v, rho0,
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
        self.a       = a
        self.b       = b
        self.b1      = b1
        self.c       = c
        self.c1      = c1
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
        #A_cte = A_cte*100

        TDM = PL_T_DM(self.r, self.M_in_kg, A_cte, alpha, self.v)
        # model temperature [K]
        Ttot = np.power(np.power(self.Teff, 4) + np.power(TDM, 4), 0.25)
        Ttot_corr = Ttot + delta_temperature_withDM(self.r, self.M, self.A,     
                        self.sigmar, self.sigmaM, self.sigmaA, self.Teff, TDM,  
                        A_cte, alpha, self.a, self.b, self.c,         
                        self.v) 
        #print("Ttot = ", Ttot, " & corrected, ", Ttot_corr)
        #import pdb; pdb.set_trace()
        sigmaT2_corr = (sigma_Tmodel2(self.r, self.M, self.A, self.sigmar,      
                            self.sigmaM, self.sigmaA, self.Teff, TDM, Ttot,     
                            A_cte, alpha, self.a, self.b, self.c,  
                            self.v) +                                           
                    delta_sigma_Tmodel2(self.r, self.M, self.A, self.sigmar,    
                        self.sigmaM, self.sigmaA, self.Teff, TDM, Ttot,         
                        A_cte, alpha, self.a, self.b, self.b1,     
                        self.c, self.c1, self.v)) 

        norm = (-0.5*self.ndata*np.log(2*np.pi)
                - 0.5*np.sum(np.log(self.Terr**2 + sigmaT2_corr)))
        chi2 = np.sum((Ttot_corr-self.T)**2/(self.Terr**2 + sigmaT2_corr)) 
        #print(norm-0.5*chi2)
        # return
        return (norm-0.5*chi2) 
