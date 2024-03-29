import sys
import numpy as np
import arviz as az
import pickle
from scipy.stats import binned_statistic
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
import pdb

def return_MAP(samples, nbins=50):
    _n, _bins = np.histogram(samples, bins=nbins)
    return(_bins[np.argmax(_n)])

def try_again_max(L, samples, bin_n=20, verbose=False, rank=400):
   # Create bins in Likelihood vs parameter space, find the max Likelihood     
    # value in each bin and the corresponding parameter values                     
    x = binned_statistic(samples, -L, 'min', bins=bin_n)[1]                        
    y = binned_statistic(samples, -L, 'min', bins=bin_n+1)[0]                      
                                                                                   
                                                                                   
    # Create Interpolation fanction Likelihood - parameter                         
    #pdb.set_trace()                                                               
    try:                                                                           
        z =  UnivariateSpline(x[~np.isnan(y)], y[~np.isnan(y)], s=0)               
    except:                                                                        
        #import pdb; pdb.set_trace()                                               
        xmax = x[np.min(np.where(np.isnan(y))[0])+1]                               
        pos = np.where(samples>xmax)                                               
        print(" ", pos, samples[pos])                                              
        pos = np.where(samples<xmax)                                               
        x = binned_statistic(samples[pos], -L[pos], 'min', bins=bin_n)[1]                     
        y = binned_statistic(samples[pos], -L[pos], 'min', bins=bin_n+1)[0]     
        z =  UnivariateSpline(x[~np.isnan(y)], y[~np.isnan(y)], s=0) 

    x_tmax     = samples[np.argmax(L)]                                          
    epsilon    = 1                                                              
    niteration = 0                                                              
    logLmin    = np.min(y[~np.isnan(y)]) + 1. 
    while epsilon >1e-4 and niteration < 100:                                   
        niteration +=1                                                          
        maximum = minimize(lambda x: (z(x)-logLmin)**2, x_tmax,                 
                           bounds=((samples[np.argmax(L)], np.max(samples)),))  
        epsilon  = maximum.fun                                                  
        x_tmax  = x_tmax*1.1
        _max    = maximum.x  
    
    if epsilon<1e-4:
        return _max[0]
    else:
        if verbose==True:
            print("rank=%i LImax bad!"%(rank+1))
        return (np.max(samples) + samples[np.argmax(L)])/2.

def try_again_min(L, samples, bin_n=10, verbose=False, rank=400):
    # Create bins in Likelihood vs parameter space, find the max Likelihood      
    # value in each bin and the corresponding parameter values                     
    x = binned_statistic(samples, -L, 'min', bins=bin_n)[1]                     
    y = binned_statistic(samples, -L, 'min', bins=bin_n+1)[0]                   
                                                                                
                                                                                
    # Create Interpolation fanction Likelihood - parameter                         
    #pdb.set_trace()                                                               
    try:                                                                        
        z =  UnivariateSpline(x[~np.isnan(y)], y[~np.isnan(y)], s=0)            
    except:                                                                     
        #import pdb; pdb.set_trace()                                               
        xmax = x[np.min(np.where(np.isnan(y))[0])+1]                            
        pos = np.where(samples>xmax)                                            
        print(" ", pos, samples[pos])                                           
        pos = np.where(samples<xmax)                                            
        x = binned_statistic(samples[pos], -L[pos], 'min', bins=bin_n)[1]       
        y = binned_statistic(samples[pos], -L[pos], 'min', bins=bin_n+1)[0]     
        z =  UnivariateSpline(x[~np.isnan(y)], y[~np.isnan(y)], s=0) 

    # Find in which points crosses 1sigma horizontal line (where 1 sigma           
    # corresponds to 1/2 -for 1 dof-)                                              
    x_tmin     = samples[np.argmax(L)]                                          
    epsilon    = 1                                                              
    niteration = 0                                                              
    logLmin    = np.min(y[~np.isnan(y)]) + 1.                                   
    while epsilon > 1e-4 and niteration < 40:                                   
        niteration +=1                                                          
        minimum = minimize(lambda x: (z(x)-logLmin)**2, x_tmin,                 
                           bounds=((np.min(samples), samples[np.argmax(L)]),))  
        epsilon = minimum.fun                                                   
        x_tmin  = x_tmin*0.9
        _min    = minimum.x                                                     
                                                                                
    if epsilon < 1e-4:                                                          
        return _min[0]                                                         
    else:           
        if verbose==True:
            print("rank=%i LImin bad!"%(rank+1))
        return ((np.min(samples) + samples[np.argmax(L)])/2.)


def LI(L, samples, bin_n=50, rank=1000, verbose=False):
    """
    For each parameter, construct profile likelihood and return the profile likelihood interval
    (i.e. region where the log Likelihood is within 1 of its maximum value)
    
    """   
    # Create bins in Likelihood vs parameter space, find the max Likelihood 
    # value in each bin and the corresponding parameter values
    x = binned_statistic(samples, -L, 'min', bins=bin_n)[1] 
    y = binned_statistic(samples, -L, 'min', bins=bin_n+1)[0]
    
    # Create Interpolation fanction Likelihood - parameter
    #pdb.set_trace()
    try:
        z =  UnivariateSpline(x[~np.isnan(y)], y[~np.isnan(y)], s=0)
    except:
        #import pdb; pdb.set_trace()
        xmax = x[np.min(np.where(np.isnan(y))[0])+1]
        pos = np.where(samples>xmax)
        print(" ", pos, samples[pos])
        pos = np.where(samples<xmax)
        x = binned_statistic(samples[pos], -L[pos], 'min', bins=bin_n)[1]                     
        y = binned_statistic(samples[pos], -L[pos], 'min', bins=bin_n+1)[0] 
        z =  UnivariateSpline(x[~np.isnan(y)], y[~np.isnan(y)], s=0)

    # Find in which points crosses 1sigma horizontal line (where 1 sigma 
    # corresponds to 1/2 -for 1 dof-)
    x_tmin     = samples[np.argmax(L)]
    epsilon    = 1
    niteration = 0
    logLmin    = np.min(y[~np.isnan(y)]) + 1.
    while epsilon > 1e-4 and niteration < 40:
        niteration +=1
        minimum = minimize(lambda x: (z(x)-logLmin)**2, x_tmin,
                           bounds=((np.min(samples), samples[np.argmax(L)]),))
        epsilon = minimum.fun
        x_tmin  = x_tmin*1.1
        _min    = minimum.x
        
    if epsilon < 1e-4:
        LImin = _min[0]
    else:
        #if verbose==True:
        #    print("rank=%i, No encuentra LImin"%rank)
        LImin = try_again_min(L, samples, verbose=verbose, rank=rank)
        
    x_tmax     = samples[np.argmax(L)]
    epsilon    = 1
    niteration = 0
    while epsilon > 1e-4 and niteration < 40:
        niteration +=1
        maximum = minimize(lambda x: (z(x)-logLmin)**2, x_tmax, 
                           bounds=((samples[np.argmax(L)], np.max(samples)),))
        epsilon  = maximum.fun
        x_tmax  = x_tmax*0.9
        _max    = maximum.x
    
    if epsilon < 1e-4:
        LImax = _max[0]
    else:
        #if verbose==True:
            #print("rank=%i, No encuentra LImax"%rank)
        LImax = try_again_max(L, samples, verbose=verbose, rank=rank)

    #Return
    return  LImin, LImax


def statistics(filepath, ex, nBDs, sigma, gamma, rs, rank=100):
    """
    Calculate mean, median, MAP & ML point estimates

    Inputs
    ------
        filepath    : directory where to save point estimates
        rel_unc_Tobs: relative uncertainty Tobs
        rank        : number of simulations
        D           : dimension parameter space
    """
    out_path = "/home/mariacst/exoplanets/results/gNFW/statistics_"
    output = open(out_path + ex + 
                ("_N%i_sigma%.1f_gamma%.1frs%.1f"%(nBDs, sigma, gamma, rs)), 
                "w") 

    for i in range(rank):
        #print(i)
        file_name  = (filepath + ("%i/" %(i+1))
                     + ex + 
                     ("_N%i_sigma%.1f_gamma%.1f_rs%.1f_v%ipost_equal_weights.dat" 
                     %(nBDs, sigma, gamma, rs, i+1)))
        samples    = np.genfromtxt(file_name, unpack=True)

        # f
        output.write("%.4f  "%samples[0][np.argmax(samples[3])]) # ML
        # profile L interval (region where log-L is within 1 of maximum)
        LI_low, LI_high = LI(samples[3], samples[0], bin_n=60)
        output.write("%.4f  "%LI_low)
        output.write("%.4f  "%LI_high)
        # gamma
        output.write("%.4f  "%samples[1][np.argmax(samples[3])]) # ML
        LI_low, LI_high = LI(samples[3], samples[1], bin_n=60, rank=i, verbose=True)
        output.write("%.4f  "%LI_low)
        output.write("%.4f  "%LI_high)
        # rs
        output.write("%.4f  "%samples[2][np.argmax(samples[3])]) # ML
        LI_low, LI_high = LI(samples[3], samples[2], bin_n=60)
        output.write("%.4f  "%LI_low)
        output.write("%.4f  "%LI_high)
        output.write("\n")

    output.close()

    print("Remember to manually check convergence of profile L intervals!")
    print("Oh no! I know ... it is boring!")

    # return
    return


if __name__ == '__main__':
    #_path    = "/local/mariacst/2022_exoplanets/results/gNFW/baseline_NL_longer/"
    _path    = "/home/mariacst/exoplanets/running/gNFW/baseline_NL/out/"
    ex       = "baseline_NL_gNFW_longerPriorG"
    nBDs     = [int(sys.argv[1])]
    sigma    = float(sys.argv[2])
    f        = 1.
    rs       = [5., 10., 20.]
    gamma    = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    for N in nBDs:
        for _rs in rs:
            for _g in gamma:
                print(N, _rs, _g)
                try:
                    statistics(_path, ex, N, sigma, _g, _rs, rank=100)
                except Exception as e:
                    print(e)
                    print("Noooooo :_(")
                    continue

