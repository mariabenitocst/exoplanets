from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from scipy.stats import binned_statistic
import numpy as np
import matplotlib.pyplot as plt

def try_again_max(L, samples, bin_n=20):
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
        print("Try MAX BAD")
        return ((np.max(samples) + samples[np.argmax(L)])/2.)

def try_again_min(L, samples, bin_n=20):
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
        return _min[0]                                                         
    else:                                                                       
        print("Try MIN BAD!")                                         
        return ((np.min(samples) + samples[np.argmax(L)])/2.)
    

def profile_L_interval(L, samples, bin_n=50, verbose=False):                                       
    """                                                                            
    For each parameter, construct profile likelihood and return the profile likelihood interval
    (i.e. region where the log Likelihood is within 1 of its maximum value)  
    --> same as LI, just name updated!
                                                                                   
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
        print(_min[0], np.min(samples))
        LImin = try_again_min(L, samples)                                                    
                                                                                   
    x_tmax     = samples[np.argmax(L)]                                             
    epsilon    = 1                                                                 
    niteration = 0                                                                 
    while epsilon >1e-4 and niteration < 100:
        niteration +=1                                                             
        maximum = minimize(lambda x: (z(x)-logLmin)**2, x_tmax,                    
                           bounds=((samples[np.argmax(L)], np.max(samples)),))  
        epsilon  = maximum.fun                                                     
        x_tmax  = x_tmax*0.9
        _max    = maximum.x                                                        
                                                                                   
    if epsilon < 1e-4:
        #print(epsilon)
        LImax = _max[0]                                                            
    else:
        print(_max[0], np.max(samples), epsilon)
        LImax = try_again_max(L, samples)                                                    
                                                                                   
    if verbose==True:                                                              
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))                               
        ax.plot(x, y, color="k", lw=2.5)                                           
        ax.axvline(LImin, color="g"); ax.axvline(LImax, color="g")                 
                                                                                   
    #Return                                                                        
    return  LImin, LImax 


