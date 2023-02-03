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


def LI(L, samples, bin_n=50, verbose=False):
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
    z =  UnivariateSpline(x[~np.isnan(y)], y[~np.isnan(y)], s=0)
    
    # Find in which points crosses 1sigma horizontal line (where 1 sigma 
    # corresponds to 1/2 -for 1 dof-)
    x_tmin     = np.min(samples)
    epsilon    = 1
    niteration = 0
    logLmin    = np.min(y[~np.isnan(y)]) + 1.
    while epsilon > 10**-6 and niteration < 20:
        niteration +=1
        minimum = minimize(lambda x: (z(x)-logLmin)**2, x_tmin,
                           bounds=((np.min(samples), samples[np.argmax(L)]),))
        epsilon = minimum.fun
        x_tmin  = x_tmin*1.1
        _min    = minimum.x
        
    if epsilon < 10**-6:
        LImin = _min[0]
    else:
        LImin = np.min(samples)
        
    x_tmax     = samples[np.argmax(L)]
    epsilon    = 1
    niteration = 0
    while epsilon > 10**-6 and niteration < 25:
        niteration +=1
        maximum = minimize(lambda x: (z(x)-logLmin)**2, x_tmax, 
                           bounds=((samples[np.argmax(L)], np.max(samples)),))
        epsilon  = maximum.fun
        x_tmax  = x_tmax*1.1
        _max    = maximum.x
    
    if epsilon < 10**-6:
        LImax = _max[0]
    else:
        LImax = np.max(samples)

    if verbose==True:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(x, y, color="k", lw=2.5)
        ax.axvline(LImin, color="g"); ax.axvline(LImax, color="g")

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
    out_path = "/home/mariacst/exoplanets/results/power_law/older_BD/v30/statistics_"
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

        # 68% highest density interval
        hdi_low, hdi_high = az.hdi(samples[1], hdi_prob=0.95)
        # profile L interval (region where log-L is within 1 of maximum)
        LI_low, LI_high = LI(samples[2], samples[1], bin_n=60)

        # calculate point estimates
        output.write("%.4f  "%np.mean(samples[1])) # mean
        output.write("%.4f  "%np.percentile(samples[1], [50])) # median
        output.write("%.4f  "%np.percentile(samples[1], [16]))
        output.write("%.4f  "%np.percentile(samples[1], [84]))
        output.write("%.4f  "%return_MAP(samples[1], nbins=20)) #MAP1
        output.write("%.4f  "%return_MAP(samples[1])) #MAP2
        output.write("%.4f  "%return_MAP(samples[1], nbins=100)) #MAP3
        output.write("%.4f  "%hdi_low)
        output.write("%.4f  "%hdi_high)
        output.write("%.4f  "%samples[1][np.argmax(samples[2])]) # ML
        output.write("%.4f  "%LI_low)
        output.write("%.4f  "%LI_high)
        output.write("\n")

    output.close()

    print("Remember to manually check convergence of profile L intervals!")
    print("Oh no! I know ... it is boring!")

    # return
    return


if __name__ == '__main__':
    _path    = "/local/mariacst/2022_exoplanets/results/power_law/v30/baseline_NL_older/"
    #_path    = "/home/mariacst/exoplanets/running/power_law/T650_NL/out/"
    ex       = "baseline_NL_olderBD_v30"
    nBDs     = [int(sys.argv[1])]
    sigma    = float(sys.argv[2])
    f        = 1.
    rs       = [5.]
    gamma    = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    for N in nBDs:
        for _rs in rs:
            for _g in gamma:
                print(N, _rs, _g)
                try:
                    statistics(_path, ex, N, sigma, _g, _rs, rank=200)
                except Exception as e:
                    print(e)
                    print("Noooooo :_(")
                    continue

