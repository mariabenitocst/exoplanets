import sys
import numpy as np
#import arviz as az
import pickle
from scipy.stats import binned_statistic
from scipy.interpolate import UnivariateSpline
from scipy.optimize import root
import pdb

def return_MAP(samples, nbins=50):
    _n, _bins = np.histogram(samples, bins=nbins)
    return(_bins[np.argmax(_n)])


def LI(L, samples, bin_n=50, verbose=False):
    """
    For each parameter, construct profile likelihood and return the profile likelihood interval
    (i.e. region where the log Likelihood is within 1 of its maximum value)
    
    """   
    x = binned_statistic(samples, -L, 'min', bins=bin_n)[1]
    y = binned_statistic(samples, -L, 'min', bins=bin_n+1)[0]

    logLmin = np.min(y[~np.isnan(y)]) + 0.5 

    # Create Interpolation fanction Likelihood - parameter
    z =  UnivariateSpline(x[~np.isnan(y)], y[~np.isnan(y)]-logLmin, s=0)      

    pos = np.where(np.abs(y-logLmin+0.5)<1e-6)
    sol = root(z, [x[pos]-0.3, x[pos]+0.3])

    if (z(np.min(x[~np.isnan(y)]))<0.) and (z(np.max(x[~np.isnan(y)]))<0.):

        sol.x[0] = np.min(x[~np.isnan(y)])
        sol.x[1] = np.max(x[~np.isnan(y)])

    elif np.abs(sol.x[0]-sol.x[1])<1e-3:
        

        if (sol.x[0]>x[np.argmin(y[~np.isnan(y)])]) and (z(np.min(x[~np.isnan(y)]))<0.):

            sol.x[0] = np.min(x[~np.isnan(y)])
        
        elif (sol.x[0]<x[np.argmin(y[~np.isnan(y)])]) and (z(np.max(x[~np.isnan(y)]))<0.):
            
            sol.x[1] = np.max(x[~np.isnan(y)])
        else:
            # ======== Try again =============================================
            if (sol.x[0]>x[np.argmin(y[~np.isnan(y)])]):
                lower=1; upper=0
            else:
                lower=0; upper=1
            sol = root(z, [x[pos]-0.3-(0.3*lower), x[pos]+0.3+(0.3*upper)])
            # ================================================================
        if np.abs(sol.x[0]-sol.x[1])<1e-3:

            if (sol.x[0]>x[np.argmin(y[~np.isnan(y)])]) and (z(np.min(x[~np.isnan(y)]))<0.):

                sol.x[0] = np.min(x[~np.isnan(y)])

            elif (sol.x[0]<x[np.argmin(y[~np.isnan(y)])]) and (z(np.max(x[~np.isnan(y)]))<0.):

                sol.x[1] = np.max(x[~np.isnan(y)])

            else:

                print("Error in %i run, unable to find roots in likelihood interval calculation"%i)

                if (sol.x[0]>x[np.argmin(y[~np.isnan(y)])]):
                    print("Problem w/ lower bound")
                else:
                    print("Problem w/ higher bound")
 
                print("Need to check this sys, break me da error!")
                sys.exit(-1) 

    if verbose==True:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        
        ax.scatter(x[~np.isnan(y)], y[~np.isnan(y)]-logLmin)
        x_plot = np.linspace(np.min(x[~np.isnan(y)]), np.max(x[~np.isnan(y)]), 50)
        ax.plot(x_plot, z(x_plot), color="red")
        ax.axhline(0., color="green")
        ax.axvline(sol.x[0], color="green", ls="--")
        ax.axvline(sol.x[1], color="green", ls="--")
        fig.savefig("%i.png"%(i+1))
        plt.close()

    #Return
    return  sol.x[0], sol.x[1]

def is_true_in_LI_1sigma(L, samples, true, bin_n=50):
    x = binned_statistic(samples, -L, 'min', bins=bin_n)[1]
    y = binned_statistic(samples, -L, 'min', bins=bin_n+1)[0]

    logLmin = np.min(y[~np.isnan(y)]) + 0.5 

    z =  UnivariateSpline(x[~np.isnan(y)], y[~np.isnan(y)]-logLmin, s=0)
    how_many = z(true)<0
    #import pdb; pdb.set_trace()
    if how_many:
        return 1
    else:
        return 0

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
    out_path = "/home/mariacst/exoplanets/results/power_law/older_BD/statistics_"
    output = open(out_path + ex + 
                ("_N%i_sigma%.1f_gamma%.1frs%.1f"%(nBDs, sigma, gamma, rs)), 
                "w") 

    how_many = 0
    for i in range(rank):

        #print(i)
        file_name  = (filepath + ("%i/" %(i+1))
                     + ex + 
                     ("_N%i_sigma%.1f_gamma%.1f_rs%.1f_v%ipost_equal_weights.dat" 
                     %(nBDs, sigma, gamma, rs, i+1)))
        samples    = np.genfromtxt(file_name, unpack=True)

        if samples.size == 0:
            how_many += 1
            print(i+1)
            continue

        # calculate point estimates
        output.write("%.4f  "%samples[1][np.argmax(samples[2])]) # ML
        # whether true value is contained in 1sigma LI 
        output.write("%i  "%is_true_in_LI_1sigma(samples[2], samples[1], gamma))
        output.write("\n")

    output.close()

    print("Number of samples empty = %i"%how_many)
    print()
    #print("Remember to manually check convergence of profile L intervals!")
    #print("Oh no! I know ... it is boring!")

    # return
    return


if __name__ == '__main__':
    #_path    = "/local/mariacst/2022_exoplanets/results/power_law/baseline_NL_older/"
    _path    = "/home/mariacst/exoplanets/running/power_law/baseline_NL_noerror/out/"
    ex       = "baseline_NL_olderBD_noerror_alsoT"
    nBDs     = [int(sys.argv[1])]
    sigma    = float(sys.argv[2])
    f        = 1.
    rs       = [5., 20.]
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

