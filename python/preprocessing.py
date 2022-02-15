import sys
#sys.path.append("/home/mariacst/exoplanets/.venv/lib/python3.6/site-packages")
import numpy as np
import pickle
from scipy.stats import binned_statistic
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
import pdb
#from pymc3.stats import hpd

def LI(L, samples, bin_n=20, verbose=False):
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
    logLmin    = np.min(y[~np.isnan(y)]) + 0.5
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

def hpd_frequency(hpd_interval, f, gamma, rs, rank=100):
    """
    Return the number of times the true value is within the HPD interval
    """
    how_many_rs    = 0
    how_many_gamma = 0
    how_many_rho0  = 0
    how_many_sigma = 0
    how_many_tau   = 0
    for i in range(rank):
        if np.round(hpd_interval[i][0, 0],2) <= f <= np.round(hpd_interval[i][0, 1],2):
            how_many_f += 1
        if np.round(hpd_interval[i][1, 0],2) <= gamma <= np.round(hpd_interval[i][1, 1],2):
            how_many_gamma += 1
        if np.round(hpd_interval[i][2, 0],2) <= rs <= np.round(hpd_interval[i][2, 1],2):
            how_many_rs += 1
    # Return
    return [how_many_f, how_many_gamma, how_many_rs, how_many_sigma]


def statistics(filepath, filepath2, ex, nBDs, rel_unc, f, gamma, rs, 
               rank=100, D=2):
    """
    Calculate mean, median, MAP & ML point estimates

    Inputs
    ------
        filepath    : directory where to save point estimates
        rel_unc_Tobs: relative uncertainty Tobs
        rank        : number of simulations
        D           : dimension parameter space
    """
    mean       = np.zeros((D, rank))
    median     = np.zeros((D, rank))
    _16th      = np.zeros((D, rank))
    _84th      = np.zeros((D, rank))
    MAP        = np.zeros((D, rank))
    ML         = np.zeros((D, rank))
    LI_min     = np.zeros((D, rank))
    LI_max     = np.zeros((D, rank))
    hpd_1sigma = []

    for i in range(rank):
        #print(i+1)
        # load posterior + likelihood
        file_name  = (filepath + ("N%isigma%.1f/posterior_" %(nBDs, rel_unc))
                     + ex + 
                     ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1fv%i" 
                     %(nBDs, rel_unc, f, gamma, rs, i+1)))
        samples    = pickle.load(open(file_name, "rb"))
        file_name2 = (filepath2 + ("N%isigma%.1f/like_" %(nBDs, rel_unc))
                     + ex +
                     ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1fv%i"
                     %(nBDs, rel_unc, f, gamma, rs, i+1)))
        like       = pickle.load(open(file_name2, "rb"))

        # calculate point estimates
        for j in range(D):
            mean[j][i]   = np.mean(samples[:, j])
            median[j][i] = np.percentile(samples[:, j], [50], axis=0)
            _16th[j][i]  = np.percentile(samples[:, j], [16], axis=0)
            _84th[j][i]  = np.percentile(samples[:, j], [84], axis=0)
            #TODO need to change # bins to see if results differ
            _n, _bins    = np.histogram(samples[:, j], bins=50)
            MAP[j][i]    = _bins[np.argmax(_n)]
            ML[j][i]     = samples[:, j][np.argmax(like)]
            _min, _max   = LI(like, samples[:, j])
            LI_min[j][i] = _min
            LI_max[j][i] = _max
        #hpd_1sigma.append(hpd(samples, alpha=0.32))

    #hpd_1sigma = np.array(hpd_1sigma)    
    #print(hpd_1sigma.shape)

    filepath = "/home/mariacst/exoplanets/results/velocity/v100/analytic/statistics_"
    output = open(filepath + ex + ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f" 
                              %(nBDs, rel_unc, f, gamma, rs)), "w")
    for i in range(rank):
        for j in range(D):
            output.write("%.4f  " %mean[j][i])
        for j in range(D):
            output.write("%.4f  " %median[j][i])
        for j in range(D):
            output.write("%.4f  " %_16th[j][i])
        for j in range(D):
            output.write("%.4f  " %_84th[j][i])
        for j in range(D):
            output.write("%.4f  " %MAP[j][i])
        for j in range(D):
            output.write("%.4f  " %ML[j][i])
        for j in range(D):
            output.write("%.4f  " %LI_min[j][i])
        for j in range(D):
            output.write("%.4f  " %LI_max[j][i])
        #for j in range(D):
        #    output.write("%.4f  " %hpd_1sigma[i][j, 0])
        #for j in range(D):
        #    output.write("%.4f  " %hpd_1sigma[i][j, 1])
        output.write("\n")
    output.close()

    # return
    return



if __name__ == '__main__':
    _path     = "/hdfs/local/mariacst/exoplanets/results/"
    _path_f   = "v100/analytic/fixedT10Tcut650_nocutTwn/"
    filepath  = _path + "posterior/" + _path_f
    filepath2 = _path + "likelihood/" + _path_f
    ex        = "fixedT10v100Tcut650_nocutTwn"
    N         = int(sys.argv[1])
    #sigma     = float(sys.argv[3])
    #print(N)
    nBDs     = [N]
    rel_unc  = [0.2]#float(sys.argv[1])]
    f        = 1.
    rs       = [10.]
    gamma    = [1.1]

    for N in nBDs:
        for rel in rel_unc:
            for _rs in rs:
                for _g in gamma:
                    print(_rs, _g)
                    try:
                        statistics(filepath, filepath2, ex, N, rel, f, _g, _rs, 
                                   100, 3)
                    except Exception as e:
                        print(e)
                        print("este no!")
                        continue

