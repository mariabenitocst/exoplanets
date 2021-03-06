import numpy as np
from scipy.stats import chisquare
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from mock_generation import mock_population_sens
from astropy.constants import L_sun, R_jup
from utils import temperature
import glob
import sys

def interp_find_x(x, y, p_interp):
    try:
        # return
        return p_interp(x) - y
    except:
        return np.inf

def sensitivity(Tobs, Teff, bins_equal, alpha=0.05):
    """
    Perform goodness-of-fit test and returns True/False if theretical cooling 
    models can/cannot explain observations
    """
    # Make number of counts histogram

    ## Calculate bins with equal probability
    #p, bins, _ = plt.hist(Teff, bins=40, density=True, cumulative=True)
    #p          = np.insert(p, 0, 0)
    #p    = np.insert(p, len(p), 1)
    #bins = np.insert(bins, len(bins), bins[len(bins)-1]+1000)
    #x0   = bins[1]
    #p    = np.insert(p, 0, 0)
    #bins = np.insert(bins, 0, 0)

    #p_interp   = interp1d(bins, p)
    #y_h        = np.linspace(0.1, 0.9, 9) # total number of bins = 10
    #bins_equal = []
    #bins_equal.append(x0)
    #for y in y_h:
    #    root = brentq(interp_find_x, x0, bins[-1], args=(y, p_interp))
    #    x0 = root
    #    bins_equal.append(root)
    #bins_equal.append(bins[-2])

    n_th, _ = np.histogram(Teff, bins=bins_equal) # theoretical counts
    n, _    = np.histogram(Tobs, bins=bins_equal)

    _chi2   = np.sum(np.power(n-n_th, 2)/n_th) # observed counts
    p_value = chi2.sf(_chi2, len(n)-1)
    #print(p_value)
    if p_value >= alpha:
        # return
        return 1
    else:
        # return
        return 0

def sensitivity_nBDs_relunc(filepath, nBDs, rel_unc, relM, points, values, 
                            Tmin=0., v=None, ex="ex3", rank=100):
    f     = 1.#[0.1, 0.3, 0.5, 0.7, 0.9]
    gamma = [0., 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
    rs    = [5., 10., 20.]
    _sens = np.ones((len(rs), len(gamma)))*1000
    j = 0
    for _rs in rs:
        k = 0
        for _g in gamma:
            print("====== nBDS=%i, rel_unc=%.2f, relM=%.2f, rs=%.1f, gamma=%.1f" 
                    %(nBDs, rel_unc, relM, _rs, _g))
            _bool = np.ones(rank)*100
            for i in range(rank):
                Tobs, Teff = mock_population_sens(nBDs, rel_unc, relM, points, 
                                                  values, f, _g, _rs, 
                                                  Tmin=Tmin, v=v)
                if i==0:
                    try:
                        ## Calculate bins with equal probability
                        p, bins, _ = plt.hist(Teff,bins=40,density=True,
                                          cumulative=True)
                        p          = np.insert(p, 0, 0)
                        p    = np.insert(p, len(p), 1)
                        bins = np.insert(bins, len(bins), bins[len(bins)-1]+1000)
                        x0   = bins[1]
                        p    = np.insert(p, 0, 0)
                        bins = np.insert(bins, 0, 0)

                        p_interp   = interp1d(bins, p)
                        y_h        = np.linspace(0.1, 0.9, 9)# total number bins = 10
                        bins_equal = []
                        bins_equal.append(x0)
                        for y in y_h:
                            root = brentq(interp_find_x, x0, bins[-1], 
                                      args=(y, p_interp))
                            x0 = root
                            bins_equal.append(root)
                        bins_equal.append(bins[-2])
                    except:
                        Tobs, Teff = mock_population_sens(nBDs, rel_unc, relM, 
                                                  points, values, f, _g, _rs, 
                                                  Tmin=Tmin, v=v)
                        ## Calculate bins with equal probability
                        p, bins, _ = plt.hist(Teff,bins=40,density=True,
                                          cumulative=True)
                        p          = np.insert(p, 0, 0)
                        p    = np.insert(p, len(p), 1)
                        bins = np.insert(bins, len(bins), bins[len(bins)-1]+1000)
                        x0   = bins[1]
                        p    = np.insert(p, 0, 0)
                        bins = np.insert(bins, 0, 0)

                        p_interp   = interp1d(bins, p)
                        y_h        = np.linspace(0.1, 0.9, 9)# total number bins = 10
                        bins_equal = []
                        bins_equal.append(x0)
                        for y in y_h:
                            root = brentq(interp_find_x, x0, bins[-1], 
                                      args=(y, p_interp))
                            x0 = root
                            bins_equal.append(root)
                        bins_equal.append(bins[-2])

                _bool[i] = sensitivity(Tobs, Teff, bins_equal)
            print("Accepted H0 : %i" %int(np.sum(_bool)))
            print("Rejected H0 : %i" %(len(_bool)-int(np.sum(_bool))))
            _sens[j, k] = int(np.sum(_bool))
            k+=1
        j+=1
    # save acceptance ratio out of rank
    np.savetxt(filepath + "sensitivity_" + ex + 
               ("_N%i_relunc%.2f_relM%.2f" %(nBDs, rel_unc, relM)), _sens)
    # return
    return


if __name__ == '__main__':
    
    ex = sys.argv[1]
    N  = int(sys.argv[2])
    v = None
    nBDs    = [N]
    rel_unc = [0.]#, 0.10, 0.20]
    relM    = [0.]#, 0.20]
    # ------------------------------------------------------------------------
    # load theoretical BD cooling model - ATMO 2020
    path  =  "./data/"
    data = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
    points = np.transpose(data[0:2, :])
    values = data[2]
    # ------------------------------------------------------------------------
    filepath = "/home/mariacst/exoplanets/results/"
    for n in nBDs:
        for rel in rel_unc:
            for rM in relM:
                print(n, rel, rM)
                sensitivity_nBDs_relunc(filepath, n, rel, rM, points, values, 
                                        Tmin=650., v=v, ex=ex)
    
