from matplotlib.lines import Line2D
from _corner import corner
import sys
import pickle
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
from matplotlib import rc
from scipy.interpolate import griddata
rc('font', family='times new roman', size=22.)

# -----------------------------------
## Sensitivity
# -----------------------------------
def display_values(XX, YY, H, ax=False):
    if ax:
        for i in range(YY.shape[0]-1):
            for j in range(XX.shape[1]-1):
                ax.text((XX[i+1][0] + XX[i][0])/2, (YY[0][j+1] + YY[0][j])/2, '%i' % H[i, j],
                     horizontalalignment='center', verticalalignment='center', size=18)
    else:
        for i in range(YY.shape[0]-1):
            for j in range(XX.shape[1]-1):
                plt.text((XX[i+1][0] + XX[i][0])/2, (YY[0][j+1] + YY[0][j])/2, '%i' % H[i, j],
                     horizontalalignment='center', verticalalignment='center', size=18)
    # return
    return


def grid_sensitivity(filepath, nBDs, rel_unc, relM, ex="ex3",
                     ax=False, y_label=True, x_label=True,
                     show_bin_values=True):
    """
    Plot # of H0 acceptance out of rank in (rs, gamma) plane
    """
    # grid points
    rs    = np.array([5., 10., 20.])
    gamma = np.array([0., 0.5, 1, 1.2, 1.4])

    zi = np.genfromtxt(filepath + "sensitivity_" + ex +
                       ("_N%i_relunc%.2f_relM%.2f" %(nBDs, rel_unc, relM)))

    #print(zi.shape)
    xi = np.array([2.5, 7.5, 15, 25])
    yi = np.array([0., 0.25, 0.75, 1.05, 1.15, 1.25, 1.35,  1.45, 1.55])
    xi, yi = np.meshgrid(xi, yi, indexing="ij")

    if ax==False:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if y_label==True:
        ax.set_ylabel(r"$\gamma$");
        ax.set_yticks(gamma)
        ax.set_yticklabels(['0', '0.5', '1', '1.2', '1.4'])
    else:
        ax.set_yticks(gamma)
        ax.set_yticklabels([])
    if x_label==True:
        ax.set_xlabel(r"$r_s$ [kpc]")
        ax.set_xticks(rs)
        ax.set_xticklabels(['5', '10', '20'])
    else:
        ax.set_xticks(rs)
        ax.set_xticklabels([])

    norm = colors.BoundaryNorm(boundaries=np.array([0, 5, 100]), ncolors=2)
    cmap = colors.ListedColormap(["#3F5F5F", "#FFFF66"])
    ax.pcolormesh(xi, yi, zi, norm=norm, cmap=cmap, edgecolor="black")

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.)

    text_box = AnchoredText((r"$N=10^{%i}$"
                            %int(np.log10(nBDs))),
                            bbox_to_anchor=(0., 0.99),
                            bbox_transform=ax.transAxes, frameon=False,
                            pad=0., loc="lower left", prop=dict(size=19))

    ax.add_artist(text_box)

    if show_bin_values:
        display_values(xi, yi, zi, ax=ax)
    # return
    return


# -----------------------------------
## FSE
# -----------------------------------
def add_hatch(ax, i, j, width, height):
    ax.add_patch(mpatches.Rectangle(
              (i, j),
              width,
              height, 
              fill=False, 
              color='yellow', linewidth=0.,
              hatch='//')) # the more slashes, the denser the hash lines 
    return

def FSE_f_gamma_rs(filepath, nBDs, sigma, ex, PE="ML"):
    # grid points
    f     = 1.
    rs    = np.array([5., 10., 20.])
    gamma = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])

    FSE = [];
    for _rs in rs:
        for _g in gamma:
            data = np.genfromtxt(filepath + "statistics_" + ex +
                                 ("_N%i_sigma%.1f_gamma%.1frs%.1f"
                                  %(nBDs, sigma, _g, _rs)), unpack=True)  
            if PE=="median":
                pe = data[1]
            elif PE=="mode1":
                pe = data[4]
            elif PE=="mode2":
                pe = data[5]
            elif PE=="mode3":
                pe = data[6]
            elif PE=="mean":
                pe = data[0]
            elif PE=="ML":
                pe = data[9]
            else:
                sys.exit("Point estimate not implemented!")
           
            rank=len(data[0]); #print(_rs, _g, rank)
            #print("rank={}".format(rank))
            FSE.append(np.sqrt(1/rank*np.sum(np.power(pe - _g, 2)))/_g)

    xi = np.array([2.5, 7.5, 15, 25])
    yi = np.array([0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 
                   1.05, 1.15, 1.25, 1.35, 1.45, 1.55])
    xi, yi = np.meshgrid(xi, yi, indexing="ij")

    zi   = np.array(FSE).reshape(len(rs), len(gamma))
    # return
    return xi, yi, zi


def FSE_f_gamma_rs_gNFW(filepath, nBDs, sigma, ex, PE="ML", _f=False,
                        _gamma=False, _scale_radius=False):     
    # grid points                                                                  
    f     = 1.                                                                     
    rs    = np.array([5., 10., 20.])                                               
    gamma = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])        
       
    if _f:
        index=0
    elif _gamma:
        index=3
    elif _scale_radius:
        index=6
                                       
    FSE = [];                                                                      
    for _rs in rs:                                                                 
        for _g in gamma:                                                           
            data = np.genfromtxt(filepath + "statistics_" + ex +                   
                                 ("_N%i_sigma%.1f_gamma%.1frs%.1f"                 
                                  %(nBDs, sigma, _g, _rs)), unpack=True)           
            #print((filepath + "statistics_" + ex +                   
            #                     ("_N%i_sigma%.1f_gamma%.1frs%.1f"                 
            #                      %(nBDs, sigma, _g, _rs))))
            #print(_rs, _g, data.shape)
            if PE=="ML":                                                       
                pe = data[index]                                                       
            else:                                                                  
                sys.exit("Point estimate not implemented!")                        
            if _f:
                _true=f
            elif _gamma:
                _true=_g
            elif _scale_radius:
                _true=_rs       
                                                                            
            rank=len(data[0]);
            FSE.append(np.sqrt(1/rank*np.sum(np.power(pe - _true, 2)))/_true)            
                                                                                   
    xi = np.array([2.5, 7.5, 15, 25])                                              
    yi = np.array([0.45, 0.55, 0.65, 0.75, 0.85, 0.95,                             
                   1.05, 1.15, 1.25, 1.35, 1.45, 1.55])                            
    xi, yi = np.meshgrid(xi, yi, indexing="ij")                                    
                                                                                   
    zi   = np.array(FSE).reshape(len(rs), len(gamma))                              
    # return                                                                       
    return xi, yi, zi 


def grid_FSE(filepath, nBDs, rel_unc, ex="baseline", ax=False, PE="median",
             ylabel=False, xlabel=False, show=False,
             gNFW=False, **kwargs):
    """
    Plot FSE grid in (rs, gamma) 
    """

    norm = colors.BoundaryNorm(boundaries=np.arange(0, 1, 0.05), ncolors=256, extend="max")

    if not gNFW:
        xi, yi, zi = FSE_f_gamma_rs(filepath, nBDs, rel_unc, ex, PE=PE)
    else:
        xi, yi, zi = FSE_f_gamma_rs_gNFW(filepath, nBDs, rel_unc, ex, **kwargs)

    if ax==False:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
    im = ax.pcolormesh(xi, yi, zi, norm=norm, cmap="viridis_r", rasterized=True)
    if show==True:
        print(xi, yi)
        print(zi)

    if ylabel==True:
        ax.set_ylabel(r"$\gamma$")
        ax.set_yticklabels(['0.5', '', '0.7', '', '0.9', '', '1.1', '', '1.3', 
                            '', '1.5'])
    else:
        ax.set_yticklabels([])
    if xlabel==True:
        ax.set_xlabel(r"$\rm r_s$ [kpc]")
        ax.set_xticklabels(['5', '10', '15', '20'])
    else:
        ax.set_xticklabels([])

    text_box = AnchoredText((r"$N=10^{%i}$, $\sigma_i$=%i"
                            %(int(np.log10(nBDs)), int(rel_unc*100))
                            + "$\% $"),
                            frameon=True, loc=3, pad=0.2, prop=dict(size=18))
    plt.setp(text_box.patch, facecolor="white")
    ax.add_artist(text_box)

    ax.tick_params(which='major',direction="out",width=2.,length=5,right=False,
                   top=False,pad=5)

    ax.set_xticks([5., 10., 15, 20.])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)

    # return
    return im

# -----------------------------------
## MARE
# -----------------------------------

def MARE_f_gamma_rs(filepath, nBDs, sigma, ex, PE="ML"):
    # grid points
    f     = 1.
    rs    = np.array([5., 10., 20.])
    gamma = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])

    MARE = [];
    for _rs in rs:
        for _g in gamma:
            data = np.genfromtxt(filepath + "statistics_" + ex +
                                 ("_N%i_sigma%.1f_gamma%.1frs%.1f"
                                  %(nBDs, sigma, _g, _rs)), unpack=True)  
            if PE=="median":
                pe = data[1]
            elif PE=="mode1":
                pe = data[4]
            elif PE=="mode2":
                pe = data[5]
            elif PE=="mode3":
                pe = data[6]
            elif PE=="mean":
                pe = data[0]
            elif PE=="ML":
                pe = data[9]
            else:
                sys.exit("Point estimate not implemented!")
           
            rank=len(data[0]); #print(_rs, _g, rank)
            #print("rank={}".format(rank))
            MARE.append(1/rank*np.sum(np.abs(pe - _g)/_g))

    xi = np.array([2.5, 7.5, 15, 25])
    yi = np.array([0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 
                   1.05, 1.15, 1.25, 1.35, 1.45, 1.55])
    xi, yi = np.meshgrid(xi, yi, indexing="ij")

    zi   = np.array(MARE).reshape(len(rs), len(gamma))
    # return
    return xi, yi, zi



def grid_MARE(filepath, nBDs, rel_unc, ex="baseline", ax=False, PE="median",
              ylabel=False, xlabel=False, show=False,
              gNFW=False, **kwargs):
    """
    Plot FSE grid in (rs, gamma) 
    """

    norm = colors.BoundaryNorm(boundaries=np.arange(0, 1, 0.05), ncolors=256, extend="max")

    if not gNFW:
        xi, yi, zi = MARE_f_gamma_rs(filepath, nBDs, rel_unc, ex, PE=PE)
    else:
        sys.exit("Not implemented!")
    if ax==False:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
    im = ax.pcolormesh(xi, yi, zi, norm=norm, cmap="viridis_r")
    if show==True:
        print(xi, yi)
        print(zi)

    if ylabel==True:
        ax.set_ylabel(r"$\gamma$")
        ax.set_yticklabels(['0.5', '', '0.7', '', '0.9', '', '1.1', '', '1.3', 
                            '', '1.5'])
    else:
        ax.set_yticklabels([])
    if xlabel==True:
        ax.set_xlabel(r"$r_s$ [kpc]")
        ax.set_xticklabels(['5', '10', '15', '20'])
    else:
        ax.set_xticklabels([])

    text_box = AnchoredText((r"$N=10^{%i}$, $\sigma_i$=%i"
                            %(int(np.log10(nBDs)), int(rel_unc*100))
                            + "$\% $"),
                            frameon=True, loc=3, pad=0.2, prop=dict(size=18))
    plt.setp(text_box.patch, facecolor="white")
    ax.add_artist(text_box)

    ax.tick_params(which='major',direction="out",width=1.5,length=5,right=False,top=False,pad=5)

    ax.set_xticks([5., 10., 15, 20.])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)

    # return
    return im


# -----------------------------------
## Coverage
# -----------------------------------
def coverage_f_gamma_rs(filepath, nBDs, rel_unc, ex, CR="symmetric"):
    # grid points
    f     = 1.
    rs    = np.array([5., 10., 20.])
    gamma = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])

    cove = []
    for _rs in rs:
        for _g in gamma:
            try:
                data = np.genfromtxt(filepath + "statistics_" + ex + 
                                 ("_N%i_sigma%.1f_gamma%.1frs%.1f" 
                                  %(nBDs, rel_unc, _g, _rs)), unpack=True)
                if CR=="symmetric":
                    low  = data[2]
                    high = data[3]
                elif CR=="HPD":
                    low  = data[7]
                    high = data[8]
                elif CR=="LI":
                    low  = data[10]
                    high = data[11]
                else:
                    sys.exit("Credible interval not implemented!")
                one = _g > low
                two = _g < high
                cove.append(len(np.where((one==True) & (two==True))[0]))
            except:
                cove.append(np.nan)

    xi = np.array([2.5, 7.5, 15, 25])
    yi = np.array([0.45, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 
                   1.35, 1.45, 1.55])
    xi, yi = np.meshgrid(xi, yi, indexing="ij")

    #print(len(data[0]))
    zi = np.array(cove).reshape(len(rs), len(gamma))/len(data[0])
    # return
    return xi, yi, zi



def coverage_f_gamma_rs_gNFW(filepath, nBDs, rel_unc, ex, CR="symmetric"):              
    # grid points                                                                  
    f     = 1.                                                                     
    rs    = np.array([5., 10., 20.])                                               
    gamma = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])        
                                                                                   
    cove = []                                                                      
    for _rs in rs:                                                                 
        for _g in gamma:                                                           
            try:                                                                   
                data = np.genfromtxt(filepath + "statistics_" + ex +               
                                 ("_N%i_sigma%.1f_gamma%.1frs%.1f"                 
                                  %(nBDs, rel_unc, _g, _rs)), unpack=True)         
                if CR=="LI":                                                     
                    low  = data[4]       
                    high = data[5]
                else:                                                              
                    sys.exit("Credible interval not implemented!")                 
                one = _g > low                                                     
                two = _g < high                                                    
                cove.append(len(np.where((one==True) & (two==True))[0]))           
            except:                                                                
                cove.append(np.nan)                                                
                                                                                   
    xi = np.array([2.5, 7.5, 15, 25])                                              
    yi = np.array([0.45, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25,                 
                   1.35, 1.45, 1.55])                                              
    xi, yi = np.meshgrid(xi, yi, indexing="ij")                                    
                                                                                   
    #print(len(data[0]))                                                           
    zi = np.array(cove).reshape(len(rs), len(gamma))/len(data[0])                  
    # return                                                                       
    return xi, yi, zi 



def grid_coverage(filepath, nBDs, rel_unc, ex="ex1",
             ax=False, CR="symmetric", ylabel=False, xlabel=False,
             _gNFW=False):
    """
    Plot coverage grid in (rs, gamma) 
    """

    #norm = colors.BoundaryNorm(boundaries=np.arange(0, 100, 5), ncolors=256)
    bounds = np.array([0., 0.1, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.])
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    if _gNFW==False:

        xi, yi, zi = coverage_f_gamma_rs(filepath, nBDs, rel_unc,
                                         ex, CR=CR)
    else:
        xi, yi, zi = coverage_f_gamma_rs_gNFW(filepath, nBDs, rel_unc,               
                                         ex, CR=CR) 

    rs = [2.5, 5., 10., 20., 25.]
    g  = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    xi_c = np.linspace(np.min(rs), np.max(rs), 10)
    yi_c = np.linspace(np.min(g), np.max(g), 10)

    cmap="RdYlGn"
    if ax==False:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    im = ax.pcolormesh(xi, yi, zi, norm=norm, cmap=cmap)
    #zi_c = np.vstack((zi_2[0], zi, zi[-1]))
    #CS = ax.contour(xi_c, yi_c, zi_c.T, levels=[68, 100], colors="k")
    x, y = np.meshgrid([5., 10, 20.], 
                       [0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5], 
                       indexing="ij")
    points = np.array((np.ravel(x), np.ravel(y))).T
    xi_c, yi_c = np.meshgrid(np.linspace(2.5, 25., 5), np.linspace(0.45, 1.55, 13), 
                                 indexing="ij")
    values = np.array((np.ravel(xi_c), np.ravel(yi_c))).T
    zi_c = griddata(points, np.ravel(zi), values, method="nearest")
    if np.any(zi_c>0.68):
        CS = ax.contour(xi_c, yi_c, zi_c.reshape(5, 13), levels=[0.68,], color="k")
        ax.clabel(CS, inline=True, fontsize=10, fmt="%.2f")
    
    if ylabel==True:
        ax.set_ylabel(r"$\gamma$")
        ax.set_yticklabels(['0.5', '', '0.7', '', '0.9', '', '1.1', '', '1.3', 
                            '', '1.5'])
    else:
        ax.set_yticklabels([])
    if xlabel==True:
        ax.set_xlabel(r"$r_s$ [kpc]")
        ax.set_xticklabels(['5', '10', '15', '20'])
    else:
        ax.set_xticklabels([])

    text_box = AnchoredText((r"$N=10^{%i}$, $\sigma_i$=%i"
                            %(int(np.log10(nBDs)), int(rel_unc*100))
                            + "$\% $"),
                            frameon=True, loc=3, pad=0.2, prop=dict(size=18))
    plt.setp(text_box.patch, facecolor="white")
    ax.add_artist(text_box)

    ax.tick_params(which='major',direction="out",width=2.,length=5,
                   right=False,top=False,pad=5)

    ax.set_xticks([5., 10., 15, 20.])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)

    # return
    return im


# -----------------------------------
## Posterior
# -----------------------------------

def plot_1Dposterior(filepath, nBDs, rel_unc, relM, ex,
                     f, gamma, rs, color="k"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    
    xvals = [np.linspace(0, 1, 100), np.linspace(0, 3, 100), 
             np.linspace(0, 50, 100)]

    true = [f, gamma, rs]
    
    filepath = filepath + ("N%irelT%.2frelM%.2f/" %(nBDs, rel_unc, relM))

    for i, ax in enumerate(axes.flat):
        print("i = ", i)
        for j in range(100):
            _file   = open(filepath + "posterior_" + ex + 
                           ("_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1fv%i" 
                           %(nBDs, rel_unc, relM, f, gamma, rs, j+1)), "rb") 
            samples = pickle.load(_file)
            kde   = gaussian_kde(samples.T[i])
            ax.plot(xvals[i], kde(xvals[i])/np.max(kde(xvals[i])), 
                    color=color, lw=2.5, 
                    alpha=0.3)
        ax.axvline(true[i], ls="--", lw=2.5, color="red")
        if i==0:
            ax.set_xlabel(r"$f$")
            ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
            ax.set_xticklabels(['0.1', '0.3', '0.5', '0.7', '0.9'])
            text_box = AnchoredText((r"N=%i, $\sigma_T$=%i" %(nBDs, int(rel_unc*100)) 
                                + "$\%, $" 
                                + "$\sigma_M$=%i" %(int(relM*100)) + "$\%$"),
                                bbox_to_anchor=(0., 0.99),
                                bbox_transform=ax.transAxes, loc='lower left', 
                                pad=0.04, prop=dict(size=20))
            plt.setp(text_box.patch, facecolor="white")
            ax.add_artist(text_box)
        elif i==1:
            ax.set_xlabel(r"$\gamma$")
            ax.set_xticks([0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.])
            ax.set_xticklabels(['0.2', '0.6', '1', '1.4', '1.8', '2.2', '2.6', '3'])
        else:
            ax.set_xlabel(r"$r_s$ [kpc]")

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2.)
    
    fig.subplots_adjust(hspace=0.25, wspace=0.08)
    fig.savefig("../../Figs/1Dposterior_" + ex + 
                ("_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%i.pdf" 
                %(nBDs, rel_unc, relM, f, gamma, int(rs))), bbox_inches="tight")
    # return
    return


def plot_corner(samples, nBDs, relT, relM, f, gamma, rs, i=1, smooth=1.):

    fig, axes = corner(samples, levels=(1-np.exp(-0.5), 1-np.exp(-2)), plot_datapoints=False, 
                       plot_density=False, fill_contours=False, smooth=smooth, color="green",
                       range=[(0., 1.01), (-0.01, 2.5), (0., 40.)])
    # plot KDE smoothed version of distributions
    for axidx, samps in zip([0, 4, 8], samples.T):
        kde   = gaussian_kde(samps)
        xvals = fig.axes[axidx].get_xlim()
        xvals = np.linspace(xvals[0], xvals[1], 100)
        fig.axes[axidx].plot(xvals, kde(xvals)/np.max(kde(xvals)), color="green", lw=2.5)    
    
    axes[0, 0].axvline(1., color="r", ls="--", lw=2.5)
    axes[1, 1].axvline(gamma, color="r", ls="--", lw=2.5)
    axes[2, 2].axvline(rs, color="r", ls="--", lw=2.5)
    axes[1, 0].scatter(f, gamma, marker="x", color="red", s=60)
    axes[2, 0].scatter(f, rs, marker="x", color="red", s=60)
    axes[2, 1].scatter(gamma, rs, marker="x", color="red", s=60)
        
    axes[1, 0].set_ylabel(r"$\gamma$")
    axes[2, 0].set_xlabel(r"$f$")
    axes[2, 0].set_ylabel(r"$r_s$ [kpc]")
    axes[2, 1].set_xlabel(r"$\gamma$")
    axes[2, 2].set_xlabel(r"$r_s$ [kpc]")
    
    colors = ['green']
    lines = [Line2D([0], [0], color=c, linewidth=2.5, linestyle='-') for c in colors]
    labels = ['N %i, relT=%0.1f, relM=%.1f' %(nBDs, relT, relM)]
    axes[0, 2].legend(lines, labels, fontsize=16)
    
    fig.savefig(("../../Figs/corner_ex15_N%irelT%.2frelM%.2f_g%.1frs%.1f_%i.png" %(nBDs, relT, relM, gamma, rs, i+1)), 
                bbox_inches="tight")



def plot_corner_each(samples, ex, nBDs, relT, relM, relA, relR, f, gamma, rs, 
                     i=1, smooth=1.):           
                                                                                    
    fig, axes = corner(samples, levels=(1-np.exp(-0.5), 1-np.exp(-2)), plot_datapoints=False, 
                       plot_density=False, fill_contours=False, smooth=smooth, color="green",
                       range=[(0., 1.01), (-0.01, 2.5), (0., 40.)])                 
    # plot KDE smoothed version of distributions                                    
    for axidx, samps in zip([0, 4, 8], samples.T):                                  
        kde   = gaussian_kde(samps)                                                 
        xvals = fig.axes[axidx].get_xlim()                                          
        xvals = np.linspace(xvals[0], xvals[1], 100)                                
        fig.axes[axidx].plot(xvals, kde(xvals)/np.max(kde(xvals)), color="green", lw=2.5)    
                                                                                    
    axes[0, 0].axvline(1., color="r", ls="--", lw=2.5)                              
    axes[1, 1].axvline(gamma, color="r", ls="--", lw=2.5)                           
    axes[2, 2].axvline(rs, color="r", ls="--", lw=2.5)                              
    axes[1, 0].scatter(f, gamma, marker="x", color="red", s=60)                     
    axes[2, 0].scatter(f, rs, marker="x", color="red", s=60)                        
    axes[2, 1].scatter(gamma, rs, marker="x", color="red", s=60)                    
                                                                                    
    axes[1, 0].set_ylabel(r"$\gamma$")                                              
    axes[2, 0].set_xlabel(r"$f$")                                                   
    axes[2, 0].set_ylabel(r"$r_s$ [kpc]")                                           
    axes[2, 1].set_xlabel(r"$\gamma$")                                              
    axes[2, 2].set_xlabel(r"$r_s$ [kpc]")                                           
                                                                                    
    colors = ['green']                                                          
    lines = [Line2D([0], [0], color=c, linewidth=2.5, linestyle='-') for c in colors]
    labels = ['N %i, relT=%0.1f, relM=%.1f' %(nBDs, relT, relM)]                
    axes[0, 2].legend(lines, labels, fontsize=16)                               
                                                                                
    fig.savefig(("../../Figs/corner_" + ex + 
                 "_N%irelT%.2frelM%.2frelA%.2frelR%.2f_g%.1frs%.1f_%i.png" 
                 %(nBDs, relT, relM, relA, relR, gamma, rs, i+1)),
                bbox_inches="tight") 
