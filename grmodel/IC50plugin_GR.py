import numpy as np
import matplotlib.pyplot as plt
from .pymcGrowth import simulate


def IC(IC50, X):
    """ Define the IC50 function """
    return IC50[2] + (IC50[1] - IC50[2]) / (1 + 10**(X - np.log10(IC50[0])))


def num(IC_Div, IC_DR, d, apopfrac, ttime, X):
    """ Define the num function to count lnum, eap and dead based on given parameters """
    out = np.empty((len(X),1,4))

    for i, x in enumerate(X):
        params = np.array([IC(IC_Div, x), d, IC(IC_DR, x), apopfrac])
        out[i] = simulate(params, ttime)

    return out


def plotCurves(IC_Div, IC_DR, d, apopfrac, ttime):
    """ Plot the curves for (lnum vs. X, eap vs. X, dead vs. X) """
    X = np.linspace(0,0.5)
    result = np.array(num(IC_Div, IC_DR, d, apopfrac, ttime, X))
    lnum = result[:,0,0]
    eap = result[:,0,1]
    dead = result[:,0,2] + result[:,0,3]

    fig, ax = plt.subplots(1,3,figsize=(10,3))
    
    ax[0].set_title("lnum vs. X")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("the number of live cells")
    ax[0].plot(X, lnum)
    ax[0].set_xscale('log')

    ax[1].set_title("eap vs. X")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("the number of early apoptosis cells")
    ax[1].plot(X, eap)
    ax[1].set_xscale('log')

    ax[2].set_title("dead vs. X")
    ax[2].set_xlabel("X")
    ax[2].set_ylabel("the number of dead cells")
    ax[2].plot(X, dead)
    ax[2].set_xscale('log')
    
    plt.tight_layout()
    plt.show()
