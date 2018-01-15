"""
This creates Figure 1.
"""

import pandas as pd
import numpy as np
import matplotlib
import pymc3 as pm
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from grmodel.fitcurves import sigmoid, residuals
from ..pymcDoseResponse import loadCellTiter


'''
# Traceplot
def traceplot(DoseResponseM, ax):
    axx = pm.traceplot(DoseResponseM.samples, varnames='IC50', ax=ax)
'''

# Plot growth rate vs. death rate
def GRvsDRFigureMaker(DoseResponseM, ax):
    for i in np.random.choice(DoseResponseM.samples['growthV'].shape[0], 80):
        axx = sns.regplot(DoseResponseM.samples['growthV'][i, :], DoseResponseM.samples['deathV'][i, :], lowess=True, ax=ax)
        axx.set_xlim([0, 0.6])
        axx.set_ylim([0, 0.6])
        axx.set_xlabel('growthRate')
        axx.set_ylabel('deathRate')
        axx.set_title('growthRate vs. deathRate')


# Check that MCMC actually fit the data provided
def DataFitCheckFigureMaker(DoseResponseM, ax1, ax2):
    """ Plot the curves for (lnum vs. X) """
    # Compare the plots of (lObs vs. X and lExp vs. X)
    # Using five sets of lExp values
    for i in np.random.choice(DoseResponseM.samples['lExp'].shape[0], 5):
        ax1.scatter(DoseResponseM.drugCs, DoseResponseM.samples['lExp'][i, :])

    ax1.set_title('lnum vs. X')
    ax1.set_xlabel('X')
    ax1.set_ylabel('the number of live cells')

    # Using all sets of lExp values
    for i in np.random.choice(DoseResponseM.samples['lExp'].shape[0], 200):
        ax2.scatter(DoseResponseM.drugCs, DoseResponseM.samples['lExp'][i, :])

    ax2.set_title('lnum vs. X')
    ax2.set_xlabel('X')
    ax2.set_ylabel('the number of live cells')

    ax1.plot(DoseResponseM.drugCs, DoseResponseM.lObs, '^', color='black')
    ax2.plot(DoseResponseM.drugCs, DoseResponseM.lObs, '^', color='black')


# Check the dimensionality of the sampling uncertainty using PCA
def PCAFigureMaker(self, ax):
    X = np.array([self.samples['IC50s'], self.samples['hill'],
                  self.samples['Emin_growth'], self.samples['Emin_growth'],
                  self.samples['Emax_death']]).transpose()

    # it is always good to scale the data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    pca = PCA()
    X_new = pca.fit_transform(X)

    # print(pca.explained_variance_ratio_)
    # Scatter plot of PC1 vs. PC2
    ax.scatter(X_new[:, 0], X_new[:, 1], c='aqua')

    # Plot the direction of each variable
    coeff = pca.components_
    n = coeff.shape[0]
    labels = ['IC50s', 'hill', 'Emin_growth', 'Emin_growth', 'Emax_death']
    for i in range(n):
        ax.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i],
                color='b', ha='center', va='center')

    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))
    ax.set_title("2 components PCA")
    ax.grid()


def makeFigure():
    from grmodel.pymcDoseResponse import doseResponseModel
    from grmodel.pymcDoseResponse import save, readSamples
    from .FigureCommon import getSetup, subplotLabel
    from string import ascii_uppercase

    DoseResponseM = readSamples()

    '''
    Generate Figure 1

    Broadly, this figure should motivate looking at cell death.
    This should be by showing that it's not captured in existing
    measurements.
    '''

    # Get list of axis objects
    ax, f, gs1 = getSetup((7, 6), (3, 3))

    GRvsDRFigureMaker(DoseResponseM, ax[5])
    #traceplot(DoseResponseM, ax[5])
    DataFitCheckFigureMaker(DoseResponseM, ax[6], ax[7])
    PCAFigureMaker(DoseResponseM, ax[8])

    # Make first cartoon
    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_uppercase[ii])

    # doseResponseTiter(ax[1])

    # Try and fix overlapping elements
    f.tight_layout()

    return f
