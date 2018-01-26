"""
This creates Figure 1.
"""

import pandas as pd
import numpy as np
import numpy_indexed as npi
import scipy as sp
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

"""
    Figure C: only the average of real data at each concentration X
    Figure D: the average of lexp at each concentration X
    Figure E: the average of expected number of dead cells at each X
    Figrue F: no lines and only one color
"""


# Print the average and 95 confidence interval
def CIPlot(_x, _y, confidence, ax):
    x_unique, y_mean = npi.group_by(_x).mean(_y)
    y_std = npi.group_by(_x).std(_y)[1]
    sample_size = npi.count(_x)[1]
    yerr = []
    for i in range(len(sample_size)):
        yerr.append((y_std[i] / np.sqrt(sample_size)[i]) * sp.stats.t._ppf((1 + confidence) / 2., sample_size[i] - 1))
    ax.errorbar(x=x_unique, y=y_mean, yerr=yerr, fmt='o')
    ax.scatter(x_unique, y_mean)


def RangePlot(_df, var, low, high, ax):
    _x = _df['concentration'].tolist()
    _y = _df[var].tolist()
    x_unique, y_mean = npi.group_by(_x).mean(_y)
    x_unq = _df.groupby('concentration')
    y_low = np.array(x_unq.quantile(low)[var].tolist())
    y_high = np.array(x_unq.quantile(high)[var].tolist())
    ax.scatter(x_unique, y_mean)
    ax.fill_between(x_unique, y_high, y_low, facecolor='blue', alpha=0.5)


# Check that MCMC actually fit the data provided
def DataFitCheckFigureMaker(DoseResponseM, ax1, ax2, ax3, ax4):
    """ Plot the curves for (lnum vs. X) and (dead vs. X)"""
    # Plot lObs at each concentration X
    X = np.array(DoseResponseM.drugCs)
    lObs = np.array(DoseResponseM.lObs)
    CIPlot(X, lObs, 0.95, ax1)
    ax1.set_title('lObs vs. concentration')
    ax1.set_xlabel('concentration')
    ax1.set_ylabel('the number of live cells')

    # Using 1000 sets of lExp values
    # Plot the average of lExp at each concentration X
    for n, i in enumerate(np.random.choice(DoseResponseM.samples['lExp'].shape[0], 1000)):
        lExp = np.array(DoseResponseM.samples['lExp'][i, :])
        df1 = pd.DataFrame({'concentration': X, 'lExp': lExp})
        if n == 0:
            df2 = df1
        else:
            df2 = df2.append(df1, ignore_index=True)
    ax2.set_title('lExp vs. concentration')
    ax2.set_xlabel('concentration')
    ax2.set_ylabel('the number of live cells')
    RangePlot(df2, 'lExp', 0.1, 0.9, ax2)

    # Plot the DeadExp at each concentration X
    for n, i in enumerate(np.random.choice(DoseResponseM.samples['growthV'].shape[0], 1000)):
        GR = np.array(DoseResponseM.samples['growthV'][i, :])
        DR = np.array(DoseResponseM.samples['deathV'][i, :])
        df3 = pd.DataFrame({'concentration': X, 'growthRate': GR})
        df4 = pd.DataFrame({'concentration': X, 'deathRate': DR})
        if n == 0:
            df5 = df3
            df6 = df4
        else:
            df5 = df5.append(df3, ignore_index=True)
            df6 = df6.append(df4, ignore_index=True)
    RangePlot(df5, 'growthRate', 0.1, 0.9, ax3)
    ax3.set_xlabel('concentration')
    RangePlot(df6, 'deathRate', 0.1, 0.9, ax4)
    ax4.set_xlabel('concentration')


# Plot growth rate vs. death rate
def GRvsDRFigureMaker(DoseResponseM, ax):
    for i in np.random.choice(DoseResponseM.samples['growthV'].shape[0], 50):
        ax.scatter(DoseResponseM.samples['growthV'][i, :], DoseResponseM.samples['deathV'][i, :], color='b')
        ax.set_xlim([0, 0.6])
        ax.set_ylim([0, 0.6])
        ax.set_xlabel('growthRate')
        ax.set_ylabel('deathRate')
        ax.set_title('growthRate vs. deathRate')


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

    DataFitCheckFigureMaker(DoseResponseM, ax[2], ax[3], ax[4], ax[5])
    GRvsDRFigureMaker(DoseResponseM, ax[6])
    PCAFigureMaker(DoseResponseM, ax[7])

    # Make first cartoon
    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_uppercase[ii])

    # Try and fix overlapping elements
    f.tight_layout()

    return f
