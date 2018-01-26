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
from grmodel.pymcDoseResponse import doseResponseModel
from grmodel.pymcDoseResponse import save, readSamples, IC, num
from .FigureCommon import getSetup, subplotLabel
from string import ascii_uppercase

DoseResponseM = readSamples()

"""
    Figure C: only the average of real data at each concentration X
    Figure D: the average of lexp at each concentration X
    Figure E: the average of expected number of dead cells at each X
    Figrue F: no lines and only one color
"""

'''
def plotCurves(list, var, ax):
    """ Plot the curves for (lnum vs. X, eap vs. X, dead vs. X) """
    # list = [IC50s_fit, hill_fit, Emin_growth_fit, Emax_growth_fit, Emax_death_fit, time]
    X = np.linspace(-1, 3)
    drugTerm = 1.0 / (1.0 + np.power(10.0, (list[0] - X) * list[1]))
    growthV = list[2] + (list[3] - list[2]) * drugTerm
    deathV = list[4] * drugTerm
    GR = growthV - deathV
    lExp = np.exp(GR * list[5])
    if var == 'lExp':
        ax.scatter(X, lExp)
    if var == 'growthV':
        ax.scatter(X, growthV)
    if var == 'deathV':
        ax.scatter(X, deathV)
'''


# Plot the average and 95 confidence interval
def CIPlot(_x, _y, confidence, ax):
    x_unique, y_mean = npi.group_by(_x).mean(_y)
    y_std = npi.group_by(_x).std(_y)[1]
    sample_size = npi.count(_x)[1]
    yerr = []
    for i in range(len(sample_size)):
        yerr.append((y_std[i] / np.sqrt(sample_size)[i]) * sp.stats.t._ppf((1 + confidence) / 2., sample_size[i] - 1))
    ax.errorbar(x=x_unique, y=y_mean, yerr=yerr, fmt='o')
    ax.scatter(x_unique, y_mean)


# Plot the average and low and high quantile
def RangePlot(_df, var, low, high, ax):
    _x = _df['concentration'].tolist()
    _y = _df[var].tolist()
    x_unique, y_mean = npi.group_by(_x).mean(_y)
    x_unq = _df.groupby('concentration')
    y_low = np.array(x_unq.quantile(low)[var].tolist())
    y_high = np.array(x_unq.quantile(high)[var].tolist())
    ax.scatter(x_unique, y_mean)
    ax.fill_between(x_unique, y_high, y_low, facecolor='blue', alpha=0.5)


# def getFitValue(var):
#    return pm.stats.quantiles(DoseResponseM.samples[var], [50]).get(50)


def getListofValues(var):
    return DoseResponseM.samples[var]


# Check that MCMC actually fit the data provided
def DataFitCheckFigureMaker(DoseResponseM, ax1, ax2, ax3, ax4, ax5):
    """ Plot the curves for (lnum vs. X) and (dead vs. X)"""
    # Plot the average lObs at each concentration X
    X = np.array(DoseResponseM.drugCs)
    lObs = np.array(DoseResponseM.lObs)
    CIPlot(X, lObs, 0.95, ax1)
    ax1.set_title('lObs vs. concentration')
    ax1.set_xlabel('concentration')
    ax1.set_ylabel('the number of live cells')

    # Plot the CI of lExp, growthV and deathV at each concentration X
    #IC50s_fit = getFitValue('IC50s')
    #hill_fit = getFitValue('hill')
    #Emin_growth_fit = getFitValue('Emin_growth')
    #Emax_growth_fit = getFitValue('Emax_growth')
    #Emax_death_fit = getFitValue('Emax_death')
    IC50s = getListofValues('IC50s')
    hill = getListofValues('hill')
    Emin_growth = getListofValues('Emin_growth')
    Emax_growth = getListofValues('Emax_growth')
    Emax_death = getListofValues('Emax_death')

    df = pd.DataFrame({'IC50s': IC50s, 'hill': hill, 'Emin_growth': Emin_growth,
                       'Emax_growth': Emax_growth, 'Emax_death': Emax_death})
    for n in range(len(df.index)):
        x = np.linspace(-1, 3)
        drugTerm = 1.0 / (1.0 + np.power(10.0, (df['IC50s'].iloc[n] - x) * df['hill'].iloc[n]))
        growthV = df['Emin_growth'].iloc[n] + ((df['Emax_growth'].iloc[n] - df['Emin_growth'].iloc[n]) * drugTerm)
        deathV = df['Emax_death'].iloc[n] * drugTerm
        GR = growthV - deathV
        lnum = np.exp(GR * 72.0)
        lExp = lnum / lnum[0]
        df1 = pd.DataFrame({'concentration': x, 'lExp': lExp})
        df2 = pd.DataFrame({'concentration': x, 'growthV': growthV})
        df3 = pd.DataFrame({'concentration': x, 'deathV': deathV})

        if n == 0:
            df4 = df1
            df5 = df2
            df6 = df3
        else:
            df4 = df4.append(df1, ignore_index=True)
            df5 = df5.append(df2, ignore_index=True)
            df6 = df6.append(df3, ignore_index=True)

    RangePlot(df4, 'lExp', 0.1, 0.9, ax2)
    RangePlot(df5, 'growthV', 0.1, 0.9, ax3)
    RangePlot(df6, 'deathV', 0.1, 0.9, ax4)

    ax2.set_title('lExp vs. concentration')
    ax2.set_xlabel('concentration')
    ax2.set_ylabel('the number of live cells')
    ax3.set_xlabel('concentration')
    ax4.set_xlabel('concentration')

    print(len(x))
    # Plot growth rate vs. death rate
    for n in range(len(x)):
        ax5.scatter(df2['growthV'].iloc[n], df3['deathV'].iloc[n], color='b')
        ax5.set_xlim([0, 0.1])
        ax5.set_ylim([0, 0.05])
        ax5.set_xlabel('growthRate')
        ax5.set_ylabel('deathRate')
        ax5.set_title('growthRate vs. deathRate')


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
    '''
    Generate Figure 1

    Broadly, this figure should motivate looking at cell death.
    This should be by showing that it's not captured in existing
    measurements.
    '''

    # Get list of axis objects
    ax, f, gs1 = getSetup((7, 6), (3, 3))

    #DataFitCheckFigureMaker(DoseResponseM, ax[2], ax[3], ax[4], ax[5])
    DataFitCheckFigureMaker(DoseResponseM, ax[2], ax[3], ax[4], ax[5], ax[6])
    PCAFigureMaker(DoseResponseM, ax[7])

    # Make first cartoon
    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_uppercase[ii])

    # Try and fix overlapping elements
    f.tight_layout()

    return f
