"""
This creates Figure 1.
"""

import pandas as pd
import numpy as np
import scipy as sp
import numpy_indexed as npi
import matplotlib


def CIPlot(_x, _y, confidence, ax):
    """ Plot the average value and confidence interval """
    # Group _y by _x and find the mean, standard deviation of _y at each _x
    x_unique, y_mean = npi.group_by(_x).mean(_y)
    y_std = npi.group_by(_x).std(_y)[1]
    sample_size = npi.count(_x)[1]
    yerr = []  # a list to store the confidence interval
    for i in range(len(sample_size)):
        yerr.append((y_std[i] / np.sqrt(sample_size)[i]) * sp.stats.t._ppf((1 + confidence) / 2., sample_size[i] - 1))
    ax.errorbar(x=x_unique, y=y_mean, yerr=yerr, fmt='o', color='black', alpha=0.5)


def RangePlot(_df, var, low, high, ax):
    """ Plot the median, low and high quantile """
    _x = _df['concentration'].tolist()
    _y = _df[var].tolist()
    x_unique, y_median = npi.group_by(_x).median(_y)
    x_unq = _df.groupby('concentration')
    y_low = np.array(x_unq.quantile(low)[var].tolist())
    y_high = np.array(x_unq.quantile(high)[var].tolist())
    ax.scatter(x_unique, y_median)
    ax.fill_between(x_unique, y_high, y_low, facecolor='b', alpha=0.3)


def RealDataPlot(M, ax2, ax3):
    """ Plot the data provided """
    X = np.array(M.drugCs)
    lObs = np.array(M.lObs)
    # Figure C: plot the mean and CI of lObs at each concentration X
    CIPlot(X, lObs, 0.95, ax2)
    ax2.set_title('concentration vs. Obs')
    ax2.set_xlabel('concentration')
    ax2.set_ylabel('the number of live cells')
    # Part of Figure D: Compare the sampling lExp with the real data lObs
    CIPlot(X, lObs, 0, ax3)


def SamplingDataPlot(df, ax3, ax4, ax5, ax6):
    """ Check that MCMC actually fit the data provided """
    # Define drug concentrations x to test MCMC sampling data fit
    x = np.arange(-1.0, 3.0, 0.01)

    # Loop through the sampling IC50s, hill, Emin_growth, Emax_growth, Emax_death in the nth sample
    for n in range(len(df.index)):

        # Drug term since we're using constant IC50 and hill slope
        drugTerm = 1.0 / (1.0 + np.power(10.0, (df['IC50s'].iloc[n] - x) * df['hill'].iloc[n]))

        # growthV = Emin_growth + (Emax_growth - Emin_growth) * drugTerm
        growthV = df['Emin_growth'].iloc[n] + ((df['Emax_growth'].iloc[n] - df['Emin_growth'].iloc[n]) * drugTerm)

        # _Assuming deathrate in the absence of drug is zero
        # deathV = Emax_death * drugTerm
        deathV = df['Emax_death'].iloc[n] * drugTerm

        # Calculate the growth rate
        GR = growthV - deathV

        # Calculate the number of live cells
        lnum = np.exp(GR * 72.0)

        # Normalize live cell data to control, as is similar to measurements
        lExp = lnum / lnum[0]

        # Temporary dataframe to store the sampling data lExp, growthV and deathV in the nth sample
        temp_df = pd.DataFrame({'concentration': x, 'lExp': lExp, 'growthV': growthV, 'deathV': deathV})

        # Append the nth sampling data to dataframe df1
        if n == 0:
            df1 = temp_df
        else:
            df1 = df1.append(temp_df, ignore_index=True)

    # Figure D: Plot the median, 0.1 and 0.9 quantile of the expected number
    # of live cells at each x
    RangePlot(df1, 'lExp', 0.1, 0.9, ax3)
    ax3.set_title('concentration vs. lExp')
    ax3.set_xlabel('concentration')
    ax3.set_ylabel('the number of live cells')

    # Figure E: Plot the median,  0.1 and 0.9 quantile of growth rate at each x
    RangePlot(df1, 'growthV', 0.1, 0.9, ax4)
    ax4.set_title('concentration vs. growth rate')
    ax4.set_xlabel('concentration')
    ax4.set_ylabel('growth rate')

    # Figure F: Plot the median,  0.1 and 0.9 quantile of death rate at each x
    RangePlot(df1, 'deathV', 0.1, 0.9, ax5)
    ax5.set_title('concentration vs. death rate')
    ax5.set_xlabel('concentration')
    ax5.set_ylabel('death rate')

    # Figure G: Plot growth rate vs. death rate
    import random
    # Randomly choose 2000 points (len(df1.index) is too large and would take too much time)
    my_randoms = random.sample(xrange(len(df1.index)), 2000)
    for n in my_randoms:
        ax6.scatter(x=df1['growthV'].iloc[n], y=df1['deathV'].iloc[n], color='b')
        ax6.set_xlim([-0.1, 0.8])
        ax6.set_ylim([-0.1, 0.8])
        ax6.set_xlabel('growth rate')
        ax6.set_ylabel('death rate')
        ax6.set_title('growth rate vs. death rate')


def PCA(df, ax):
    """ Check the dimensionality of the sampling uncertainty using PCA """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    features = ['IC50s', 'hill', 'Emin_growth', 'Emin_growth', 'Emax_death']
    # Separating out the features
    m = df.loc[:, features].values

    # Standardizing the features
    m_new = StandardScaler().fit_transform(m)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(m)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    # print out explained_variance
    # print(pca.explained_variance_ratio_)

    # Scatter plot of PC1 vs. PC2
    ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'], alpha=0.5)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('2 component PCA')
    ax.grid(True)


def makeFigure():
    '''
    Generate Figure 1
    Broadly, this figure should motivate looking at cell death.
    This should be by showing that it's not captured in existing
    measurements.
    '''
    from grmodel.pymcDoseResponse import doseResponseModel
    from grmodel.pymcDoseResponse import save, readSamples
    from .FigureCommon import getSetup, subplotLabel
    from string import ascii_uppercase

    M = readSamples()

    # Store the sampling data for priors to calculate the lExp, growthV and deathV at each concentration
    IC50s = M.samples['IC50s']
    hill = M.samples['hill']
    Emin_growth = M.samples['Emin_growth']
    Emax_growth = M.samples['Emax_growth']
    Emax_death = M.samples['Emax_death']

    df = pd.DataFrame({'IC50s': IC50s, 'hill': hill, 'Emin_growth': Emin_growth,
                       'Emax_growth': Emax_growth, 'Emax_death': Emax_death})

    # Get list of axis objects
    ax, f, gs1 = getSetup((7, 6), (3, 3))

    RealDataPlot(M, ax[2], ax[3])
    SamplingDataPlot(df, ax[3], ax[4], ax[5], ax[6])
    PCA(df, ax[7])

    # Make first cartoon
    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_uppercase[ii])

    # Try and fix overlapping elements
    f.tight_layout()

    return f
