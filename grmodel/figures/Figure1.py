"""
This creates Figure 1.
"""

import pandas as pd
import numpy as np
import scipy as sp
import numpy_indexed as npi
import matplotlib


def plot_mean_and_CI(_x, _y, confidence, ax):
    """ Plot the mean value and confidence interval """
    # Group _y by _x and find the mean, standard deviation of _y at each _x
    x_unique, y_mean = npi.group_by(_x).mean(_y)
    y_std = npi.group_by(_x).std(_y)[1]
    sample_size = npi.count(_x)[1]
    yerr = []  # a list to store the confidence interval
    for i in range(len(sample_size)):
        yerr.append((y_std[i] / np.sqrt(sample_size)[i]) * sp.stats.t._ppf((1 + confidence) / 2., sample_size[i] - 1))
    ax.errorbar(x=x_unique, y=y_mean, yerr=yerr, fmt='o', color='black', alpha=0.5)


def plot_median_and_quantile(_df, var, range1, range2, ax):
    """ Plot the median, low and high quantile """
    _x = _df['concentration'].tolist()
    _y = _df[var].tolist()
    x_unique, y_median = npi.group_by(_x).median(_y)
    x_unq = _df.groupby('concentration')
    ax.scatter(x_unique, y_median, s=2)
    if var != 'lExp':
        y_low1 = np.array(x_unq.quantile(range1[0])[var].tolist())
        y_high1 = np.array(x_unq.quantile(range1[1])[var].tolist())
        y_low2 = np.array(x_unq.quantile(range2[0])[var].tolist())
        y_high2 = np.array(x_unq.quantile(range2[1])[var].tolist())
        ax.fill_between(x_unique, y_high1, y_low1, color='b', alpha=0.2, label='90% quantile')
        ax.fill_between(x_unique, y_high2, y_low2, color='g', alpha=0.5, label='50% quantile')
        ax.legend()


def plot_exact_data(M, ax2, ax3):
    """ Plot the data provided """
    X = np.array(M.drugCs)
    lObs = np.array(M.lObs)
    # Figure C: plot the mean and 95% CI of lObs at each concentration X
    plot_mean_and_CI(X, lObs, 0.95, ax2)
    ax2.set_title('Mean and 95% CI of # of live cells')
    ax2.set_xlabel(r'$log_{10}$[DOX(nM)]')
    ax2.set_ylabel('# of live cells')
    # Part of Figure D: Compare the sampling lExp with the exact data lObs
    plot_mean_and_CI(X, lObs, 0, ax3)


def plot_sampling_data(df, ax3, ax4, ax5, ax6):
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

    # Figure D: Plot the median, 90% and 50% quantile of the expected number
    # of live cells at each x
    plot_median_and_quantile(df1, 'lExp', [0.05, 0.95], [0.25, 0.75], ax3)
    ax3.set_title('Median of predicted # of live cells ')
    ax3.set_xlabel(r'$log_{10}$[DOX(nM)]')
    ax3.set_ylabel('Predicted # of live cells')

    # Figure E: Plot the median, 90% and 50% quantile of growth rate at each x
    plot_median_and_quantile(df1, 'growthV', [0.05, 0.95], [0.25, 0.75], ax4)
    ax4.set_title('Median of predicted growth rate')
    ax4.set_xlabel(r'$log_{10}$[DOX(nM)]')
    ax4.set_ylabel('Predicted growth rate')


# Figure F: Plot the median, 90% and 50% quantile of growth rate at each x
    plot_median_and_quantile(df1, 'deathV', [0.05, 0.95], [0.25, 0.75], ax5)
    ax5.set_title('Median of predicted death rate')
    ax5.set_xlabel(r'$log_{10}$[DOX(nM)]')
    ax5.set_ylabel('Predicted death rate')

    # Figure G: Plot growth rate vs. death rate
    import random
    # Randomly choose 2000 points (len(df1.index) is too large and would take too much time)
    my_randoms = random.sample(range(len(df1.index)), 2000)
    for n in my_randoms:
        ax6.scatter(x=df1['growthV'].iloc[n], y=df1['deathV'].iloc[n], color='b', s=30)
        ax6.set_xlim([0, 0.6])
        ax6.set_ylim([0, 0.6])
        ax6.set_xlabel('Growth Rate')
        ax6.set_ylabel('Death Rate')
        ax6.set_title('Growth Rate vs. Death Rate')


def plot_PCA(df, ax):
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

    plot_exact_data(M, ax[2], ax[3])
    plot_sampling_data(df, ax[3], ax[4], ax[5], ax[6])
    plot_PCA(df, ax[7])

    # Make first cartoon
    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_uppercase[ii])

    # Try and fix overlapping elements
    f.tight_layout()

    return f
