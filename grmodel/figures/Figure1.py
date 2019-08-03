"""
This creates Figure 1.
"""

import pandas as pd
import numpy as np
import scipy as sp
import numpy_indexed as npi


def makeFigure():
    '''
    Generate Figure 1
    Broadly, this figure should motivate looking at cell death.
    This should be by showing that it's not captured in existing
    measurements.
    '''
    from matplotlib.ticker import FormatStrFormatter
    from ..pymcDoseResponse import doseResponseModel
    from .FigureCommon import getSetup, subplotLabel
    from string import ascii_lowercase

    M = doseResponseModel()
    M.readSamples()

    # Store the sampling data for priors to calculate the lExp, growthV and deathV at each concentration
    df = pd.DataFrame({'IC50s': M.trace['IC50s'],
                       'hill': M.trace['hill'],
                       'Emin_growth': M.trace['Emin_growth'],
                       'Emax_death': M.trace['Emax_death']})
    df['Emax_growth'] = M.Emax_growth

    # Get list of axis objects
    ax, f = getSetup((7, 3.5), (2, 4))

    # set significant figures for xtick
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[4].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # Going to put a cartoons in A and B
    ax[0].axis('off')
    ax[1].axis('off')

    plot_exact_data(M, ax[2], ax[3])
    plot_sampling_data(df, ax[3], ax[4], ax[5], ax[6])
    alphaFig(M, ax[7])

    # Make first cartoon
    ax.pop(1)
    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_lowercase[ii])

    return f


def plot_mean_and_CI(ax, _x, _y, confidence=True):
    """ Plot the mean value and confidence interval """
    # Group _y by _x and find the mean, standard deviation of _y at each _x
    x_unique, y_mean = npi.group_by(_x).mean(_y)
    sample_size = npi.count(_x)[1]
    y_sem = npi.group_by(_x).std(_y)[1] / np.sqrt(sample_size)
    
    if not confidence:
        y_sem = None

    ax.errorbar(x=x_unique, y=y_mean, yerr=y_sem, fmt='.', color='black')


def plot_data_and_quantile(df2, xvar, yvar, ax, c='b', quantiles=[0.90, 0.75, 0.50], lb=None):
    """ Plot the median, low and high quantile """
    _x = np.array(list(xvar.groups.keys()))
    _y = df2[yvar]

    if lb is None:
        ax.plot(_x, _y, color=c, linewidth=1, alpha=0.9)
    else:
        ax.plot(_x, _y, color=c, linewidth=1, alpha=0.9, label=lb)

    alphas = np.arange(0.2, 1.0, 0.2)
    for i in range(len(quantiles)):
        y_low = np.array(xvar.quantile((1 - quantiles[i]) / 2)[yvar])
        y_high = np.array(xvar.quantile(1 - (1 - quantiles[i]) / 2)[yvar])
        if(lb is not None):
            ax.fill_between(_x, y_high, y_low, color=c, alpha=alphas[i])
        else:
            ax.fill_between(_x, y_high, y_low, color=c, alpha=alphas[i],
                            label=str(int(quantiles[i] * 100)) + '% CI')


def plot_exact_data(M, ax2, ax3):
    """ Plot the data provided """
    X = np.array(M.drugCs)
    lObs = np.array(M.lObs)
    # Figure C: plot the mean and SEM of lObs at each concentration X
    plot_mean_and_CI(ax2, X, lObs, )
    ax2.set_xlabel(r'$\mathregular{Log_{10}}$[DOX(nM)]')
    ax2.set_ylabel(r'Cell viability' + '\n' + r'normalized to untreated cells')
    ax2.set_ylim(0, 1.1)
    # Part of Figure D: Compare the sampling lExp with the exact data lObs
    plot_mean_and_CI(ax3, X, lObs, confidence=False)


def df_crossjoin(df1, df2, **kwargs):
    """
    Make a cross join (cartesian product) between two dataframes by using a constant temporary key.
    Also sets a MultiIndex which is the cartesian product of the indices of the input dataframes.
    See: https://github.com/pydata/pandas/issues/5401
    :param df1 dataframe 1
    :param df1 dataframe 2
    :param kwargs keyword arguments that will be passed to pd.merge()
    :return cross join of df1 and df2
    """
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res


def plot_sampling_data(df, ax3, ax4, ax5, ax6):
    """ Check that MCMC actually fit the data provided """
    # Define drug concentrations x to test MCMC sampling data fit
    df1 = df_crossjoin(df, pd.DataFrame({'concentration': np.arange(-1.0, 3.0, 0.01)}))

    # Drug term since we're using constant IC50 and hill slope
    df1['drugTerm'] = 1.0 / (1.0 + np.power(10.0, (df1['IC50s'] - df1['concentration']) * df1['hill']))

    # Minimum drug term
    df1['controlDrugTerm'] = 1.0 / (1.0 + np.power(10.0, (df1['IC50s'] - np.min(df1['concentration'])) * df1['hill']))

    # growthV = Emin_growth + (Emax_growth - Emin_growth) * drugTerm
    df1['growthV'] = df1['Emax_growth'] + ((df1['Emin_growth'] - df1['Emax_growth']) * df1['drugTerm'])

    # Control growth rate
    df1['growthControl'] = df1['Emax_growth'] + ((df1['Emin_growth'] - df1['Emax_growth']) * df1['controlDrugTerm'])

    # Range of growth effect
    df1['growthRange'] = df1['Emax_growth'] - df1['Emin_growth']

    # _Assuming deathrate in the absence of drug is zero
    # deathV = Emax_death * drugTerm
    df1['deathV'] = df1['Emax_death'] * df1['drugTerm']

    # Calculate the growth rate
    df1['GR'] = df1['growthV'] - df1['deathV']

    # Calculate the number of live cells, normalized to T=0
    df1['lExp'] = np.exp(df1['GR'] * 72.0 - df1['growthControl'] * 72.0)

    df2 = df1.groupby(['concentration']).agg({'lExp': 'median', 'growthV': 'median', 'deathV': 'median'})
    conc = df1.groupby(['concentration'])

    # Figure D: Plot the median, 90% and 50% quantile of the expected number
    # of live cells at each x
    plot_data_and_quantile(df2, conc, 'lExp', ax3)
    ax3.set_xlabel(r'$\mathregular{Log_{10}}$[DOX(nM)]')
    ax3.set_ylabel('Fit CellTiter quantitation')
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc=6)

    # Figure E: Plot the median, 90% and 50% quantile of growth rate at each x
    plot_data_and_quantile(df2, conc, 'growthV', ax4)
    ax4.set_xlabel(r'$\mathregular{Log_{10}}$[DOX(nM)]')
    ax4.set_ylabel('Predicted growth rate (1/min)')
    ax4.set_ylim(0., ax4.get_ylim()[1])
    ax4.legend(loc=6)

    # Figure F: Plot the median, 90% and 50% quantile of growth rate at each x
    plot_data_and_quantile(df2, conc, 'deathV', ax5)
    ax5.set_xlabel(r'$\mathregular{Log_{10}}$[DOX(nM)]')
    ax5.set_ylabel('Predicted death rate (1/min)')
    ax5.legend(loc=6)

    # Figure G: Plot growth rate vs. death rate
    ax6.scatter(x=df['Emax_growth'] - df['Emin_growth'],
                y=df['Emax_death'], color='b', s=1)
    ax6.set_xlim(0., df['Emax_growth'][0])
    ax6.set_ylim(0., 0.03)
    ax6.set_xlabel('Drug growth effect (1/min)')
    ax6.set_ylabel('Drug death effect (1/min)')


def alphaFig(M, ax1):
    drug_lnum_effect = 0.25

    # alpha = (R_g0 - R_gd) / R_dD ---- R_d0 is 0
    # -ln(drug_lnum_effect) / t = (1 + alpha) R_dD

    alpha = np.logspace(-2, 2)
    R_dD = -np.log(drug_lnum_effect) / M.time / (1 + alpha)
    R_gD = M.Emax_growth - R_dD * alpha

    cellDiv = R_gD * 72.
    deadCells = R_dD * (np.exp((R_gD - R_dD) * 72.) - 1) / (R_gD - R_dD)

    ax1.semilogx(alpha, deadCells, label="cum. # dead")
    ax1.set_xlabel(r'$\alpha$ (ratio growth to death effect)')
    ax1.set_ylabel('Quantity per starting cell')
    ax1.semilogx(alpha, cellDiv, 'r', label="avg. divisions")
    ax1.legend(handlelength=0.5)
