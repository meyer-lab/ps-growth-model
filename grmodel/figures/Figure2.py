"""
This creates Figure 2.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from ..fcseAnalysis import importFCS

def makeFigure():
    ''' Make Figure 2. This should generally be initial analysis
    of the data we've been collecting. '''
    from string import ascii_uppercase
    from .FigureCommon import getSetup, subplotLabel

    # Get list of axis objects
    ax, f, _ = getSetup((9, 8), (5, 6))

    # Blank out for cartoon
    for axis in ax[0:12]:
        axis.axis('off')

    # Show simulation plots (predicted vs experimental)
    simulationPlots(axes=[ax[12], ax[13], ax[14], ax[18], ax[19], ax[20]])

    # Show violin plots for model parameters
    violinPlots(axes=[ax[15], ax[16], ax[17], ax[21], ax[22], ax[23]])

    for ii, item in enumerate([ax[0], ax[12], ax[15], ax[18], ax[21], ax[24], ax[27]]):
        subplotLabel(item, ascii_uppercase[ii])

    CFSEcurve(ax[24])

    CFSEsamples(ax[25])

    CFSEcorr(ax[26])

    for axis in ax[27:30]:
        axis.axis('off')

    # Try and fix overlapping elements
    f.tight_layout(pad=0.1)

    return f


def simulationPlots(axes):
    from ..sampleAnalysis import readModel, simulation
    # Get the list of drugs for 101117_H1299 experiment
    classM, _ = readModel(ff="101117_H1299")
    drugs = list(sorted(set(classM.drugs)))
    drugs.remove('Control')

    # Iterate over each drug
    for i, drug in enumerate(drugs):
        # Make simulation plots and legend
        axis, patches = simulation('101117_H1299', drug, ax=axes[3*i: 3*i+3], unit='nM')
        #Show legend
        axis[2].legend(handles=patches, labelspacing=0.15, prop={'size': 4})


def violinPlots(axes):
    """ Create violin plots of model posterior. """
    from ..utils import violinplot
    dfdict, drugs, params = violinplot('101117_H1299')
    # Plot params vs. drug dose
    for j, drug in enumerate(drugs):
        # Get drug
        dfplot = dfdict[drug]
        # Iterate over each parameter in params
        for i, param in enumerate(params):
            idx = 3*j + i
            # Set y-axis confluence limits for each parameter
            if param == 'div':
                axes[idx].set_ylim([-2.5, -1.2])
            elif param == 'deathRate':
                axes[idx].set_ylim([-5.0, -0.7])
            elif param == 'apopfrac':
                axes[idx].set_ylim([0, 1])
            # Make violin plots
            sns.violinplot(x='dose', y=param, data=dfplot, ax=axes[idx], cut=0)
            axes[idx].set_xlabel(drug + ' dose')
            # Rotate dosage labels
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45,
                                      rotation_mode="anchor", ha="right",
                                      position=(0, 0.05), fontsize=4.5)


def CFSEcurve(ax):
    """ Plot the CFSE standard curve. """
    data = importFCS()

    data_mean = data.groupby(['Sample'])['sFITC'].mean()

    y = [data_mean['Day0 STD'], data_mean['Day1 STD'], data_mean['Day2 STD'],
         data_mean['Day3 STD'], data_mean['Day4 STD'], data_mean['Dox  CTRL'],
         data_mean['NVB  CTRL']]

    df = pd.DataFrame({'Days': [0, 1, 2, 3, 4, 4, 4], 'CFSE': y})


    sns.regplot(x="CFSE", y="Days", data=df, ax=ax)
    ax.set_ylim(-0.4, 5)


def CFSEsamples(ax):
    """ Plot the distribution of CFSE values for each experimental sample. """
    data = importFCS()

    dataFilt = data.loc[data['Sample'].str.contains('Dox'), :]
    dataFilt = dataFilt.append(data.loc[data['Sample'].str.contains('NVB'), :])

    dataFilt.sort_values(inplace=True, by='Sample')

    sns.lvplot(data=dataFilt, x='Sample', y='sFITC', ax=ax)

    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    ax.set_ylim(-1.5, 1.5)


def CFSEcorr(ax):
    """ Correlate CFSE signal with the inferred growth rate. """
    data = importFCS()

    data_mean = data.groupby(['Sample'])['sFITC'].mean().to_frame()

    data_mean['Params'] = np.nan # TODO: Replace with extracing from fit
    data_mean.loc['Dox a100 nM', 'Params'] = -1.8
    data_mean.loc['NVB 40 nM', 'Params'] = -1.6
    data_mean.loc['Dox 50 nM', 'Params'] = -1.6
    data_mean.loc['NVB  CTRL', 'Params'] = -1.5
    data_mean.loc['Dox  CTRL', 'Params'] = -1.5
    data_mean.loc['Dox 25 nM', 'Params'] = -1.5
    data_mean.loc['NVB 20 nM', 'Params'] = -1.5
    data_mean.loc['NVB 10 nM', 'Params'] = -1.5

    data_mean['Params'] = np.exp(data_mean['Params'])

    # Plot of predicted vs. actual
    data_mean.plot.scatter(x='sFITC', y='Params')
    ax.plot([-0.5, 2.34], [0.227, 0.0]) # TODO: Check how we should draw the line here
    ax.set_ylim(0.15, 0.23)
    ax.set_xlim(-0.5, 0.3)
    ax.set_xlabel('Log(CFSE / SSC-W)')
    ax.set_ylabel('Fit Growth Rate (1/min)')
