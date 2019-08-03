"""
This creates Figure 2.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from ..fcseAnalysis import importFCS


def makeFigure():
    ''' Make Figure 2. This should generally be initial analysis
    of the data we've been collecting. '''
    from string import ascii_uppercase
    from .FigureCommon import getSetup, subplotLabel

    # Get list of axis objects
    ax, f, _ = getSetup((12, 8), (4, 5))

    for axis in ax[0:20]:
        axis.tick_params(axis='both', which='major', pad=-2)  # set ticks style

    # Blank out for cartoon
    for axis in ax[0:5]:
        axis.axis('off')

    # Show simulation plots (predicted vs experimental)
    simulationPlots(axes=[ax[5], ax[6], ax[7], ax[10], ax[11], ax[12]])

    # Show violin plots for model parameters
    violinPlots(axes=[ax[8], ax[9], ax[13], ax[14]])

    # TODO: change labels for each subplot
    for ii, item in enumerate([ax[0], ax[5], ax[3], ax[8], ax[15], ax[17], ax[18]]):
        subplotLabel(item, ascii_uppercase[ii])

    for axis in ax[15:20]:
        axis.axis('off')

    # Try and fix overlapping elements
    f.tight_layout(pad=0.1)

    return f


def simulationPlots(axes, ff='101117_H1299', drugAname='Dox', drugBname='NVB', sg=None):
    """ Make plots of experimental data. """
    from ..sampleAnalysis import readModel

    # Load model and dataset
    if sg is None:
        sg = False

    classM, _ = readModel(ff=ff, singles=sg)

    df = pd.DataFrame(classM.expTable)

    df['time'] = np.tile(classM.timeV, int(df.shape[0] / classM.timeV.size))
    df['dose'] = np.repeat(classM.doses, classM.timeV.size).astype(np.float64)
    df['drug'] = np.repeat(classM.drugs, int(df.shape[0] / len(classM.drugs)))

    # help to name title
    quant_tt = ['Phase', 'Annexin V', 'YOYO-3']

    # array of all time points
    times = np.unique(df['time'])

    for ii, ax in enumerate(axes):
        quant = ['confl', 'apop', 'dna'][ii % 3]

        if ii < 3:
            dfcur = df.loc[df['drug'] != drugAname, :]
        else:
            dfcur = df.loc[df['drug'] != drugBname, :]

        # array of all doses for the drug
        doses = np.unique(dfcur['dose'])

        # take average of quant for all data points
        grpCols = ['time', 'dose', 'drug']
        this_dfcur = dfcur.groupby(grpCols)
        this_dfcur_avg = this_dfcur.agg({quant: 'mean'}).unstack(0)

        # subtratc ctrl for apop and dna
        if quant == 'apop':  # apop (annexin v)
            ctrl = np.array(this_dfcur_avg.apop.iloc[0])
            this_dfcur_avg.apop = this_dfcur_avg.apop - ctrl
        elif quant == 'dna':  # dna (vovo-3)
            ctrl = np.array(this_dfcur_avg.dna.iloc[0])
            this_dfcur_avg.dna = this_dfcur_avg.dna - ctrl

        # plot simulations
        quantile = 0.95
        palette = plt.get_cmap('tab10')  # color palette

        for k in range(len(doses)):
            # plot simulations for each drug dose
            qt = this_dfcur_avg[quant].iloc[k]

            if quant == 'confl':
                ax.plot(times, qt, color=palette(k), linewidth=1, alpha=0.9,
                        label=str(doses[k]))
            else:
                ax.plot(times, qt, color=palette(k), linewidth=1, alpha=0.9)

            # plot confidence intervals for simulations for each drug dose
            dfci = dfcur[dfcur.dose == doses[k]].groupby('time')
            y_low = dfci[quant].quantile((1 - quantile) / 2).values
            y_high = dfci[quant].quantile(1 - (1 - quantile) / 2).values
            if quant != 'confl':
                y_low = [a_i - b_i for a_i, b_i in zip(y_low, ctrl)]
                y_high = [a_i - b_i for a_i, b_i in zip(y_high, ctrl)]
            ax.fill_between(times, y_high, y_low, color=palette(k), alpha=0.2)

        # drugs with unitu nM
        drugs_nM = ['Dox', 'NVB', 'Paclitaxel', 'Erl']
        # drugs with lowercase drugname
        drugs_lowercase = ['Erl', 'Paclitaxel', 'Binimetinib']

        # add legends
        if quant == 'confl':
            if (ii < 3 and drugBname in drugs_nM) or (ii >= 3 and drugAname in drugs_nM):
                title = 'Doses (nM)'
            else:
                title = r'Doses ($\mu$M)'
            legend = ax.legend(loc=2, ncol=2, title=title, handletextpad=0.3,
                               handlelength=0.5, columnspacing=0.5, prop={'size': 7})
            legend.get_title().set_fontsize('8')

        # set titles and labels
        ax.set_xlabel('Time (hrs)')

        if ii < 3:
            if drugBname in drugs_lowercase:
                ax.set_title(quant_tt[ii % 3] + ' (' + drugBname + ')')
            else:
                ax.set_title(quant_tt[ii % 3] + ' (' + drugBname.upper() + ')')
        else:
            if drugAname in drugs_lowercase:
                ax.set_title(quant_tt[ii % 3] + ' (' + drugAname + ')')
            else:
                ax.set_title(quant_tt[ii % 3] + ' (' + drugAname.upper() + ')')

        if quant == 'confl':
            ax.set_ylim(0., 100.)
        else:
            if ff == '101117_H1299':
                ax.set_ylim(-0.1, 0.5)
            else:
                ax.set_ylim(-1.0, 10.0)

        ax.set_ylabel('Percent Image Positive')


def violinPlots(axes, ff='101117_H1299', sg=None):
    """ Create violin plots of model posterior. """
    from ..utils import violinplot

    # Load model and dataset
    if sg is None:
        sg = False

    dfdict, drugs, _ = violinplot(ff, singles=sg)

    # Plot params vs. drug dose
    for j, drug in enumerate(drugs):
        # Get drug
        dfplot = dfdict[drug]

        # combine div and deathRate in one dataframe
        # take exponential
        dose = np.array([float(ds) for ds in np.array(dfplot['dose'])])
        df1 = pd.DataFrame({'rate': np.append(dfplot['div'], dfplot['deathRate']),
                            'type': np.append(np.repeat('div', len(dfplot)),
                                              np.repeat('deathRate', len(dfplot))),
                            'dose': np.append(dose, dose)})

        df2 = pd.DataFrame({'apopfrac': dfplot['apopfrac'], 'dose': dose})
        df1 = df1.sort_values(by='dose')
        df2 = df2.sort_values(by='dose')

        # Iterate over each parameter in params
        for i, param in enumerate(['rate', 'apopfrac']):
            idx = 2 * j + i
            # Set y-axis confluence limits for each parameter
            if param == 'rate':
                # Make violin plots
                sns.violinplot(x='dose', y='rate', hue='type', data=df1, ax=axes[idx],
                               palette='Set2', linewidth=0.2)
                # Set legend
                axes[idx].legend(loc=4, handletextpad=0.3, handlelength=0.8, prop={'size': 8})
                # Set y label
                axes[idx].set_ylabel('Rate')
            elif param == 'apopfrac':
                # Make violin plots
                sns.violinplot(x='dose', y=param, data=df2, ax=axes[idx],
                               color=sns.color_palette('Set2')[2], linewidth=0.2)
                # Set y label
                axes[idx].set_ylabel('Apopfrac')
                # Set ylim
                axes[idx].set_ylim([0, 1])

            drugs_nM = ['Dox', 'NVB', 'Paclitaxel', 'Erl']
            drugs_lowercase = ['Erl', 'Paclitaxel', 'Binimetinib']
            if drug in drugs_nM:
                if drug in drugs_lowercase:
                    axes[idx].set_xlabel(drug + ' (nM)')
                else:
                    axes[idx].set_xlabel(drug.upper() + ' (nM)')
            else:
                if drug in drugs_lowercase:
                    axes[idx].set_xlabel(drug + r' ($\mu$M)')
                else:
                    axes[idx].set_xlabel(drug.upper() + r' ($\mu$M)')

            axes[idx].tick_params(axis='x', labelsize=6)


'''
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

    sns.boxenplot(x='Sample', y='sFITC', data=dataFilt, ax=ax)

    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    ax.set_ylim(-1.5, 1.5)


def CFSEcorr(ax):
    """ Correlate CFSE signal with the inferred growth rate. """
    data = importFCS()

    data_mean = data.groupby(['Sample'])['sFITC'].mean().to_frame()

    data_mean['Params'] = np.nan  # TODO: Replace with extracing from fit
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
    ax.plot([-0.5, 2.34], [0.227, 0.0])  # TODO: Check how we should draw the line here
    ax.set_ylim(0.15, 0.23)
    ax.set_xlim(-0.5, 0.3)
    ax.set_xlabel('Log(CFSE / SSC-W)')
    ax.set_ylabel('Fit Growth Rate (1/min)')
'''
