"""
.. module:: Figure2

.. moduleauthor:: Guan Ning; Rui Yan <rachelyan@ucla.edu>; Aaron Meyer <ameyer@ucla.edu>

This module generates Figure2 which should generally be initial analysis
    of the data we've been collecting.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def makeFigure():
    """This function generetes Figure 2.

    Args None
    Returns:
        A figure
    """
    from string import ascii_lowercase
    from .FigureCommon import getSetup, subplotLabel

    # Get list of axis objects
    ax, f = getSetup((12, 8), (4, 5))

    for axis in ax[0:20]:
        axis.tick_params(axis='both', which='major', pad=-2)  # set ticks style

    # Blank out for cartoon
    for axis in ax[0:5] + ax[15:20]:
        axis.axis('off')

    # Show simulation plots (predicted vs experimental)
    simulationPlots(axes=[ax[5], ax[6], ax[7], ax[10], ax[11], ax[12]])

    # Show violin plots for model parameters
    violinPlots(axes=[ax[8], ax[9], ax[13], ax[14]])

    for ii, item in enumerate([ax[0], ax[5], ax[3], ax[8], ax[15]]):
        subplotLabel(item, ascii_lowercase[ii])

    return f


def simulationPlots(axes, ff='101117_H1299'):
    """ Make plots of experimental data. """
    from ..sampleAnalysis import readModel
    from more_itertools import unique_everseen

    # Load model and dataset
    classM, _ = readModel(ff=ff, model='growthModel')

    df = pd.DataFrame(classM.expTable)

    df['time'] = np.tile(classM.timeV, int(df.shape[0] / classM.timeV.size))
    df['dose'] = np.repeat(classM.doses, classM.timeV.size).astype(np.float64)
    df['drug'] = np.repeat(classM.drugs, int(df.shape[0] / len(classM.drugs)))

    # Get drug names
    drugs = list(unique_everseen(classM.drugs))
    drugs.remove('Control')
    drugAname, drugBname = drugs
    print('drugname: ' + drugAname + ', ' + drugBname)

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
                ax.plot(times, qt, color=palette(k), linewidth=1, alpha=0.9, label=str(doses[k]))
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
        ax.set_xlabel('Time (h)')

        if ii < 3:
            ax.set_title(quant_tt[ii % 3] + ' (' + drugBname + ')')

        else:
            ax.set_title(quant_tt[ii % 3] + ' (' + drugAname + ')')

        if quant == 'confl':
            ax.set_ylim(0., 100.)
        else:
            if ff == '101117_H1299':
                ax.set_ylim(-0.1, 0.5)
            else:
                ax.set_ylim(-1.0, 10.0)

        ax.set_ylabel('Percent Image Positive')


def violinPlots(axes, ff='101117_H1299'):
    """ Create violin plots of model posterior. """
    from ..utils import violinplot

    # Load model and dataset
    dfdict, drugs, _ = violinplot(ff)

    # Plot params vs. drug dose
    for j, drug in enumerate(drugs):
        # Get drug
        dfplot = dfdict[drug]

        # Combine div and deathRate in one dataframe
        # Convert div and deathRate from log scale to linear
        dose = np.array([float(ds) for ds in np.array(dfplot['dose'])])
        df1 = pd.DataFrame({'rate': np.append(10 ** dfplot['div'],
                                              10 ** dfplot['deathRate']),
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
                axes[idx].legend(handletextpad=0.3, handlelength=0.8, prop={'size': 8})
                # Set y label
                axes[idx].set_ylabel(r'Rate (1/h)')
                # Set ylim
                axes[idx].set_ylim(bottom=0)
            elif param == 'apopfrac':
                # Make violin plots
                sns.violinplot(x='dose', y=param, data=df2, ax=axes[idx],
                               color=sns.color_palette('Set2')[2], linewidth=0.2)
                # Set y label
                axes[idx].set_ylabel('Apopfrac')
                # Set ylim
                axes[idx].set_ylim([0, 1])

            # Set x labels
            drugs_nM = ['Dox', 'NVB', 'Paclitaxel', 'Erl']

            if drug in drugs_nM:
                axes[idx].set_xlabel(drug + ' (nM)')
            else:
                axes[idx].set_xlabel(drug + r' ($\mu$M)')

            if ff == '101117_H1299':
                axes[idx].set_ylim(bottom=-0.005)
            else:
                axes[idx].set_ylim(bottom=-0.01)

            axes[idx].tick_params(axis='x', labelsize=6)
