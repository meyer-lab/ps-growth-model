"""
This creates Figure 3.
"""
import numpy as np
import pandas as pd
import seaborn as sns


def makeFigure():
    ''' Generate Figure 3: This figure should show different drugs
        have different effects by looking at the division rate and
        death rate of cancer cells under their treatments. '''

    from string import ascii_uppercase
    from .FigureCommon import getSetup, subplotLabel

    # plot division rate, rate of cells entering apoptosis, rate of cells straight to death
    ax, f, _ = getSetup((12, 4), (1, 3))

    for axis in ax[0:3]:
        axis.tick_params(axis='both', which='major', pad=-2)  # set ticks style

    # Show line plots of rates for each drug
    ratePlots(axes=[ax[0], ax[1], ax[2]])

    # Labels for each subplot
    for ii, item in enumerate([ax[0], ax[1], ax[2]]):
        subplotLabel(item, ascii_uppercase[ii])

    # Try and fix overlapping elements
    f.tight_layout(pad=0.1)

    return f


def ratePlots(axes, files=['072718_PC9_BYL_PIM', '081118_PC9_LCL_TXL', '071318_PC9_OSI_Bin', '090618_PC9_TXL_Erl']):
    """ Create line plots of model posterior. """
    from ..utils import violinplot

    df = None
    for i, ff in enumerate(files):

        # Load model and dataset
        dfdict, drugs, _ = violinplot(ff, singles=True)

        # Plot params vs. drug dose
        for j, drug in enumerate(drugs):
            # Get drug
            dfplot = dfdict[drug]

            if df is not None:
                drug_lab = 1
                while drug in np.array(df['drugName'][1]):
                    drug = drug + str(drug_lab)
                    drug_lab = drug_lab + 1

            df_temp = pd.DataFrame({'div': np.array(dfplot['div']),
                                    'apopRate': np.multiply(np.array(dfplot['deathRate']), np.array(dfplot['apopfrac'])),
                                    'deathRate': np.multiply(np.array(dfplot['deathRate']),
                                                             np.ones(len(dfplot['apopfrac'])) - np.array(dfplot['apopfrac'])),
                                    'drugName': np.repeat(drug, len(dfplot['div'])),
                                    'dose': np.log([float(ds) for ds in np.array(dfplot['dose'])])})

            # Sort the data set by the value of doses
            df_temp = df_temp.sort_values(by='dose')

            if df is None:
                df = df_temp
            else:
                df = df.append(df_temp)

    # Make line plots
    # Division rate
    sns.lineplot(x='dose', y='div', hue='drugName', marker='o', data=df, ax=axes[0], palette='muted')
    # Rate of cells entering apoptosis
    sns.lineplot(x='dose', y='apopRate', hue='drugName', marker='o', data=df, ax=axes[1], palette='muted')
    # Rate of cells straight to death
    sns.lineplot(x='dose', y='deathRate', hue='drugName', marker='o', data=df, ax=axes[2], palette='muted')

    # Set legend
    for i in range(3):
        axes[i].legend(handletextpad=0.3, handlelength=0.8, prop={'size': 8})
        # Set x, y labels and title
        axes[i].set_ylabel('Rate')
        axes[i].set_xlabel('Log[dose]')

    axes[0].set_title('Division rate')
    axes[1].set_title('Rate of cells entering apoptosis')
    axes[2].set_title('Rate of cells straight to death')
