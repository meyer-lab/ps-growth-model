"""
This creates Figure 4.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from ..sampleAnalysis import readModel
from ..pymcInteraction import blissInteract


def makeFigure(loadFiles=['050719_PC9_LCL_OSI', '050719_PC9_PIM_OSI']):
    ''' Generate Figure 4: This figure should show looking at cell death can
    tell something about the cells' responses to drug interactions that are
    not captured by the traditional cell number measurements. '''
    from .FigureCommon import getSetup, subplotLabel
    from string import ascii_lowercase

    # plot phase, green and red confl for three drug interactions
    ax, f = getSetup((10, 10), (2, 2))

    for idx, loadFile in enumerate(loadFiles):

        # Read model from saved pickle file
        M, trace = readModel(loadFile, model='interactionModel')

        # Randomly sample 10 rows from pymc sampling results
        if trace.shape[0] > 100:
            trace = trace.sample(100)

        def transform(name):
            ''' Transforms the data structure of parameters generated from pymc model'''
            return np.vstack((np.array(trace[name + '__0']),
                              np.array(trace[name + '__1'])))

        E_con = transform('E_con')
        hill_death = transform('hill_death')
        hill_growth = transform('hill_growth')
        IC50_death = transform('IC50_death')
        IC50_growth = transform('IC50_growth')

        X1 = np.unique(M.X1)
        X2 = np.unique(M.X2)

        print('filename:', loadFiles)

        N_obs = 100

        # Compute death and growth rate
        death_rates = np.empty([N_obs, len(X1) * len(X2)])
        growth_rates = np.empty([N_obs, len(X1) * len(X2)])

        for i in range(N_obs):
            this_hill_death = np.vstack([[x[i]] for x in hill_death])
            this_IC50_death = np.vstack([[x[i]] for x in IC50_death])
            this_death_rate = [E_con[0][i]] * blissInteract(M.X1, M.X2, this_hill_death, this_IC50_death, numpyy=True)
            death_rates[i] = this_death_rate

            this_hill_growth = np.vstack([[x[i]] for x in hill_growth])
            this_IC50_growth = np.vstack([[x[i]] for x in IC50_growth])
            this_growth_rate = [E_con[1][i]] * (1 - blissInteract(M.X1, M.X2, this_hill_growth, this_IC50_growth, numpyy=True))
            growth_rates[i] = this_growth_rate

        # Initialize a dataframe
        params = ['div', 'deathRate', 'X1', 'X2']
        dfplot = pd.DataFrame(columns=params)

        for i, dose2 in enumerate(X2):
            dftemp = pd.DataFrame(columns=params)
            for j, dose1 in enumerate(X1):
                dftemp2 = pd.DataFrame(columns=params)
                k = j * len(X2) + i
                try:
                    dftemp2['div'] = [x[k] for x in growth_rates]
                    dftemp2['deathRate'] = [x[k] for x in death_rates]
                    dftemp2['X1'] = dose1
                    dftemp2['X2'] = dose2
                except BaseException:
                    print('this idx is out of bound:', k)
                dftemp = pd.concat([dftemp, dftemp2], axis=0)
            dfplot = pd.concat([dfplot, dftemp], axis=0)

        dfplot['X1'] = round(dfplot['X1'], 1)
        dfplot['X2'] = round(dfplot['X2'], 1)

        print(dfplot)

        # Make violin plots
        sns.violinplot(x='X2', y='div', hue='X1', data=dfplot, ax=ax[2 * idx],
                       palette='Set2', linewidth=0.2)
        sns.violinplot(x='X2', y='deathRate', hue='X1', data=dfplot, ax=ax[2 * idx + 1],
                       palette='Set2', linewidth=0.2)

        for axes in [ax[2 * idx], ax[2 * idx + 1]]:
            # Set legend
            axes.legend(handletextpad=0.3, title=M.drugs[0] + r'($\mu$M)', handlelength=0.8, prop={'size': 8})
            # Set x label
            axes.set_xlabel(M.drugs[1] + r'($\mu$M)')
            # Set ylim
            axes.set_ylim(bottom=0)

        # Set y label
        ax[2 * idx].set_ylabel(r'Division rate (1/h)')
        ax[2 * idx + 1].set_ylabel(r'Death rate (1/h)')

    # Make third figure
    for ii, item in enumerate([ax[0], ax[2]]):
        subplotLabel(item, ascii_lowercase[ii])

    return f
