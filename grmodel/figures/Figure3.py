"""
This creates Figure 3.
"""
import numpy as np
import pymc3 as pm
from ..pymcInteraction import blissInteract, drugInteractionModel


def build(loadFile='BYLvPIM', drug1='PIM447', drug2='BYL749'):
    ''' Build and save the drugInteractionModel '''
    M = drugInteractionModel(loadFile, drug1, drug2)
    # Save the drug interaction model
    M.save()


def add_corrmedian(trace, ax):
    ''' Compute the median of correlation coefficient of pymc fitting '''
    median_confl_corr = np.median(np.array(trace['confl_corr']))
    median_apop_corr = np.median(np.array(trace['apop_corr']))
    median_dna_corr = np.median(np.array(trace['dna_corr']))

    # Add text box for displaying the corr
    textstr = '\n'.join((r'median_confl_corr=%.3f' % (median_confl_corr, ),
                         r'median_apop_corr=%.3f' % (median_apop_corr, ),
                         r'median_dna_corr=%.3f' % (median_dna_corr, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)


def plot_cellnumVStime(M, celltype, printname, loadFile, ax):
    ''' Plot the number of live and dead cells by time for different drug interactions '''
    for i in range(len(M.X1)):
        ax.plot(M.timeV, celltype[i])
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('The number of ' + printname + ' cells')
    ax.set_title('The number of ' + printname + ' cells by time (' + loadFile + ')')


def makeFigure(loadFiles=['BYLvPIM', 'OSIvBIN', 'LCLvTXL']):
    ''' Generate Figure 3: This figure should show looking at cell death can
    tell something about the cells' responses to drug interactions that are
    not captured by the traditional cell number measurements. '''
    from ..sampleAnalysis import read_dataset
    from ..pymcGrowth import theanoCore
    from .FigureCommon import getSetup, subplotLabel
    from string import ascii_uppercase

    ax, f, _ = getSetup((8, 9), (3, 2))

    for idx, loadFile in enumerate(loadFiles):
        # Read model from saved pickle file
        M = read_dataset(loadFile)

        # Trace is drawn from 1000 pymc samplings
        trace = pm.backends.tracetab.trace_to_dataframe(M.fit)

        # Traceplot
        pm.plots.traceplot(M.fit)

        def transform(name):
            ''' Transforms the data structure of parameters generated from pymc model'''
            return np.vstack((np.array(trace[name + '__0'])[0],
                              np.array(trace[name + '__1'])[0]))

        E_con = transform('E_con')
        hill_death = transform('hill_death')
        hill_growth = transform('hill_growth')
        IC50_death = transform('IC50_death')
        IC50_growth = transform('IC50_growth')

        death_rates = E_con[0] * blissInteract(M.X1, M.X2, hill_death,
                                               IC50_death, numpyy=True)
        growth_rates = E_con[1] * (1 - blissInteract(M.X1, M.X2, hill_growth,
                                                     IC50_growth, numpyy=True))

        # Compute the number of live cells, dead cells and early apoptosis cells
        # given growth and death rate
        lnum, eap, deadapop, deadnec = theanoCore(M.timeV, growth_rates, death_rates,
                                                  np.array(trace['apopfrac'])[0],
                                                  np.array(trace['d'])[0], numpyy=True)
        dead = deadapop + deadnec

        # Plot the graphs of the number of live and dead cells by time for drug interactions
        plot_cellnumVStime(M, lnum, 'live', loadFile, ax[2 * idx])
        add_corrmedian(trace, ax[2 * idx])
        plot_cellnumVStime(M, dead, 'dead', loadFile, ax[2 * idx + 1])

    # Make third figure
    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_uppercase[ii])

    # Try and fix overlapping elements
    f.tight_layout()

    return f
