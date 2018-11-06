"""
This creates Figure 3.
"""
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from ..pymcInteraction import blissInteract, drugInteractionModel


def build(loadFile='BYLvPIM', drug1='PIM447', drug2='BYL749', timepoint_start=0):
    ''' Build and save the drugInteractionModel '''
    M = drugInteractionModel(loadFile, drug1, drug2, timepoint_start)
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


def plot_cellnumVStime(X1, celltype, printname, loadFile, timeV, ax):
    ''' Plot the number of live and dead cells by time for different drug interactions '''
    for i in range(len(X1)):
        ax.plot(timeV, celltype[i], label=str(X1[i]))

    ax.legend(title='Drug1 doses', loc='upper right', framealpha=0.3)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('The number of ' + printname + ' cells')
    ax.set_title('The number of ' + printname + ' cells by time (' + loadFile + ')')


def plot_cellnumVSdoses(X1, celltype, printname, loadFile, timeV, ax):
    ''' Plot the number of live and dead cells by doses at 72 hours '''
    ax.plot(X1, celltype, 'r*',)
    ax.plot(X1, celltype)
    ax.set_xlabel('Drug1 doses')
    ax.set_ylabel('The number of ' + printname + ' cells')
    ax.set_title('The number of ' + printname + ' cells by doses (' + loadFile + ') , t = ' + str(timeV[:]))


def makeFigure(loadFiles=['BYLvPIM', 'OSIvBIN', 'LCLvTXL'], timepoint_start=0):
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
        trace = pm.backends.tracetab.trace_to_dataframe(M.fit)

        if timepoint_start == 0:
            timeV = M.timeV
            # Trace is drawn from pymc samplings, this is only used to compute corr
            trace_corr = trace
        elif timepoint_start == 72:
            # We are not interested in corr at only one time point
            timeV = np.array([72.])
        else:
            # This is only used to compute corr
            M2 = read_dataset(loadFile, timepoint_start=timepoint_start)
            timeV = M2.timeV
            # Trace is drawn from pymc samplings, this is only used to compute corr
            trace_corr = pm.backends.tracetab.trace_to_dataframe(M2.fit)

        # Traceplot
        # pm.plots.traceplot(M.fit)

        def transform(name):
            ''' Transforms the data structure of parameters generated from pymc model'''
            return np.vstack((np.array(trace[name + '__0'])[0],
                              np.array(trace[name + '__1'])[0]))

        E_con = transform('E_con')
        hill_death = transform('hill_death')
        hill_growth = transform('hill_growth')
        IC50_death = transform('IC50_death')
        IC50_growth = transform('IC50_growth')

        death_rates = E_con[0] * blissInteract(M.X1, M.X2, hill_death, IC50_death, numpyy=True)
        growth_rates = E_con[1] * (1 - blissInteract(M.X1, M.X2, hill_growth, IC50_growth, numpyy=True))

        # Compute the number of live cells, dead cells and early apoptosis cells
        # given growth and death rate
        lnum, eap, deadapop, deadnec = theanoCore(timeV, growth_rates, death_rates,
                                                  np.array(trace['apopfrac'])[0],
                                                  np.array(trace['d'])[0], numpyy=True)
        dead = deadapop + deadnec

        # Get one fitting data for each X1
        size = len(np.unique(M.X1))
        duplicate = int(len(M.X1) / size)
        subset = [x * duplicate - 1 for x in range(1, size + 1)]

        X1 = np.array([M.X1[i] for i in subset])
        lnum = np.array([lnum[i] for i in subset])
        dead = np.array([dead[i] for i in subset])

        # Plot the graphs of the number of live and dead cells by time for drug interactions
        if timepoint_start == 72:
            plot_cellnumVSdoses(X1, lnum, 'live', loadFile, timeV, ax[2 * idx])
            ax[2 * idx].set_ylim(min(min(dead.flatten()), min(lnum.flatten())) - 0.5,
                                 max(max(dead.flatten()), max(lnum.flatten())) + 0.5)
            plot_cellnumVSdoses(X1, dead, 'dead', loadFile, timeV, ax[2 * idx + 1])
            ax[2 * idx + 1].set_ylim(min(min(dead.flatten()), min(lnum.flatten())) - 0.5,
                                     max(max(dead.flatten()), max(lnum.flatten())) + 0.5)
        else:
            plot_cellnumVStime(X1, lnum, 'live', loadFile, timeV, ax[2 * idx])
            ax[2 * idx].set_ylim(min(min(dead.flatten()), min(lnum.flatten())) - 0.5,
                                 max(max(dead.flatten()), max(lnum.flatten())) + 0.5)
            add_corrmedian(trace_corr, ax[2 * idx])
            plot_cellnumVStime(X1, dead, 'dead', loadFile, timeV, ax[2 * idx + 1])
            ax[2 * idx + 1].set_ylim(min(min(dead.flatten()), min(lnum.flatten())) - 0.5,
                                     max(max(dead.flatten()), max(lnum.flatten())) + 0.5)

    # Make third figure
    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_uppercase[ii])

    # Try and fix overlapping elements
    f.tight_layout()
    plt.show()

    return f
