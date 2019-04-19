"""
This creates Figure 4.
"""
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ..pymcInteraction import blissInteract, drugInteractionModel
from ..pymcGrowth import convSignal


def makeFigure(loadFiles=['072718_PC9_BYL_PIM'], drugAname='PIM447', drugBname='BYL749', timepoint_start=72):
    ''' Generate Figure 4: This figure should show looking at cell death can
    tell something about the cells' responses to drug interactions that are
    not captured by the traditional cell number measurements. '''
    from ..sampleAnalysis import read_dataset
    from ..pymcGrowth import theanoCore
    from .FigureCommon import getSetup, subplotLabel
    from string import ascii_uppercase

    if timepoint_start == 72:
        # plot phase, green and red confl for three drug interactions
        ax, f, _ = getSetup((10, 8), (3, 3))
    else:
        # plot lnum and dead for three drug interactions
        ax, f, _ = getSetup((8, 9), (3, 2))

    for idx, loadFile in enumerate(loadFiles):

        # Read model from saved pickle file
        M = read_dataset(loadFile)
        trace = pm.backends.tracetab.trace_to_dataframe(M.fit)
        trace_corr = trace

        if timepoint_start == 0:
            timeV = M.timeV
            # Trace is drawn from pymc samplings, this is only used to compute corr
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
            return np.vstack((np.array(trace[name + '__0'])[-1],
                              np.array(trace[name + '__1'])[-1]))

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
                                                  np.array(trace['apopfrac'])[-1],
                                                  np.array(trace['d'])[-1], numpyy=True)
        dead = deadapop + deadnec

        # Compute the conversions, the expected phase & green & red confluence
        conversions = (np.array(trace['confl_conv'])[-1], np.array(trace['apop_conv'])[-1],
                       np.array(trace['dna_conv'])[-1])

        confl_exp, apop_exp, dna_exp = convSignal(lnum, eap, deadapop, deadnec,
                                                  conversions, offset=False)

        X1 = np.unique(M.X1)
        X2 = np.unique(M.X2)

        # Plot the graphs of the observed and expected value of phase, green and red confluence
        if timepoint_start == 72:
            # Get observed values of phase, green and red confl at the last time point
            confl_obs = np.array([i[-1] for i in M.phase])
            apop_obs = np.array([i[-1] for i in M.green])
            dna_obs = np.array([i[-1] for i in M.red])

            # Reshape
            N_X1 = len(X1)  # the number of drug 1 doses
            N_X2 = len(X2)  # the number of drug 2 doses

            lnum = lnum.reshape(N_X1, N_X2)  # number of live cells
            dead = dead.reshape(N_X1, N_X2)  # number of dead cells
            confl_obs = confl_obs.reshape(N_X1, N_X2)  # observed phase confl
            apop_obs = apop_obs.reshape(N_X1, N_X2)  # observed green confl
            dna_obs = dna_obs.reshape(N_X1, N_X2)  # observed red confl

            confl_exp = confl_exp.reshape(N_X1, N_X2)  # expected phase confl
            apop_exp = apop_exp.reshape(N_X1, N_X2)  # expected green confl
            dna_exp = dna_exp.reshape(N_X1, N_X2)  # expected red confl

            # Plot the observed and expected value
            plot_endpoints(X1, X2, confl_obs, confl_exp, 'Phase', loadFile, drugAname, drugBname,
                           [ax[3 + 3 * idx], ax[3 + 3 * (idx + 1)]], legend=True)
            plot_endpoints(X1, X2, apop_obs, apop_exp, 'Annexin V', loadFile, drugAname, drugBname,
                           [ax[3 + 3 * idx + 1], ax[3 + 3 * (idx + 1) + 1]], legend=False)
            plot_endpoints(X1, X2, dna_obs, dna_exp, 'YOYO-3', loadFile, drugAname, drugBname,
                           [ax[3 + 3 * idx + 2], ax[3 + 3 * (idx + 1) + 2]], legend=False)

            add_corrmedian(trace_corr, 'confl_corr', ax[3 + 3 * (idx + 1)])
            add_corrmedian(trace_corr, 'apop_corr', ax[3 + 3 * (idx + 1) + 1])
            add_corrmedian(trace_corr, 'dna_corr', ax[3 + 3 * (idx + 1) + 2])

        else:

            plot_cellnumVStime(X1, lnum, 'live', loadFile, timeV, ax[2 * idx])
            ax[2 * idx].set_ylim(min(min(dead.flatten()), min(lnum.flatten())) - 0.5,
                                 max(max(dead.flatten()), max(lnum.flatten())) + 0.5)
            # add_corrmedian(trace_corr, ax[2 * idx])
            plot_cellnumVStime(X1, dead, 'dead', loadFile, timeV, ax[2 * idx + 1])
            ax[2 * idx + 1].set_ylim(min(min(dead.flatten()), min(lnum.flatten())) - 0.5,
                                     max(max(dead.flatten()), max(lnum.flatten())) + 0.5)

    # Make third figure
    for ii, item in enumerate([ax[0], ax[1], ax[2], ax[3], ax[6]]):
        subplotLabel(item, ascii_uppercase[ii])

    for axis in ax[0:3]:
        axis.axis('off')

    # Try and fix overlapping elements
    f.tight_layout()
    plt.show()

    return f


def build(loadFile='072718_PC9_BYL_PIM', drug1='PIM447', drug2='BYL749', timepoint_start=0):
    ''' Build and save the drugInteractionModel '''
    M = drugInteractionModel(loadFile, drug1, drug2, timepoint_start)
    # Save the drug interaction model
    M.save()


def add_corrmedian(trace, item, ax):
    ''' Compute the median of correlation coefficient of pymc fitting '''
    median_corr = np.median(np.array(trace[item]))

    # Add text box for displaying the corr
    textstr = 'median_' + item + '=%.3f' % (median_corr, )

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)


def plot_cellnumVStime(X1, celltype, printname, loadFile, timeV, ax):
    ''' Plot the number of live and dead cells by time for different drug interactions '''
    for i in range(len(X1)):
        ax.plot(timeV, celltype[i], label=str(X1[i]))

    ax.legend(title='Drug1(uM)', loc='upper right', framealpha=0.3)
    ax.set_xlabel('Time Ellapsed Post-Treatment (hrs)')
    ax.set_ylabel('The number of ' + printname + ' cells')
    ax.set_title('The number of ' + printname + ' cells by time (' + loadFile + ')')


def plot_endpoints(X1, X2, obs, exp, itemname, loadFile, drugAname, drugBname, axes, legend):
    ''' Plot the fitted vs. observed confl, apop and dna for different drug
        interactions at t=72h '''
    col = cm.rainbow(np.linspace(0, 1, len(X2)))

    for i in range(len(X2)):
        # plot for observed data
        axes[0].scatter(X1, [x[i] for x in obs], color=col[i], label=str(round(X2[i], 1)))
        axes[0].plot(X1, [x[i] for x in obs], color=col[i])
        # axes[0].scatter(X1, [x[i] for x in exp], alpha=0.1, marker="^", color=col[i])
        # plot for fitted data
        # axes[1].scatter(X1, [x[i] for x in obs], alpha=0.1, color=col[i])
        axes[1].scatter(X1, [x[i] for x in exp], marker="^", color=col[i], label=str(round(X2[i], 1)))
        axes[1].plot(X1, [x[i] for x in exp], color=col[i])

    for i in range(2):
        axes[i].set_xlabel(drugAname + r' ($\mu$M)')
        axes[i].set_ylabel("Percent Image Positive")
        axes[i].set_ylim(bottom=-0.5)
        axes[i].set_xticks([round(x, 1) for x in X1])

        if i == 0:
            axes[i].set_title('72h ' + itemname)
        else:
            axes[i].set_title('72h ' + itemname + ' (fitted)')

        if legend:
            axes[i].legend(title=drugBname + '\n' + r'   ($\mu$M)', loc='center left',
                           bbox_to_anchor=(1, 0.5), fancybox=True)
