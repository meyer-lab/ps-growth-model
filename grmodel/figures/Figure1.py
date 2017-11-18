"""
This creates Figure 1.
"""

from string import ascii_uppercase
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from .FigureCommon import getSetup, subplotLabel
import seaborn as sns
import scipy.optimize as opt
from grmodel.fitcurves import sigmoid, residuals
from ..pymcDoseResponse import loadCellTiter


def makeFigure():
    '''
    Generate Figure 1

    Broadly, this figure should motivate looking at cell death.
    This should be by showing that it's not captured in existing
    measurements.
    '''

    # Get list of axis objects
    ax, f, gs1 = getSetup((7, 6), (3, 3))

    # Make first cartoon
    cartoonFig(gs1[0], f)

    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_uppercase[ii])

    doseResponseTiter(ax[1])

    # Try and fix overlapping elements
    f.tight_layout()

    return f


def cartoonFig(gs1, fig):
    """
    A first cartoon to show that the cell numbers don't really give you all the info
    """

    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs1,
                                             wspace=0.1, hspace=0.1)

    ax1 = plt.Subplot(fig, inner[0])
    ax2 = plt.Subplot(fig, inner[1])
    ax3 = plt.Subplot(fig, inner[2])

    # Let's calculate everything within three logs of the IC50
    xx = np.linspace(-3, 3, num=30)

    # Let's calculate everything for an experiment in which we have normaled cell numbers
    response = 10.0 + (1.0 - 10.0) / (1 + np.power(10, -xx))

    ax1.plot(xx, response)
    ax1.set_xlabel('Test')
    ax1.set_ylabel('Cells End / Cells Beginning')

    # Assuming that the death rate is zero
    gr = np.log(response)

    ax2.plot(xx, gr)
    ax2.plot(xx, np.full(response.shape, 0.0))
    ax2.set_xlabel('Test')
    ax2.set_ylabel('Growth Rate')

    # Assuming that the growth rate stays constant

    ax3.plot(xx, np.full(response.shape, gr[0]))
    ax3.plot(xx, gr[0] - gr)
    ax3.set_xlabel('Test')
    ax3.set_ylabel('Growth Rate')


def doseResponseTiter(ax=None):
    """ Dose response of celltiter glo data. """

    if ax is None:
        ax = plt.axes()

    data = loadCellTiter()

    refDose = np.linspace(min(data.logDose),max(data.logDose), 300)

    g = sns.FacetGrid(data, hue="Drug")

    g = (g.map(ax.scatter, 'logDose', 'response', s=3).add_legend())

    for _, group in data.groupby(['Drug']):
        retVal = opt.least_squares(residuals, x0=[0, 0.1, 0.1, 0], args=(group.logDose, group.response))
        
        ax.plot(refDose, sigmoid(retVal['x'], refDose))

    ax.set_ylabel('Viability')
    ax.set_xlabel('Log Drug Concentration')
    ax.set_ylim(0, 1)

