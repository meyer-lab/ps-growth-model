""" This creates Figure S1. """

from string import ascii_uppercase
from .FigureCommon import getSetup, subplotLabel
from ..utils import violinplot_split


def makeFigure():
    """ Make figure S3. """

    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 3))

    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_uppercase[ii])

    # violinplot_split("101117_H1299", ax)

    return f
