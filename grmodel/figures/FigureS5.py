"""
This creates Figure S5.
"""

from string import ascii_uppercase
from .FigureCommon import getSetup, subplotLabel


def makeFigure():
    ''' Make figure S5. '''

    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 3))

    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_uppercase[ii])

    return f
