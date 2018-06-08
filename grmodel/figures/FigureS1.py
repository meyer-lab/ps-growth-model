"""
This creates Figure S1.
"""

from string import ascii_uppercase
from .FigureCommon import getSetup, subplotLabel


def makeFigure():
    ''' Make figure S1. Show that endpoint measurements can be
    as useful as the full kinetics. '''

    # Get list of axis objects
    ax, f, _ = getSetup((7, 6), (3, 3))

    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_uppercase[ii])

    # Try and fix overlapping elements
    f.tight_layout()

    return f
