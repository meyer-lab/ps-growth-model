"""
This creates Figure 1.
"""

from string import ascii_uppercase
from .FigureCommon import getSetup, subplotLabel


def makeFigure():
    ''' XXX '''

    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 3))

    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_uppercase[ii])

    # Try and fix overlapping elements
    f.tight_layout()

    return f
