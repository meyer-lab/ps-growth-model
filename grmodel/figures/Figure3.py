"""
This creates Figure 3.
"""

from string import ascii_uppercase
from .FigureCommon import getSetup, subplotLabel


def makeFigure():
    ''' Make figure 3. This probably should focus on validation of what's found
    in figure 2. '''

    # Get list of axis objects
    ax, f, _ = getSetup((7, 6), (3, 3))

    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_uppercase[ii])

    # Try and fix overlapping elements
    f.tight_layout()

    return f
