"""
This creates Figure 4.
"""

from string import ascii_uppercase
from .FigureCommon import getSetup, subplotLabel


def makeFigure():
    ''' Make figure 4. This should generally be focused on analyzing drug interactions.
    Might start this figure by showing live cell measurements can make additive effects
    look synergistic. '''

    # Get list of axis objects
    ax, f, _ = getSetup((7, 6), (3, 3))

    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_uppercase[ii])

    # Try and fix overlapping elements
    f.tight_layout()

    return f
