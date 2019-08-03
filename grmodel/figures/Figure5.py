"""
This creates Figure 5.
"""

from string import ascii_lowercase
from .FigureCommon import getSetup, subplotLabel


def makeFigure():
    ''' Make figure 5. Bring home all the analysis by showing the value it adds
    in being able to predict the outcome of in vivo response. '''

    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 3))

    for ii, item in enumerate(ax):
        subplotLabel(item, ascii_lowercase[ii])

    return f
