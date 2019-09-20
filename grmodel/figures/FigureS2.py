"""
This creates Figure S2.
"""
from .Figure2 import simulationPlots
from .FigureCommon import getSetup, subplotLabel


def makeFigure():
    """ Make Figure S2. This should be the experimental data of
        single drug in each drug combinations """

    # Get list of axis objects
    ax, f = getSetup((12, 9), (5, 6))

    for axis in ax[0:30]:
        axis.tick_params(axis="both", which="major", pad=-2)  # set ticks style

    # Show simulation plots (predicted vs experimental)
    simulationPlots(axes=ax[0:6], ff="072718_PC9_BYL_PIM")
    simulationPlots(axes=ax[6:12], ff="050719_PC9_PIM_OSI")
    simulationPlots(axes=ax[12:18], ff="050719_PC9_LCL_OSI")
    simulationPlots(axes=ax[18:24], ff="071318_PC9_OSI_Bin")
    simulationPlots(axes=ax[24:30], ff="090618_PC9_TXL_Erl")

    subplotLabel(ax[::6])

    return f
