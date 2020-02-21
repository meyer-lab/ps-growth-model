"""
This creates Figure S2.
"""
from .Figure2 import simulationPlots
from .FigureCommon import getSetup, subplotLabel


def makeFigure():
    """ Make Figure S2. This should be the experimental data of
        single drug in each drug combinations """

    # Get list of axis objects
    ax, f = getSetup((12, 7), (4, 6))

    for axis in ax:
        axis.tick_params(axis="both", which="major", pad=-2)  # set ticks style

    # Show simulation plots (predicted vs experimental)
    simulationPlots(axes=ax[0:3], ff="072718_PC9_BYL_PIM")
    simulationPlots(axes=ax[3:9], ff="050719_PC9_PIM_OSI")
    simulationPlots(axes=ax[9:12], ff="050719_PC9_LCL_OSI", swapDrugs=True)
    simulationPlots(axes=ax[12:15], ff="071318_PC9_OSI_Bin")
    simulationPlots(axes=ax[15:21], ff="090618_PC9_TXL_Erl")
    simulationPlots(axes=ax[21:24], ff="020720_PC9_Erl_THZ1")

    subplotLabel(ax)

    return f
