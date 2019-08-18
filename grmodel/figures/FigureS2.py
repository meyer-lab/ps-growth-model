"""
This creates Figure S2.
"""


def makeFigure():
    """ Make Figure S2. This should be the experimental data of
        single drug in each drug combinations """
    from .Figure2 import simulationPlots
    from .FigureCommon import getSetup, subplotLabel

    # Get list of axis objects
    ax, f = getSetup((12, 12), (6, 6))

    for axis in ax[0:24]:
        axis.tick_params(axis="both", which="major", pad=-2)  # set ticks style

    files = ["072718_PC9_BYL_PIM", "081118_PC9_LCL_TXL", "071318_PC9_OSI_Bin", "090618_PC9_TXL_Erl", "050719_PC9_PIM_OSI", "050719_PC9_LCL_OSI"]

    # Show simulation plots (predicted vs experimental)
    simulationPlots(axes=[ax[0], ax[1], ax[2], ax[6], ax[7], ax[8]], ff=files[0])
    simulationPlots(axes=[ax[3], ax[4], ax[5], ax[9], ax[10], ax[11]], ff=files[1])
    simulationPlots(axes=[ax[12], ax[13], ax[14], ax[18], ax[19], ax[20]], ff=files[2])
    simulationPlots(axes=[ax[15], ax[16], ax[17], ax[21], ax[22], ax[23]], ff=files[3])
    simulationPlots(axes=[ax[24], ax[25], ax[26], ax[30], ax[31], ax[32]], ff=files[4])
    simulationPlots(axes=[ax[27], ax[28], ax[29], ax[33], ax[34], ax[35]], ff=files[5])

    subplotLabel([ax[0], ax[3], ax[12], ax[15], ax[24], ax[27]])

    return f
