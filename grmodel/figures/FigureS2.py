"""
This creates Figure S2.
"""


def makeFigure():
    """ Make Figure S2. This should be the experimental data of
        single drug in each drug combinations """
    from .Figure2 import simulationPlots
    from string import ascii_lowercase
    from .FigureCommon import getSetup, subplotLabel

    # Get list of axis objects
    ax, f = getSetup((12, 8), (4, 6))

    for axis in ax[0:24]:
        axis.tick_params(axis="both", which="major", pad=-2)  # set ticks style

    files = ["072718_PC9_BYL_PIM", "081118_PC9_LCL_TXL", "071318_PC9_OSI_Bin", "090618_PC9_TXL_Erl"]

    # Show simulation plots (predicted vs experimental)
    simulationPlots(axes=[ax[0], ax[1], ax[2], ax[6], ax[7], ax[8]], ff=files[0])
    simulationPlots(axes=[ax[3], ax[4], ax[5], ax[9], ax[10], ax[11]], ff=files[1])
    simulationPlots(axes=[ax[12], ax[13], ax[14], ax[18], ax[19], ax[20]], ff=files[2])
    simulationPlots(axes=[ax[15], ax[16], ax[17], ax[21], ax[22], ax[23]], ff=files[3])

    for ii, item in enumerate([ax[0], ax[3], ax[12], ax[15]]):
        subplotLabel(item, ascii_lowercase[ii])

    return f
