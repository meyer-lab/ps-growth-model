""" This creates Figure S3. """
from .Figure2 import violinPlots
from .FigureCommon import getSetup, subplotLabel


def makeFigure():
    """ Make Figure S3. This would be the violinplot of model posterior
        estimates of the data shown in Figure S2 """

    # Get list of axis objects
    ax, f = getSetup((10, 8), (4, 4))

    for axis in ax[0:16]:
        axis.tick_params(axis="both", which="major", pad=-2)  # set ticks style

    # Show violin plots (predicted vs experimental)
    violinPlots(axes=ax[0:2], ff="072718_PC9_BYL_PIM", remm="PIM447")
    violinPlots(axes=ax[2:4], ff="050719_PC9_PIM_OSI", remm="OSI-906")
    violinPlots(axes=ax[4:6], ff="050719_PC9_LCL_OSI", remm="OSI-906")
    violinPlots(axes=ax[6:10], ff="071318_PC9_OSI_Bin")
    violinPlots(axes=ax[10:14], ff="090618_PC9_TXL_Erl")
    violinPlots(axes=ax[14:16], ff="020720_PC9_Erl_THZ1", remm="Erlotinib")

    subplotLabel(ax[::2])

    return f
