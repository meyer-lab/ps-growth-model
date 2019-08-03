""" This creates Figure S3. """


def makeFigure():
    ''' Make Figure S3. This would be the violinplot of model posterior
        estimates of the data shown in Figure S1 '''
    from .Figure2 import violinPlots
    from string import ascii_lowercase
    from .FigureCommon import getSetup, subplotLabel

    # Get list of axis objects
    ax, f, _ = getSetup((12, 8), (4, 4))

    for axis in ax[0:16]:
        axis.tick_params(axis='both', which='major', pad=-2)  # set ticks style

    # files include the filenames of each drug combination
    files = ['072718_PC9_BYL_PIM', '081118_PC9_LCL_TXL', '071318_PC9_OSI_Bin', '090618_PC9_TXL_Erl']

    # Show violin plots (predicted vs experimental)
    violinPlots(axes=[ax[0], ax[1], ax[4], ax[5]], ff=files[0], sg=True)
    violinPlots(axes=[ax[2], ax[3], ax[6], ax[7]], ff=files[1], sg=True)
    violinPlots(axes=[ax[8], ax[9], ax[12], ax[13]], ff=files[2], sg=True)
    violinPlots(axes=[ax[10], ax[11], ax[14], ax[15]], ff=files[3], sg=True)

    # TODO: change labels for each subplot
    for ii, item in enumerate([ax[0], ax[2], ax[8], ax[10]]):
        subplotLabel(item, ascii_lowercase[ii])

    # Try and fix overlapping elements
    f.tight_layout(pad=0.1)

    return f
