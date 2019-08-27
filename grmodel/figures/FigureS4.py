"""
This creates Figure S4.
"""
import numpy as np
import matplotlib.pyplot as plt
from .Figure2 import simulationPlots
from .FigureCommon import getSetup, subplotLabel
from ..sampleAnalysis import readModel


def makeFigure():
    """ Make Figure S4. This should be the experimental data of
        each drug combinations """

    # Get list of axis objects
    ax, f = getSetup((12, 12), (6, 6))

    for axis in ax[0:36]:
        axis.tick_params(axis="both", which="major", pad=-2)  # set ticks style

    files = ["050719_PC9_LCL_OSI", "050719_PC9_PIM_OSI"]

    # Show simulation plots (predicted vs experimental)
    simulationPlots(axes=ax[0:6], ff=files[0])
    simulationPlots(axes=ax[18:24], ff=files[1])
    simulationPlots_comb(files[0], ax[6:18])
    simulationPlots_comb(files[1], ax[24:36])

    subplotLabel([ax[0], ax[18]])

    return f


def simulationPlots_comb(loadFile, axes):
    if loadFile == "050719_PC9_LCL_OSI":
        drug1 = "LCL161"
        drug2 = "OSI-906"
    elif loadFile == "050719_PC9_PIM_OSI":
        drug1 = "PIM447"
        drug2 = "OSI-906"
    else:
        raise ValueError("Unrecognized file.")

    # Read model
    M = readModel(loadFile, model="interactionModel", drug1=drug1, drug2=drug2, fit=False)

    drugAname, drugBname = M.drugs

    X1 = np.unique(M.X1)
    X2 = np.unique(M.X2)
    timeV = M.timeV

    # Reshape
    N_X1 = len(X1)  # the number of drug 1 doses
    N_X2 = len(X2)  # the number of drug 2 doses

    confl_obs = M.phase.reshape(N_X1, N_X2, len(timeV))  # observed phase confl
    apop_obs = M.green.reshape(N_X1, N_X2, len(timeV))  # observed green confl
    dna_obs = M.red.reshape(N_X1, N_X2, len(timeV))  # observed red confl

    palette = plt.get_cmap("tab10")  # color palette

    for i in range(1, len(X1)):
        this_obs = [confl_obs[i], apop_obs[i], dna_obs[i]]

        for j in range(len(X2)):
            n = i - 1
            for k in range(3):
                ii = (3 * n) + k
                axes[ii].plot(timeV, this_obs[k][j], color=palette(j), linewidth=1, alpha=0.9, label=str(X2[j]))

                if ii % 3 == 0:
                    axes[ii].set_ylim(0.0, 100.0)
                else:
                    axes[ii].set_ylim(-1.0, 10.0)

                axes[ii].set_xlabel("Time (h)")
                axes[ii].set_ylabel("Percent Image Positive")
                axes[ii].set_title(drugAname + " " + str(round(X1[i], 1)) + r"$\mu$M + " + drugBname)
