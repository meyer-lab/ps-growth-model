"""
This creates Figure S5.
"""
import numpy as np
import matplotlib.cm as cm


def makeFigure():
    """ Make figure S5. """
    from .FigureCommon import getSetup

    # Get list of axis objects
    ax, f = getSetup((8, 4), (2, 3))

    for axis in ax[0:36]:
        axis.tick_params(axis="both", which="major", pad=-2)  # set ticks style

    files = ["050719_PC9_LCL_OSI", "050719_PC9_PIM_OSI"]

    plot_endpoints_without_fitted(files[0], ax[0:3])
    plot_endpoints_without_fitted(files[1], ax[3:6])

    return f


def plot_endpoints_without_fitted(loadFile, axes):
    """ Plot the fitted vs. observed confl, apop and dna for different drug
        interactions at t=72h """
    from ..sampleAnalysis import read_dataset

    # Read model from saved pickle file
    M = read_dataset(loadFile, model="interactionModel")

    drugAname, drugBname = M.drugs
    print("drugname: " + drugAname + ", " + drugBname)

    X1 = np.unique(M.X1)
    X2 = np.unique(M.X2)

    # Reshape
    N_X1 = len(X1)  # the number of drug 1 doses
    N_X2 = len(X2)  # the number of drug 2 doses

    confl_obs = np.array([i[-1] for i in M.phase])
    apop_obs = np.array([i[-1] for i in M.green])
    dna_obs = np.array([i[-1] for i in M.red])

    confl_obs = confl_obs.reshape(N_X1, N_X2)  # observed phase confl
    apop_obs = apop_obs.reshape(N_X1, N_X2)  # observed green confl
    dna_obs = dna_obs.reshape(N_X1, N_X2)  # observed red confl

    obs = [confl_obs, apop_obs, dna_obs]

    # help to name title
    quant_tt = ["Phase", "Annexin V", "YOYO-3"]

    col = cm.rainbow(np.linspace(0, 1, len(X1)))

    for i in range(len(quant_tt)):
        for j in range(len(X1)):
            # plot for observed data
            axes[i].scatter(X2, obs[i][j], color=col[j], label=str(round(X2[j], 1)))
            axes[i].plot(X2, obs[i][j], color=col[j])
            axes[i].set_xlabel(drugBname + r" ($\mu$M)")
            axes[i].set_ylabel("Percent Image Positive")
            axes[i].set_xticks([round(x, 1) for x in X2])
            axes[i].set_title("72h " + quant_tt[i])

            if i == 0:
                axes[i].legend(title=drugAname + "\n" + r"   ($\mu$M)", loc="center left", bbox_to_anchor=(1, 0.5), fancybox=True)
            else:
                axes[i].set_ylim([-0.2, 4.5])
