"""
This creates Figure 4.
"""
import pandas as pd
import seaborn as sns
from ..sampleAnalysis import readModel
from ..pymcInteraction import blissInteract
from .FigureCommon import getSetup, subplotLabel


def makeFigure(loadFiles=["050719_PC9_LCL_OSI", "050719_PC9_PIM_OSI", "071318_PC9_OSI_Bin"]):
    """ Generate Figure 4: This figure should show looking at cell death can
    tell something about the cells' responses to drug interactions that are
    not captured by the traditional cell number measurements. """

    # plot phase, green and red confl for three drug interactions
    ax, f = getSetup((10, 7.5), (3, 2))

    for idx, loadFile in enumerate(loadFiles):
        if loadFile == "050719_PC9_LCL_OSI":
            drug1 = "LCL161"
            drug2 = "OSI-906"
        elif loadFile == "050719_PC9_PIM_OSI":
            drug1 = "PIM447"
            drug2 = "OSI-906"
        elif loadFile == "071318_PC9_OSI_Bin":
            drug1 = "OSI-906"
            drug2 = "Binimetinib"
        else:
            raise ValueError("Unrecognized file.")

        # Read model from saved pickle file
        M = readModel(loadFile, model="interactionModel", drug1=drug1, drug2=drug2)

        E_con = M.samples["E_con"]
        hill_death = M.samples["hill_death"]
        hill_growth = M.samples["hill_growth"]
        IC50_death = M.samples["IC50_death"]
        IC50_growth = M.samples["IC50_growth"]

        N_obs = 100

        # Initialize a dataframe
        params = ["div", "deathRate", "X1", "X2"]
        dfplot = pd.DataFrame(columns=params)

        for i in range(N_obs):
            dftemp2 = pd.DataFrame(columns=params)

            dftemp2["deathRate"] = E_con[i, 0] * blissInteract(M.X1, M.X2, hill_death[i, :], IC50_death[i, :], numpyy=True)
            dftemp2["div"] = E_con[i, 1] * (1 - blissInteract(M.X1, M.X2, hill_growth[i, :], IC50_growth[i, :], numpyy=True))
            dftemp2["X1"] = M.X1
            dftemp2["X2"] = M.X2

            dfplot = pd.concat([dfplot, dftemp2], axis=0)

        dfplot["X1"] = round(dfplot["X1"], 1)
        dfplot["X2"] = round(dfplot["X2"], 1)

        # Make violin plots
        sns.violinplot(x="X2", y="div", hue="X1", data=dfplot, ax=ax[2 * idx], palette="Set2", linewidth=0.2)
        sns.violinplot(x="X2", y="deathRate", hue="X1", data=dfplot, ax=ax[2 * idx + 1], palette="Set2", linewidth=0.2)

        for axes in [ax[2 * idx], ax[2 * idx + 1]]:
            # Set legend
            axes.legend(handletextpad=0.3, title=M.drugs[0] + r"($\mu$M)", handlelength=0.8, prop={"size": 8})
            # Set x label
            axes.set_xlabel(M.drugs[1] + r"($\mu$M)")
            # Set ylim
            axes.set_ylim(bottom=0)

        # Set y label
        ax[2 * idx].set_ylabel(r"Division rate (1/h)")
        ax[2 * idx + 1].set_ylabel(r"Death rate (1/h)")

    subplotLabel([ax[0], ax[2], ax[4]])

    return f
