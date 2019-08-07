"""
This creates Figure 3.
"""
from string import ascii_lowercase
import numpy as np
import pandas as pd
import seaborn as sns
from .FigureCommon import getSetup, subplotLabel
from ..utils import violinplot


def makeFigure():
    """ Generate Figure 3: This figure should show different drugs
        have different effects by looking at the division rate and
        death rate of cancer cells under their treatments. """

    # plot division rate, rate of cells entering apoptosis, rate of cells straight to death
    ax, f = getSetup((6, 2.5), (1, 2))

    for axis in ax[0:3]:
        axis.tick_params(axis="both", which="major", pad=-2)  # set ticks style

    # Show line plots of rates for each drug
    ratePlots(axes=[ax[0], ax[1]])

    # Labels for each subplot
    for ii, item in enumerate([ax[0], ax[1]]):
        subplotLabel(item, ascii_lowercase[ii])

    return f


def ratePlots(axes, files=["072718_PC9_BYL_PIM", "081118_PC9_LCL_TXL", "071318_PC9_OSI_Bin", "090618_PC9_TXL_Erl"]):
    """ Create line plots of model posterior. """
    df = None
    for i, ff in enumerate(files):

        # Load model and dataset
        dfdict, drugs, _ = violinplot(ff)

        # Plot params vs. drug dose
        for j, drug in enumerate(drugs):

            # Get drug
            dfplot = dfdict[drug]

            # Change dose to the same unit muM (change nanoM to microM)
            dfplot["dose"] = [float(ds) for ds in np.array(dfplot["dose"])]
            if drug in ["Paclitaxel", "Erl"]:
                dfplot["dose"] /= 1000.0

            # Deal with duplicated drug for different drug combinations
            if df is not None:
                drug_lab = 1
                while drug in np.array(df["drugName"][1]):
                    drug = drug + str(drug_lab)
                    drug_lab = drug_lab + 1

            # Convert div and deathRate from log scale to linear;
            # Convert dose from linear to log scale
            df_temp = pd.DataFrame(
                {
                    "div": 10 ** np.array(dfplot["div"]),
                    "deathRate": 10 ** np.array(dfplot["deathRate"]),
                    "drugName": np.repeat(drug, len(dfplot["div"])),
                    "dose": dfplot["dose"].apply(np.log10),
                }
            )

            # Sort the data set by the value of doses
            df_temp = df_temp.sort_values(by="dose")

            if df is None:
                df = df_temp
            else:
                df = df.append(df_temp)

    # Make line plots
    # Remove the data for Pacitaxel and only keep the one for Pacitacel1
    df = df.loc[df["drugName"] != "Paclitaxel"]
    df.loc[df["drugName"] == "Paclitaxel1", "drugName"] = "Paclitaxel"

    # Division rate
    sns.lineplot(x="dose", y="div", hue="drugName", marker="o", data=df, ax=axes[0], palette="muted")
    # Death rate
    sns.lineplot(x="dose", y="deathRate", hue="drugName", marker="o", data=df, ax=axes[1], palette="muted")

    # Set legend
    for i in range(2):
        axes[i].legend(handletextpad=0.3, handlelength=0.8, prop={"size": 8})
        # Set x, y labels and title
        axes[i].set_ylabel(r"Rate (1/h)")
        axes[i].set_xlabel(r"$\mathregular{Log_{10}}$[dose($\mu$M))]")
        axes[i].set_ylim(bottom=-0.005)

    axes[0].set_title("Division rate")
    axes[1].set_title("Death rate")
