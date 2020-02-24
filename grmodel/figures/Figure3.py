"""
This creates Figure 3.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from seaborn import lineplot
from .FigureCommon import getSetup, subplotLabel
from ..utils import violinplot


def makeFigure():
    """ Generate Figure 3: This figure should show different drugs
        have different effects by looking at the division rate and
        death rate of cancer cells under their treatments. """

    # plot division rate, rate of cells entering apoptosis, rate of cells straight to death
    ax, f = getSetup((7.5, 2.5), (1, 3))

    for axis in ax[0:3]:
        axis.tick_params(axis="both", which="major", pad=-2)  # set ticks style

    # Show line plots of rates for each drug
    ratePlots(ax)

    # Remove legend title
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles=handles[1:], labels=labels[1:])

    ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax[0].set_ylim(bottom=-0.0015, top=0.03)
    ax[1].set_ylim(bottom=0.0, top=0.025)
    ax[2].set_ylim(bottom=0.0, top=0.025)
    ax[2].set_xlim(left=0.0, right=0.03)

    # Labels for each subplot
    subplotLabel(ax)

    return f


def ratePlots(axes):
    """ Create line plots of model posterior. """
    files = ["072718_PC9_BYL_PIM", "050719_PC9_PIM_OSI", "050719_PC9_LCL_OSI", "071318_PC9_OSI_Bin", "090618_PC9_TXL_Erl", "020720_PC9_Erl_THZ1"]

    df = None
    for i, ff in enumerate(files):
        # Load model and dataset
        dfdict = violinplot(ff)

        # Plot params vs. drug dose
        for drug, dfplot in dfdict.items():
            # Change dose to the same unit uM (change nM to uM)
            dfplot["dose"] = dfplot["dose"].astype(float).apply(np.log10)
            if drug in ["Paclitaxel", "Erlotinib", "THZ1"]:
                dfplot["dose"] -= 3.0

            # Deal with duplicated drug for different drug combinations
            if df is not None:
                drug_lab = 1
                while drug in np.array(df["drugName"][1]):
                    drug = drug + str(drug_lab)
                    drug_lab = drug_lab + 1

            # Convert div and deathRate from log scale to linear;
            # Convert dose from linear to log scale
            df_temp = dfplot[["div", "deathRate", "dose"]]
            df_temp["drugName"] = drug

            # Sort the data set by the value of doses
            df_temp.sort_values(by="dose", inplace=True)

            if df is None:
                df = df_temp
            else:
                df = df.append(df_temp)

    # Make line plots
    # Remove the data for Pacitaxel and only keep the one for Pacitacel1
    for ddd in ("OSI-90612", "OSI-9061", "PIM447", "Erlotinib1"):
        df = df.loc[df.drugName != ddd]

    df.loc[df.drugName == "PIM4471", "drugName"] = "PIM447"

    # Division rate
    lineplot(x="dose", y="div", hue="drugName", marker="o", data=df, ax=axes[0], palette="muted")
    # Death rate
    lineplot(x="dose", y="deathRate", hue="drugName", marker="o", data=df, ax=axes[1], palette="muted")
    # Division vs. death
    df2 = df.groupby(["drugName", "dose"], sort=False).median().reset_index()
    lineplot(x="div", y="deathRate", hue="drugName", marker="o", data=df2, ax=axes[2], palette="muted")

    # Set legend
    for i in range(2):
        axes[i].legend(handletextpad=0.3, handlelength=0.8, prop={"size": 8})
        # Set x, y labels and title
        axes[i].set_ylabel(r"Rate (1/hr)")
        axes[i].set_xlabel(r"Log$_{10}$[$\mu$M]")
        axes[i].set_ylim(bottom=0.0)
        axes[i].set_xlim(right=2.0)
        axes[i].xaxis.set_major_locator(plt.MultipleLocator(1.0))

    axes[0].set_title("Division Rate")
    axes[1].set_title("Death Rate")
    axes[1].get_legend().remove()

    axes[2].set_xlabel(r"Division Rate (1/hr)")
    axes[2].set_ylabel(r"Death Rate (1/hr)")
    axes[2].plot(0.027, 0.001, 'ko')
    axes[2].set_ylim(bottom=-0.001)
    axes[2].set_xlim(left=-0.001)
    axes[2].get_legend().remove()
