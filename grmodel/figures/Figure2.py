"""
.. module:: Figure2

.. moduleauthor:: Guan Ning; Rui Yan <rachelyan@ucla.edu>; Aaron Meyer <ameyer@ucla.edu>

This module generates Figure2 which should generally be initial analysis
    of the data we've been collecting.
"""

from collections import OrderedDict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ..pymcGrowth import GrowthModel
from ..utils import violinplot
from .FigureCommon import getSetup, subplotLabel


def makeFigure():
    """This function generetes Figure 2.

    Args None
    Returns:
        A figure
    """

    # Get list of axis objects
    ax, f = getSetup((12, 8), (4, 5))

    for axis in ax[0:20]:
        axis.tick_params(axis="both", which="major", pad=-2)  # set ticks style

    # Blank out for cartoon
    for axis in ax[0:5] + ax[15:20]:
        axis.axis("off")

    # Show simulation plots (predicted vs experimental)
    simulationPlots(axes=[ax[5], ax[6], ax[7], ax[10], ax[11], ax[12]], swapDrugs=True)

    # Show violin plots for model parameters
    violinPlots(axes=[ax[13], ax[14], ax[8], ax[9]], swapDrugs=True)

    subplotLabel([ax[0], ax[5], ax[3], ax[8], ax[15], ax[17], ax[18]])

    return f


def simulationPlots(axes, ff="101117_H1299", swapDrugs=False):
    """ Make plots of experimental data. """

    # Load model and dataset
    classM = GrowthModel(loadFile=ff)
    classM.importData()

    df = pd.DataFrame(classM.expTable)

    df["time"] = np.tile(classM.timeV, int(df.shape[0] / classM.timeV.size))
    df["dose"] = np.repeat(classM.doses, classM.timeV.size).astype(np.float64)
    df["drug"] = np.repeat(classM.drugs, int(df.shape[0] / len(classM.drugs)))

    # Get drug names
    drugs = list(OrderedDict.fromkeys(classM.drugs))
    drugs.remove("Control")

    if swapDrugs:
        drugs = list(reversed(drugs))

    # help to name title
    quant_tt = ["Phase", "Annexin V", "YOYO-3"]

    # array of all time points
    times = np.unique(df["time"])

    for ii, ax in enumerate(axes):
        quant = ["confl", "apop", "dna"][ii % 3]

        if ii < 3:
            curDrug = drugs[1]
        else:
            curDrug = drugs[0]

        dfcur = df.loc[np.logical_or(df["drug"] == curDrug, df["drug"] == "Control"), :]

        if curDrug == "Erl":
            curDrug = "Erlotinib"

        # array of all doses for the drug
        doses = np.unique(dfcur["dose"])

        # take average of quant for all data points
        this_dfcur_avg = dfcur.groupby(["time", "dose", "drug"]).agg({quant: "mean"}).unstack(0)

        # subtratc ctrl for apop and dna
        if quant == "apop":  # apop (Annexin v)
            ctrl = np.array(this_dfcur_avg.apop.iloc[0])
            this_dfcur_avg.apop = this_dfcur_avg.apop - ctrl
        elif quant == "dna":  # dna (YOYO-3)
            ctrl = np.array(this_dfcur_avg.dna.iloc[0])
            this_dfcur_avg.dna = this_dfcur_avg.dna - ctrl

        # plot simulations
        quantile = 0.95
        palette = plt.get_cmap("tab10")  # color palette

        for k, dose in enumerate(doses):
            # plot simulations for each drug dose
            qt = this_dfcur_avg[quant].iloc[k]

            if quant == "confl":
                ax.plot(times, qt, color=palette(k), linewidth=1, alpha=0.9, label=str(round(float(dose), 1)))
            else:
                ax.plot(times, qt, color=palette(k), linewidth=1, alpha=0.9)

            # plot confidence intervals for simulations for each drug dose
            dfci = dfcur[dfcur.dose == dose].groupby("time")
            y_low = dfci[quant].quantile((1 - quantile) / 2).values
            y_high = dfci[quant].quantile(1 - (1 - quantile) / 2).values
            if quant != "confl":
                y_low = [a_i - b_i for a_i, b_i in zip(y_low, ctrl)]
                y_high = [a_i - b_i for a_i, b_i in zip(y_high, ctrl)]
            ax.fill_between(times, y_high, y_low, color=palette(k), alpha=0.2)

        # add legends
        if quant == "confl":
            if curDrug in ["Dox", "NVB", "Paclitaxel", "Erlotinib"]: # drugs with nM units
                title = "Doses (nM)"
            else:
                title = r"Doses ($\mu$M)"
            legend = ax.legend(loc=2, ncol=2, title=title, handletextpad=0.3, handlelength=0.5, columnspacing=0.5, prop={"size": 7})
            legend.get_title().set_fontsize("8")

        # set titles and labels
        ax.set_xlabel("Time (hr)")

        ax.set_title(quant_tt[ii % 3] + " (" + curDrug + ")")

        if quant == "confl":
            ax.set_ylim(-0.1, 100.0)
        else:
            if ff == "101117_H1299":
                ax.set_ylim(-0.05, 0.5)
            else:
                ax.set_ylim(-0.5, 5.0)

        ax.set_ylabel("Percent Image Positive")


def violinPlots(axes, ff="101117_H1299", remm=None, swapDrugs=False):
    """ Create violin plots of model posterior. """
    # Load model and dataset
    dfdict, drugs, _ = violinplot(ff, swapDrugs=swapDrugs)

    if remm is not None:
        drugs.remove(remm)

    # Plot params vs. drug dose
    for j, drug in enumerate(drugs):
        # Get drug
        dfplot = dfdict[drug]

        # Combine div and deathRate in one dataframe
        # Convert div and deathRate from log scale to linear
        dose = dfplot["dose"].to_numpy(dtype=np.float)
        df1 = pd.DataFrame(
            {
                "rate": np.append(dfplot["div"], dfplot["deathRate"]),
                "type": np.append(np.repeat("Division", len(dfplot)), np.repeat("Death", len(dfplot))),
                "dose": np.append(dose, dose),
            }
        )

        df2 = pd.DataFrame({"apopfrac": dfplot["apopfrac"], "dose": dose})
        df1 = df1.sort_values(by="dose")
        df2 = df2.sort_values(by="dose")

        # Iterate over each parameter in params
        for i, param in enumerate(["rate", "apopfrac"]):
            idx = 2 * j + i

            # Set y-axis confluence limits for each parameter
            if param == "rate":
                # Make violin plots
                sns.violinplot(x="dose", y="rate", hue="type", data=df1, ax=axes[idx], palette="Set2", linewidth=0.2)
                # Set legend
                axes[idx].legend(handletextpad=0.3, handlelength=0.8, prop={"size": 8})
                # Set y label
                axes[idx].set_ylabel(r"Rate (1/hr)")
                # Set ylim
                axes[idx].set_ylim(bottom=0)
            elif param == "apopfrac":
                # Make violin plots
                sns.violinplot(x="dose", y=param, data=df2, ax=axes[idx], color=sns.color_palette("Set2")[2], linewidth=0.2)
                # Set y label
                axes[idx].set_ylabel("Apopfrac")
                # Set ylim
                axes[idx].set_ylim([0, 1])

            # Set x labels
            if drug in ["Dox", "NVB", "Paclitaxel", "Erl"]:
                if drug == "Erl":
                    drug = "Erlotinib"

                axes[idx].set_xlabel(drug + " (nM)")
            else:
                axes[idx].set_xlabel(drug + r" ($\mu$M)")

            axes[idx].set_ylim(bottom=-0.002)
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=25, horizontalalignment="right")
