"""
This creates Figure 4.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from ..pymcInteraction import drugInteractionModel
from .FigureCommon import getSetup, subplotLabel


def makeFigure():
    """ Generate Figure 4: This figure should show looking at cell death can
    tell something about the cells' responses to drug interactions that are
    not captured by the traditional cell number measurements. """

    # plot phase, green and red confl for three drug interactions
    ax, f = getSetup((10, 4), (2, 5))

    A = simPlots_comb("050719_PC9_LCL_OSI", ax[0:4], "LCL161", "OSI-906")
    B = simPlots_comb("050719_PC9_PIM_OSI", ax[5:9], "PIM447", "OSI-906")

    fittingPlots([ax[2], ax[4]], "050719_PC9_LCL_OSI", "LCL161", "OSI-906", A)
    fittingPlots([ax[7], ax[9]], "050719_PC9_PIM_OSI", "PIM447", "OSI-906", B)

    subplotLabel(ax)

    return f


def simPlots_comb(loadFile, axes, drug1, drug2):
    """ Output raw data plotting for Bliss additivity. """
    # Read model
    M = drugInteractionModel(loadFile, drug1=drug1, drug2=drug2, fit=False)

    drug1 = drug1 + r" ($\mu$M)"
    drug2 = drug2 + r" ($\mu$M)"

    dfplot = pd.DataFrame()
    dfplot["confl"] = M.phase.flatten()
    dfplot["apop"] = M.green.flatten()
    dfplot["dna"] = M.red.flatten()
    dfplot["time"] = np.tile(M.timeV, M.X1.size)
    dfplot[drug1] = np.round(np.repeat(M.X1, M.timeV.size), decimals=1)
    dfplot[drug2] = np.round(np.repeat(M.X2, M.timeV.size), decimals=1)

    ddd = dfplot.loc[dfplot["time"] == 72.0, :]
    ddd = ddd.groupby([drug1, drug2, "time"]).mean().reset_index()
    confldf = ddd.pivot(drug1, drug2, "confl")

    sns.heatmap(confldf, ax=axes[0], vmin=0.0, square=True, xticklabels=1)
    axes[0].set_title("Phase")
    sns.heatmap(ddd.pivot(drug1, drug2, "apop"), ax=axes[3], vmin=0.0, square=True, xticklabels=1)
    axes[3].set_title("Annexin V")

    confl = confldf.to_numpy()
    confl /= confl[0, 0]
    confl = 1.0 - confl

    assert np.all(confl >= 0.0) and np.all(confl <= 1.0)

    additive = (confl[:, 0][:, None] + confl[0, :][None, :]) - np.outer(confl[:, 0], confl[0, :])

    assert np.all(additive >= 0.0) and np.all(additive <= 1.0)

    confldf.iloc[:, :] = confl - additive

    sns.heatmap(confldf, ax=axes[1], cmap="PiYG", vmin=-0.5, vmax=0.5, square=True, xticklabels=1)
    axes[1].set_title("Just Viability")

    return confldf


def fittingPlots(ax, loadFile, drug1, drug2, df):
    """ Plots of additive interaction fit. """
    # Read model from saved pickle file
    M = drugInteractionModel(loadFile, drug1=drug1, drug2=drug2, fit=True)

    df.iloc[:, :] = np.median(M.samples["conflResid"], axis=0).reshape(5, 7)

    sns.heatmap(df, ax=ax[0], cmap="PiYG", vmin=-0.5, vmax=0.5, cbar=False, square=True)
    ax[0].set_title("Full Model")

    df1 = pd.DataFrame({"drug": drug1, "param": "IC50 [mM]", "value": M.samples["IC50"][:, 0] / 1000.0})
    df2 = pd.DataFrame({"drug": drug2, "param": "IC50 [mM]", "value": M.samples["IC50"][:, 1] / 1000.0})
    df3 = pd.DataFrame({"drug": drug1, "param": "Growth Emax [1/hr]", "value": M.samples["EmaxGrowthEffect"][:, 0]})
    df4 = pd.DataFrame({"drug": drug2, "param": "Growth Emax [1/hr]", "value": M.samples["EmaxGrowthEffect"][:, 1]})
    df5 = pd.DataFrame({"drug": drug1, "param": "Death Emax [1/hr]", "value": M.samples["EmaxDeath"][:, 0]})
    df6 = pd.DataFrame({"drug": drug2, "param": "Death Emax [1/hr]", "value": M.samples["EmaxDeath"][:, 1]})

    dfplot = pd.concat([df1, df2, df3, df4, df5, df6])
    dfplot["value"] = np.log10(dfplot["value"])

    sns.violinplot(x="param", y="value", hue="drug", data=dfplot, ax=ax[1], linewidth=0.1)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=25, horizontalalignment="right")
    ax[1].set_ylabel(r"Log$_{10}$[Value]")
    ax[1].set_ylim(-2.0, -0.5)
    ax[1].set_xlabel("")

    # Remove legend title
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles=handles, labels=labels)
