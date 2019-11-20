"""
This creates Figure 4.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from ..sampleAnalysis import readModel
from .FigureCommon import getSetup, subplotLabel


def makeFigure():
    """ Generate Figure 4: This figure should show looking at cell death can
    tell something about the cells' responses to drug interactions that are
    not captured by the traditional cell number measurements. """

    # plot phase, green and red confl for three drug interactions
    ax, f = getSetup((10, 4), (2, 5))

    fittingPlots([ax[2], ax[4]], "050719_PC9_LCL_OSI", "LCL161", "OSI-906")
    fittingPlots([ax[7], ax[9]], "050719_PC9_PIM_OSI", "PIM447", "OSI-906")

    simPlots_comb("050719_PC9_LCL_OSI", ax[0:4], "LCL161", "OSI-906")
    simPlots_comb("050719_PC9_PIM_OSI", ax[5:9], "PIM447", "OSI-906")

    subplotLabel(ax)

    return f


def simPlots_comb(loadFile, axes, drug1, drug2):
    """ Output raw data plotting for Bliss additivity. """
    # Read model
    M = readModel(loadFile, model="interactionModel", drug1=drug1, drug2=drug2, fit=False)

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

    sns.heatmap(confldf, ax=axes[0], vmin=0.0, square=True)
    sns.heatmap(ddd.pivot(drug1, drug2, "apop"), ax=axes[3], vmin=0.0, square=True)

    confl = confldf.to_numpy()
    confl /= confl[0, 0]
    confl = 1.0 - confl

    assert np.all(confl >= 0.0) and np.all(confl <= 1.0)

    additive = (confl[:, 0][:, None] + confl[0, :][None, :]) - np.outer(confl[:, 0], confl[0, :])

    assert np.all(additive >= 0.0) and np.all(additive <= 1.0)

    confldf.iloc[:, :] = confl - additive

    sns.heatmap(confldf, ax=axes[1], cmap="PiYG", vmin=-0.5, vmax=0.5, square=True)


def fittingPlots(ax, loadFile, drug1, drug2):
    """ Plots of additive interaction fit. """
    # Read model from saved pickle file
    M = readModel(loadFile, model="interactionModel", drug1=drug1, drug2=drug2)

    resid = np.median(M.samples["conflResid"], axis=0).reshape(5, 7)
    sns.heatmap(resid, ax=ax[0], cmap="PiYG", vmin=-0.5, vmax=0.5, cbar=False, square=True)

    df1 = pd.DataFrame({"drug": drug1, "rate": "IC50_growth", "value": M.samples["IC50_growth"][:, 0]})
    df2 = pd.DataFrame({"drug": drug2, "rate": "IC50_growth", "value": M.samples["IC50_growth"][:, 1]})
    df3 = pd.DataFrame({"drug": drug1, "rate": "IC50_death", "value": M.samples["IC50_death"][:, 0]})
    df4 = pd.DataFrame({"drug": drug2, "rate": "IC50_death", "value": M.samples["IC50_death"][:, 1]})

    dfplot = pd.concat([df1, df2, df3, df4])
    dfplot["value"] = np.log10(dfplot["value"])

    sns.violinplot(x="drug", y="value", hue="rate", data=dfplot, ax=ax[1], linewidth=0.1)
    ax[1].set_ylabel(r'log_{10}(nM)')
