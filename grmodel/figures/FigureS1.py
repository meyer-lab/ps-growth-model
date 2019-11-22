""" This creates Figure S1. """

from collections import OrderedDict
import seaborn as sns
import pandas as pd
from .FigureCommon import getSetup, subplotLabel
from ..pymcGrowth import GrowthModel
from ..utils import reformatData


def makeFigure():
    """ Make figure S1. """

    # Get list of axis objects
    ax, f = getSetup((7, 5), (2, 3))

    subplotLabel(ax)

    violinplot_split("101117_H1299", ax)

    for axis in ax:
        axis.set_xticklabels(axis.get_xticklabels(), rotation=25, horizontalalignment="right")

    return f


def violinplot_split(filename, axis):
    """
    Make split violin plots for comparison of sampling distributions from
    analyses of kinetic data and endpoint data.
    """
    # Read in model and kinetic dataframe
    classM = GrowthModel(loadFile=filename)
    classM.importData()
    classM.performFit()
    df = classM.df
    df["Data Type"] = "Kinetic"

    # Read in dataframe for endpoint data
    classM2 = GrowthModel(loadFile=filename)
    classM2.importData(interval=False)
    classM2.performFit()
    df2 = classM2.df
    df2["Data Type"] = "Endpoints"

    # Concatinate the two data frames
    df = pd.concat([df, df2], axis=0)

    # Get a list of drugs
    drugs = list(OrderedDict.fromkeys(classM.drugs))
    drugs.remove("Control")

    params = ["div", "deathRate", "apopfrac"]

    # Interate over each drug
    for j, drug in enumerate(drugs):
        dfplot = reformatData(df, classM.doses, classM.drugs, drug, params)

        # Iterate over each parameter in params
        for i, param in enumerate(params):
            # Make violin plots
            sns.violinplot(x="dose", y=param, hue="Data Type", data=dfplot, palette="muted", split=True, ax=axis[3 * j + i], cut=0, linewidth=0.2)
            axis[3 * j + i].set_xlabel(drug + " (nM)")

        axis[3 * j].set_ylabel("Division Rate (1/hr)")
        axis[3 * j + 1].set_ylabel("Death Rate (1/hr)")
        axis[3 * j + 2].set_ylabel("Apoptosis Fraction")

    for i, ax in enumerate(axis):
        ax.set_ylim(bottom=0.0)

        if i > 0:
            ax.legend_.remove()

    axis[2].set_ylim(top=1.0)
    axis[5].set_ylim(top=1.0)

    axis[1].set_ylim(top=0.07)
    axis[4].set_ylim(top=0.07)

    axis[0].set_ylim(top=0.04)
    axis[3].set_ylim(top=0.04)
