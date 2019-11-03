"""
.. module:: Figure1

.. moduleauthor:: Rui Yan <rachelyan@ucla.edu>; Aaron Meyer <ameyer@ucla.edu>

This module generates Figure1 which motivates looking at cell death by showing that the cell death is not captured in existing measurements, and the markers of live cell number are insufficient to distinguish cell growth and death effect.
"""

import numpy as np
import pandas as pd
import numpy_indexed as npi
from matplotlib.ticker import FormatStrFormatter
from ..pymcDoseResponse import doseResponseModel
from .FigureCommon import getSetup, subplotLabel


def makeFigure():
    """This function generetes Figure 1.

    Args None
    Returns: A figure
    """
    # Build and read the PyMC3 model for dose response sampling
    M = doseResponseModel()
    M.sample()

    # Store the MCMC sampling priors to compute the lExp (fit celltiter quantitation), growthV (predicted growth rate) and deathV (predicted death rate) at each drug concentration.
    df = pd.DataFrame(
        {
            "IC50s": M.trace["IC50s"],
            "hill": M.trace["hill"],
            "Emin_growth": M.trace["Emin_growth"],
            "Emax_death": M.trace["Emax_death"],
            "Emax_growth": M.Emax_growth,
        }
    )

    # Plots arrangements
    # Get list of axis objects
    ax, f = getSetup((10, 5), (2, 4))
    # Set significant figures for xtick
    ax[2].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax[3].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax[4].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    # Empty several axes to put cartoons
    ax[0].axis("off")
    ax[1].axis("off")

    # Subplots
    # Fig. 1b plots the dose response measurement of H1299 cells to DOX:
    plot_exact_data(M, ax[2], ax[3])
    # Fig. 1c-e plot the fitting data to our does response model for lExp, growthV and deathV.
    plot_sampling_data(df, ax[3], ax[4], ax[5], ax[6])
    # Fig. 1g
    alphaFig(M, ax[7])

    # Add subplot labels
    ax.pop(1)
    subplotLabel(ax)

    return f


def plot_mean_and_CI(ax, _x, _y, confidence=True):
    """This helper function plots the mean and p% confidence interval for _y grouped by index _x.

    Args:
        _x (numpy array): Drug concentrations which group different observations _y.
        _y (numpy array): The data that we would like to find the mean and confidence interval of.
        p: the percentage of confidence interval.
    Returns: None

    """
    # Group _y by _x and find the mean, standard deviation of _y at each _x
    x_unique, y_mean = npi.group_by(_x).mean(_y)
    sample_size = npi.count(_x)[1]
    y_sem = npi.group_by(_x).std(_y)[1] / np.sqrt(sample_size)

    if not confidence:
        y_sem = None

    ax.errorbar(x=x_unique, y=y_mean, yerr=y_sem, fmt=".", color="black")


def plot_data_and_quantile(data, yvar, q, ax):
    """This helper function plots the median and q quantile for yvar in df2 grouped by xvar.

    Args:
        _x (numpy array): Drug concentrations which group different observations _y.
        _y (numpy array): The data that we would like to find the mean and confidence interval of.

    Returns: None

    """
    conc = np.array(list(data.groups.keys()))
    y_median = data.agg({yvar: "median"})

    # Plot the data _y vs. _x
    ax.plot(conc, y_median, color="b", linewidth=1, alpha=0.9)

    alphas = np.arange(0.2, 1.0, 0.8 / len(q))

    for i, qi in enumerate(q):
        y_low = np.array(data.quantile((1 - qi) / 2)[yvar])
        y_high = np.array(data.quantile(1 - (1 - qi) / 2)[yvar])

        ax.fill_between(conc, y_high, y_low, color="b", alpha=alphas[i], label=str(int(qi * 100)) + "% CI")


def plot_exact_data(M, ax2, ax3):
    """ Plot the data provided """
    X = np.array(M.drugCs)
    lObs = np.array(M.lObs)
    # Figure C: plot the mean and SEM of lObs at each concentration X
    plot_mean_and_CI(ax2, X, lObs)
    ax2.set_xlabel(r"$\mathregular{Log_{10}}$[DOX(nM)]")
    ax2.set_ylabel(r"Cell viability" + "\n" + r"normalized to untreated cells")
    ax2.set_ylim(0, 1.1)
    # Part of Figure D: Compare the sampling lExp with the exact data lObs
    plot_mean_and_CI(ax3, X, lObs, confidence=False)


def df_crossjoin(df1, df2):
    """
    Make a cross join (cartesian product) between two dataframes by using a constant temporary key.
    Also sets a MultiIndex which is the cartesian product of the indices of the input dataframes.
    See: https://github.com/pydata/pandas/issues/5401
    :param df1 dataframe 1
    :param df1 dataframe 2
    :param kwargs keyword arguments that will be passed to pd.merge()
    :return cross join of df1 and df2
    """
    df1["_tmpkey"] = 1
    df2["_tmpkey"] = 1

    res = pd.merge(df1, df2, on="_tmpkey").drop("_tmpkey", axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    return res


def plot_sampling_data(df, ax3, ax4, ax5, ax6):
    """ Check that MCMC actually fit the data provided """
    # Define drug concentrations x to test MCMC sampling data fit
    df1 = df_crossjoin(df, pd.DataFrame({"concentration": np.arange(-1.0, 3.0, 0.01)}))

    # Drug term since we're using constant IC50 and hill slope
    df1["drugTerm"] = 1.0 / (1.0 + np.power(10.0, (df1["IC50s"] - df1["concentration"]) * df1["hill"]))

    # Minimum drug term
    df1["controlDrugTerm"] = 1.0 / (1.0 + np.power(10.0, (df1["IC50s"] - np.min(df1["concentration"])) * df1["hill"]))

    # growthV = Emin_growth + (Emax_growth - Emin_growth) * drugTerm
    df1["growthV"] = df1["Emax_growth"] + ((df1["Emin_growth"] - df1["Emax_growth"]) * df1["drugTerm"])

    # Control growth rate
    gControl = df1["Emax_growth"] + ((df1["Emin_growth"] - df1["Emax_growth"]) * df1["controlDrugTerm"])

    # Range of growth effect
    df1["growthRange"] = df1["Emax_growth"] - df1["Emin_growth"]

    # _Assuming deathrate in the absence of drug is zero
    df1["deathV"] = df1["Emax_death"] * df1["drugTerm"]

    # Calculate the growth rate
    GR = df1["growthV"] - df1["deathV"]

    # Calculate the number of live cells, normalized to T=0
    df1["lExp"] = np.exp(GR * 72.0 - gControl * 72.0)

    df2 = df1.groupby(["concentration"])

    # Plot the median, 90%, 75% and 50% quantiles of lExp, growthV, and deathV:
    quantiles = [0.90, 0.75, 0.50]

    # lExp (Figure 1c)
    plot_data_and_quantile(df2, "lExp", quantiles, ax3)
    ax3.set_xlabel(r"$\mathregular{Log_{10}}$[DOX(nM)]")
    ax3.set_ylabel("Fit CellTiter quantitation")
    ax3.set_ylim(bottom=0.0)
    ax3.legend(loc=6)

    # growthV (Figure 1d)
    df1["growthV"] *= 24.0
    plot_data_and_quantile(df2, "growthV", quantiles, ax4)
    ax4.set_xlabel(r"$\mathregular{Log_{10}}$[DOX(nM)]")
    ax4.set_ylabel("Predicted growth rate (1/day)")
    ax4.set_ylim(bottom=0.0)
    ax4.legend(loc=6)

    # deathV (Figure 1e)
    df1["deathV"] *= 24.0
    plot_data_and_quantile(df2, "deathV", quantiles, ax5)
    ax5.set_xlabel(r"$\mathregular{Log_{10}}$[DOX(nM)]")
    ax5.set_ylabel("Predicted death rate (1/day)")
    ax5.legend(loc=6)

    # Figure G: Plot growth rate vs. death rate
    df["Emax_growth"] *= 24.0
    df["Emin_growth"] *= 24.0
    df["Emax_death"] *= 24.0
    ax6.scatter(x=df["Emax_growth"] - df["Emin_growth"], y=df["Emax_death"], color="b", s=1)
    ax6.set_xlim(0.0, 0.72)
    ax6.set_ylim(0.0, 0.72)
    ax6.set_xlabel("Drug growth effect (1/day)")
    ax6.set_ylabel("Drug death effect (1/day)")


def alphaFig(M, ax1):
    """ Explore the consequences of a growth- vs. death-centric drug. """
    drug_lnum_effect = 0.25

    # alpha = (R_g0 - R_gd) / R_dD ---- R_d0 is 0
    # -ln(drug_lnum_effect) / t = (1 + alpha) R_dD

    alpha = np.logspace(-2, 2)
    R_dD = -np.log(drug_lnum_effect) / M.time / (1 + alpha)
    R_gD = M.Emax_growth - R_dD * alpha

    cellDiv = R_gD * 72.0
    deadCells = R_dD * (np.exp((R_gD - R_dD) * 72.0) - 1) / (R_gD - R_dD)

    ax1.semilogx(alpha, deadCells, label="cum. # dead")
    ax1.set_xlabel(r"Growth/death effect ratio")
    ax1.set_ylabel("Quantity per starting cell")
    ax1.semilogx(alpha, cellDiv, "r", label="avg. divisions")
    ax1.legend(handlelength=0.5)
