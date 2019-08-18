"""
Various utility functions, probably mostly for plotting.
"""
from collections import OrderedDict
import numpy as np
import pandas as pd
import seaborn as sns
from .sampleAnalysis import readModel


def reformatData(dfd, alldoses, alldrugs, drug, params, dTypes=False):
    """
    Sample nsamples number of points from sampling results,
    Reformat dataframe so that columns of the dataframe are params
    Returns dataframe of columns: parameters and dose
    """
    # Set up ordered dictionary for dose:idx
    doseidx = OrderedDict()
    flag = True
    # Iterate from the last condition to the first condition
    for i in range(len(alldrugs) - 1, -1, -1):
        # Condition matches drug of interest
        if alldrugs[i] == drug:
            doseidx[alldoses[i]] = i
        # Include the first control after drug conditions
        elif alldrugs[i] == "Control" and flag and bool(doseidx):
            doseidx[alldoses[i]] = i
            flag = False
    # Put dictionary items in order of increasing dosage
    doseidx = OrderedDict(reversed(list(doseidx.items())))

    # Reshape table for violinplot
    # Columns: div, deathRate, apopfrac, dose
    # Initialize a dataframe
    dfplot = pd.DataFrame()
    # Randomly sample 1000 rows from pymc sampling results
    if dfd.shape[0] > 1000:
        dfd = dfd.sample(1000)

    # Interate over each dose
    # Columns: div, d, deathRate, apopfrac, dose
    for dose in doseidx:
        dftemp = pd.DataFrame()
        for param in params:
            dftemp[param] = dfd[param + "__" + str(doseidx[dose])]
        dftemp["dose"] = dose
        if dTypes:
            dftemp["Data Type"] = dfd["Data Type"]
        dfplot = pd.concat([dfplot, dftemp], axis=0)

    # Log transformation
    for param in ["div", "deathRate"]:
        dfplot[param] = dfplot[param].apply(np.log10)
    return dfplot


def violinplot(filename, drugs=None, model="growthModel"):
    """
    Takes in a list of drugs
    Makes 1*len(parameters) violinplots for each drug
    """

    # Load model and dataset
    # Read in dataframe
    classM, df = readModel(filename, model=model)
    alldrugs = classM.drugs
    alldoses = classM.doses

    # Get a list of drugs
    if drugs is None:
        drugs = set(classM.drugs)
        drugs.remove("Control")

    params = ["div", "deathRate", "apopfrac"]

    dfdict = {}

    # Interate over each drug
    for drug in drugs:
        dfplot = reformatData(df, alldoses, alldrugs, drug, params)

        dfdict[drug] = dfplot
    return (dfdict, drugs, params)


def violinplot_split(filename, axis, drugs=None):
    """
    Make split violin plots for comparison of sampling distributions from
    analyses of kinetic data and endpoint data.
    """
    # Read in model and kinetic dataframe
    classM, df = readModel(filename)

    # Append a Data Type variable to dataframe
    df["Data Type"] = "Kinetic"
    # Read in dataframe for endpoint data
    _, df2 = readModel(filename, interval=False)
    df2["Data Type"] = "Endpoints"
    # Concatinate the two data frames
    df = pd.concat([df, df2], axis=0)

    # Get variables from model
    alldrugs = classM.drugs
    alldoses = classM.doses
    # Get a list of drugs
    if drugs is None:
        drugs = list(sorted(set(classM.drugs)))
        drugs.remove("Control")

    params = ["div", "deathRate", "apopfrac"]

    # Interate over each drug
    for drug in drugs:
        dfplot = reformatData(df, alldoses, alldrugs, drug, params, dTypes=True)

        # Plot params vs. drug dose
        # Get drug index
        j = drugs.index(drug)
        # Iterate over each parameter in params
        for i in range(len(params)):
            # For apopfrac, set y-axis limit to [0,1]
            if params[i] == "apopfrac":
                axis[3 * i + j].set_ylim([0, 1])
            # Make violin plots
            sns.violinplot(x="dose", y=params[i], hue="Data Type", data=dfplot, palette="muted", split=True, ax=axis[3 * i + j], cut=0, linewidth=0.2)
            axis[3 * i + j].set_xlabel(drug + " dose")
            axis[3 * i + j].legend_.remove()
