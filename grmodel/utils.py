"""
Various utility functions, probably mostly for plotting.
"""
from collections import OrderedDict
import pandas as pd
from .sampleAnalysis import readModel


def reformatData(dfd, alldoses, alldrugs, drug, params):
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

    # Interate over each dose
    # Columns: div, d, deathRate, apopfrac, dose
    for dose in doseidx:
        dftemp = pd.DataFrame()
        for param in params:
            dftemp[param] = dfd[param + "__" + str(doseidx[dose])]
        dftemp["dose"] = dose
        if "Data Type" in dfd.columns:
            dftemp["Data Type"] = dfd["Data Type"]
        dfplot = pd.concat([dfplot, dftemp], axis=0)

    return dfplot


def violinplot(filename, model="growthModel"):
    """
    Takes in a list of drugs
    Makes 1*len(parameters) violinplots for each drug
    """

    # Load model and dataset
    # Read in dataframe
    classM = readModel(filename, model=model)
    df = classM.df

    # Get a list of drugs
    drugs = set(classM.drugs)
    drugs.remove("Control")

    params = ["div", "deathRate", "apopfrac"]

    dfdict = {}

    # Interate over each drug
    for drug in drugs:
        dfplot = reformatData(df, classM.doses, classM.drugs, drug, params)

        dfdict[drug] = dfplot
    return (dfdict, drugs, params)
