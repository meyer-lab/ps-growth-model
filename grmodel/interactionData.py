"""
This module handles reading drug combination data.
"""
import re
import numpy as np
import pandas as pd
from os.path import join, dirname, abspath


def readCombo(name='BYLvPIM'):
    ''' Read in data file, melt each table across conditions, and then merge all measurements into one table. '''
    filename = join(dirname(abspath(__file__)), 'data/combinations/' + name + '_raw.xlsx')

    data = pd.read_excel(filename, sheet_name=None)

    del data['Conditions']

    for key in data:
        data[key]['Well'] = np.repeat([1, 2, 3], 25)
        data[key] = pd.melt(data[key], id_vars=['Elapsed', 'Well'], var_name='Condition', value_name='Measure')
        data[key].dropna(inplace=True)

    data = pd.concat(data).reset_index().drop(['level_1'], axis=1)

    data['Type'], data['drug_amt'] = data['level_0'].str.split('_', 1).str

    data.drop(['level_0'], axis=1, inplace=True)

    return data


def filterDrugC(df, drugAname, drugBname):
    df['drugA'] = 0

    for index, row in df.iterrows():
        m = re.search(drugAname + ' (\d*\.?\d*)', row['Condition'])

        if m is not None:
            df.loc[index, 'drugA'] = float(m.group(1))

    df['drugB'] = 0

    for index, row in df.iterrows():
        m = re.search(drugBname + ' (\d*\.?\d*)', row['Condition'])

        if m is not None:
            df.loc[index, 'drugB'] = float(m.group(1))

    df.loc[df['Condition'] == 'blank', 'drugA'] = np.nan
    df.loc[df['Condition'] == 'blank', 'drugB'] = np.nan

    df.drop('Condition', axis=1, inplace=True)

    return df


def dataSplit(df, timepoint_start=0):
    ''' This will pull out the data at certain timepoints '''
    keepCols = ['drugA', 'drugB', 'Elapsed', 'Measure']
    grpCols = ['Elapsed', 'drugA', 'drugB']
    df.dropna(inplace=True)

    timeV = np.sort(np.array(df.Elapsed.unique(), dtype=np.float64))

    if timepoint_start != 0:
        # Take the subset of timeV
        try:
            idx_start = np.where(timeV == timepoint_start)[0][0]
            timeV = timeV[idx_start:]
            # Take the subset of df
            df = df.loc[df['Elapsed'].isin(timeV), :]
        except IndexError:
            print('wrong timepoint to start')

    dfPhase = df.loc[df['Type'] == 'phase', :]
    dfRed = df.loc[df['Type'] == 'red', :]
    dfGreen = df.loc[df['Type'] == 'green', :]

    dfMAT = dfPhase[keepCols].groupby(grpCols).agg({"Measure": "mean"}).unstack(0)

    phase = dfMAT.values

    red = dfRed[keepCols].groupby(grpCols).agg({"Measure": "mean"}).unstack(0).values
    green = dfGreen[keepCols].groupby(grpCols).agg({"Measure": "mean"}).unstack(0).values

    dfMAT.reset_index(inplace=True)

    X1 = dfMAT['drugA'].values + 0.01
    X2 = dfMAT['drugB'].values + 0.01

    assert(phase.shape == red.shape)
    assert(phase.shape == green.shape)

    return (X1, X2, timeV, phase, red, green)
