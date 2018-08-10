import glob
from os.path import dirname, abspath
import numpy as np
import pandas as pd
import fcsparser


def importFCS():
    """ Import FCS files for CFSE dilution. """
    fullpath = dirname(abspath(__file__))
    files = glob.glob(fullpath + "/data/FCSE/*.fcs")

    assert len(files) > 0

    data = pd.DataFrame()

    for _, filename in enumerate(files):
        _, df = fcsparser.parse(filename)
        
        df['file'] = filename
        
        data = data.append(df)

    data = data.loc[data['SSC-A'] > 200, :] # Some very low side-scatter items were causing issues
    data = data.loc[data['FITC-A'] > 1, :]  # Get rid of nonsensical fluorescence values

    # Just keep data we're going to use
    data = data[['FSC-A', 'FSC-H', 'SSC-A', 'SSC-H', 'SSC-W', 'FITC-A', 'file']]

    # Strip filename so we can match it
    data['file'] = data['file'].str.replace(fullpath, '')
    data['file'] = data['file'].str.replace('/data/FCSE/', '')

    # Derive normalized FITC values
    data['sFITC'] = np.log(data['FITC-A']/data['SSC-W'])

    # Open file descriptions
    table = pd.read_csv(fullpath + '/data/FCSE/20171106.csv')

    # Merge file descriptions with measurements
    data = data.merge(table, on='file')

    return data
