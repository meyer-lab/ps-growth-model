"""
This module handles reading drug combination data.
"""
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
