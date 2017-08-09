##! /usr/bin/env python3
#
#import matplotlib
#matplotlib.use('Agg')
#
#from grmodel.pymcGrowth import MultiSample
#from grmodel.sampleAnalysis import diagnostics, read_dataset
#
#a = MultiSample()
#
#print(a.loadCols(2))
#a.sample()
#a.save()
#
#cL = read_dataset()
#
#diagnostics(cL)

from grmodel.fitcurves import MultiDrugs
#from grmodel.utils import plot_dose_fits 


a = MultiDrugs(list(range(2,14)), ['Dox', 'NVB'], ['div', 'deathRate'])
a.get_tables()
curves = a.fitCurves()
print(curves)
#plot_dose_fits(list(range(2,14)), ['Dox', 'NVB'], ['div', 'deathRate'], curves)
    