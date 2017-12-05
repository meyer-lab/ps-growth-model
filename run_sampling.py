#! /usr/bin/env python3

import matplotlib
matplotlib.use('Agg')

from grmodel.pymcGrowth import MultiSample
from grmodel.sampleAnalysis import diagnostics, read_dataset

a = MultiSample()
print(a.loadModels(2))
a.sample()
a.save()

cL = read_dataset()

diagnostics(cL)
