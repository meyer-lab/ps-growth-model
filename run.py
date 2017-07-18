#! /usr/bin/env python3

from grmodel.pymcGrowth import MultiSample

a = MultiSample()

print(a.loadCols(2))
a.sample()
filename = './grmodel/data/062117_second_chain.h5'
a.save(filename)